"""Encode text with one cached SentenceTransformers model, over stdio.

This script deliberately imports nothing from SparkDeck and nothing outside the
standard library at module scope: it runs under the private embedding virtual
environment, which holds only ``sentence-transformers`` and its dependencies,
and it is executed as a file rather than as ``sparkdeck.embedding_worker`` so
importing it never pulls in the server's own package.

The protocol is newline-delimited JSON. The first line is a handshake
(``{"ready": true, "dimension": 384}``), then one request per line
(``{"id": 0, "inputs": ["hello world"], "normalize": true}``) answered by one
line carrying either ``embeddings`` or ``error``. The operating system's
standard output is kept for this protocol alone; ``sys.stdout`` is repointed at
stderr so anything a library decides to print lands in the worker log instead of
desynchronizing the stream.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# Keep the protocol channel private before importing anything that might print.
_PROTOCOL = sys.stdout
sys.stdout = sys.stderr

# Embedding models share the machine with the LLM deployments that SparkDeck
# exists to run, and a PyPI ``torch`` wheel ships CUDA support by default, so
# tensors would otherwise land in the VRAM an inference server is using. CPU
# encodes short passages in milliseconds; an operator who wants the GPU can ask
# for it explicitly.
_DEFAULT_DEVICE = "cpu"


def _emit(payload: dict[str, Any]) -> None:
    _PROTOCOL.write(json.dumps(payload, allow_nan=False) + "\n")
    _PROTOCOL.flush()


class InputRejected(ValueError):
    """The request cannot be encoded as asked; the caller must change it."""


def _prompted_texts(model: Any, inputs: list[str]) -> list[str]:
    """Reproduce the text the model will actually tokenize.

    A pipeline can carry a default prompt, and ``encode`` prepends it before
    tokenizing. Counting the raw inputs would then undercount by the prompt's
    length, so an input that fits on paper can still be truncated in practice.
    """
    name = getattr(model, "default_prompt_name", None)
    prompts = getattr(model, "prompts", None)
    if name and isinstance(prompts, dict):
        prompt = prompts.get(name)
        if isinstance(prompt, str) and prompt:
            return [prompt + text for text in inputs]
    return list(inputs)


def _input_token_counts(model: Any, inputs: list[str]) -> list[int] | None:
    """Tokenize every input as it will be encoded, or None without a tokenizer."""
    try:
        encoded = model.tokenizer(
            _prompted_texts(model, inputs), truncation=False, padding=False,
        )["input_ids"]
    except Exception:
        return None
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        return [len(ids) for ids in encoded]
    return [len(encoded)] if isinstance(encoded, list) else None


def _check_token_window(model: Any, counts: list[int] | None) -> None:
    """Refuse inputs that the model would silently truncate.

    ``SentenceTransformer.encode`` truncates every text to the pipeline's
    ``max_seq_length`` without saying so, which would answer a long document
    with the vector of its opening fragment and quietly corrupt whatever
    retrieval index consumed it. A caller can chunk deliberately; SparkDeck
    must not decide that on its behalf.
    """
    limit = getattr(model, "max_seq_length", None)
    if not isinstance(limit, int) or limit <= 0 or counts is None:
        return
    oversized = [index for index, count in enumerate(counts) if count > limit]
    if oversized:
        raise InputRejected(
            f"input {oversized[0]} has {counts[oversized[0]]} tokens, beyond this "
            f"model's {limit}-token window; chunk long text before embedding it"
        )


def _encode(model: Any, request: dict[str, Any]) -> dict[str, Any]:
    inputs = request.get("inputs")
    if not isinstance(inputs, list) or not inputs or any(
        not isinstance(item, str) for item in inputs
    ):
        raise InputRejected("inputs must be a non-empty array of strings")
    normalize = request.get("normalize", True)
    if not isinstance(normalize, bool):
        raise InputRejected("normalize must be a boolean")
    counts = _input_token_counts(model, inputs)
    _check_token_window(model, counts)
    vectors = model.encode(
        inputs,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return {
        "id": request.get("id"),
        "embeddings": vectors.tolist(),
        "prompt_tokens": sum(counts) if counts is not None else 0,
    }


def main() -> int:
    if len(sys.argv) != 2:
        _emit({"ready": False, "error": "usage: embedding_worker.py <snapshot-dir>"})
        return 2
    snapshot = sys.argv[1]
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            snapshot,
            device=os.environ.get("SPARKDECK_EMBEDDING_DEVICE", _DEFAULT_DEVICE),
            # A cached repository is data, never code: remote-module execution
            # stays off even for a snapshot the operator downloaded themselves.
            trust_remote_code=False,
        )
    except Exception as exc:  # noqa: BLE001 - reported to the parent verbatim
        _emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"[:800]})
        return 1

    dimension = 0
    try:
        dimension = int(model.get_sentence_embedding_dimension() or 0)
    except Exception:
        pass
    _emit({"ready": True, "dimension": dimension})

    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        request: Any = None
        try:
            request = json.loads(text)
            if not isinstance(request, dict):
                raise InputRejected("request must be a JSON object")
            _emit(_encode(model, request))
        except Exception as exc:  # noqa: BLE001 - one bad request must not kill the worker
            # The kind tells the parent whether the caller must change its
            # request or the cluster merely failed, which is the difference
            # between a 400 and a retryable server error.
            kind = "input" if isinstance(exc, InputRejected) else "runtime"
            _emit({
                "id": request.get("id") if isinstance(request, dict) else None,
                "kind": kind,
                "error": f"{type(exc).__name__}: {exc}"[:800],
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
