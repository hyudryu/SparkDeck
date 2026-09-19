"""OpenAI-compatible decision server for Laya System 1 models.

Laya is not a text generator: it scores typed questions over a state in one
forward pass and returns calibrated probabilities. This module puts that
decision engine behind the same ``/v1`` surface SparkDeck already proxies, so
``POST http://<node>:7878/v1/chat/completions`` reaches Laya through the normal
cluster router with no special-case code in the router itself.

Request contract (several accepted spellings, checked in this order):

1. ``{"state": ..., "questions": {...}}`` as top-level body fields.
2. A ``system`` (or single ``user``) message whose text is a JSON object
   containing ``state`` and ``questions``.
3. ``presets`` naming built-in Laya question sets, layered over any state.

The assistant message carries the Laya result as compact JSON. ``usage``
reports Laya's real input-token count, which keeps SparkDeck's token
accounting and benchmark history meaningful for a non-generating model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("laya-decide")

OBJECT_NAME = "laya-decide"
QUESTION_TYPES = frozenset({"choice", "score", "noul"})
PRESET_NAMES = ("router", "guard", "moderation", "triage")
MAX_QUESTIONS = 256
MAX_STATE_CHARS = 4_000_000
# Questions are tokenized with the state, so they need their own bound: a single
# question can carry a very large instruction or rubric without exceeding the
# question-count limit.
MAX_QUESTIONS_CHARS = 1_000_000
MAX_STATE_DEPTH = 64

# Set by the container entrypoint. ``WEIGHTS`` is the repository or local path
# Laya loads; ``MODEL_ID`` is the OpenAI model id published by ``/v1/models``.
# SparkDeck discovers the served name from that response, so an operator who
# sets ``--served-model-name`` gets the alias they deployed under and everyone
# else keeps the Hugging Face repository as the request id.
WEIGHTS = os.environ.get("LAYA_MODEL") or "convaiinnovations/laya"
MODEL_ID = (os.environ.get("LAYA_SERVED_MODEL_NAME") or "").strip() or WEIGHTS
MODEL_DEVICE = (os.environ.get("LAYA_DEVICE") or "").strip() or None
MODEL_REVISION = (os.environ.get("LAYA_REVISION") or "").strip() or None
SERVE_PORT = int(os.environ.get("LAYA_PORT") or 8080)

# Opt-in multilingual routing. Off by default: a single checkpoint has a much
# smaller footprint, and the English checkpoint is the strongest on English.
ROUTER_ENABLED = (os.environ.get("LAYA_ROUTER") or "").strip().casefold() in {
    "1", "true", "yes", "on",
}
ROUTER_DEFAULT = (os.environ.get("LAYA_ROUTER_DEFAULT") or "english").strip()


def _router_max_loaded() -> int:
    try:
        return max(1, int(os.environ.get("LAYA_ROUTER_MAX_LOADED") or 1))
    except (TypeError, ValueError):
        return 1


ROUTER_MAX_LOADED = _router_max_loaded()


class DecisionRequestError(ValueError):
    """A caller-supplied decision payload could not be evaluated."""


def _max_concurrency() -> int:
    try:
        return max(1, int(os.environ.get("LAYA_MAX_CONCURRENCY") or 1))
    except (TypeError, ValueError):
        return 1


_agent: Any = None
_agent_lock = asyncio.Lock()
_inference_gate: asyncio.Semaphore | None = None
# Resolved once from the ``laya`` package so preset validation stays eager and
# a caller can install its own factories (tests, or a future preset source)
# without importing torch.
_preset_factories: dict[str, Any] | None = None


def set_agent(agent: Any) -> None:
    """Install a decision agent. Tests use this to avoid loading torch."""
    global _agent
    _agent = agent


def _preset_factory(name: str) -> Any:
    """Return the question factory for a built-in preset name."""
    global _preset_factories
    if _preset_factories is None:
        import laya

        _preset_factories = {
            "router": laya.router_questions,
            "guard": laya.guard_questions,
            "moderation": laya.moderation_questions,
            "triage": laya.triage_questions,
        }
    return _preset_factories.get(name)


def get_agent() -> Any:
    # Read the module global directly so a test or embedder that replaces
    # ``_agent`` after import is honoured without re-importing the module.
    agent = globals()["_agent"]
    if agent is None:
        raise HTTPException(503, "Laya model is still loading")
    return agent


def _resolve_weights() -> str:
    """Return the path or repository id to load, honouring a revision pin.

    Laya's own loader has no revision argument and always takes the current
    default revision, so a pinned launch resolves the exact snapshot here first
    and hands Laya the resulting local directory. SparkDeck appends
    ``--revision`` for a cached bookmark launch, and silently loading a newer
    snapshot than the operator pinned would be worse than failing.
    """
    if not MODEL_REVISION:
        return WEIGHTS
    if os.path.exists(WEIGHTS):
        # A local directory is already an exact, immutable checkout.
        return WEIGHTS
    from huggingface_hub import snapshot_download

    logger.info("resolving %s at revision %s", WEIGHTS, MODEL_REVISION)
    return snapshot_download(
        WEIGHTS,
        revision=MODEL_REVISION,
        token=os.environ.get("HF_TOKEN") or None,
    )


def _load_agent() -> Any:
    """Import Laya lazily so the HTTP surface is usable without torch."""
    if ROUTER_ENABLED and MODEL_REVISION:
        # The router resolves each of its own checkpoints, so the single
        # revision this deployment was pinned to cannot apply to all of them,
        # and passing it as the English repository path would break every routed
        # request. An explicit failure beats loading a commit the operator did
        # not ask for. Checked before importing Laya so a misconfigured launch
        # reports the configuration problem rather than a missing dependency.
        raise RuntimeError(
            "a revision pin cannot be combined with router mode: the router "
            "resolves its own checkpoints. Remove --revision, or pin a revision "
            "by deploying the specific checkpoint repository without --router."
        )

    import laya

    started = time.monotonic()
    if ROUTER_ENABLED:
        # The English checkpoint does not degrade off English, it collapses and
        # stays confident while doing so, so an operator serving mixed-language
        # traffic should route before the forward pass. The shipped English
        # checkpoint stays the default because it is the strongest on English.
        agent = laya.Router(
            device=MODEL_DEVICE,
            max_loaded=ROUTER_MAX_LOADED,
            default=ROUTER_DEFAULT,
        )
        logger.info(
            "Laya router ready (max_loaded=%d, default=%s) in %.1fs",
            ROUTER_MAX_LOADED, ROUTER_DEFAULT, time.monotonic() - started,
        )
        return agent
    source = _resolve_weights()
    logger.info("loading Laya model from %s", source)
    agent = laya.load(source, device=MODEL_DEVICE)
    logger.info(
        "Laya model %s ready on %s in %.1fs",
        WEIGHTS, getattr(agent, "device", "unknown"), time.monotonic() - started,
    )
    return agent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _inference_gate
    _inference_gate = asyncio.Semaphore(_max_concurrency())
    set_agent(await asyncio.to_thread(_load_agent))
    yield
    set_agent(None)


app = FastAPI(title=OBJECT_NAME, version="1.0", lifespan=lifespan)


# ---------------------------------------------------------------- request parsing


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # OpenAI content parts: keep only text segments.
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text")
        )
    return ""


def _embedded_payload(messages: Any) -> dict[str, Any]:
    """Read the decision payload from a JSON chat message if one is present."""
    if not isinstance(messages, list):
        return {}
    selected: list[dict[str, Any]] = []
    users: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "system":
            selected.append(message)
        elif message.get("role") == "user":
            users.append(message)
    # A system message is the documented carrier and always wins. Within the
    # fallback, the newest user turn is the caller's current request, so earlier
    # turns of the same conversation must not supply stale state or questions.
    selected.extend(reversed(users))
    for message in selected:
        text = _message_text(message.get("content")).strip()
        if not text.startswith("{"):
            continue
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict) and (
            "questions" in decoded or "presets" in decoded
        ):
            return decoded
    return {}


def _preset_questions(presets: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(presets, list) or not presets:
        raise DecisionRequestError("presets must be a non-empty array of names")
    combined: dict[str, dict[str, Any]] = {}
    for raw in presets:
        name = str(raw or "").strip().lower()
        if name not in PRESET_NAMES:
            raise DecisionRequestError(
                f"unknown preset {name!r}; expected one of {', '.join(PRESET_NAMES)}"
            )
        factory = _preset_factory(name)
        if factory is None:
            raise DecisionRequestError(f"preset {name!r} is unavailable")
        generated = factory()
        if not isinstance(generated, dict):
            raise DecisionRequestError(f"preset {name!r} produced no questions")
        for question_id, definition in generated.items():
            # Namespace preset ids so two presets can be combined in one call
            # without silently overwriting each other.
            key = question_id if question_id not in combined else f"{name}_{question_id}"
            combined[key] = definition
    return combined


def _validated_questions(questions: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(questions, dict) or not questions:
        raise DecisionRequestError(
            "questions must be a non-empty object of typed question definitions"
        )
    if len(questions) > MAX_QUESTIONS:
        raise DecisionRequestError(f"questions cannot exceed {MAX_QUESTIONS} entries")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_id, definition in questions.items():
        question_id = str(raw_id or "").strip()
        if not question_id:
            raise DecisionRequestError("question ids must be non-empty strings")
        if not isinstance(definition, dict):
            raise DecisionRequestError(f"question {question_id!r} must be an object")
        question_type = str(definition.get("type") or "").strip().lower()
        if question_type not in QUESTION_TYPES:
            raise DecisionRequestError(
                f"question {question_id!r} has unsupported type {question_type!r}; "
                f"expected one of {', '.join(sorted(QUESTION_TYPES))}"
            )
        instructions = definition.get("instructions")
        if not isinstance(instructions, (str, dict, list)) or instructions == "":
            raise DecisionRequestError(
                f"question {question_id!r} requires instructions"
            )
        criteria = definition.get("criteria")
        if question_type == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise DecisionRequestError(
                    f"choice question {question_id!r} requires a criteria object "
                    "mapping option -> description"
                )
        elif question_type == "score":
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise DecisionRequestError(
                    f"score question {question_id!r} requires an ordered criteria "
                    "array of at least two levels"
                )
        elif criteria is not None:
            raise DecisionRequestError(
                f"noul question {question_id!r} does not take criteria"
            )
        normalized[question_id] = {
            "type": question_type,
            "instructions": instructions,
            "criteria": criteria,
        }
    return normalized


def _payload_size(value: Any, *, label: str, budget: int) -> int:
    """Measure a request payload without trusting its shape.

    A JSON object or array is the documented ticket/JSON use case, so the bound
    has to apply to it too: measuring only strings would let a caller hand an
    arbitrarily large nested value to the tokenizer. The same walk bounds the
    questions, because a single question can carry a huge instruction, option
    description, or rubric while staying well inside the question-count limit.
    The walk is bounded by depth and by the running total, so a pathological
    structure cannot make the check itself expensive.
    """
    stack: list[tuple[Any, int]] = [(value, 0)]
    total = 0
    while stack:
        item, depth = stack.pop()
        if depth > MAX_STATE_DEPTH:
            raise DecisionRequestError(f"{label} is nested too deeply")
        if isinstance(item, str):
            total += len(item)
        elif isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise DecisionRequestError(f"{label} object keys must be strings")
            for key, nested in item.items():
                total += len(key)
                stack.append((nested, depth + 1))
        elif isinstance(item, (list, tuple)):
            for nested in item:
                stack.append((nested, depth + 1))
        elif isinstance(item, bool) or item is None:
            total += 4
        elif isinstance(item, (int, float)):
            total += 24
        else:
            raise DecisionRequestError(
                f"{label} may only contain strings, numbers, booleans, null, "
                "objects, and arrays"
            )
        if total > budget:
            raise DecisionRequestError(f"{label} is too large")
    return total


def _state_size(state: Any) -> int:
    return _payload_size(state, label="state", budget=MAX_STATE_CHARS)


def _questions_size(questions: dict[str, dict[str, Any]]) -> int:
    return _payload_size(questions, label="questions", budget=MAX_QUESTIONS_CHARS)


def _decision_payload(body: dict[str, Any]) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Resolve the state and typed questions for one decision request."""
    embedded = _embedded_payload(body.get("messages"))
    state = body.get("state", embedded.get("state"))
    questions = body.get("questions", embedded.get("questions"))
    presets = body.get("presets", embedded.get("presets"))

    if questions is None and presets is not None:
        questions = _preset_questions(presets)
    elif questions is not None and presets is not None:
        merged = _preset_questions(presets)
        merged.update(_validated_questions(questions))
        questions = merged

    if state is None:
        raise DecisionRequestError(
            "state is required: pass the text, email, ticket, or JSON document "
            "to decide over"
        )
    if not isinstance(state, (str, dict, list)):
        raise DecisionRequestError("state must be a string, object, or array")
    _state_size(state)
    validated = _validated_questions(questions)
    # A question's instruction, option descriptions, and rubric are tokenized
    # alongside the state, so the question-count limit alone does not bound the
    # work one request can ask for.
    _questions_size(validated)
    return state, validated


def _routing_overrides(body: dict[str, Any]) -> dict[str, Any]:
    """Per-request routing overrides, honoured only when the router is enabled.

    Milestone: an operator who pinned one checkpoint asked for that checkpoint,
    so an override must never quietly load a different one. Rejecting it is
    better than ignoring it.
    """
    routing = body.get("routing")
    if routing is None:
        return {}
    if not isinstance(routing, dict):
        raise DecisionRequestError("routing must be an object")
    unknown = sorted(set(routing) - {"model", "task", "lang"})
    if unknown:
        raise DecisionRequestError(
            f"unsupported routing field(s): {', '.join(unknown)}; "
            "expected model, task, or lang"
        )
    if not ROUTER_ENABLED:
        raise DecisionRequestError(
            "this deployment serves a single checkpoint, so routing overrides "
            "are unavailable; start it with --router to enable multilingual "
            "routing"
        )
    return routing


def _decide(
    agent: Any, state: Any, questions: dict[str, dict[str, Any]],
    routing: dict[str, Any],
) -> dict[str, Any]:
    """Run one decision, routing first when the deployment asked for it."""
    if ROUTER_ENABLED:
        return agent.predict(state, questions, **routing)
    return agent.predict(state, questions)


def _render_content(result: dict[str, Any], include_state: bool) -> str:
    payload = {
        "answers": result.get("answers") or {},
        "usage": result.get("usage") or {},
    }
    if include_state:
        payload["model"] = result.get("model") or MODEL_ID
    # A routed deployment reports which checkpoint answered, and why, so a
    # caller can see that a non-English state did not go to the English model.
    if result.get("routing"):
        payload["routing"] = result["routing"]
    return json.dumps(payload, separators=(",", ":"), sort_keys=False)


def _completion(
    body: dict[str, Any], result: dict[str, Any],
) -> dict[str, Any]:
    usage = result.get("usage") or {}
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    answer_text = _render_content(result, include_state=True)
    extension = {
        "model": result.get("model") or MODEL_ID,
        "answers": result.get("answers") or {},
    }
    if result.get("routing"):
        extension["routing"] = result["routing"]
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(body.get("model") or MODEL_ID),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": answer_text,
                # Laya returns structure, not prose. This extension keeps the
                # probabilities and confidences typed for callers that can use
                # them, while `content` stays valid OpenAI text.
                "laya": extension,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "laya": extension,
    }


def _stream_chunks(body: dict[str, Any], completion: dict[str, Any]):
    """Replay a finished decision as OpenAI SSE data-only chunks."""
    chunk_id = completion["id"]
    created = completion["created"]
    model = completion["model"]
    usage = completion["usage"]

    def frame(delta: dict[str, Any], finish: str | None) -> str:
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    message = completion["choices"][0]["message"]
    yield frame({"role": "assistant"}, None)
    yield frame({"content": message["content"], "laya": message["laya"]}, "stop")
    # A final usage-only frame matches the include_usage convention SparkDeck
    # already enables for streaming requests.
    tail = {
        "id": chunk_id, "object": "chat.completion.chunk", "created": created,
        "model": model, "choices": [], "usage": usage,
    }
    yield f"data: {json.dumps(tail)}\n\n"
    yield "data: [DONE]\n\n"


# ------------------------------------------------------------------- HTTP surface


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, Any]:
    return {"status": "ok" if _agent is not None else "loading", "model": MODEL_ID}


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": 0,
            "owned_by": OBJECT_NAME,
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(400, "request body is not valid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be a JSON object")

    try:
        state, questions = _decision_payload(body)
        routing = _routing_overrides(body)
    except DecisionRequestError as exc:
        raise HTTPException(400, str(exc)) from exc

    agent = get_agent()
    gate = _inference_gate

    def decide() -> dict[str, Any]:
        result = _decide(agent, state, questions, routing)
        if not isinstance(result, dict) or "answers" not in result:
            raise HTTPException(500, "Laya returned an unexpected result")
        return result

    try:
        if gate is None:
            result = await asyncio.to_thread(decide)
        else:
            async with gate:
                result = await asyncio.to_thread(decide)
    except DecisionRequestError as exc:
        raise HTTPException(400, str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Laya decision failed")
        raise HTTPException(500, f"Laya decision failed: {exc}") from exc

    completion = _completion(body, result)
    if body.get("stream"):
        return StreamingResponse(
            _stream_chunks(body, completion),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(completion)


@app.post("/laya/predict")
async def predict(request: Request):
    """Laya-native decision endpoint: typed answers without the OpenAI envelope."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(400, "request body is not valid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be a JSON object")
    try:
        state, questions = _decision_payload({**body, "messages": []})
        routing = _routing_overrides(body)
    except DecisionRequestError as exc:
        raise HTTPException(400, str(exc)) from exc
    agent = get_agent()
    gate = _inference_gate

    def decide() -> dict[str, Any]:
        return _decide(agent, state, questions, routing)

    try:
        if gate is None:
            return await asyncio.to_thread(decide)
        async with gate:
            return await asyncio.to_thread(decide)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Laya decision failed")
        raise HTTPException(500, f"Laya decision failed: {exc}") from exc


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=os.environ.get("LAYA_LOG_LEVEL") or "info",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        app, host="0.0.0.0", port=SERVE_PORT,
        log_level=(os.environ.get("LAYA_LOG_LEVEL") or "info").lower(),
    )


if __name__ == "__main__":
    main()
