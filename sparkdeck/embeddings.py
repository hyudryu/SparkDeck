"""Serve SentenceTransformers embedding models that are already cached on disk.

SparkDeck recognizes an embedding model the same way it recognizes any other
cached repository — by what a node actually holds — but the marker differs.
A SentenceTransformers repository is the only kind that publishes a
``modules.json`` manifest, so its presence in a complete snapshot is what makes
a cached repository servable through ``/v1/embeddings``. Recognition reads a
few kilobytes of manifest; it never loads the model, installs anything, or
touches the network, so it is safe to evaluate while listing models.

The runtime is a different story. ``sentence-transformers`` pulls ``torch`` and
about a gigabyte of dependencies, so it is installed on the first embedding
request into a virtual environment of SparkDeck's own beside its data
directory, leaving the server's dependency set untouched. Each model is then
loaded by a separate worker process: a model that exhausts memory, segfaults,
or hangs can be killed without taking the controller down with it, and the
worker is reused across requests so a warm model is not reloaded per call.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import struct
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Advertised by every node agent that can turn a cached repository into
# embeddings. The controller only forwards to nodes that report it, so a mixed
# cluster keeps working while its agents are updated one at a time.
EMBEDDINGS_CAPABILITY = "embeddings-v1"

REQUIREMENT = "sentence-transformers"
_IMPORT_NAME = "sentence_transformers"

_MODULES_MARKER = "modules.json"
_CONFIG_MARKER = "config.json"
# ``SentenceTransformer.load`` imports every module's ``type`` by dotted path
# and instantiates it, so a manifest is an execution surface, not just data.
# The prefix every published repository uses is enforced in
# :func:`_load_modules_manifest`; config and manifest reads are capped so a
# hostile cache entry cannot be used to read an unbounded file.
_SENTENCE_TRANSFORMERS_PREFIX = "sentence_transformers."
_ENCODER_MODULE = "sentence_transformers.models.Transformer"
_MANIFEST_MAX_BYTES = 64 * 1024

DEFAULT_INSTALL_TIMEOUT = 3600.0
# A controller's request to a node may be the one that installs the runtime on
# that node, so it has to outlast a full install plus the model load and the
# encode that follows it.
DEFAULT_REQUEST_TIMEOUT = DEFAULT_INSTALL_TIMEOUT + 600.0
DEFAULT_VENV_TIMEOUT = 300.0
DEFAULT_INSPECT_TIMEOUT = 300.0
DEFAULT_ENCODE_TIMEOUT = 300.0
DEFAULT_LOAD_TIMEOUT = 300.0
DEFAULT_WORKER_IDLE_SECONDS = 900.0
# Each live worker holds a copy of the model and the torch runtime, so a client
# cycling through many cached models must not be able to pin one process per
# model. Least-recently-used workers beyond this limit are stopped.
MAX_WORKERS = 2
_WORKER_SCRIPT = Path(__file__).with_name("embedding_worker.py")
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

# OpenAI's embeddings contract: one string, or a list of strings, per request.
# The caps bound a single request's work and its JSON response on the wire.
MAX_EMBEDDING_INPUTS = 512
MAX_EMBEDDING_INPUT_CHARS = 200_000
MAX_EMBEDDING_TOTAL_CHARS = 1_000_000


class EmbeddingError(ValueError):
    """Raised for an unusable embedding request; maps to HTTP 400."""


@dataclass(frozen=True)
class CachedEmbedding:
    """One servable SentenceTransformers snapshot found in a node's cache."""

    revision: str
    snapshot: Path
    dimension: int | None
    module_count: int

    def public(self) -> dict[str, Any]:
        """Return the path-free projection inventory and ``/v1/models`` publish.

        Cache paths are agent-private everywhere else in SparkDeck, so the
        public descriptor carries the resolved revision and shape only.
        """
        descriptor: dict[str, Any] = {
            "revision": self.revision,
            "module_count": self.module_count,
        }
        if self.dimension is not None:
            descriptor["dimension"] = self.dimension
        return descriptor


def _safe_revision_name(revision: str) -> bool:
    """Accept only a single safe snapshot directory name."""
    return bool(
        revision
        and revision not in {".", ".."}
        and "/" not in revision
        and "\\" not in revision
        and not any(ord(character) < 32 or ord(character) == 127 for character in revision)
    )


def _safe_snapshot_entry(
    repository: Path, snapshot: Path, name: str, limit: int,
) -> Path | None:
    """Resolve one snapshot entry, following the hub's blob symlink safely.

    A Hugging Face snapshot normally stores each file as a symlink into the
    repository's own ``blobs`` directory, so a symlink is expected here and only
    a target that escapes the repository is rejected. The returned path is the
    resolved file, never the caller-controlled entry.
    """
    try:
        candidate = snapshot / name
        if not candidate.is_file():
            return None
        resolved = candidate.resolve(strict=True)
        root = repository.resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_relative_to(root):
        return None
    try:
        if resolved.stat().st_size > limit:
            return None
    except OSError:
        return None
    return resolved


def _load_modules_manifest(path: Path) -> int | None:
    """Return the module count of a trustworthy SentenceTransformers manifest.

    Rejects everything that is not the shape every published repository
    produces: a non-empty list of module entries whose ``type`` lives in the
    ``sentence_transformers.`` namespace and that includes the encoder module.
    A repository failing this check is simply not advertised as an embedding
    model rather than being loaded on faith.
    """
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, list) or not manifest:
        return None
    encoders = 0
    for entry in manifest:
        if not isinstance(entry, dict):
            return None
        module_type = entry.get("type")
        if not isinstance(module_type, str) or not module_type.startswith(
            _SENTENCE_TRANSFORMERS_PREFIX
        ):
            return None
        module_path = entry.get("path", "")
        if not isinstance(module_path, str):
            return None
        if module_type == _ENCODER_MODULE:
            encoders += 1
    return len(manifest) if encoders else None


def _hidden_size(repository: Path, snapshot: Path) -> int | None:
    """Read the encoder width from a snapshot's config, when it is published."""
    config_path = _safe_snapshot_entry(
        repository, snapshot, _CONFIG_MARKER, _MANIFEST_MAX_BYTES,
    )
    if config_path is None:
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(config, dict):
        return None
    value = config.get("hidden_size")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _ordered_revisions(repository: Path, revisions: Iterable[str]) -> list[str]:
    """Order complete revisions newest first, so the freshest copy is served."""
    entries: list[tuple[float, str]] = []
    for revision in revisions:
        if not _safe_revision_name(revision):
            continue
        snapshot = repository / "snapshots" / revision
        try:
            if not snapshot.is_dir() or snapshot.is_symlink():
                continue
            entries.append((snapshot.stat().st_mtime, revision))
        except OSError:
            continue
    entries.sort(reverse=True)
    return [revision for _modified, revision in entries]


def embedding_descriptor(
    repository: Path, revisions: Iterable[str],
) -> CachedEmbedding | None:
    """Describe one cached repository as a servable embedding model.

    @param repository - hub cache directory for a single repository.
    @param revisions - complete snapshot revisions the caller already resolved.
    @returns the descriptor, or None when no complete snapshot publishes a
        SentenceTransformers manifest.
    """
    for revision in _ordered_revisions(repository, revisions):
        snapshot = repository / "snapshots" / revision
        manifest = _safe_snapshot_entry(
            repository, snapshot, _MODULES_MARKER, _MANIFEST_MAX_BYTES,
        )
        if manifest is None:
            continue
        module_count = _load_modules_manifest(manifest)
        if module_count is None:
            continue
        return CachedEmbedding(
            revision=revision,
            snapshot=snapshot,
            dimension=_hidden_size(repository, snapshot),
            module_count=module_count,
        )
    return None


# ---------- OpenAI-compatible request and response shaping ----------


@dataclass(frozen=True)
class EmbeddingRequest:
    """A validated ``/v1/embeddings`` request."""

    model: str
    inputs: list[str]
    normalize: bool
    encoding_format: str


def normalize_embedding_inputs(value: Any) -> list[str]:
    """Validate OpenAI's ``input`` field into a non-empty list of strings.

    Token-id arrays are rejected rather than reinterpreted: a SentenceTransformers
    model tokenizes text itself, so accepting ids would silently encode
    unrelated text.
    """
    if isinstance(value, str):
        inputs = [value]
    elif isinstance(value, list) and value and all(
        isinstance(item, str) for item in value
    ):
        inputs = list(value)
    else:
        raise EmbeddingError("input must be a string or a non-empty array of strings")
    if any(item == "" for item in inputs):
        # An empty string tokenizes to nothing, and pooling over an empty
        # sequence yields NaN — a vector no caller can use. Refuse it here
        # rather than returning a poisoned embedding.
        raise EmbeddingError("input strings must not be empty")
    if len(inputs) > MAX_EMBEDDING_INPUTS:
        raise EmbeddingError(
            f"input must contain at most {MAX_EMBEDDING_INPUTS} strings"
        )
    for item in inputs:
        if len(item) > MAX_EMBEDDING_INPUT_CHARS:
            raise EmbeddingError(
                f"each input must be at most {MAX_EMBEDDING_INPUT_CHARS} characters"
            )
    if sum(len(item) for item in inputs) > MAX_EMBEDDING_TOTAL_CHARS:
        raise EmbeddingError(
            f"input must total at most {MAX_EMBEDDING_TOTAL_CHARS} characters"
        )
    return inputs


def parse_embedding_request(body: Any) -> EmbeddingRequest:
    """Validate an OpenAI-compatible embeddings body."""
    if not isinstance(body, dict):
        raise EmbeddingError("request body must be a JSON object")
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise EmbeddingError("model is required")
    inputs = normalize_embedding_inputs(body.get("input"))
    encoding_format = body.get("encoding_format", "float")
    if encoding_format not in {"float", "base64"}:
        raise EmbeddingError("encoding_format must be 'float' or 'base64'")
    if body.get("dimensions") is not None:
        # Silently truncating would hand a caller vectors that no longer match
        # the model's similarity space, so this is refused rather than ignored.
        raise EmbeddingError("dimensions is not supported by cached embedding models")
    normalize = body.get("normalize", True)
    if not isinstance(normalize, bool):
        raise EmbeddingError("normalize must be a boolean")
    return EmbeddingRequest(
        model=model.strip(),
        inputs=inputs,
        normalize=normalize,
        encoding_format=encoding_format,
    )


def _encoded_vector(vector: list[float], encoding_format: str) -> Any:
    if encoding_format != "base64":
        return list(vector)
    packed = struct.pack(f"<{len(vector)}f", *vector)
    return base64.b64encode(packed).decode("ascii")


def embeddings_response(
    model: str,
    vectors: list[list[float]],
    prompt_tokens: int,
    encoding_format: str = "float",
) -> dict[str, Any]:
    """Shape one answer in the OpenAI embeddings response format."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": index,
                "embedding": _encoded_vector(vector, encoding_format),
            }
            for index, vector in enumerate(vectors)
        ],
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    }


# ---------- worker process ----------


class _EmbeddingWorker:
    """One live SentenceTransformers process, reused across embedding requests.

    A worker loads exactly one snapshot and answers one request at a time. The
    protocol is newline-delimited JSON on stdin/stdout; the worker redirects
    anything a library prints to stdout over to stderr so a stray line cannot
    desynchronize the stream.
    """

    def __init__(
        self,
        interpreter: Path,
        snapshot: Path,
        *,
        load_timeout: float,
        encode_timeout: float,
    ):
        self.interpreter = interpreter
        self.snapshot = snapshot
        self.load_timeout = float(load_timeout)
        self.encode_timeout = float(encode_timeout)
        self.process: asyncio.subprocess.Process | None = None
        self.dimension: int | None = None
        self.last_used = time.monotonic()
        self._lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future] = {}
        self._ready: asyncio.Future | None = None
        self._next_id = 0
        self._stderr: deque[str] = deque(maxlen=40)
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Spawn the worker and wait until its model is loaded."""
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        environment = _worker_environment()
        try:
            self.process = await asyncio.create_subprocess_exec(
                str(self.interpreter),
                str(_WORKER_SCRIPT),
                str(self.snapshot),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(_REPOSITORY_ROOT),
                env=environment,
            )
        except OSError as exc:
            raise EmbeddingError(f"could not start the embedding worker: {exc}") from exc
        self._tasks = [
            asyncio.create_task(self._read_stdout()),
            asyncio.create_task(self._read_stderr()),
        ]
        try:
            handshake = await asyncio.wait_for(
                asyncio.shield(self._ready), timeout=self.load_timeout,
            )
        except asyncio.TimeoutError as exc:
            await self.stop()
            raise EmbeddingError(
                f"loading {self.snapshot.name} timed out after "
                f"{self.load_timeout:g} seconds"
            ) from exc
        except asyncio.CancelledError:
            await self.stop()
            raise
        if handshake.get("ready") is not True:
            reason = str(handshake.get("error") or "the embedding worker failed to load")
            await self.stop()
            raise EmbeddingError(reason)
        dimension = handshake.get("dimension")
        if isinstance(dimension, int) and dimension > 0:
            self.dimension = dimension

    async def request(
        self, inputs: list[str], *, normalize: bool,
    ) -> dict[str, Any]:
        """Encode one batch, restarting the worker if it died meanwhile."""
        async with self._lock:
            self.last_used = time.monotonic()
            if self.process is None or self.process.returncode is not None:
                await self.stop()
                await self.start()
            request_id = self._next_id
            self._next_id += 1
            loop = asyncio.get_running_loop()
            future: asyncio.Future = loop.create_future()
            self._pending[request_id] = future
            payload = json.dumps({
                "id": request_id, "inputs": inputs, "normalize": normalize,
            })
            try:
                assert self.process is not None and self.process.stdin is not None
                self.process.stdin.write(payload.encode("utf-8") + b"\n")
                await self.process.stdin.drain()
                answer = await asyncio.wait_for(
                    asyncio.shield(future), timeout=self.encode_timeout,
                )
            except asyncio.TimeoutError as exc:
                # A wedged encode cannot be interrupted safely in-process, so
                # the worker is discarded rather than left holding the lock.
                # The pending entry goes first: `stop()` settles whatever is
                # still registered, and a future nobody awaits would otherwise
                # be left holding an unretrieved exception.
                self._pending.pop(request_id, None)
                await self.stop()
                raise EmbeddingError(
                    f"embedding timed out after {self.encode_timeout:g} seconds"
                ) from exc
            except (BrokenPipeError, ConnectionResetError) as exc:
                self._pending.pop(request_id, None)
                await self.stop()
                raise EmbeddingError(self.failure_detail()) from exc
            finally:
                self._pending.pop(request_id, None)
            self.last_used = time.monotonic()
            if answer.get("error"):
                raise EmbeddingError(str(answer["error"]))
            return answer

    @property
    def busy(self) -> bool:
        """Whether a request is being encoded right now."""
        return self._lock.locked()

    def failure_detail(self) -> str:
        """Describe why the worker stopped, using its own last output.

        Called after `stop()` has already detached the process, so it must not
        assume one is still attached.
        """
        process = self.process
        detail = " ".join(line for line in self._stderr if line).strip()
        if not detail:
            if process is not None and process.returncode is not None:
                detail = f"worker exit code {process.returncode}"
            else:
                detail = "the worker exited before it reported a reason"
        return f"the embedding worker stopped: {detail[:500]}"

    async def _read_stdout(self) -> None:
        stream = self.process.stdout if self.process is not None else None
        if stream is None:
            return
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                try:
                    message = json.loads(line.decode("utf-8", "replace"))
                except json.JSONDecodeError:
                    continue
                if not isinstance(message, dict):
                    continue
                if "id" not in message:
                    # The handshake and any load failure carry no request id.
                    self._resolve_ready(message)
                    continue
                request_id = message.get("id")
                future = self._pending.get(request_id) if isinstance(
                    request_id, int
                ) else None
                if future is not None and not future.done():
                    future.set_result(message)
        except (asyncio.CancelledError, OSError):
            pass
        finally:
            self._fail_pending()

    async def _read_stderr(self) -> None:
        stream = self.process.stderr if self.process is not None else None
        if stream is None:
            return
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").strip()
                if text:
                    self._stderr.append(text)
        except (asyncio.CancelledError, OSError):
            pass

    def _resolve_ready(self, message: dict[str, Any]) -> None:
        if self._ready is not None and not self._ready.done():
            self._ready.set_result(message)

    def _fail_pending(self) -> None:
        if not self._pending:
            return
        error = EmbeddingError(self.failure_detail())
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._resolve_ready({"ready": False, "error": self.failure_detail()})

    async def stop(self) -> None:
        """Terminate the worker and settle anything still waiting on it."""
        process, self.process = self.process, None
        tasks, self._tasks = self._tasks, []
        if process is not None:
            try:
                if process.stdin is not None and not process.stdin.is_closing():
                    process.stdin.close()
            except (OSError, RuntimeError):
                pass
            if process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    process.kill()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=10)
                    except asyncio.TimeoutError:
                        logger.warning("embedding worker did not exit after SIGKILL")
        for task in tasks:
            task.cancel()
        self._fail_pending()
        self._pending.clear()


def _worker_environment() -> dict[str, str]:
    """Build a worker environment that can only use what is already on disk.

    Offline mode is the point: an embedding request is served from the snapshot
    the controller matched on disk, so a worker must fail loudly on a missing
    file instead of silently fetching it from the Hub.
    """
    environment = dict(os.environ)
    environment.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TRANSFORMERS_VERBOSITY": "error",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.pathsep.join(
            [str(_REPOSITORY_ROOT), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep),
    })
    return environment


# ---------- runtime ----------


class EmbeddingRuntime:
    """Own the private embedding environment and the workers that use it."""

    def __init__(
        self,
        data_dir: Path,
        *,
        install_timeout: float = DEFAULT_INSTALL_TIMEOUT,
        venv_timeout: float = DEFAULT_VENV_TIMEOUT,
        inspect_timeout: float = DEFAULT_INSPECT_TIMEOUT,
        encode_timeout: float = DEFAULT_ENCODE_TIMEOUT,
        load_timeout: float = DEFAULT_LOAD_TIMEOUT,
        idle_seconds: float = DEFAULT_WORKER_IDLE_SECONDS,
        base_python: str | None = None,
        max_workers: int = MAX_WORKERS,
    ):
        self.data_dir = Path(data_dir)
        self.runtime_dir = self.data_dir / "embeddings"
        self.venv_dir = self.runtime_dir / "venv"
        self.marker_path = self.runtime_dir / "runtime.json"
        self.install_timeout = float(install_timeout)
        self.venv_timeout = float(venv_timeout)
        self.inspect_timeout = float(inspect_timeout)
        self.encode_timeout = float(encode_timeout)
        self.load_timeout = float(load_timeout)
        self.idle_seconds = float(idle_seconds)
        self.max_workers = max(1, int(max_workers))
        self._base_python = base_python or sys.executable
        self._install_lock = asyncio.Lock()
        self._worker_lock = asyncio.Lock()
        self._workers: dict[str, _EmbeddingWorker] = {}
        self._monitor: asyncio.Task | None = None
        self.install_error: str | None = None

    # -- environment --

    def _venv_python(self) -> Path:
        if os.name == "nt":
            return self.venv_dir / "Scripts" / "python.exe"
        return self.venv_dir / "bin" / "python"

    def _installed(self) -> dict[str, Any] | None:
        """Report the recorded runtime, validated against the filesystem."""
        if not self._venv_python().is_file():
            return None
        try:
            marker = json.loads(self.marker_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(marker, dict) or marker.get("requirement") != REQUIREMENT:
            return None
        version = marker.get("version")
        return {
            "installed": True,
            "version": version if isinstance(version, str) else None,
            "installed_at": marker.get("installed_at"),
        }

    def state(self) -> dict[str, Any]:
        """Return path-free install state for status endpoints."""
        installed = self._installed()
        return {
            "installed": installed is not None,
            "version": (installed or {}).get("version"),
            "installed_at": (installed or {}).get("installed_at"),
            "installing": self._install_lock.locked(),
            "workers": len(self._workers),
            "error": self.install_error,
        }

    async def install(self) -> dict[str, Any]:
        """Ensure the private environment exists, installing it on first use.

        The install is serialized, so a burst of concurrent embedding requests
        performs it once; later calls find the recorded runtime and return
        immediately.
        """
        async with self._install_lock:
            installed = self._installed()
            if installed is not None:
                return installed
            self.install_error = None
            try:
                self.runtime_dir.mkdir(parents=True, exist_ok=True)
                interpreter = self._venv_python()
                if not interpreter.is_file():
                    await self._run(
                        [self._base_python, "-m", "venv", str(self.venv_dir)],
                        timeout=self.venv_timeout,
                        label="Python virtual environment",
                    )
                await self._run(
                    [
                        str(interpreter), "-m", "pip", "install", "--upgrade",
                        "--disable-pip-version-check", REQUIREMENT,
                    ],
                    timeout=self.install_timeout,
                    label=REQUIREMENT,
                )
                version = await self._probe_version(interpreter)
            except EmbeddingError as exc:
                self.install_error = str(exc)
                raise
            self._write_marker({"requirement": REQUIREMENT, "version": version})
            return {"installed": True, "version": version, "installed_at": None}

    async def _probe_version(self, interpreter: Path) -> str | None:
        """Confirm the install actually imports, and record its version.

        Importing `sentence_transformers` pulls in `torch`, so this fails on a
        partially installed or incompatible environment right here, where the
        message can name the install, instead of surfacing later as an opaque
        worker crash.
        """
        output = await self._run(
            [
                str(interpreter), "-c",
                f"import {_IMPORT_NAME} as s; print('{REQUIREMENT}', s.__version__)",
            ],
            timeout=self.inspect_timeout,
            label=f"verifying {REQUIREMENT}",
        )
        # Read the answer off its own marked line: pip and the import itself
        # both write to this stream, so the last line is not reliably ours.
        for line in reversed(output.splitlines()):
            match = re.match(rf"{re.escape(REQUIREMENT)}\s+(\S+)", line.strip())
            if match:
                return match.group(1)
        return None

    async def _run(self, argv: list[str], *, timeout: float, label: str) -> str:
        """Run one setup command, surfacing a bounded tail of its output."""
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            raise EmbeddingError(f"could not run the {label} command: {exc}") from exc
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            await self._kill(process)
            raise EmbeddingError(
                f"{label} timed out after {timeout:g} seconds"
            ) from exc
        except asyncio.CancelledError:
            # Shutdown must not leave pip mutating the shared environment.
            await self._kill(process)
            raise
        text = stdout.decode("utf-8", "replace") if stdout else ""
        if process.returncode != 0:
            tail = " ".join(text.strip().splitlines()[-4:])
            raise EmbeddingError(
                f"{label} failed (exit {process.returncode}): {tail[:800]}"
            )
        return text

    @staticmethod
    async def _kill(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            process.kill()
        except (OSError, ProcessLookupError):
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            logger.warning("embedding setup process ignored SIGKILL")

    def _write_marker(self, marker: dict[str, Any]) -> None:
        payload = {**marker, "installed_at": time.time()}
        temporary = self.marker_path.with_suffix(".json.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary, self.marker_path)
        except OSError as exc:
            # The environment is usable even when the marker cannot be
            # recorded; the next call simply re-verifies it.
            logger.warning("could not record the embedding runtime: %s", exc)

    # -- workers --

    async def _worker(self, snapshot: Path) -> _EmbeddingWorker:
        key = str(snapshot)
        worker = self._workers.get(key)
        if worker is not None and worker.process is not None:
            worker.last_used = time.monotonic()
            return worker
        # Creation is serialized because two concurrent requests for the same
        # model would otherwise each start a process and one would be orphaned
        # while still holding a loaded model. A request that arrives while
        # another model is loading simply waits for that load to finish.
        async with self._worker_lock:
            worker = self._workers.get(key)
            if worker is not None and worker.process is not None:
                worker.last_used = time.monotonic()
                return worker
            if worker is not None:
                # A worker left in place after a timeout is replaced rather
                # than reused; its reader tasks are already cancelled.
                await worker.stop()
            worker = _EmbeddingWorker(
                self._venv_python(), snapshot,
                load_timeout=self.load_timeout, encode_timeout=self.encode_timeout,
            )
            await worker.start()
            self._workers[key] = worker
            self._start_monitor()
            await self._evict_workers(protected=key)
            return worker

    async def _evict_workers(self, *, protected: str | None = None) -> None:
        """Stop least-recently-used idle workers beyond the configured limit.

        The worker that was just created is never a candidate — it exists to
        serve the request that created it — and neither is one that is encoding
        right now. Failing an in-flight request to satisfy a memory bound would
        turn a resource limit into a surprising client error, so the limit
        yields until those requests finish.
        """
        while len(self._workers) > self.max_workers:
            candidates = [
                (key, worker) for key, worker in self._workers.items()
                if key != protected and not worker.busy
            ]
            if not candidates:
                return
            key, worker = min(
                candidates, key=lambda item: (item[1].last_used, item[0]),
            )
            self._workers.pop(key, None)
            await worker.stop()

    def _start_monitor(self) -> None:
        if self._monitor is None or self._monitor.done():
            self._monitor = asyncio.create_task(self._monitor_workers())

    async def _monitor_workers(self) -> None:
        """Release idle workers so a warm model does not hold memory forever."""
        interval = max(1.0, min(60.0, self.idle_seconds))
        while True:
            await asyncio.sleep(interval)
            idle = [
                key for key, worker in self._workers.items()
                if not worker.busy
                and time.monotonic() - worker.last_used >= self.idle_seconds
            ]
            for key in idle:
                worker = self._workers.pop(key, None)
                if worker is not None:
                    await worker.stop()

    async def encode(
        self,
        *,
        model_id: str,
        snapshot: Path,
        inputs: list[str],
        normalize: bool = True,
    ) -> dict[str, Any]:
        """Encode ``inputs`` with the model cached at ``snapshot``."""
        if self._installed() is None:
            await self.install()
        worker = await self._worker(snapshot)
        answer = await worker.request(inputs, normalize=normalize)
        vectors = answer.get("embeddings")
        if (
            not isinstance(vectors, list)
            or len(vectors) != len(inputs)
            or any(not isinstance(vector, list) or not vector for vector in vectors)
        ):
            raise EmbeddingError("the embedding worker returned an unusable result")
        prompt_tokens = answer.get("prompt_tokens")
        return {
            "model": model_id,
            "embeddings": vectors,
            "prompt_tokens": (
                prompt_tokens if isinstance(prompt_tokens, int) and prompt_tokens >= 0
                else 0
            ),
            "dimension": len(vectors[0]),
        }

    async def stop(self) -> None:
        """Release every worker; called on controller shutdown."""
        monitor, self._monitor = self._monitor, None
        if monitor is not None:
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass
        workers, self._workers = self._workers, {}
        for worker in workers.values():
            try:
                await worker.stop()
            except Exception:  # pragma: no cover - shutdown must not raise
                logger.exception("stopping an embedding worker failed")
