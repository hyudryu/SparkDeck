"""Secure, durable model-cache transfers between SparkDeck nodes."""

from __future__ import annotations

import asyncio
import hashlib
import httpx
import io
import json
import os
import posixpath
import queue
import re
import shutil
import secrets
import tarfile
import tempfile
import threading
import time
import uuid
import weakref
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator, Awaitable, Callable
from urllib.parse import quote, urlencode


LOCAL_NODE_ID = "local"
_MODEL_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_HUB_BLOB_KEY = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
VIRTUAL_NAS_DOWNLOAD_CAPABILITY = "virtual-nas-download-v1"
VIRTUAL_NAS_DOWNLOAD_BASELINE_CAPABILITY = "virtual-nas-download-baseline-v1"
VIRTUAL_NAS_FILES_DOWNLOAD_CAPABILITY = "virtual-nas-files-download-v1"
VIRTUAL_NAS_DIRECT_TRANSFER_CAPABILITY = "virtual-nas-direct-transfer-v1"
_ACTIVE_TRANSFER_STATES = {"queued", "running"}
_FINAL_TRANSFER_STATES = {"completed", "failed", "canceled"}
_WEIGHT_SHARD = re.compile(
    r"^(?P<prefix>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})"
    r"(?P<suffix>\.(?:safetensors|bin|gguf))$",
    re.IGNORECASE,
)
_TOKENIZER_FILES = {
    "tokenizer.json", "tokenizer.model", "spiece.model",
    "sentencepiece.bpe.model", "tekken.json", "vocab.txt",
}
# Imports hold both the received tar and the extracted repository until the
# final atomic rename. Reserve additional room for tar headers, metadata, and
# filesystem allocation rounding rather than admitting at an exact 2x edge.
TRANSFER_STAGING_RESERVE_BYTES = 64 * 1024 * 1024
DOWNLOAD_STAGING_RESERVE_BYTES = 64 * 1024 * 1024
_SELECTIVE_SNAPSHOT_MARKER = ".sparkdeck-selective.incomplete"
_SELECTIVE_MARKER_CONTENT = b"selective\n"
# Marks a snapshot exported from an externally managed ComfyUI install: bare
# weight files under a synthetic pseudo-revision with no Hugging Face
# metadata, so completeness is judged by the weight checks alone.
_EXTERNAL_SNAPSHOT_MARKER = ".sparkdeck-external"
_EXTERNAL_MARKER_CONTENT = b"comfyui\n"
_EXTERNAL_PSEUDO_REVISION = "comfyui"
_HF_XET_HIGH_PERFORMANCE = "HF_XET_HIGH_PERFORMANCE"
_COMFYUI_MODEL_BUNDLES = (
    (
        "Lightricks/LTX-2.5",
        (
            "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
            "text_encoders/gemma4_e2b_it_bf16.safetensors",
            "vae/LTX2.5/ltx-2.5-video-vae-bf16.safetensors",
            "vae/LTX2.5/ltx-2.5-audio-vae-bf16.safetensors",
        ),
        (
            (
                "diffusion_models/LTX2.5/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
                "diffusion_models/LTX2.5/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
                "diffusion_models/LTX2.5/LTX-2.5-Distilled-Q8_0.gguf",
            ),
        ),
        (
            "vae/LTX2.5/ltx-2.5-video-vae-conv-bf16.safetensors",
            "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        ),
    ),
    (
        "Comfy-Org/MiniMax-Music-3",
        (
            "diffusion_models/minimax_music3_dit_fp16.safetensors",
            "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
            "vae/minimax_music3_dav.safetensors",
        ),
        (),
        (),
    ),
)
# Conventional ComfyUI model-store sections scanned by the generic fallback
# inventory. The scan stays bounded to these directories only.
_COMFYUI_SECTION_DIRS = (
    "checkpoints", "diffusion_models", "unet", "loras", "vae",
    "text_encoders", "clip", "clip_vision", "controlnet", "upscale_models",
    "embeddings", "latent_upscale_models", "style_models", "hypernetworks",
)
# Every file referenced by a known bundle, relative to a ComfyUI model root.
# The generic fallback never reports these, even while a bundle is incomplete.
_COMFYUI_BUNDLE_PATHS = frozenset(
    relative
    for _, required, alternative_groups, optional in _COMFYUI_MODEL_BUNDLES
    for relative in (
        *required, *optional,
        *(path for group in alternative_groups for path in group),
    )
)


def _enable_hf_xet_high_performance() -> None:
    """Use the Hub's highest-throughput downloader settings on SparkDeck nodes.

    SparkDeck targets NVMe-equipped, high-memory nodes for large model pulls.
    Set this before importing Hugging Face download helpers so hf-xet can apply
    its high-performance configuration to every new download process.
    """
    os.environ[_HF_XET_HIGH_PERFORMANCE] = "1"


# The process must set this before any lazy huggingface_hub import. Every
# SparkDeck controller and agent imports this module on startup, so restarting
# nodes applies the high-throughput Xet configuration uniformly.
_enable_hf_xet_high_performance()


def _default_comfyui_model_roots() -> list[Path]:
    """Return bounded, conventional ComfyUI model stores without exposing paths."""
    configured = str(os.environ.get("SPARKDECK_COMFYUI_MODELS_ROOTS") or "")
    candidates = [
        *(Path(value.strip()) for value in configured.split(os.pathsep) if value.strip()),
        Path("/bridge-mcp/ComfyUI/models"),
        Path.home() / "ComfyUI" / "models",
    ]
    roots = []
    seen = set()
    for candidate in candidates:
        expanded = candidate.expanduser()
        key = os.path.normcase(str(expanded))
        if key not in seen:
            seen.add(key)
            roots.append(expanded)
    return roots


def partial_download_size_bytes(
    model: dict[str, Any] | None, revision: str | None = None,
) -> int:
    """Return bytes already reusable by a resumable Hub download."""
    if not model or not (model.get("partial") or model.get("has_partial_download")):
        return 0
    if revision is not None:
        return _nonnegative_int(
            (model.get("partial_revision_size_bytes") or {}).get(revision)
        )
    value = model.get("size_bytes") if model.get("partial") else model.get("partial_size_bytes")
    return _nonnegative_int(value)


def download_required_free_bytes(expected_bytes: int, cached_bytes: int = 0) -> int:
    """Return staging capacity needed after accounting for reusable cache data."""
    expected = _nonnegative_int(expected_bytes)
    cached = min(expected, _nonnegative_int(cached_bytes))
    return expected * 2 + DOWNLOAD_STAGING_RESERVE_BYTES - cached


def cached_download_bytes(
    model: dict[str, Any] | None, baseline_bytes: int | None = None,
    revision: str | None = None,
) -> int:
    """Return target-download bytes added since its immutable attempt began."""
    if baseline_bytes is None:
        return partial_download_size_bytes(model, revision)
    current = _nonnegative_int((model or {}).get("size_bytes"))
    return max(0, current - _nonnegative_int(baseline_bytes))


class TransferCanceled(Exception):
    pass


def validate_model_id(model_id: str) -> str:
    """Validate the canonical two-component Hugging Face repository ID."""
    value = str(model_id or "").strip()
    parts = value.split("/")
    if (
        len(parts) != 2
        or any(part in {".", ".."} or "--" in part for part in parts)
        or any(not _MODEL_PART.fullmatch(part) for part in parts)
    ):
        raise ValueError("model_id must be a safe Hugging Face owner/repository ID")
    return value


def validate_revision(revision: str | None) -> str:
    value = str(revision or "main").strip() or "main"
    components = value.split("/")
    if (
        len(value) > 200
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or any(character.isspace() for character in value)
        or any(character in "~^:?*[\\<>|\"" for character in value)
        or value == "@"
        or ".." in value
        or "//" in value
        or "@{" in value
        or value.startswith("/")
        or value.endswith("/")
        or value.endswith(".")
        or any(
            component in {"", ".", ".."}
            or component.startswith(".")
            or component.endswith(".lock")
            for component in components
        )
    ):
        raise ValueError("revision must be a bounded Hugging Face revision")
    return value


def _is_commit_sha(revision: str | None) -> bool:
    return bool(
        isinstance(revision, str) and _COMMIT_SHA.fullmatch(revision.strip())
    )


def _cache_name(model_id: str) -> str:
    owner, repository = validate_model_id(model_id).split("/", 1)
    return f"models--{owner}--{repository}"


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class _TarQueueWriter:
    def __init__(self, chunks: queue.Queue, stopped: threading.Event):
        self.chunks = chunks
        self.stopped = stopped

    def write(self, value: bytes) -> int:
        data = bytes(value)
        while data and not self.stopped.is_set():
            try:
                self.chunks.put(data, timeout=0.1)
                return len(data)
            except queue.Full:
                continue
        if data:
            raise BrokenPipeError("archive consumer closed")
        return 0

    def flush(self) -> None:
        return None


class _TrackedExport:
    def __init__(self, owner: "VirtualNAS", model_id: str, stream: AsyncIterator[bytes]):
        self._stream = stream
        self._finalizer = weakref.finalize(self, owner._release_stream, model_id)

    def __aiter__(self) -> "_TrackedExport":
        return self

    async def __anext__(self) -> bytes:
        try:
            return await self._stream.__anext__()
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        try:
            await self._stream.aclose()
        finally:
            self._finalizer()


class VirtualNAS:
    """Inventory, archive safety, and the controller's durable transfer queue."""

    def __init__(
        self,
        data_dir: Path,
        hub_path_provider: Callable[[], Path],
        node_registry: Any,
        enabled_provider: Callable[[], bool],
        token_provider: Callable[[], str] | None = None,
        external_model_roots_provider: Callable[[], list[Path]] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self._hub_path_provider = hub_path_provider
        self.node_registry = node_registry
        self._enabled_provider = enabled_provider
        self._token_provider = token_provider or (lambda: "")
        self._external_model_roots_provider = (
            external_model_roots_provider or _default_comfyui_model_roots
        )
        self.path = self.data_dir / "virtual_nas_transfers.json"
        self.jobs = self._load_jobs()
        self._wake = asyncio.Event()
        self._dispatcher: asyncio.Task | None = None
        self._active: dict[str, asyncio.Task] = {}
        self._direct_export_capabilities: dict[str, dict[str, Any]] = {}
        self._peer_imports: dict[str, dict[str, Any]] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._last_progress_save: dict[str, float] = {}
        self._streaming_models: dict[str, int] = {}
        self._download_locks: dict[str, threading.Lock] = {}
        self._download_locks_guard = threading.Lock()
        self._download_size_cache: dict[
            tuple[str, str], tuple[float, str, int]
        ] = {}
        self._queue_lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self._enabled_provider())

    def _hub(self) -> Path:
        return Path(self._hub_path_provider()).expanduser().resolve()

    async def _await_uncancelable(self, operation: Awaitable[Any]) -> Any:
        """Finish work that cannot be safely stopped after it has begun."""
        task = asyncio.ensure_future(operation)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            return await task

    def _model_path(self, model_id: str) -> Path:
        hub = self._hub()
        candidate = hub / _cache_name(model_id)
        if candidate.parent.resolve() != hub:
            raise ValueError("model cache path escapes the Hugging Face hub")
        return candidate

    @staticmethod
    def _has_revision(
        model: dict[str, Any], resolved_revision: str,
        requested_revision: str | None = None,
    ) -> bool:
        if model.get("partial") or resolved_revision not in (model.get("revisions") or []):
            return False
        if not requested_revision or requested_revision == resolved_revision:
            return True
        return (model.get("revision_refs") or {}).get(requested_revision) == resolved_revision

    def _write_revision_ref(
        self, model_id: str, requested_revision: str, resolved_revision: str,
    ) -> None:
        requested_revision = validate_revision(requested_revision)
        resolved_revision = validate_revision(resolved_revision)
        if requested_revision == resolved_revision:
            return
        repository = self._model_path(model_id)
        snapshot = repository / "snapshots" / resolved_revision
        if not snapshot.is_dir() or snapshot.is_symlink():
            raise RuntimeError("download finished without the resolved snapshot")
        refs_root = repository / "refs"
        if refs_root.exists() and (not refs_root.is_dir() or refs_root.is_symlink()):
            raise RuntimeError("Hugging Face refs directory is not safe")
        refs_root.mkdir(parents=True, exist_ok=True)
        parts = PurePosixPath(requested_revision).parts
        parent = refs_root
        for component in parts[:-1]:
            candidate = parent / component
            if candidate.exists():
                if not candidate.is_dir() or candidate.is_symlink():
                    raise RuntimeError("Hugging Face revision ref contains an unsafe directory")
            else:
                candidate.mkdir()
            parent = candidate
        destination = parent / parts[-1]
        refs_resolved = refs_root.resolve()
        try:
            destination.parent.resolve().relative_to(refs_resolved)
        except ValueError as exc:
            raise RuntimeError("Hugging Face revision ref escapes the cache") from exc
        if destination.exists() and (destination.is_symlink() or not destination.is_file()):
            raise RuntimeError("Hugging Face revision ref is not a safe file")
        handle, temporary = tempfile.mkstemp(prefix=".sparkdeck-ref-", dir=destination.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(f"{resolved_revision}\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _load_jobs(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(value, list):
            return []
        recovered = False
        jobs = []
        for raw in value:
            if not isinstance(raw, dict):
                continue
            try:
                validate_model_id(raw.get("model_id"))
                revision = validate_revision(raw.get("revision")) if raw.get("revision") else None
                requested_revision = (
                    validate_revision(raw.get("requested_revision"))
                    if raw.get("requested_revision") else revision
                )
            except ValueError:
                continue
            status = str(raw.get("status") or "failed")
            legacy_mutable_workflow = bool(
                raw.get("workflow_id")
                and revision
                and not _is_commit_sha(revision)
                and status in _ACTIVE_TRANSFER_STATES
            )
            if legacy_mutable_workflow:
                status = "failed"
                raw["completed_at"] = time.time()
                raw["error"] = (
                    "model preparation was created before immutable revision "
                    "pinning; retry the recipe preparation"
                )
                recovered = True
            if status == "running":
                status = "queued"
                raw["bytes_transferred"] = 0
                recovered = True
            if status not in _ACTIVE_TRANSFER_STATES | _FINAL_TRANSFER_STATES:
                status = "failed"
            jobs.append({
                "id": str(raw.get("id") or uuid.uuid4()),
                "kind": "download" if raw.get("kind") == "download" else "transfer",
                "model_id": raw["model_id"],
                "source_node_id": str(raw.get("source_node_id") or LOCAL_NODE_ID),
                "target_node_id": str(raw.get("target_node_id") or ""),
                "revision": revision,
                "requested_revision": requested_revision,
                "depends_on_job_id": raw.get("depends_on_job_id"),
                "workflow_id": raw.get("workflow_id"),
                "workflow_node_ids": list(raw.get("workflow_node_ids") or []),
                "require_partial_cache": bool(raw.get("require_partial_cache")),
                "download_cache_baseline_bytes": (
                    _nonnegative_int(raw.get("download_cache_baseline_bytes"))
                    if raw.get("download_cache_baseline_bytes") is not None else None
                ),
                "download_attempted_at": raw.get("download_attempted_at"),
                "download_attempt_start_bytes": (
                    _nonnegative_int(raw.get("download_attempt_start_bytes"))
                    if raw.get("download_attempt_start_bytes") is not None else None
                ),
                "legacy_download_attempt_tracking": bool(
                    raw.get(
                        "legacy_download_attempt_tracking",
                        "download_attempted_at" not in raw,
                    )
                ),
                "status": status,
                "bytes_total": _nonnegative_int(raw.get("bytes_total")),
                "bytes_transferred": _nonnegative_int(raw.get("bytes_transferred")),
                "created_at": float(raw.get("created_at") or time.time()),
                "started_at": raw.get("started_at"),
                "completed_at": raw.get("completed_at"),
                "error": raw.get("error"),
            })
        if recovered:
            _atomic_json_write(self.path, jobs)
        return jobs

    def _save(self) -> None:
        _atomic_json_write(self.path, self.jobs)

    def start(self) -> None:
        if not self.enabled or (self._dispatcher and not self._dispatcher.done()):
            return
        self._dispatcher = asyncio.create_task(self._dispatch_loop())
        self._wake.set()

    async def stop(self) -> None:
        dispatcher = self._dispatcher
        self._dispatcher = None
        if dispatcher and not dispatcher.done():
            dispatcher.cancel()
        active = list(self._active.values())
        for task in active:
            task.cancel()
        if dispatcher:
            await asyncio.gather(dispatcher, return_exceptions=True)
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        self._active.clear()

    def list_transfers(self) -> dict[str, Any]:
        return {"items": [dict(job) for job in self.jobs]}

    async def resolve_download_revision(
        self, model_id: str, revision: str = "main",
        explicit_token: str | None = None,
        *, force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Resolve a requested Hub ref once to an immutable commit and size."""
        model_id = validate_model_id(model_id)
        revision = validate_revision(revision)
        cache_key = (model_id, revision)
        cached = self._download_size_cache.get(cache_key)
        if (
            _is_commit_sha(revision)
            and not force_refresh
            and cached
            and time.monotonic() - cached[0] < 300
        ):
            return {
                "requested_revision": revision,
                "resolved_revision": cached[1],
                "size_bytes": cached[2],
            }
        token = str(
            explicit_token if explicit_token is not None else self._token_provider() or ""
        ).strip()

        def inspect() -> tuple[str, int]:
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:
                raise RuntimeError("huggingface-hub is required to download model weights") from exc
            try:
                info = HfApi(token=token or None).model_info(
                    model_id, revision=revision, files_metadata=True,
                )
                resolved_value = getattr(info, "sha", None)
                if not isinstance(resolved_value, str) or not _COMMIT_SHA.fullmatch(
                    resolved_value.strip()
                ):
                    raise RuntimeError("Hugging Face did not report an immutable revision")
                resolved_revision = resolved_value.strip().lower()
                siblings = list(getattr(info, "siblings", None) or [])
                sizes = [getattr(item, "size", None) for item in siblings]
                if not siblings or any(
                    isinstance(size, bool) or not isinstance(size, int) or size < 0
                    for size in sizes
                ):
                    raise RuntimeError("Hugging Face did not report complete file sizes")
                total = sum(sizes)
                if total <= 0:
                    raise RuntimeError("Hugging Face reported an empty model repository")
                return resolved_revision, total
            except Exception as exc:
                if isinstance(exc, RuntimeError):
                    raise
                raise RuntimeError(
                    "could not inspect Hugging Face model; verify repository access, revision, credentials, and network"
                ) from exc

        resolved_revision, size_bytes = await asyncio.to_thread(inspect)
        if _is_commit_sha(revision):
            self._download_size_cache[cache_key] = (
                time.monotonic(), resolved_revision, size_bytes,
            )
        return {
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "size_bytes": size_bytes,
        }

    async def estimate_download_size(
        self, model_id: str, revision: str = "main",
        explicit_token: str | None = None,
        *, force_refresh: bool = False,
    ) -> int:
        """Return the exact Hub file total or fail closed when it is unknown."""
        resolved = await self.resolve_download_revision(
            model_id, revision, explicit_token, force_refresh=force_refresh,
        )
        return int(resolved["size_bytes"])

    async def estimate_selected_files_size(
        self, model_id: str, revision: str, filenames: list[str],
        explicit_token: str | None = None,
    ) -> int:
        """Sum the Hub-reported sizes of exactly the requested files."""
        model_id = validate_model_id(model_id)
        revision = validate_revision(revision)
        selected = list(dict.fromkeys(
            _validate_repo_relative_file(filename) for filename in filenames
        ))
        if not selected:
            raise ValueError("at least one repository file must be selected")
        token = str(
            explicit_token if explicit_token is not None else self._token_provider() or ""
        ).strip()

        def inspect() -> int:
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:
                raise RuntimeError("huggingface-hub is required to download model weights") from exc
            try:
                info = HfApi(token=token or None).model_info(
                    model_id, revision=revision, files_metadata=True,
                )
                sizes: dict[str, int] = {}
                for sibling in list(getattr(info, "siblings", None) or []):
                    filename = getattr(sibling, "rfilename", None)
                    size = getattr(sibling, "size", None)
                    if isinstance(filename, str) and isinstance(size, int) and size >= 0:
                        sizes[filename] = size
                missing = [name for name in selected if name not in sizes]
                if missing:
                    raise RuntimeError(
                        "Hugging Face did not report every selected repository file: "
                        + ", ".join(missing)
                    )
                return sum(sizes[name] for name in selected)
            except Exception as exc:
                if isinstance(exc, RuntimeError):
                    raise
                raise RuntimeError(
                    "could not inspect Hugging Face model; verify repository access, revision, credentials, and network"
                ) from exc

        return await asyncio.to_thread(inspect)

    async def download_model_checked(
        self, model_id: str, revision: str = "main",
        explicit_token: str | None = None,
        requested_revision: str | None = None,
        download_cache_baseline_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Agent-side capacity gate using only this cache filesystem."""
        model_id = validate_model_id(model_id)
        revision = validate_revision(revision)
        if not _is_commit_sha(revision):
            raise ValueError("download revision must be an immutable Hugging Face commit SHA")
        requested_revision = validate_revision(requested_revision or revision)
        inventory = await asyncio.to_thread(self.inventory)
        cached_model = next((
            item for item in inventory if item.get("model_id") == model_id
        ), None)
        existing = next((
            item for item in inventory
            if item.get("model_id") == model_id
            and self._has_revision(item, revision, requested_revision)
        ), None)
        if existing is None:
            expected_bytes = await self.estimate_download_size(
                model_id, revision, explicit_token, force_refresh=True,
            )
            free_bytes = await asyncio.to_thread(self.free_bytes)
            required = download_required_free_bytes(
                expected_bytes,
                cached_download_bytes(
                    cached_model, download_cache_baseline_bytes, revision,
                ),
            )
            if free_bytes is None:
                raise RuntimeError("download node did not report free cache capacity")
            if free_bytes < required:
                raise RuntimeError(
                    f"download node has insufficient free cache space "
                    f"({free_bytes} bytes available; {required} bytes required)"
                )
        return await self._await_uncancelable(
            asyncio.to_thread(
                self.download_model, model_id, revision, explicit_token,
                requested_revision,
            ),
        )

    async def download_model_files_checked(
        self, model_id: str, revision: str, filenames: list[str],
        explicit_token: str | None = None,
        requested_revision: str | None = None,
    ) -> dict[str, Any]:
        """Download an exact subset of one immutable Hub revision.

        Selective snapshots remain explicitly partial so the normal model
        preparation path cannot mistake one GGUF quantization for a complete
        repository and skip a later unrestricted download.
        """
        model_id = validate_model_id(model_id)
        revision = validate_revision(revision)
        if not _is_commit_sha(revision):
            raise ValueError("download revision must be an immutable Hugging Face commit SHA")
        requested_revision = validate_revision(requested_revision or revision)
        selected = list(dict.fromkeys(
            _validate_repo_relative_file(filename) for filename in filenames
        ))
        if not selected:
            raise ValueError("at least one repository file must be selected")
        token = str(
            explicit_token if explicit_token is not None else self._token_provider() or ""
        ).strip()

        def inspect() -> tuple[dict[str, int], list[str], dict[str, str]]:
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:
                raise RuntimeError(
                    "huggingface-hub is required to download model weights"
                ) from exc
            try:
                info = HfApi(token=token or None).model_info(
                    model_id, revision=revision, files_metadata=True,
                )
                resolved_value = getattr(info, "sha", None)
                if (
                    not isinstance(resolved_value, str)
                    or resolved_value.strip().lower() != revision.lower()
                ):
                    raise RuntimeError(
                        "Hugging Face did not report the requested immutable revision"
                    )
                repository_files: list[str] = []
                sizes: dict[str, int] = {}
                blob_keys: dict[str, str] = {}
                for sibling in list(getattr(info, "siblings", None) or []):
                    filename = getattr(sibling, "rfilename", None)
                    if not isinstance(filename, str):
                        continue
                    repository_files.append(filename)
                    if filename not in selected:
                        continue
                    size = getattr(sibling, "size", None)
                    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                        raise RuntimeError(
                            "Hugging Face did not report complete selected file sizes"
                        )
                    sizes[filename] = size
                    lfs = getattr(sibling, "lfs", None)
                    if lfs is not None:
                        cache_key = getattr(lfs, "sha256", None)
                        if (
                            isinstance(cache_key, str)
                            and re.fullmatch(r"[0-9a-fA-F]{64}", cache_key)
                        ):
                            blob_keys[filename] = cache_key.lower()
                    else:
                        cache_key = getattr(sibling, "blob_id", None)
                        if (
                            isinstance(cache_key, str)
                            and _HUB_BLOB_KEY.fullmatch(cache_key)
                        ):
                            blob_keys[filename] = cache_key.lower()
                missing = [filename for filename in selected if filename not in sizes]
                if missing:
                    raise RuntimeError(
                        "Hugging Face did not report every selected repository file"
                    )
                return sizes, repository_files, blob_keys
            except Exception as exc:
                if isinstance(exc, RuntimeError):
                    raise
                raise RuntimeError(
                    "could not inspect Hugging Face model; verify repository access, revision, credentials, and network"
                ) from exc

        sizes, repository_files, blob_keys = await asyncio.to_thread(inspect)

        def download_selected() -> dict[str, Any]:
            _enable_hf_xet_high_performance()
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError(
                    "huggingface-hub is required to download model weights"
                ) from exc
            with self._download_locks_guard:
                download_lock = self._download_locks.setdefault(model_id, threading.Lock())
            with download_lock:
                repository = self._model_path(model_id)
                snapshot = repository / "snapshots" / revision
                repository_complete = bool(repository_files) and all(
                    _safe_cached_snapshot_file(repository, revision, filename) is not None
                    for filename in repository_files
                )
                missing = [
                    filename for filename in selected
                    if _safe_cached_snapshot_file(repository, revision, filename) is None
                ]
                if missing:
                    expected_bytes = sum(sizes[filename] for filename in missing)
                    cached_bytes = sum(
                        _safe_incomplete_blob_bytes(
                            repository, blob_keys.get(filename), sizes[filename],
                        )
                        for filename in missing
                    )
                    free_bytes = self.free_bytes()
                    required = download_required_free_bytes(
                        expected_bytes, cached_bytes,
                    )
                    if free_bytes is None:
                        raise RuntimeError(
                            "download node did not report free cache capacity"
                        )
                    if free_bytes < required:
                        raise RuntimeError(
                            f"download node has insufficient free cache space "
                            f"({free_bytes} bytes available; {required} bytes required)"
                        )
                marker = snapshot / _SELECTIVE_SNAPSHOT_MARKER
                if repository_complete:
                    marker.unlink(missing_ok=True)
                else:
                    _ensure_safe_snapshot_directory(repository, revision)
                    if marker.is_symlink() or (
                        marker.exists() and not marker.is_file()
                    ):
                        raise RuntimeError(
                            "Hugging Face selective snapshot marker is not safe"
                        )
                    existing_files = _selective_files_by_revision(
                        repository, _snapshot_files_by_revision(repository),
                    ).get(revision, [])
                    marker.write_text(
                        json.dumps(
                            {"files": list(dict.fromkeys((*existing_files, *selected)))},
                            separators=(",", ":"),
                        ),
                        encoding="utf-8",
                    )
                if missing:
                    try:
                        for filename in missing:
                            hf_hub_download(
                                repo_id=model_id,
                                filename=filename,
                                revision=revision,
                                cache_dir=str(self._hub()),
                                token=token or None,
                            )
                    except Exception as exc:
                        raise RuntimeError(
                            "Hugging Face download failed; verify repository access, revision, credentials, and network"
                        ) from exc
                if any(
                    _safe_cached_snapshot_file(repository, revision, filename) is None
                    for filename in selected
                ):
                    raise RuntimeError(
                        "download finished without every selected repository file"
                    )
                if repository_files and all(
                    _safe_cached_snapshot_file(repository, revision, filename) is not None
                    for filename in repository_files
                ):
                    marker.unlink(missing_ok=True)
                self._write_revision_ref(
                    model_id, requested_revision, revision,
                )
                return {
                    "ok": True,
                    "model_id": model_id,
                    "revision": revision,
                    "size_bytes": sum(sizes.values()),
                    "files": selected,
                }

        return await self._await_uncancelable(
            asyncio.to_thread(download_selected),
        )

    def has_model_files(
        self, model_id: str, revision: str, filenames: list[str],
    ) -> dict[str, Any]:
        """Report which selected files of one revision are already cached.

        A pure local filesystem check so the controller can decide between
        transferring an existing copy and seeding a fresh Hub download
        without touching the network.
        """
        model_id = validate_model_id(model_id)
        revision = validate_revision(revision)
        selected = list(dict.fromkeys(
            _validate_repo_relative_file(filename) for filename in filenames
        ))
        if not selected:
            raise ValueError("at least one repository file must be selected")
        repository = self._model_path(model_id)
        present = [
            filename for filename in selected
            if _safe_cached_snapshot_file(repository, revision, filename) is not None
        ]
        return {
            "model_id": model_id,
            "revision": revision,
            "present_files": present,
            "missing_files": [
                filename for filename in selected if filename not in present
            ],
            "complete": not any(filename not in present for filename in selected),
        }

    def download_model(
        self, model_id: str, revision: str = "main", explicit_token: str | None = None,
        requested_revision: str | None = None,
    ) -> dict[str, Any]:
        """Resume a Hub snapshot into this node's configured cache."""
        model_id = validate_model_id(model_id)
        revision = validate_revision(revision)
        requested_revision = validate_revision(requested_revision or revision)
        token = str(
            explicit_token if explicit_token is not None else self._token_provider() or ""
        ).strip()
        existing = next((
            item for item in self.inventory()
            if item.get("model_id") == model_id
            and self._has_revision(item, revision, requested_revision)
        ), None)
        if existing is not None:
            return {
                "ok": True, "model_id": model_id, "revision": revision,
                "size_bytes": _nonnegative_int(existing.get("size_bytes")),
            }
        try:
            _enable_hf_xet_high_performance()
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface-hub is required to download model weights") from exc
        try:
            with self._download_locks_guard:
                download_lock = self._download_locks.setdefault(model_id, threading.Lock())
            # snapshot_download uses resumable temporary blobs and Hub file
            # locks. This additional per-process lock makes controller retries
            # idempotent when an earlier HTTP request is still unwinding.
            with download_lock:
                snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    cache_dir=str(self._hub()),
                    token=token or None,
                )
                (
                    self._model_path(model_id) / "snapshots" / revision
                    / _SELECTIVE_SNAPSHOT_MARKER
                ).unlink(missing_ok=True)
                self._write_revision_ref(
                    model_id, requested_revision, revision,
                )
        except Exception as exc:
            raise RuntimeError(
                "Hugging Face download failed; verify repository access, revision, credentials, and network"
            ) from exc
        model = next((
            item for item in self.inventory()
            if item.get("model_id") == model_id
            and self._has_revision(item, revision, requested_revision)
        ), None)
        if model is None:
            raise RuntimeError("download finished without a complete requested revision")
        return {
            "ok": True, "model_id": model_id, "revision": revision,
            "size_bytes": _nonnegative_int(model.get("size_bytes")),
        }

    def free_bytes(self) -> int | None:
        """Free space on the filesystem hosting the model cache hub.

        Measured at the hub path rather than "/" because the cache commonly
        lives on its own mount; None means the reading is unavailable.
        """
        try:
            return _local_free_bytes(self._hub())
        except (OSError, RuntimeError):
            return None

    def inventory(self) -> list[dict[str, Any]]:
        hub = self._hub()
        if not hub.is_dir():
            return _external_comfyui_inventory(self._external_model_roots_provider())
        models = []
        for repository in sorted(hub.glob("models--*--*")):
            if not repository.is_dir() or repository.is_symlink():
                continue
            encoded = repository.name.removeprefix("models--")
            parts = encoded.split("--")
            if len(parts) != 2:
                continue
            try:
                model_id = validate_model_id(f"{parts[0]}/{parts[1]}")
            except ValueError:
                continue
            snapshot_revisions = _complete_snapshot_revisions(repository)
            partial = not bool(snapshot_revisions)
            incomplete_revisions = _safe_incomplete_snapshot_revisions(
                repository, set(snapshot_revisions),
            )
            size_bytes = 0
            incomplete_size_bytes = 0
            file_count = 0
            last_modified = 0.0
            for root, directories, files in os.walk(repository, followlinks=False):
                directories[:] = [
                    name for name in directories
                    if not (Path(root) / name).is_symlink()
                ]
                for name in files:
                    try:
                        stat = (Path(root) / name).lstat()
                    except OSError:
                        continue
                    size_bytes += stat.st_size
                    if (
                        name.endswith(".incomplete")
                        and name != _SELECTIVE_SNAPSHOT_MARKER
                    ):
                        incomplete_size_bytes += stat.st_size
                    file_count += 1
                    last_modified = max(last_modified, stat.st_mtime)
            incomplete_revision_bytes = _incomplete_snapshot_reusable_bytes(
                repository, set(snapshot_revisions), incomplete_revisions,
            )
            if len(incomplete_revisions) == 1:
                only_revision = next(iter(incomplete_revisions))
                incomplete_revision_bytes[only_revision] = (
                    incomplete_revision_bytes.get(only_revision, 0)
                    + incomplete_size_bytes
                )
            revisions = set(snapshot_revisions)
            revision_refs: dict[str, str] = {}
            partial_revision_refs: dict[str, str] = {}
            refs_root = repository / "refs"
            if refs_root.is_dir() and not refs_root.is_symlink():
                for ref in refs_root.rglob("*"):
                    try:
                        if not ref.is_file() or ref.is_symlink() or ref.stat().st_size > 4096:
                            continue
                        target = ref.read_text(encoding="utf-8").strip()
                        if target in snapshot_revisions:
                            ref_name = ref.relative_to(refs_root).as_posix()
                            revisions.add(ref_name)
                            revision_refs[ref_name] = target
                        elif target in incomplete_revisions:
                            ref_name = ref.relative_to(refs_root).as_posix()
                            partial_revision_refs[ref_name] = target
                    except (OSError, UnicodeError, ValueError):
                        continue
            partial_revision = None
            if len(incomplete_revisions) == 1:
                incomplete_revision = next(iter(incomplete_revisions))
                aliases = sorted(
                    name for name, target in partial_revision_refs.items()
                    if target == incomplete_revision
                )
                partial_revision = aliases[0] if aliases else incomplete_revision
            snapshot_files = _snapshot_files_by_revision(repository)
            selective_files = _selective_files_by_revision(
                repository, snapshot_files,
            )
            model = {
                "model_id": model_id,
                "size_bytes": size_bytes,
                "file_count": file_count,
                "snapshot_files": snapshot_files,
                "partial": partial,
                "has_partial_download": (
                    partial or bool(incomplete_revisions)
                ),
                "partial_size_bytes": (
                    size_bytes if partial else sum(incomplete_revision_bytes.values())
                ),
                "partial_revision_size_bytes": incomplete_revision_bytes,
                "partial_revisions": sorted(incomplete_revisions),
                "partial_revision_refs": partial_revision_refs,
                "revision": partial_revision,
                "revisions": sorted(revisions),
                "revision_refs": revision_refs,
                "last_modified": (
                    datetime.fromtimestamp(last_modified, timezone.utc).isoformat()
                    if last_modified else None
                ),
            }
            if selective_files:
                model["selective_files_by_revision"] = selective_files
            models.append(model)
        external_models = _external_comfyui_inventory(
            self._external_model_roots_provider()
        )
        model_indexes = {
            model["model_id"]: index for index, model in enumerate(models)
        }
        for external in external_models:
            index = model_indexes.get(external["model_id"])
            if index is None:
                model_indexes[external["model_id"]] = len(models)
                models.append(external)
            elif models[index].get("partial"):
                # A usable external installation is authoritative for runtime
                # availability even if the same repository also has resumable
                # Hugging Face cache residue on this node.
                models[index] = {
                    **external,
                    "size_bytes": external["size_bytes"] + models[index]["size_bytes"],
                    "file_count": external["file_count"] + models[index]["file_count"],
                    "has_partial_download": True,
                    "partial_size_bytes": models[index]["partial_size_bytes"],
                    "partial_revision_size_bytes": models[index][
                        "partial_revision_size_bytes"
                    ],
                    "partial_revisions": models[index]["partial_revisions"],
                    "partial_revision_refs": models[index]["partial_revision_refs"],
                }
        return models

    def model_info(self, model_id: str) -> dict[str, Any]:
        model_id = validate_model_id(model_id)
        model = next(
            (item for item in self.inventory() if item["model_id"] == model_id),
            None,
        )
        if model is None:
            raise LookupError("cached model not found")
        return model

    def export_model(self, model_id: str) -> AsyncIterator[bytes]:
        model_id = validate_model_id(model_id)
        repository = self._model_path(model_id)
        external_files: dict[str, Path] | None = None
        if (
            not repository.is_dir() or repository.is_symlink()
            or not _is_complete_repository(repository)
        ):
            # Fall back to an externally managed ComfyUI install: its files
            # stream as a synthetic Hugging Face cache so the target imports a
            # normal SparkDeck-managed copy. Re-resolved on every export.
            external_files = _external_comfyui_bundle_files(
                self._external_model_roots_provider(), model_id,
            )
            if external_files is None:
                raise LookupError("cached model not found")
        self._reserve_stream(model_id)

        async def stream() -> AsyncIterator[bytes]:
            chunks: queue.Queue[Any] = queue.Queue(maxsize=8)
            stopped = threading.Event()
            finished = object()

            def produce() -> None:
                def publish(value: Any) -> None:
                    while not stopped.is_set():
                        try:
                            chunks.put(value, timeout=0.1)
                            return
                        except queue.Full:
                            continue

                try:
                    writer = _TarQueueWriter(chunks, stopped)
                    with tarfile.open(fileobj=writer, mode="w|") as archive:
                        if external_files is None:
                            archive.add(repository, arcname=repository.name, recursive=True)
                        else:
                            _write_external_bundle_archive(
                                archive, model_id, external_files,
                            )
                except BrokenPipeError:
                    pass
                except BaseException as exc:
                    publish(exc)
                finally:
                    publish(finished)

            producer = asyncio.create_task(asyncio.to_thread(produce))
            try:
                while True:
                    item = await asyncio.to_thread(chunks.get)
                    if item is finished:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield item
            finally:
                stopped.set()
                await asyncio.gather(producer, return_exceptions=True)

        return _TrackedExport(self, model_id, stream())

    async def import_model(
        self,
        model_id: str,
        chunks: AsyncIterator[bytes],
        expected_bytes: int | None = None,
        required_model_bytes: int | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> dict[str, Any]:
        model_id = validate_model_id(model_id)
        self._reserve_stream(model_id)
        hub = self._hub()
        try:
            if required_model_bytes is not None:
                model_bytes = _nonnegative_int(required_model_bytes)
                required_free = model_bytes * 2 + TRANSFER_STAGING_RESERVE_BYTES
                free_bytes = self.free_bytes()
                if free_bytes is None or free_bytes < required_free:
                    raise RuntimeError(
                        "target cache volume no longer has enough free space for import"
                    )
            hub.mkdir(parents=True, exist_ok=True)
            archive_name = _cache_name(model_id)
            if (hub / archive_name).exists():
                raise FileExistsError("cached model already exists on target node")
            stage = Path(tempfile.mkdtemp(prefix=".sparkdeck-vnas-stage-", dir=hub))
            descriptor, archive_path_value = tempfile.mkstemp(
                prefix=".sparkdeck-vnas-", suffix=".tar", dir=hub,
            )
            archive_path = Path(archive_path_value)
            received = 0
            try:
                with os.fdopen(descriptor, "wb") as output:
                    async for chunk in chunks:
                        if not isinstance(chunk, (bytes, bytearray, memoryview)):
                            raise ValueError("archive stream yielded non-bytes data")
                        data = bytes(chunk)
                        received += len(data)
                        output.write(data)
                        if progress:
                            progress(received)
                    output.flush()
                    os.fsync(output.fileno())
                if expected_bytes is not None and received != int(expected_bytes):
                    raise ValueError("archive byte count does not match expected size")
                await asyncio.to_thread(
                    self._extract_and_finalize, archive_path, stage, archive_name,
                )
                return {"ok": True, "model_id": model_id, "bytes_received": received}
            finally:
                archive_path.unlink(missing_ok=True)
                shutil.rmtree(stage, ignore_errors=True)
        finally:
            self._release_stream(model_id)

    async def import_model_from_peer(
        self,
        model_id: str,
        source_agent_url: str,
        source_capability: str,
        coordinator_id: str,
        required_model_bytes: int | None = None,
        transfer_id: str | None = None,
    ) -> dict[str, Any]:
        """Pull an archive from a source agent without relaying it via the controller."""
        model_id = validate_model_id(model_id)
        source_agent_url = str(source_agent_url or "").strip().rstrip("/")
        source_capability = str(source_capability or "").strip()
        transfer_id = str(transfer_id or "").strip()
        coordinator_id = str(coordinator_id or "").strip()
        if not source_agent_url.startswith("http://") or not source_capability or not coordinator_id or not transfer_id:
            raise ValueError("direct transfer requires an authenticated HTTP peer endpoint")
        record = {"status": "running", "bytes_transferred": 0, "cancel": asyncio.Event()}
        self._peer_imports[transfer_id] = record
        url = f"{source_agent_url}/api/agent/virtual-nas/models/{quote(model_id, safe='')}/export-direct"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(3600, connect=15)) as client:
                async with client.stream("GET", url, headers={
                    "X-SparkDeck-Direct-Transfer-Capability": source_capability,
                }) as response:
                    response.raise_for_status()
                    async def tracked() -> AsyncIterator[bytes]:
                        async for chunk in response.aiter_bytes():
                            if record["cancel"].is_set():
                                raise TransferCanceled()
                            record["bytes_transferred"] += len(chunk)
                            yield chunk
                    result = await self.import_model(
                        model_id, tracked(), required_model_bytes=required_model_bytes,
                    )
                    record["status"] = "completed"
                    return result
        except TransferCanceled:
            record["status"] = "canceled"
            raise
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"source peer rejected direct transfer (HTTP {exc.response.status_code})"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"could not reach source peer for direct transfer: {exc}") from exc
        finally:
            if record["status"] == "running":
                record["status"] = "failed"

    def peer_import_status(self, transfer_id: str) -> dict[str, Any]:
        record = self._peer_imports.get(str(transfer_id or ""))
        if record is None:
            raise LookupError("direct transfer is not active")
        return {"status": record["status"], "bytes_transferred": record["bytes_transferred"]}

    def cancel_peer_import(self, transfer_id: str) -> dict[str, Any]:
        record = self._peer_imports.get(str(transfer_id or ""))
        if record is None:
            raise LookupError("direct transfer is not active")
        record["cancel"].set()
        return {"status": record["status"]}

    def issue_direct_export_capability(self, model_id: str, revision: str | None) -> str:
        """Create a single-use, short-lived capability for a fabric peer pull."""
        model_id = validate_model_id(model_id)
        revision = validate_revision(revision) if revision else None
        model = next((item for item in self.inventory() if item.get("model_id") == model_id), None)
        if model is None or model.get("partial") or (revision and not self._has_revision(model, revision, revision)):
            raise LookupError("source node does not have the complete requested revision")
        capability = secrets.token_urlsafe(32)
        self._direct_export_capabilities[hashlib.sha256(capability.encode()).hexdigest()] = {
            "model_id": model_id, "revision": revision, "expires_at": time.monotonic() + 300,
        }
        return capability

    def export_model_with_capability(
        self, model_id: str, capability: str,
    ) -> AsyncIterator[bytes]:
        model_id = validate_model_id(model_id)
        key = hashlib.sha256(str(capability or "").encode()).hexdigest()
        grant = self._direct_export_capabilities.pop(key, None)
        if (
            grant is None or grant.get("model_id") != model_id
            or time.monotonic() > float(grant.get("expires_at") or 0)
        ):
            raise PermissionError("direct transfer capability is invalid or expired")
        return self.export_model(model_id)

    def _extract_and_finalize(
        self, archive_path: Path, stage: Path, archive_name: str
    ) -> None:
        with tarfile.open(archive_path, mode="r:*") as archive:
            members = archive.getmembers()
            if not members:
                raise ValueError("model archive is empty")
            for member in members:
                _validate_tar_member(member, archive_name)
            archive.extractall(stage, members=members, filter="data")
        extracted = stage / archive_name
        if not extracted.is_dir() or extracted.is_symlink():
            raise ValueError("model archive does not contain the expected repository")
        if not _is_complete_repository(extracted):
            raise ValueError("model archive does not contain a complete Hugging Face cache")
        destination = self._hub() / archive_name
        if destination.exists():
            raise FileExistsError("cached model already exists on target node")
        os.replace(extracted, destination)

    def _selected_snapshot_entries(
        self, repository: Path, revision: str, filenames: list[str],
    ) -> tuple[dict[str, Path], bool]:
        """Resolve selected snapshot entries plus whether the snapshot is selective."""
        snapshot = repository / "snapshots" / revision
        entries: dict[str, Path] = {}
        for filename in filenames:
            resolved = _safe_cached_snapshot_file(repository, revision, filename)
            if resolved is None:
                raise LookupError(
                    "snapshot does not contain every requested repository file"
                )
            entries[filename] = snapshot / filename
        marker = snapshot / _SELECTIVE_SNAPSHOT_MARKER
        selective = marker.is_file() and not marker.is_symlink()
        return entries, selective

    def _model_files_members(
        self, repository: Path, revision: str, filenames: list[str],
        requested_revision: str,
    ) -> tuple[list[Path], list[str], int, bool]:
        """Collect archive members for one file-scoped snapshot transfer.

        Returns the filesystem paths to archive, their arcnames, the byte
        total of the real files, and whether the target snapshot must carry
        the selective marker. A file-scoped export is selective unless it
        ships every file of an already-complete source snapshot, otherwise
        targets would advertise a partial copy as a complete revision.
        """
        entries, marker_present = self._selected_snapshot_entries(
            repository, revision, filenames,
        )
        snapshot = repository / "snapshots" / revision
        marker = snapshot / _SELECTIVE_SNAPSHOT_MARKER
        snapshot_files = {
            item.relative_to(snapshot).as_posix()
            for item in snapshot.rglob("*")
            if (item.is_symlink() or item.is_file())
            and item != marker
        }
        ships_whole_snapshot = snapshot_files == set(filenames)
        selective = marker_present or not ships_whole_snapshot
        synthetic_marker = selective and not marker_present
        paths: list[Path] = []
        arcnames: list[str] = []
        total = 0
        archived_reals: set[Path] = set()
        snapshot = repository / "snapshots" / revision

        def add(path: Path, arcname: str, *, size: bool) -> None:
            nonlocal total
            paths.append(path)
            arcnames.append(arcname)
            if size and not path.is_symlink():
                total += path.stat().st_size

        root = repository.name
        add(repository, root, size=False)
        snapshots_dir = repository / "snapshots"
        blobs_dir = repository / "blobs"
        if blobs_dir.is_dir() and not blobs_dir.is_symlink():
            add(blobs_dir, f"{root}/blobs", size=False)
        add(snapshots_dir, f"{root}/snapshots", size=False)
        add(snapshot, f"{root}/snapshots/{revision}", size=False)
        directories: set[str] = set()
        for filename, entry in entries.items():
            relative = PurePosixPath(filename).parts
            for depth in range(1, len(relative)):
                directory = "/".join(relative[:depth])
                if directory not in directories:
                    directories.add(directory)
                    add(
                        snapshot / directory,
                        f"{root}/snapshots/{revision}/{directory}",
                        size=False,
                    )
        for filename, entry in entries.items():
            real = entry.resolve(strict=True)
            if real.is_file() and real not in archived_reals:
                archived_reals.add(real)
                blob_relative = real.relative_to(repository).as_posix()
                add(real, f"{root}/{blob_relative}", size=True)
        for filename, entry in entries.items():
            if entry in archived_reals:
                # A regular-file entry (degraded symlink layouts) was already
                # archived under its snapshot path; adding it again would
                # duplicate the payload in the archive.
                continue
            add(
                entry,
                f"{root}/snapshots/{revision}/{filename}",
                size=True,
            )
        if marker_present:
            add(
                marker,
                f"{root}/snapshots/{revision}/{_SELECTIVE_SNAPSHOT_MARKER}",
                size=False,
            )
        refs_root = repository / "refs"
        if refs_root.is_dir() and not refs_root.is_symlink():
            for ref in sorted(refs_root.rglob("*")):
                if not ref.is_file() or ref.is_symlink():
                    continue
                try:
                    content = ref.read_text(encoding="utf-8").strip()
                except OSError:
                    continue
                if content == revision:
                    add(
                        ref,
                        f"{root}/refs/{ref.relative_to(refs_root).as_posix()}",
                        size=True,
                    )
        return paths, arcnames, total, synthetic_marker

    def export_model_files(
        self, model_id: str, revision: str, filenames: list[str],
        requested_revision: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream a tar of exactly one snapshot's selected files and blobs."""
        model_id = validate_model_id(model_id)
        revision = validate_revision(revision)
        requested_revision = validate_revision(requested_revision or revision)
        selected = list(dict.fromkeys(
            _validate_repo_relative_file(filename) for filename in filenames
        ))
        if not selected:
            raise ValueError("at least one repository file must be selected")
        repository = self._model_path(model_id)
        if not repository.is_dir() or repository.is_symlink():
            raise LookupError("cached model not found")
        entries, _ = self._selected_snapshot_entries(
            repository, revision, selected,
        )
        paths, arcnames, total, synthetic_marker = self._model_files_members(
            repository, revision, selected, requested_revision,
        )
        self._reserve_stream(model_id)

        async def stream() -> AsyncIterator[bytes]:
            chunks: queue.Queue[Any] = queue.Queue(maxsize=8)
            stopped = threading.Event()
            finished = object()

            def produce() -> None:
                def publish(value: Any) -> None:
                    while not stopped.is_set():
                        try:
                            chunks.put(value, timeout=0.1)
                            return
                        except queue.Full:
                            continue

                try:
                    writer = _TarQueueWriter(chunks, stopped)
                    with tarfile.open(fileobj=writer, mode="w|") as archive:
                        for path, arcname in zip(paths, arcnames):
                            archive.add(
                                path, arcname=arcname, recursive=False,
                            )
                        if synthetic_marker:
                            info = tarfile.TarInfo(
                                f"{repository.name}/snapshots/{revision}/"
                                f"{_SELECTIVE_SNAPSHOT_MARKER}"
                            )
                            info.size = len(_SELECTIVE_MARKER_CONTENT)
                            archive.addfile(
                                info, io.BytesIO(_SELECTIVE_MARKER_CONTENT),
                            )
                except BrokenPipeError:
                    pass
                except BaseException as exc:
                    publish(exc)
                finally:
                    publish(finished)

            producer = asyncio.create_task(asyncio.to_thread(produce))
            try:
                while True:
                    item = await asyncio.to_thread(chunks.get)
                    if item is finished:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield item
            finally:
                stopped.set()
                await asyncio.gather(producer, return_exceptions=True)

        return _TrackedExport(self, model_id, stream())

    async def import_model_files(
        self,
        model_id: str,
        chunks: AsyncIterator[bytes],
        expected_bytes: int | None = None,
        required_model_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Receive one file-scoped archive, placing or merging it atomically.

        Targets without the repository receive the archive as-is; targets
        already caching the repository (for example a different GGUF
        quantization) get the missing blobs, snapshot entries, and revision
        refs merged in without touching their existing content.
        """
        model_id = validate_model_id(model_id)
        self._reserve_stream(model_id)
        hub = self._hub()
        try:
            if required_model_bytes is not None:
                model_bytes = _nonnegative_int(required_model_bytes)
                required_free = model_bytes * 2 + TRANSFER_STAGING_RESERVE_BYTES
                free_bytes = self.free_bytes()
                if free_bytes is None or free_bytes < required_free:
                    raise RuntimeError(
                        "target cache volume no longer has enough free space for import"
                    )
            hub.mkdir(parents=True, exist_ok=True)
            archive_name = _cache_name(model_id)
            stage = Path(tempfile.mkdtemp(prefix=".sparkdeck-vnas-stage-", dir=hub))
            descriptor, archive_path_value = tempfile.mkstemp(
                prefix=".sparkdeck-vnas-", suffix=".tar", dir=hub,
            )
            archive_path = Path(archive_path_value)
            received = 0
            try:
                with os.fdopen(descriptor, "wb") as output:
                    async for chunk in chunks:
                        if not isinstance(chunk, (bytes, bytearray, memoryview)):
                            raise ValueError("archive stream yielded non-bytes data")
                        data = bytes(chunk)
                        received += len(data)
                        output.write(data)
                        if expected_bytes is not None and received > int(expected_bytes):
                            raise ValueError(
                                "archive byte count exceeds the expected size"
                            )
                    output.flush()
                    os.fsync(output.fileno())
                await asyncio.to_thread(
                    self._place_or_merge_model_files,
                    archive_path, stage, archive_name,
                )
                return {
                    "ok": True, "model_id": model_id, "bytes_received": received,
                }
            finally:
                archive_path.unlink(missing_ok=True)
                shutil.rmtree(stage, ignore_errors=True)
        finally:
            self._release_stream(model_id)

    def _place_or_merge_model_files(
        self, archive_path: Path, stage: Path, archive_name: str,
    ) -> None:
        with tarfile.open(archive_path, mode="r:*") as archive:
            members = archive.getmembers()
            if not members:
                raise ValueError("model archive is empty")
            for member in members:
                _validate_tar_member(member, archive_name)
            archive.extractall(stage, members=members, filter="data")
        extracted = stage / archive_name
        if not extracted.is_dir() or extracted.is_symlink():
            raise ValueError("model archive does not contain the expected repository")
        snapshot_dirs = [
            item for item in (extracted / "snapshots").iterdir()
            if item.is_dir() and not item.is_symlink()
        ]
        if not snapshot_dirs:
            raise ValueError("model archive does not contain a snapshot")
        destination = self._hub() / archive_name
        if not destination.exists():
            # Capacity was gated before the stream started; the staging copy
            # is already part of that budget, so placement is a rename. The
            # archive's own marker state defines snapshot completeness.
            os.replace(extracted, destination)
            return
        # Mirror the download, inventory, and delete paths: a repository
        # entry that is not a real cache-local directory must never be
        # written through, or a tampered link would export merge writes
        # outside the configured Hub directory.
        if destination.is_symlink() or not destination.is_dir():
            raise RuntimeError("cached model repository is not a safe directory")
        self._merge_model_directory(extracted, destination)
        shutil.rmtree(extracted, ignore_errors=True)

    def _merge_model_directory(self, extracted: Path, destination: Path) -> None:
        """Merge one cached repository into an existing one, never deleting.

        Snapshot markers are governed by the destination for pre-existing
        snapshot directories (they keep their own marker state so an
        incoming selective subset can never downgrade a complete target
        snapshot), while snapshot directories newly created by this merge
        take the archive's marker because they hold only the shipped
        subset. Revision refs are updated to the incoming alias mapping
        when it differs.
        """
        destination_snapshots = destination / "snapshots"
        pre_existing = {
            item.name
            for item in destination_snapshots.iterdir()
            if item.is_dir() and not item.is_symlink()
        } if destination_snapshots.is_dir() else set()

        for source_path in sorted(extracted.rglob("*")):
            relative = source_path.relative_to(extracted)
            target_path = destination / relative
            if (
                len(relative.parts) >= 3
                and relative.parts[0] == "snapshots"
                and relative.parts[2] == _SELECTIVE_SNAPSHOT_MARKER
                and relative.parts[1] in pre_existing
            ):
                # Pre-existing snapshots keep their own marker state so an
                # incoming selective subset cannot downgrade a complete
                # target snapshot; newly created snapshots accept the
                # archive's marker because they hold only a subset.
                continue
            if source_path.is_dir() and not source_path.is_symlink():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_path.exists():
                os.replace(source_path, target_path)
                continue
            if relative.parts[0] == "refs" and source_path.is_file():
                content = source_path.read_text(encoding="utf-8").strip()
                if (
                    target_path.is_file()
                    and target_path.read_text(encoding="utf-8").strip() == content
                ):
                    continue
                if re.fullmatch(r"[0-9a-fA-F]{40}", content):
                    staging_ref = target_path.with_name(
                        target_path.name + ".sparkdeck-ref"
                    )
                    staging_ref.write_text(content + "\n", encoding="utf-8")
                    os.replace(staging_ref, target_path)
                continue
            if source_path.is_symlink() or target_path.is_symlink():
                if source_path.is_symlink() and target_path.is_symlink() and (
                    os.readlink(source_path) == os.readlink(target_path)
                ):
                    continue
                raise RuntimeError(
                    "refusing to merge a snapshot entry that differs on the target"
                )
            if (
                source_path.is_file() and target_path.is_file()
                and source_path.stat().st_size == target_path.stat().st_size
            ):
                continue
            raise RuntimeError(
                "refusing to merge a cached file that differs on the target"
            )
        shutil.rmtree(extracted, ignore_errors=True)

    async def transfer_model_files(
        self, model_id: str, revision: str, filenames: list[str],
        source_node_id: str, target_node_id: str,
        requested_revision: str | None = None,
    ) -> dict[str, Any]:
        """Stream one file-scoped snapshot subset between two cluster nodes.

        Unlike the whole-repository transfer jobs, this path accepts
        selective (marker-flagged) snapshots and merges into targets that
        already cache the same repository under another quantization.
        """
        model_id = validate_model_id(model_id)
        revision = validate_revision(revision)
        requested_revision = validate_revision(requested_revision or revision)
        selected = list(dict.fromkeys(
            _validate_repo_relative_file(filename) for filename in filenames
        ))
        if not selected:
            raise ValueError("at least one repository file must be selected")
        if source_node_id == target_node_id:
            return {"ok": True, "model_id": model_id, "bytes_transferred": 0}
        expected = await self._model_files_transfer_bytes(
            model_id, revision, selected, source_node_id,
        )
        target_storage = await self._node_storage(target_node_id)
        free_bytes = _optional_nonnegative_int(target_storage.get("free_size"))
        if free_bytes is None:
            raise RuntimeError("target node did not report free cache capacity")
        if free_bytes < expected * 2 + TRANSFER_STAGING_RESERVE_BYTES:
            raise RuntimeError(
                f"target node has insufficient free cache space "
                f"({free_bytes} bytes available; "
                f"{expected * 2 + TRANSFER_STAGING_RESERVE_BYTES} bytes required)"
            )
        source_response = None
        target_response = None
        source: AsyncIterator[bytes] | None = None
        try:
            if source_node_id == LOCAL_NODE_ID:
                source = self.export_model_files(
                    model_id, revision, selected, requested_revision,
                )
            else:
                query = urlencode([
                    ("revision", revision),
                    ("requested_revision", requested_revision),
                    *[("files", filename) for filename in selected],
                ])
                source_response = await self.node_registry.open_stream(
                    source_node_id, "GET",
                    f"{self._model_agent_path(model_id, 'files/export')}?{query}",
                    timeout=3600,
                )
                await _raise_remote_status(source_response, "source")
                source = source_response.aiter_bytes()
            if target_node_id == LOCAL_NODE_ID:
                return await self.import_model_files(
                    model_id, source, required_model_bytes=expected,
                )
            target_response = await self.node_registry.open_stream(
                target_node_id, "PUT",
                self._model_agent_path(model_id, "files/import"),
                content=source,
                headers={
                    "Content-Type": "application/x-tar",
                    "X-SparkDeck-Model-Bytes": str(expected),
                },
                timeout=3600,
            )
            await _raise_remote_status(target_response, "target")
            await target_response.aread()
            imported_storage = await self._node_storage(target_node_id)
            if not any(
                item.get("model_id") == model_id
                for item in imported_storage.get("models") or []
            ):
                raise RuntimeError("target did not report the merged repository")
            return {"ok": True, "model_id": model_id, "bytes_transferred": expected}
        finally:
            if source is not None and hasattr(source, "aclose"):
                await source.aclose()
            if target_response is not None:
                await target_response.aclose()
            if source_response is not None:
                await source_response.aclose()

    def estimate_model_files_bytes(
        self, model_id: str, revision: str, filenames: list[str],
    ) -> dict[str, Any]:
        """Report the byte total of the real files backing selected snapshot entries.

        Blobs shared by several snapshot entries are counted once, matching
        what a file-scoped archive actually contains.
        """
        model_id = validate_model_id(model_id)
        revision = validate_revision(revision)
        selected = list(dict.fromkeys(
            _validate_repo_relative_file(filename) for filename in filenames
        ))
        if not selected:
            raise ValueError("at least one repository file must be selected")
        repository = self._model_path(model_id)
        entries, _ = self._selected_snapshot_entries(repository, revision, selected)
        total = 0
        seen: set[Path] = set()
        for entry in entries.values():
            real = entry.resolve(strict=True)
            if real in seen or not real.is_file():
                continue
            seen.add(real)
            total += real.stat().st_size
        if total <= 0:
            raise RuntimeError("source node did not report a usable cached size")
        return {"model_id": model_id, "revision": revision, "size_bytes": total}

    async def _model_files_transfer_bytes(
        self, model_id: str, revision: str, filenames: list[str],
        source_node_id: str,
    ) -> int:
        if source_node_id == LOCAL_NODE_ID:
            result = await asyncio.to_thread(
                self.estimate_model_files_bytes, model_id, revision, filenames,
            )
            return _nonnegative_int(result["size_bytes"])
        result = await self.node_registry.request(
            source_node_id, "GET",
            f"{self._model_agent_path(model_id, 'files/size')}?"
            + urlencode([
                ("revision", revision),
                *[("files", filename) for filename in filenames],
            ]),
            timeout=60,
        )
        size = _nonnegative_int((result or {}).get("size_bytes"))
        if size <= 0:
            raise RuntimeError("source node did not report a usable cached size")
        return size

    def delete_model(self, model_id: str) -> dict[str, Any]:
        model_id = validate_model_id(model_id)
        if self.model_in_transfer(model_id, LOCAL_NODE_ID):
            raise RuntimeError("model is in use by a virtual NAS transfer")
        repository = self._model_path(model_id)
        if repository.exists():
            if repository.is_symlink() or not repository.is_dir():
                raise ValueError("cached model repository is not a safe directory")
            hub = self._hub()
            if repository.resolve().parent != hub:
                raise ValueError("model cache path escapes the Hugging Face hub")
            if (
                _is_complete_repository(repository)
                or _external_comfyui_bundle_files(
                    self._external_model_roots_provider(), model_id,
                ) is None
            ):
                shutil.rmtree(repository)
                return {"ok": True, "model_id": model_id}
            # The hub copy is partial residue while inventory displays the
            # complete external install: delete the externally managed files.
        return self._delete_external_model(model_id)

    def _delete_external_model(self, model_id: str) -> dict[str, Any]:
        """Unlink an externally managed ComfyUI bundle's real files.

        Every complete copy across the configured roots is re-resolved at
        delete time and every path is re-verified before anything is
        removed: regular files only, never symlinks, and always inside one
        of the currently configured, resolved ComfyUI model roots. Files are
        then staged by an in-directory rename and only unlinked once every
        copy staged successfully, so a mid-delete failure rolls back instead
        of leaving a half-deleted bundle. Directories and unrelated files
        are left untouched.
        """
        raw_roots = self._external_model_roots_provider()
        roots = []
        for raw_root in raw_roots:
            try:
                if raw_root.is_symlink() or not raw_root.is_dir():
                    continue
                roots.append(raw_root.resolve(strict=True))
            except OSError:
                continue
        copies = _external_comfyui_bundle_copies(raw_roots, model_id)
        if not copies:
            raise LookupError("cached model not found")
        paths = [path for files in copies for path in files.values()]
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError("external model file is not a safe regular file")
            if not any(path.is_relative_to(root) for root in roots):
                raise ValueError("external model file escapes the ComfyUI model roots")
        staged: list[tuple[Path, Path]] = []
        try:
            for path in paths:
                staged_path = path.with_name(f"{path.name}.sparkdeck-deleting")
                os.replace(path, staged_path)
                staged.append((staged_path, path))
        except OSError as exc:
            for staged_path, original in reversed(staged):
                try:
                    os.replace(staged_path, original)
                except OSError:
                    pass
            raise RuntimeError(
                "could not stage external model files for deletion"
            ) from exc
        for staged_path, _ in staged:
            staged_path.unlink()
        return {"ok": True, "model_id": model_id}

    async def queue_transfer(
        self, model_id: str, source_node_id: str,
        target_node_ids: list[str], revision: str | None = None,
        workflow_id: str | None = None,
        workflow_node_ids: list[str] | None = None,
        requested_revision: str | None = None,
    ) -> dict[str, Any]:
        async with self._queue_lock:
            return await self._queue_transfer(
                model_id, source_node_id, target_node_ids, revision,
                workflow_id, workflow_node_ids, requested_revision,
            )

    async def _queue_transfer(
        self, model_id: str, source_node_id: str,
        target_node_ids: list[str], revision: str | None = None,
        workflow_id: str | None = None,
        workflow_node_ids: list[str] | None = None,
        requested_revision: str | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("virtual NAS is disabled")
        model_id = validate_model_id(model_id)
        revision = validate_revision(revision) if revision else None
        requested_revision = (
            validate_revision(requested_revision or revision)
            if revision else None
        )
        source_node_id = str(source_node_id or LOCAL_NODE_ID)
        targets = list(dict.fromkeys(str(value) for value in target_node_ids if value))
        if not targets:
            raise ValueError("target_node_ids must contain at least one node")
        await self._validate_online_node(source_node_id)
        target_statuses: dict[str, dict[str, Any]] = {}
        for target in targets:
            target_statuses[target] = await self._validate_online_node(target)
            if target == source_node_id:
                raise ValueError("source and target nodes must be different")
        source_storage = await self._node_storage(source_node_id)
        source_inventory = source_storage["models"]
        source_model = next(
            (
                item for item in source_inventory
                if item.get("model_id") == model_id
                and (not revision or self._has_revision(
                    item, revision, requested_revision,
                ))
            ),
            None,
        )
        if source_model is None:
            raise LookupError("cached source model revision not found")
        if source_model.get("partial"):
            raise RuntimeError("partial cached models cannot be transferred")
        if source_model.get("transferable") is False:
            raise RuntimeError(
                "model weights cannot be transferred from their current storage location"
            )
        model_size = _nonnegative_int(
            source_model.get("transfer_size_bytes")
            or source_model.get("size_bytes")
        )
        if model_size <= 0:
            raise RuntimeError("source node did not report a usable cached model size")
        existing_targets: list[str] = []
        target_storage: dict[str, dict[str, Any]] = {}
        for target in targets:
            duplicate = next((
                job for job in self.jobs
                if job["model_id"] == model_id
                and job["target_node_id"] == target
                and job["status"] in _ACTIVE_TRANSFER_STATES
            ), None)
            if duplicate:
                raise ValueError(
                    f"an active model preparation for '{model_id}' to node "
                    f"'{target}' already exists"
                )
            storage = await self._node_storage(target)
            target_storage[target] = storage
            target_inventory = storage["models"]
            if any(
                item.get("model_id") == model_id for item in target_inventory
            ):
                existing_targets.append(target)
        if existing_targets:
            raise FileExistsError(
                f"cached model already exists on target node(s): {', '.join(existing_targets)}"
            )
        required_free = model_size * 2 + TRANSFER_STAGING_RESERVE_BYTES
        for target in target_statuses:
            raw_free = target_storage[target].get("free_size")
            free_bytes = None if raw_free is None else _nonnegative_int(raw_free)
            if free_bytes is None:
                raise RuntimeError(
                    f"target node '{target}' did not report free disk capacity"
                )
            if free_bytes < required_free:
                raise RuntimeError(
                    f"target node '{target}' has insufficient free disk space "
                    f"({free_bytes} bytes available; {required_free} bytes required "
                    "for archive staging and extraction)"
                )

        created = time.time()
        jobs = []
        for target in targets:
            job = {
                "id": str(uuid.uuid4()), "kind": "transfer", "model_id": model_id,
                "source_node_id": source_node_id, "target_node_id": target,
                "revision": revision,
                "requested_revision": requested_revision,
                "depends_on_job_id": None,
                "workflow_id": workflow_id,
                "workflow_node_ids": list(workflow_node_ids or []),
                "status": "queued",
                "bytes_total": model_size,
                "bytes_transferred": 0, "created_at": created,
                "started_at": None,
                "completed_at": None, "error": None,
            }
            self.jobs.append(job)
            jobs.append(dict(job))
        self._save()
        self.start()
        self._wake.set()
        return {"job_ids": [job["id"] for job in jobs], "jobs": jobs}

    async def queue_download_and_transfer(
        self,
        model_id: str,
        revision: str,
        download_node_id: str,
        transfer_target_node_ids: list[str],
        expected_bytes: int,
        workflow_id: str | None = None,
        workflow_node_ids: list[str] | None = None,
        additional_download_node_ids: list[str] | None = None,
        source_node_id: str | None = None,
        requested_revision: str | None = None,
        require_partial_cache: bool = False,
        download_cache_baseline_bytes: int | None = None,
    ) -> dict[str, Any]:
        async with self._queue_lock:
            return await self._queue_download_and_transfer(
                model_id, revision, download_node_id,
                transfer_target_node_ids, expected_bytes, workflow_id,
                workflow_node_ids, additional_download_node_ids,
                source_node_id, requested_revision, require_partial_cache,
                download_cache_baseline_bytes,
            )

    async def _queue_download_and_transfer(
        self,
        model_id: str,
        revision: str,
        download_node_id: str,
        transfer_target_node_ids: list[str],
        expected_bytes: int,
        workflow_id: str | None = None,
        workflow_node_ids: list[str] | None = None,
        additional_download_node_ids: list[str] | None = None,
        source_node_id: str | None = None,
        requested_revision: str | None = None,
        require_partial_cache: bool = False,
        download_cache_baseline_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Persist resumable Hub downloads and any dependent NAS fan-out."""
        if not self.enabled:
            raise RuntimeError("virtual NAS is disabled")
        model_id = validate_model_id(model_id)
        revision = validate_revision(revision)
        if not _is_commit_sha(revision):
            raise ValueError("download revision must be an immutable Hugging Face commit SHA")
        requested_revision = validate_revision(requested_revision or revision)
        expected_bytes = _nonnegative_int(expected_bytes)
        if expected_bytes <= 0:
            raise ValueError("expected_bytes must be positive")
        download_node_id = str(download_node_id or LOCAL_NODE_ID)
        download_nodes = list(dict.fromkeys([
            download_node_id,
            *(str(value) for value in (additional_download_node_ids or []) if value),
        ]))
        explicit_baseline = (
            _nonnegative_int(download_cache_baseline_bytes)
            if download_cache_baseline_bytes is not None else None
        )
        download_baselines: dict[str, int] = {}
        download_progress_bytes: dict[str, int] = {}
        targets = list(dict.fromkeys(
            str(value) for value in transfer_target_node_ids if value
        ))
        overlap = sorted(set(download_nodes) & set(targets))
        if overlap:
            raise ValueError(
                "download nodes cannot also be transfer targets: "
                + ", ".join(overlap)
            )
        transfer_source = str(source_node_id) if source_node_id else download_node_id
        if transfer_source in targets:
            raise ValueError("source node cannot also be a transfer target")

        for node_id in download_nodes:
            await self._validate_download_node(node_id)
        if targets and transfer_source not in download_nodes:
            await self._validate_online_node(transfer_source)
        for target in targets:
            await self._validate_online_node(target)

        active_targets = {
            job["target_node_id"]
            for job in self.jobs
            if job.get("model_id") == model_id
            and job.get("status") in _ACTIVE_TRANSFER_STATES
        }
        requested_targets = {*download_nodes, *targets}
        duplicate_targets = sorted(active_targets & requested_targets)
        if duplicate_targets:
            raise ValueError(
                "an active model preparation already exists for node(s): "
                + ", ".join(duplicate_targets)
            )

        for node_id in download_nodes:
            download_storage = await self._node_storage(node_id)
            cached_model = next((
                item for item in download_storage["models"]
                if item.get("model_id") == model_id
            ), None)
            if require_partial_cache and not (
                cached_model
                and (cached_model.get("partial") or cached_model.get("has_partial_download"))
            ):
                raise LookupError(
                    f"partial model cache no longer exists on node '{node_id}'"
                )
            fallback_cached = partial_download_size_bytes(cached_model, revision)
            baseline = (
                explicit_baseline
                if node_id == download_node_id and explicit_baseline is not None
                else max(
                    0,
                    _nonnegative_int((cached_model or {}).get("size_bytes"))
                    - fallback_cached,
                )
            )
            download_baselines[node_id] = baseline
            cached_bytes = min(
                expected_bytes,
                cached_download_bytes(cached_model, baseline, revision),
            )
            download_progress_bytes[node_id] = cached_bytes
            download_required = download_required_free_bytes(
                expected_bytes, cached_bytes,
            )
            download_free = _optional_nonnegative_int(download_storage.get("free_size"))
            if download_free is None:
                raise RuntimeError(
                    f"download node '{node_id}' did not report free cache capacity"
                )
            if download_free < download_required:
                raise RuntimeError(
                    f"download node '{node_id}' has insufficient free cache space "
                    f"({download_free} bytes available; {download_required} bytes required)"
                )

        transfer_bytes = expected_bytes
        if targets and transfer_source not in download_nodes:
            source_storage = await self._node_storage(transfer_source)
            source_model = next((
                item for item in source_storage["models"]
                if item.get("model_id") == model_id
                and self._has_revision(item, revision, requested_revision)
            ), None)
            if source_model is None:
                raise LookupError("cached source model revision not found")
            transfer_bytes = _nonnegative_int(source_model.get("size_bytes"))
            if transfer_bytes <= 0:
                raise RuntimeError("source node did not report a usable cached model size")
        transfer_required = transfer_bytes * 2 + TRANSFER_STAGING_RESERVE_BYTES
        for target in targets:
            storage = await self._node_storage(target)
            if any(item.get("model_id") == model_id for item in storage["models"]):
                raise FileExistsError(
                    f"cached model already exists on target node '{target}'"
                )
            free_bytes = _optional_nonnegative_int(storage.get("free_size"))
            if free_bytes is None:
                raise RuntimeError(
                    f"target node '{target}' did not report free cache capacity"
                )
            if free_bytes < transfer_required:
                raise RuntimeError(
                    f"target node '{target}' has insufficient free cache space "
                    f"({free_bytes} bytes available; {transfer_required} bytes required)"
                )

        created = time.time()
        jobs = [{
            "id": str(uuid.uuid4()), "kind": "download", "model_id": model_id,
            "source_node_id": "huggingface", "target_node_id": node_id,
            "revision": revision, "depends_on_job_id": None,
            "requested_revision": requested_revision,
            "require_partial_cache": bool(require_partial_cache),
            "download_cache_baseline_bytes": download_baselines[node_id],
            "download_attempted_at": None,
            "download_attempt_start_bytes": None,
            "legacy_download_attempt_tracking": False,
            "workflow_id": workflow_id,
            "workflow_node_ids": list(workflow_node_ids or []),
            "status": "queued", "bytes_total": expected_bytes,
            "bytes_transferred": download_progress_bytes[node_id],
            "created_at": created,
            "started_at": None, "completed_at": None, "error": None,
        } for node_id in download_nodes]
        primary_download_job = jobs[0]
        for target in targets:
            jobs.append({
                "id": str(uuid.uuid4()), "kind": "transfer", "model_id": model_id,
                "source_node_id": transfer_source, "target_node_id": target,
                "revision": revision,
                "requested_revision": requested_revision,
                "depends_on_job_id": (
                    None if source_node_id else primary_download_job["id"]
                ),
                "workflow_id": workflow_id,
                "workflow_node_ids": list(workflow_node_ids or []),
                "status": "queued", "bytes_total": transfer_bytes,
                "bytes_transferred": 0, "created_at": created,
                "started_at": None, "completed_at": None, "error": None,
            })
        self.jobs.extend(jobs)
        self._save()
        self.start()
        self._wake.set()
        return {"job_ids": [job["id"] for job in jobs], "jobs": [dict(job) for job in jobs]}

    async def cancel_transfer(self, job_id: str) -> dict[str, Any]:
        job = next((item for item in self.jobs if item["id"] == job_id), None)
        if job is None:
            raise LookupError("transfer not found")
        if job["status"] in _FINAL_TRANSFER_STATES:
            return dict(job)
        if job.get("kind") == "download" and job["status"] == "running":
            raise RuntimeError(
                "a running Hugging Face download cannot be canceled safely; "
                "it is resumable and will finish or fail"
            )
        job["status"] = "canceled"
        job["completed_at"] = time.time()
        job["error"] = None
        event = self._cancel_events.get(job_id)
        if event:
            event.set()
        self._save()
        self._wake.set()
        return dict(job)

    def model_in_transfer(self, model_id: str, node_id: str | None = None) -> bool:
        model_id = validate_model_id(model_id)
        local_stream = bool(
            node_id in (None, LOCAL_NODE_ID)
            and self._streaming_models.get(model_id, 0)
        )
        return local_stream or any(
            job["model_id"] == model_id and job["status"] in _ACTIVE_TRANSFER_STATES
            and (
                node_id is None
                or node_id in {job["source_node_id"], job["target_node_id"]}
            )
            for job in self.jobs
        )

    def _reserve_stream(self, model_id: str) -> None:
        self._streaming_models[model_id] = self._streaming_models.get(model_id, 0) + 1

    def _release_stream(self, model_id: str) -> None:
        remaining = self._streaming_models.get(model_id, 0) - 1
        if remaining > 0:
            self._streaming_models[model_id] = remaining
        else:
            self._streaming_models.pop(model_id, None)

    def _validate_node(self, node_id: str) -> None:
        if node_id == LOCAL_NODE_ID:
            return
        node = self.node_registry.get(node_id)
        if not node:
            raise ValueError(f"unknown node_id '{node_id}'")
        if not node.get("enabled", True):
            raise ValueError(f"node_id '{node_id}' is disabled")

    async def _validate_online_node(self, node_id: str) -> dict[str, Any]:
        self._validate_node(node_id)
        if node_id == LOCAL_NODE_ID:
            return {
                "id": LOCAL_NODE_ID, "online": True,
                "disk": {"free": _local_free_bytes(self._hub())},
            }
        node = self.node_registry.get(node_id)
        # Queueing is preceded by a cluster/preflight inventory. Reuse that
        # fresh status instead of bypassing the probe cache and allowing a
        # second transient callback failure to misclassify a live node.
        status = await self.node_registry.probe(node)
        if not status.get("online"):
            raise RuntimeError(f"node '{node.get('name', node_id)}' is offline")
        return status

    async def _validate_download_node(self, node_id: str) -> dict[str, Any]:
        status = await self._validate_online_node(node_id)
        if (
            node_id != LOCAL_NODE_ID
            and VIRTUAL_NAS_DOWNLOAD_CAPABILITY
            not in (status.get("capabilities") or [])
        ):
            raise RuntimeError(
                f"node '{status.get('name', node_id)}' must be updated before "
                "it can download models from Hugging Face"
            )
        return status

    async def _node_storage(self, node_id: str) -> dict[str, Any]:
        if node_id == LOCAL_NODE_ID:
            return {
                "models": await asyncio.to_thread(self.inventory),
                "free_size": await asyncio.to_thread(self.free_bytes),
            }
        payload = await self.node_registry.request(
            node_id, "GET", "/api/agent/virtual-nas/inventory", timeout=30,
        )
        return {
            "models": list((payload or {}).get("models") or []),
            "free_size": (payload or {}).get("free_size"),
        }

    async def _dispatch_loop(self) -> None:
        try:
            while True:
                for target, task in list(self._active.items()):
                    if task.done():
                        self._active.pop(target, None)
                # A single global transfer prevents a multi-target copy from
                # saturating the source disk and cluster network.
                if not self._active:
                    job = None
                    for candidate in self.jobs:
                        if candidate["status"] != "queued":
                            continue
                        dependency_id = candidate.get("depends_on_job_id")
                        if dependency_id:
                            dependency = next((
                                item for item in self.jobs
                                if item["id"] == dependency_id
                            ), None)
                            if dependency is None or dependency["status"] in {"failed", "canceled"}:
                                candidate["status"] = "failed"
                                candidate["completed_at"] = time.time()
                                candidate["error"] = "required source download did not complete"
                                self._save()
                                continue
                            if dependency["status"] != "completed":
                                continue
                        job = candidate
                        break
                    if job is not None:
                        target = job["target_node_id"]
                        runner = self._run_download if job.get("kind") == "download" else self._run_transfer
                        task = asyncio.create_task(runner(job))
                        self._active[target] = task
                        task.add_done_callback(lambda _task: self._wake.set())
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=0.5)
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise

    async def _run_download(self, job: dict[str, Any]) -> None:
        event = asyncio.Event()
        self._cancel_events[job["id"]] = event
        job.update({
            "status": "running", "started_at": time.time(),
            "completed_at": None, "error": None,
        })
        self._save()
        token = str(self._token_provider() or "").strip()
        try:
            storage = await self._node_storage(job["target_node_id"])
            already_complete = next((
                item for item in storage["models"]
                if item.get("model_id") == job["model_id"]
                and self._has_revision(
                    item, job.get("revision") or "main",
                    job.get("requested_revision") or job.get("revision") or "main",
                )
            ), None)
            immutable_complete = next((
                item for item in storage["models"]
                if item.get("model_id") == job["model_id"]
                and self._has_revision(
                    item, job.get("revision") or "main",
                    job.get("revision") or "main",
                )
            ), None)
            cached_model = next((
                item for item in storage["models"]
                if item.get("model_id") == job["model_id"]
            ), None)
            if job.get("require_partial_cache") and not (
                immutable_complete is not None
                or (
                    cached_model
                    and (cached_model.get("partial") or cached_model.get("has_partial_download"))
                )
            ):
                raise LookupError("partial model cache no longer exists")
            if already_complete is None:
                current_size = await self.estimate_download_size(
                    job["model_id"], job.get("revision") or "main", token,
                    force_refresh=True,
                )
                job["bytes_total"] = current_size
                self._save()
                # Re-read the authoritative cache filesystem after the Hub
                # metadata call so both operands of the capacity gate are
                # current when the download actually begins.
                storage = await self._node_storage(job["target_node_id"])
                cached_model = next((
                    item for item in storage["models"]
                    if item.get("model_id") == job["model_id"]
                ), None)
                already_complete = next((
                    item for item in storage["models"]
                    if item.get("model_id") == job["model_id"]
                    and self._has_revision(
                        item, job.get("revision") or "main",
                        job.get("requested_revision") or job.get("revision") or "main",
                    )
                ), None)
                immutable_complete = next((
                    item for item in storage["models"]
                    if item.get("model_id") == job["model_id"]
                    and self._has_revision(
                        item, job.get("revision") or "main",
                        job.get("revision") or "main",
                    )
                ), None)
                if already_complete is None and immutable_complete is None:
                    if job.get("require_partial_cache") and not (
                        cached_model and (
                            cached_model.get("partial")
                            or cached_model.get("has_partial_download")
                        )
                    ):
                        raise LookupError("partial model cache no longer exists")
                    free_bytes = _optional_nonnegative_int(storage.get("free_size"))
                    required = download_required_free_bytes(
                        current_size, cached_download_bytes(
                        cached_model, job.get("download_cache_baseline_bytes"),
                        job.get("revision") or "main",
                        ),
                    )
                    if free_bytes is None:
                        raise RuntimeError("download node did not report free cache capacity")
                    if free_bytes < required:
                        raise RuntimeError(
                            f"download node has insufficient free cache space "
                            f"({free_bytes} bytes available; {required} bytes required)"
                        )
            if already_complete is not None:
                completed_size = _nonnegative_int(job.get("bytes_total"))
                job.update({
                    "status": "completed",
                    "bytes_transferred": completed_size,
                    "completed_at": time.time(), "error": None,
                })
                self._save()
                return
            if job["target_node_id"] == LOCAL_NODE_ID:
                operation = asyncio.to_thread(
                    self.download_model, job["model_id"],
                    job.get("revision") or "main", token,
                    job.get("requested_revision") or job.get("revision") or "main",
                )
            else:
                download_status = await self._validate_download_node(job["target_node_id"])
                download_body = {
                    "revision": job.get("revision") or "main",
                    "requested_revision": (
                        job.get("requested_revision")
                        or job.get("revision") or "main"
                    ),
                    "hf_token": token,
                }
                if VIRTUAL_NAS_DOWNLOAD_BASELINE_CAPABILITY in (
                    download_status.get("capabilities") or []
                ):
                    download_body["download_cache_baseline_bytes"] = job.get(
                        "download_cache_baseline_bytes"
                    )
                operation = self.node_registry.request(
                    job["target_node_id"], "POST",
                    self._model_agent_path(job["model_id"], "download"),
                    json_body=download_body,
                    timeout=24 * 60 * 60,
                )
            job["download_attempt_start_bytes"] = (
                _nonnegative_int(immutable_complete.get("size_bytes"))
                if immutable_complete is not None
                else min(
                    _nonnegative_int(job.get("bytes_total")),
                    cached_download_bytes(
                        cached_model,
                        job.get("download_cache_baseline_bytes"),
                        job.get("revision") or "main",
                    ),
                )
            )
            job["download_attempted_at"] = time.time()
            self._save()
            # Neither a local snapshot_download worker nor a remote agent's
            # snapshot worker can be safely canceled once it begins. Retain
            # ownership until it finishes so stop/restart cannot duplicate it.
            result = await self._await_uncancelable(operation)
            if event.is_set() or job["status"] == "canceled":
                raise TransferCanceled()
            size_bytes = _nonnegative_int((result or {}).get("size_bytes"))
            job["status"] = "completed"
            job["bytes_transferred"] = size_bytes or job["bytes_total"]
            job["bytes_total"] = size_bytes or job["bytes_total"]
            job["completed_at"] = time.time()
            for dependent in self.jobs:
                if dependent.get("depends_on_job_id") == job["id"]:
                    dependent["bytes_total"] = job["bytes_total"]
        except TransferCanceled:
            job["status"] = "canceled"
            job["completed_at"] = time.time()
            job["error"] = None
        except asyncio.CancelledError:
            if job["status"] != "canceled":
                job["status"] = "queued"
                job["started_at"] = None
                job["error"] = None
            raise
        except Exception as exc:
            message = str(exc)
            if token:
                message = message.replace(token, "[REDACTED]")
            job["status"] = "failed"
            job["completed_at"] = time.time()
            job["error"] = message[:500]
        finally:
            self._cancel_events.pop(job["id"], None)
            self._save()
            self._wake.set()

    async def _run_transfer(self, job: dict[str, Any]) -> None:
        event = asyncio.Event()
        self._cancel_events[job["id"]] = event
        job.update({
            "status": "running", "started_at": time.time(),
            # Archive copies are not resumable. A requeued attempt streams the
            # full archive again, so its progress and rate restart at byte zero.
            "bytes_transferred": 0,
            "completed_at": None, "error": None,
        })
        self._save()
        source_response = None
        target_response = None
        source = None
        tracked_stream = None
        try:
            source_storage = await self._node_storage(job["source_node_id"])
            source_model = next((
                item for item in source_storage["models"]
                if item.get("model_id") == job["model_id"]
                and (not job.get("revision") or self._has_revision(
                    item, job["revision"],
                    job.get("requested_revision") or job["revision"],
                ))
            ), None)
            if source_model is None:
                raise RuntimeError("source node does not have the complete requested revision")
            if source_model.get("partial"):
                raise RuntimeError("source node no longer has a complete requested revision")
            if source_model.get("transferable") is False:
                raise RuntimeError(
                    "model weights cannot be transferred from their current storage location"
                )
            actual_size = _nonnegative_int(
                source_model.get("transfer_size_bytes")
                or source_model.get("size_bytes")
            )
            if actual_size <= 0:
                raise RuntimeError("source node did not report a usable cached model size")
            job["bytes_total"] = actual_size
            target_storage = await self._node_storage(job["target_node_id"])
            if any(
                item.get("model_id") == job["model_id"]
                for item in target_storage["models"]
            ):
                raise FileExistsError("cached model already exists on target node")
            free_bytes = _optional_nonnegative_int(target_storage.get("free_size"))
            required = actual_size * 2 + TRANSFER_STAGING_RESERVE_BYTES
            if free_bytes is None:
                raise RuntimeError("target node did not report free cache capacity")
            if free_bytes < required:
                raise RuntimeError(
                    f"target node has insufficient free cache space "
                    f"({free_bytes} bytes available; {required} bytes required)"
                )
            if job["source_node_id"] == LOCAL_NODE_ID:
                source = self.export_model(job["model_id"])
            else:
                direct_source = None
                if job["target_node_id"] != LOCAL_NODE_ID:
                    direct_source = getattr(
                        self.node_registry, "direct_transfer_source", lambda _node_id: None,
                    )(
                        job["source_node_id"]
                    )
                target_status = (
                    await self._validate_online_node(job["target_node_id"])
                    if direct_source is not None else None
                )
                source_status = (
                    await self._validate_online_node(job["source_node_id"])
                    if direct_source is not None else None
                )
                if (
                    direct_source is not None
                    and getattr(self.node_registry, "direct_transfer_source", lambda _node_id: None)(job["target_node_id"])
                    and VIRTUAL_NAS_DIRECT_TRANSFER_CAPABILITY
                    in (target_status.get("capabilities") or [])
                    and VIRTUAL_NAS_DIRECT_TRANSFER_CAPABILITY
                    in (source_status.get("capabilities") or [])
                ):
                    capability_result = await self.node_registry.request(
                        job["source_node_id"], "POST",
                        self._model_agent_path(job["model_id"], "export-capability"),
                        json_body={"revision": job.get("revision")}, timeout=30,
                    )
                    source_capability = str((capability_result or {}).get("capability") or "")
                    if not source_capability:
                        raise RuntimeError("source did not issue a direct transfer capability")
                    operation = asyncio.create_task(self.node_registry.request(
                        job["target_node_id"], "POST",
                        self._model_agent_path(job["model_id"], "import-from-peer"),
                        json_body={
                            "source_agent_url": direct_source,
                            "source_capability": source_capability,
                            "model_bytes": actual_size,
                            "transfer_id": job["id"],
                        },
                        timeout=3600,
                    ))
                    while not operation.done():
                        try:
                            await asyncio.wait_for(event.wait(), timeout=0.5)
                        except TimeoutError:
                            pass
                        if event.is_set():
                            await self.node_registry.request(job["target_node_id"], "DELETE", self._model_agent_path(job["model_id"], f"peer-imports/{job['id']}"), timeout=15)
                            operation.cancel()
                            await asyncio.gather(operation, return_exceptions=True)
                            raise TransferCanceled()
                        try:
                            status = await self.node_registry.request(job["target_node_id"], "GET", self._model_agent_path(job["model_id"], f"peer-imports/{job['id']}"), timeout=5)
                            job["bytes_transferred"] = min(actual_size, _nonnegative_int((status or {}).get("bytes_transferred")))
                            self._save_progress(job)
                        except Exception:
                            pass
                    result = await operation
                    received = _nonnegative_int((result or {}).get("bytes_received"))
                    if received != actual_size:
                        raise RuntimeError("target did not receive the complete direct transfer")
                    job["bytes_transferred"] = received
                    source = None
                else:
                    source_response = await self.node_registry.open_stream(
                        job["source_node_id"], "GET",
                        self._model_agent_path(job["model_id"], "export"), timeout=3600,
                    )
                    await _raise_remote_status(source_response, "source")
                    source = source_response.aiter_bytes()

            async def tracked() -> AsyncIterator[bytes]:
                async for chunk in source:
                    if event.is_set():
                        raise TransferCanceled()
                    job["bytes_transferred"] += len(chunk)
                    self._save_progress(job)
                    yield chunk

            if source is None:
                tracked_stream = None
            else:
                tracked_stream = tracked()
            if source is None:
                pass
            elif job["target_node_id"] == LOCAL_NODE_ID:
                await self.import_model(
                    job["model_id"], tracked_stream,
                    required_model_bytes=actual_size,
                )
            else:
                target_response = await self.node_registry.open_stream(
                    job["target_node_id"], "PUT",
                    self._model_agent_path(job["model_id"], "import"),
                    content=tracked_stream,
                    headers={
                        "Content-Type": "application/x-tar",
                        "X-SparkDeck-Model-Bytes": str(actual_size),
                    },
                    timeout=3600,
                )
                await _raise_remote_status(target_response, "target")
                await target_response.aread()
            imported_storage = await self._node_storage(job["target_node_id"])
            imported_model = next((
                item for item in imported_storage["models"]
                if item.get("model_id") == job["model_id"]
                and (not job.get("revision") or self._has_revision(
                    item, job["revision"],
                    job.get("requested_revision") or job["revision"],
                ))
            ), None)
            if imported_model is None:
                raise RuntimeError("target did not report the complete requested revision")
            if event.is_set() or job["status"] == "canceled":
                raise TransferCanceled()
            job["status"] = "completed"
            job["bytes_total"] = job["bytes_transferred"]
            job["completed_at"] = time.time()
        except TransferCanceled:
            job["status"] = "canceled"
            job["completed_at"] = time.time()
            job["error"] = None
        except asyncio.CancelledError:
            if job["status"] != "canceled":
                job["status"] = "queued"
                job["started_at"] = None
                job["error"] = None
            raise
        except Exception as exc:
            job["status"] = "failed"
            job["completed_at"] = time.time()
            job["error"] = str(exc)[:500]
        finally:
            if tracked_stream is not None:
                await tracked_stream.aclose()
            if source is not None and hasattr(source, "aclose"):
                await source.aclose()
            if target_response is not None:
                await target_response.aclose()
            if source_response is not None:
                await source_response.aclose()
            self._cancel_events.pop(job["id"], None)
            self._last_progress_save.pop(job["id"], None)
            self._save()
            self._wake.set()

    def _save_progress(self, job: dict[str, Any]) -> None:
        now = time.monotonic()
        if now - self._last_progress_save.get(job["id"], 0.0) >= 0.25:
            self._last_progress_save[job["id"]] = now
            self._save()

    @staticmethod
    def _model_agent_path(model_id: str, action: str) -> str:
        return f"/api/agent/virtual-nas/models/{quote(model_id, safe='')}/{action}"


def _validate_tar_member(member: tarfile.TarInfo, archive_name: str) -> None:
    if "\\" in member.name or not member.name:
        raise ValueError("model archive contains an unsafe path")
    path = PurePosixPath(member.name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("model archive contains an unsafe path")
    if not path.parts or path.parts[0] != archive_name:
        raise ValueError("model archive contains an unexpected repository")
    if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
        raise ValueError("model archive contains an unsupported special file")
    if member.issym() or member.islnk():
        if "\\" in member.linkname:
            raise ValueError("model archive contains an unsafe link")
        link = PurePosixPath(member.linkname)
        if link.is_absolute():
            raise ValueError("model archive contains an unsafe link")
        base = path.parent if member.issym() else PurePosixPath()
        normalized = PurePosixPath(posixpath.normpath(str(base / link)))
        if not normalized.parts or normalized.parts[0] != archive_name or ".." in normalized.parts:
            raise ValueError("model archive link escapes its repository")


async def _raise_remote_status(response: Any, role: str) -> None:
    if 200 <= int(response.status_code) < 300:
        return
    payload = await response.aread()
    detail = payload.decode("utf-8", errors="replace")[:500]
    raise RuntimeError(f"virtual NAS {role} error ({response.status_code}): {detail}")


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _disk_free_bytes(disk: Any) -> int | None:
    if not isinstance(disk, dict):
        return None
    value = disk.get("free")
    if value is None:
        value = disk.get("free_bytes")
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _local_free_bytes(path: Path) -> int:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    try:
        return int(shutil.disk_usage(candidate).free)
    except OSError as exc:
        raise RuntimeError("local node free disk capacity is unavailable") from exc


def _validate_repo_relative_file(filename: str) -> str:
    value = str(filename or "")
    relative = PurePosixPath(value)
    if (
        len(value) > 1024
        or relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in value
    ):
        raise ValueError("repository file must be a safe relative path")
    return relative.as_posix()


def _ensure_safe_snapshot_directory(repository: Path, revision: str) -> Path:
    if repository.exists() and (not repository.is_dir() or repository.is_symlink()):
        raise RuntimeError("Hugging Face cache repository is not safe")
    repository.mkdir(parents=True, exist_ok=True)
    snapshots = repository / "snapshots"
    if snapshots.exists() and (not snapshots.is_dir() or snapshots.is_symlink()):
        raise RuntimeError("Hugging Face snapshots directory is not safe")
    snapshots.mkdir(exist_ok=True)
    snapshot = snapshots / revision
    if snapshot.exists() and (not snapshot.is_dir() or snapshot.is_symlink()):
        raise RuntimeError("Hugging Face snapshot directory is not safe")
    snapshot.mkdir(exist_ok=True)
    return snapshot


def _safe_cached_snapshot_file(
    repository: Path, revision: str, filename: str,
) -> Path | None:
    try:
        relative = PurePosixPath(_validate_repo_relative_file(filename))
        snapshot = repository / "snapshots" / revision
        candidate = snapshot.joinpath(*relative.parts)
        resolved = candidate.resolve(strict=True)
        allowed = (
            (repository / "blobs").resolve(strict=True)
            if candidate.is_symlink()
            else snapshot.resolve(strict=True)
        )
        resolved.relative_to(allowed)
        if not candidate.is_file():
            return None
        return candidate
    except (OSError, RuntimeError, ValueError):
        return None


def _safe_incomplete_blob_bytes(
    repository: Path, cache_key: str | None, expected_bytes: int,
) -> int:
    """Return resumable bytes for one exact Hub sibling, failing closed."""
    if not isinstance(cache_key, str) or not _HUB_BLOB_KEY.fullmatch(cache_key):
        return 0
    try:
        if not repository.is_dir() or repository.is_symlink():
            return 0
        blobs = repository / "blobs"
        if not blobs.is_dir() or blobs.is_symlink():
            return 0
        blob_root = blobs.resolve(strict=True)
        candidate = blobs / f"{cache_key.lower()}.incomplete"
        if candidate.is_symlink() or not candidate.is_file():
            return 0
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(blob_root)
        return min(_nonnegative_int(expected_bytes), candidate.stat().st_size)
    except (OSError, RuntimeError, ValueError):
        return 0


def _is_complete_repository(repository: Path) -> bool:
    return bool(_complete_snapshot_revisions(repository))


# Safety valve for pathologies like a repository snapshot holding thousands
# of loose files: inventory stays a summary, not a full file dump.
_INVENTORY_FILE_LIMIT = 2000


def _snapshot_files_by_revision(repository: Path) -> dict[str, list[str]]:
    """Map each cached snapshot revision to its repository-relative files.

    Inventory responses stay path-free: these are the same repo-relative
    artifact names the Hub API publishes, so the UI can compare a resolved
    revision's files and mark which quantizations are already on disk.
    Download residue (``.incomplete`` blobs and ``.lock`` files, including
    the selective-snapshot marker) is excluded; files from selective
    snapshots still count because a downloaded GGUF artifact is usable
    as-is.
    """
    files_by_revision: dict[str, list[str]] = {}
    total_names = 0
    try:
        snapshots = repository / "snapshots"
        if not snapshots.is_dir() or snapshots.is_symlink():
            return {}
        for revision in sorted(snapshots.iterdir()):
            if not revision.is_dir() or revision.is_symlink():
                continue
            names: list[str] = []
            for item in revision.rglob("*"):
                if item.is_dir() or not item.is_file():
                    continue
                if item.name.endswith((".incomplete", ".lock")):
                    continue
                # Mirror the launch path's containment validation so a name
                # reported as cached is exactly a file the launcher will
                # accept: symlinked snapshot entries must resolve inside the
                # repository's blobs directory, never outside the cache.
                relative = item.relative_to(revision).as_posix()
                if _safe_cached_snapshot_file(repository, revision.name, relative) is None:
                    continue
                names.append(relative)
                total_names += 1
                # Enforce the cap inside the walk so a single huge snapshot
                # cannot make traversal collect an unbounded name list.
                if total_names >= _INVENTORY_FILE_LIMIT:
                    break
            if names:
                files_by_revision[revision.name] = names
            if total_names >= _INVENTORY_FILE_LIMIT:
                break
    except OSError:
        return files_by_revision
    return files_by_revision


def _selective_files_by_revision(
    repository: Path, snapshot_files: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Return exact selected files for partial snapshots without exposing paths.

    New selective pulls persist their chosen files in the marker. Older caches
    used a text-only marker, so fall back to their safely cached snapshot files
    instead of treating every quantization in the repository as required.
    """
    result: dict[str, list[str]] = {}
    try:
        snapshots = repository / "snapshots"
        if not snapshots.is_dir() or snapshots.is_symlink():
            return result
        for snapshot in snapshots.iterdir():
            if not snapshot.is_dir() or snapshot.is_symlink():
                continue
            marker = snapshot / _SELECTIVE_SNAPSHOT_MARKER
            if not marker.is_file() or marker.is_symlink():
                continue
            selected: list[str] = []
            try:
                if marker.stat().st_size <= 1024 * 1024:
                    value = json.loads(marker.read_text(encoding="utf-8"))
                    raw_files = value.get("files") if isinstance(value, dict) else None
                    if isinstance(raw_files, list):
                        selected = list(dict.fromkeys(
                            _validate_repo_relative_file(filename)
                            for filename in raw_files
                            if isinstance(filename, str)
                        ))
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                selected = []
            # Text markers from previous SparkDeck versions have no manifest.
            # The cache inventory is still a strictly better denominator than
            # the full multi-quantization Hub repository.
            if not selected:
                selected = list(snapshot_files.get(snapshot.name) or [])
            if selected:
                result[snapshot.name] = selected
    except OSError:
        pass
    return result


def _complete_snapshot_revisions(repository: Path) -> set[str]:
    """Return only revisions containing a usable, fully resolved model snapshot."""
    try:
        snapshots = repository / "snapshots"
        blobs = repository / "blobs"
        if not snapshots.is_dir() or snapshots.is_symlink():
            return set()
        blob_root = None
        if blobs.exists():
            if not blobs.is_dir() or blobs.is_symlink():
                return set()
            blob_root = blobs.resolve(strict=True)
        complete = set()
        for snapshot in snapshots.iterdir():
            if (
                snapshot.is_dir() and not snapshot.is_symlink()
                and _is_complete_snapshot(snapshot, blob_root)
            ):
                complete.add(snapshot.name)
        return complete
    except OSError:
        return set()


def _resolve_comfyui_bundle_files(
    root: Path,
    required_files: tuple[str, ...],
    alternative_groups: tuple[tuple[str, ...], ...],
    optional_files: tuple[str, ...],
) -> dict[str, Path] | None:
    """Resolve one bundle under an already-resolved ComfyUI models root.

    Returns each installed file as relative POSIX name -> resolved absolute
    path, or None when a required file or a whole alternative group is
    missing, symlinked, escaping the root, or empty.
    """
    files: dict[str, Path] = {}

    def installed(relative: str) -> Path | None:
        candidate = root.joinpath(*PurePosixPath(relative).parts)
        if candidate.is_symlink() or not candidate.is_file():
            return None
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root) or resolved.stat().st_size <= 0:
            return None
        return resolved

    try:
        for relative in required_files:
            resolved = installed(relative)
            if resolved is None:
                return None
            files[relative] = resolved
        for alternatives in alternative_groups:
            installed_alternatives = []
            for relative in alternatives:
                try:
                    resolved = installed(relative)
                except (OSError, RuntimeError, ValueError):
                    resolved = None
                if resolved is not None:
                    installed_alternatives.append((relative, resolved))
            if not installed_alternatives:
                return None
            files.update(installed_alternatives)
        for relative in optional_files:
            try:
                resolved = installed(relative)
            except (OSError, RuntimeError, ValueError):
                continue
            if resolved is not None:
                files[relative] = resolved
    except (OSError, RuntimeError, ValueError):
        return None
    return files


def _external_comfyui_bundle_copies(
    roots: list[Path], model_id: str,
) -> list[dict[str, Path]]:
    """Resolve every complete installation of a known ComfyUI bundle.

    Returns one relative POSIX name -> resolved absolute path mapping per
    root holding a complete, safe installation; empty when the model is not
    a known bundle or no root has it fully installed. Never trust a stale
    inventory snapshot: callers re-resolve at operation time.
    """
    bundle = next(
        (entry for entry in _COMFYUI_MODEL_BUNDLES if entry[0] == model_id), None,
    )
    if bundle is None:
        return []
    _, required_files, alternative_groups, optional_files = bundle
    copies: list[dict[str, Path]] = []
    seen: set[str] = set()
    for raw_root in roots:
        try:
            if raw_root.is_symlink() or not raw_root.is_dir():
                continue
            root = raw_root.resolve(strict=True)
        except OSError:
            continue
        key = os.path.normcase(str(root))
        if key in seen:
            continue
        seen.add(key)
        files = _resolve_comfyui_bundle_files(
            root, required_files, alternative_groups, optional_files,
        )
        if files is not None:
            copies.append(files)
    return copies


def _external_comfyui_bundle_files(
    roots: list[Path], model_id: str,
) -> dict[str, Path] | None:
    """Resolve the copy of a known bundle that inventory would report.

    Inventory keeps the largest complete installation across roots, so
    export and transfer read from the largest copy too; otherwise the
    streamed bytes could differ from the reported inventory entry.
    """
    best: dict[str, Path] | None = None
    best_size = 0
    for files in _external_comfyui_bundle_copies(roots, model_id):
        try:
            size = sum(path.stat().st_size for path in files.values())
        except OSError:
            continue
        if best is None or size > best_size:
            best = files
            best_size = size
    return best


def _write_external_bundle_archive(
    archive: tarfile.TarFile, model_id: str, files: dict[str, Path],
) -> None:
    """Stream external bundle files as a synthetic Hugging Face cache repo.

    The layout matches a regular whole-model export so targets import a
    normal SparkDeck-managed copy: bundle files land under
    ``snapshots/<pseudo-revision>/`` alongside a marker that relaxes the
    config/tokenizer completeness requirement for bare weight bundles, and
    ``refs/main`` points at the pseudo-revision so consumers resolving the
    default revision find the snapshot.
    """
    root = _cache_name(model_id)
    snapshot = f"{root}/snapshots/{_EXTERNAL_PSEUDO_REVISION}"

    def add_directory(name: str) -> None:
        info = tarfile.TarInfo(name)
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        archive.addfile(info)

    def add_file(name: str, content: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

    add_directory(root)
    add_directory(f"{root}/snapshots")
    add_directory(snapshot)
    directories: set[str] = set()
    for relative in files:
        parts = PurePosixPath(relative).parts
        for depth in range(1, len(parts)):
            directory = "/".join(parts[:depth])
            if directory not in directories:
                directories.add(directory)
                add_directory(f"{snapshot}/{directory}")
    for relative, resolved in files.items():
        archive.add(resolved, arcname=f"{snapshot}/{relative}", recursive=False)
    add_file(f"{snapshot}/{_EXTERNAL_SNAPSHOT_MARKER}", _EXTERNAL_MARKER_CONTENT)
    add_directory(f"{root}/refs")
    add_file(f"{root}/refs/main", _EXTERNAL_PSEUDO_REVISION.encode("utf-8"))


def _external_comfyui_inventory(roots: list[Path]) -> list[dict[str, Any]]:
    """Report known complete ComfyUI bundles as installed node weights."""
    models: dict[str, dict[str, Any]] = {}
    for raw_root in roots:
        try:
            if raw_root.is_symlink() or not raw_root.is_dir():
                continue
            root = raw_root.resolve(strict=True)
        except OSError:
            continue
        claimed: set[Path] = set()
        for model_id, required_files, alternative_groups, optional_files in _COMFYUI_MODEL_BUNDLES:
            files = _resolve_comfyui_bundle_files(
                root, required_files, alternative_groups, optional_files,
            )
            if files is None:
                continue
            try:
                stats = [path.stat() for path in files.values()]
            except OSError:
                continue
            size_bytes = sum(stat.st_size for stat in stats)
            last_modified = max(stat.st_mtime for stat in stats)
            candidate = {
                "model_id": model_id,
                "size_bytes": size_bytes,
                # The transferable payload is the external bundle alone, even
                # when inventory merges in partial Hugging Face cache residue.
                "transfer_size_bytes": size_bytes,
                "file_count": len(files),
                "partial": False,
                "has_partial_download": False,
                "partial_size_bytes": 0,
                "partial_revision_size_bytes": {},
                "partial_revisions": [],
                "partial_revision_refs": {},
                "revision": "ComfyUI",
                "revisions": [],
                "revision_refs": {},
                "last_modified": datetime.fromtimestamp(
                    last_modified, timezone.utc,
                ).isoformat(),
                "source": "ComfyUI",
                "externally_managed": True,
            }
            current = models.get(model_id)
            if current is None or candidate["size_bytes"] > current["size_bytes"]:
                models[model_id] = candidate
            claimed.update(files.values())
        for entry in _comfyui_fallback_entries(root, claimed):
            entry_id = entry["model_id"]
            current = models.get(entry_id)
            if current is None or entry["size_bytes"] > current["size_bytes"]:
                models[entry_id] = entry
    return sorted(models.values(), key=lambda item: str(item["model_id"]))


def _comfyui_fallback_entries(root: Path, claimed: set[Path]) -> list[dict[str, Any]]:
    """Report unrecognized ComfyUI weights grouped by section directory.

    Weight files inside a dedicated subdirectory of a conventional section
    (e.g. ``checkpoints/Minimax Video/``) collapse into one entry named by
    that directory; loose files at a section root become one entry per file
    stem. Files already claimed by a known bundle are never duplicated.
    """
    entries: list[dict[str, Any]] = []
    for section in _COMFYUI_SECTION_DIRS:
        try:
            section_dir = root / section
            if section_dir.is_symlink() or not section_dir.is_dir():
                continue
            resolved_section = section_dir.resolve(strict=True)
            if not resolved_section.is_relative_to(root):
                continue
        except OSError:
            continue
        groups: dict[str, list[Path]] = {}
        try:
            for item in resolved_section.rglob("*"):
                try:
                    if item.is_symlink() or not item.is_file():
                        continue
                    resolved = item.resolve(strict=True)
                    if not resolved.is_relative_to(root) or resolved in claimed:
                        continue
                    relative = resolved.relative_to(resolved_section)
                    if (
                        any(part.startswith(".") for part in relative.parts)
                        or not _is_weight_file(relative.as_posix().casefold())
                        or f"{section}/{relative.as_posix()}" in _COMFYUI_BUNDLE_PATHS
                        or resolved.stat().st_size <= 0
                    ):
                        continue
                except (OSError, RuntimeError, ValueError):
                    continue
                parent = relative.parent
                group = parent.parts[0] if str(parent) != "." else relative.stem
                groups.setdefault(group, []).append(resolved)
        except (OSError, RuntimeError, ValueError):
            continue
        for name, files in groups.items():
            try:
                stats = [path.stat() for path in files]
            except OSError:
                continue
            last_modified = max(stat.st_mtime for stat in stats)
            entries.append({
                "model_id": f"{section}/{name}",
                "size_bytes": sum(stat.st_size for stat in stats),
                "file_count": len(files),
                "partial": False,
                "has_partial_download": False,
                "partial_size_bytes": 0,
                "partial_revision_size_bytes": {},
                "partial_revisions": [],
                "partial_revision_refs": {},
                "revision": "ComfyUI",
                "revisions": [],
                "revision_refs": {},
                "last_modified": datetime.fromtimestamp(
                    last_modified, timezone.utc,
                ).isoformat(),
                "source": "ComfyUI",
                "externally_managed": True,
                "transferable": False,
                "deletable": False,
            })
    return entries


def _safe_incomplete_snapshot_revisions(
    repository: Path, complete_revisions: set[str],
) -> set[str]:
    """Return incomplete snapshot directories whose contents stay cache-local."""
    try:
        snapshots = repository / "snapshots"
        blobs = repository / "blobs"
        if (
            not snapshots.is_dir() or snapshots.is_symlink()
            or not blobs.is_dir() or blobs.is_symlink()
        ):
            return set()
        blob_root = blobs.resolve(strict=True)
        incomplete = set()
        for snapshot in snapshots.iterdir():
            if (
                snapshot.name in complete_revisions
                or not _is_commit_sha(snapshot.name)
                or not snapshot.is_dir() or snapshot.is_symlink()
            ):
                continue
            try:
                safe = True
                has_evidence = False
                for item in snapshot.rglob("*"):
                    if item.is_symlink():
                        if item.name == _SELECTIVE_SNAPSHOT_MARKER or not item.is_file():
                            safe = False
                            break
                        resolved = item.resolve(strict=True)
                        if not resolved.is_relative_to(blob_root) or not resolved.is_file():
                            safe = False
                            break
                        has_evidence = True
                    elif item.is_dir():
                        continue
                    elif not item.is_file():
                        safe = False
                        break
                    elif item.name == _SELECTIVE_SNAPSHOT_MARKER:
                        has_evidence = True
                    elif not item.name.endswith((".incomplete", ".lock")):
                        has_evidence = True
                if safe and has_evidence:
                    incomplete.add(snapshot.name)
            except (OSError, RuntimeError, ValueError):
                continue
        return incomplete
    except OSError:
        return set()


def _incomplete_snapshot_reusable_bytes(
    repository: Path, complete_revisions: set[str],
    incomplete_revisions: set[str],
) -> dict[str, int]:
    """Count unique completed blobs referenced only by incomplete snapshots."""
    try:
        snapshots = repository / "snapshots"
        blobs = repository / "blobs"
        if not snapshots.is_dir() or snapshots.is_symlink():
            return {}
        blob_root = None
        if blobs.exists():
            if not blobs.is_dir() or blobs.is_symlink():
                return {}
            blob_root = blobs.resolve(strict=True)
        complete_blobs: set[Path] = set()
        for revision in complete_revisions:
            snapshot = snapshots / revision
            if not snapshot.is_dir() or snapshot.is_symlink():
                continue
            for item in snapshot.rglob("*"):
                if not item.is_symlink() or not item.is_file():
                    continue
                if blob_root is None:
                    continue
                resolved = item.resolve(strict=True)
                if resolved.is_relative_to(blob_root) and resolved.is_file():
                    complete_blobs.add(resolved)
        reusable_by_revision: dict[str, set[Path]] = {}
        for revision in incomplete_revisions:
            snapshot = snapshots / revision
            reusable = reusable_by_revision.setdefault(snapshot.name, set())
            for item in snapshot.rglob("*"):
                if item.name.endswith((".incomplete", ".lock")):
                    continue
                if item.is_symlink():
                    if blob_root is None or not item.is_file():
                        continue
                    resolved = item.resolve(strict=True)
                    if (
                        resolved.is_relative_to(blob_root)
                        and resolved.is_file()
                        and resolved not in complete_blobs
                    ):
                        reusable.add(resolved)
                elif item.is_file():
                    reusable.add(item)
        reusable_sizes: dict[str, int] = {}
        for revision, reusable in reusable_by_revision.items():
            size_bytes = sum(item.stat().st_size for item in reusable)
            if size_bytes > 0:
                reusable_sizes[revision] = size_bytes
        return reusable_sizes
    except (OSError, RuntimeError, ValueError):
        return {}


def _is_complete_snapshot(snapshot: Path, blob_root: Path | None) -> bool:
    files: dict[str, Path] = {}
    try:
        for item in snapshot.rglob("*"):
            relative = item.relative_to(snapshot).as_posix()
            if item.is_symlink():
                if blob_root is None or not item.is_file():
                    return False
                resolved = item.resolve(strict=True)
                if not resolved.is_relative_to(blob_root) or not resolved.is_file():
                    return False
            elif item.is_dir():
                continue
            elif not item.is_file():
                return False
            files[relative] = item
    except (OSError, RuntimeError, ValueError):
        return False
    if not files or any(name.endswith((".incomplete", ".lock")) for name in files):
        return False

    lowered = {name.casefold(): path for name, path in files.items()}
    weights = {name: path for name, path in lowered.items() if _is_weight_file(name)}
    if not weights or not _required_files_are_nonempty(weights.values()):
        return False
    if not _weight_shards_are_complete(weights):
        return False
    if not _weight_indexes_are_complete(lowered):
        return False

    if _EXTERNAL_SNAPSHOT_MARKER in files:
        # Snapshots exported from an externally managed ComfyUI install are
        # bare weight files: no config, tokenizer, or Diffusers metadata, so
        # the weight checks above are the whole completeness contract.
        return True

    if _is_complete_diffusers_snapshot(lowered):
        return True

    # GGUF contains its own tensor metadata and tokenizer vocabulary. Other
    # Transformers runtimes need both configuration and tokenizer assets.
    if any(name.endswith(".gguf") for name in weights):
        return True
    if "model_index.json" in lowered:
        # A declared Diffusers pipeline that failed validation is an
        # incomplete pipeline, not a raw weights-only repository.
        return False
    config = lowered.get("config.json")
    if config is None:
        # A raw weights-only repository is complete only when the cache has
        # no unfinished Hub blobs. Those blobs live outside the snapshot, so
        # checking the snapshot alone would misclassify an interrupted pull.
        if blob_root is None:
            # Minimal test/portable caches may not materialize a blobs mount;
            # there is then no external unfinished-blob evidence to reject.
            return True
        try:
            if any(blob.is_file() for blob in blob_root.rglob("*.incomplete")):
                return False
        except (OSError, RuntimeError):
            return False
        return True
    if not _required_files_are_nonempty([config]):
        return False
    tokenizer = [
        path for name, path in lowered.items()
        if Path(name).name in _TOKENIZER_FILES
    ]
    if not tokenizer:
        by_filename = {Path(name).name: path for name, path in lowered.items()}
        if not {"vocab.json", "merges.txt"} <= by_filename.keys():
            return False
        tokenizer = [by_filename["vocab.json"], by_filename["merges.txt"]]
    if not _required_files_are_nonempty(tokenizer):
        return False
    return True


def _is_complete_diffusers_snapshot(files: dict[str, Path]) -> bool:
    """Recognize a self-contained Diffusers pipeline rooted by model_index.json."""
    index = files.get("model_index.json")
    if index is None or not _required_files_are_nonempty([index]):
        return False
    try:
        manifest = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if (
        not isinstance(manifest, dict)
        or not isinstance(manifest.get("_class_name"), str)
        or not manifest["_class_name"].strip()
    ):
        return False
    components = [
        (str(name).casefold(), descriptor)
        for name, descriptor in manifest.items()
        if not str(name).startswith("_")
        and isinstance(descriptor, list)
        and len(descriptor) >= 2
        and any(value is not None for value in descriptor[:2])
    ]
    if not components:
        return False
    for component, descriptor in components:
        component_files = {
            name: path for name, path in files.items()
            if name.startswith(f"{component}/")
        }
        if not component_files or not _required_files_are_nonempty(component_files.values()):
            return False
        filenames = {PurePosixPath(name).name for name in component_files}
        descriptor_name = " ".join(str(value or "") for value in descriptor[:2]).casefold()
        if "scheduler" in component or "scheduler" in descriptor_name:
            if "scheduler_config.json" not in filenames:
                return False
            continue
        if "tokenizer" in component or "tokenizer" in descriptor_name:
            if not (
                filenames & _TOKENIZER_FILES
                or {"vocab.json", "merges.txt"} <= filenames
            ):
                return False
            continue
        if any(
            marker in component or marker in descriptor_name
            for marker in ("feature_extractor", "image_processor", "processor")
        ):
            if not filenames & {
                "preprocessor_config.json", "feature_extractor_config.json",
                "processor_config.json", "image_processor_config.json",
            }:
                return False
            continue
        component_weights = {
            name: path for name, path in component_files.items()
            if _is_weight_file(name)
        }
        if (
            not component_weights
            or not _weight_shards_are_complete(component_weights)
            or not _weight_indexes_are_complete(component_files)
        ):
            return False
    return True


def _is_weight_file(name: str) -> bool:
    return (
        name.endswith((".safetensors", ".gguf", ".pt", ".pth", ".ckpt", ".onnx"))
        or (
            name.endswith(".bin")
            and Path(name).name.startswith(("pytorch_model", "model", "adapter_model"))
        )
    )


def _required_files_are_nonempty(paths: Any) -> bool:
    try:
        return all(path.stat().st_size > 0 for path in paths)
    except OSError:
        return False


def _weight_shards_are_complete(weights: dict[str, Path]) -> bool:
    groups: dict[tuple[str, str, int], set[int]] = {}
    for name in weights:
        match = _WEIGHT_SHARD.match(name)
        if not match:
            continue
        total = int(match.group("total"))
        part = int(match.group("part"))
        key = (match.group("prefix"), match.group("suffix").casefold(), total)
        groups.setdefault(key, set()).add(part)
    return all(parts == set(range(1, total + 1)) for (*_, total), parts in groups.items())


def _weight_indexes_are_complete(files: dict[str, Path]) -> bool:
    indexes = {
        name: path for name, path in files.items()
        if name.endswith((".safetensors.index.json", ".bin.index.json"))
    }
    required_indexes = set()
    for name in files:
        match = _WEIGHT_SHARD.match(name)
        if match and match.group("suffix").casefold() != ".gguf":
            required_indexes.add(
                f"{match.group('prefix')}{match.group('suffix')}.index.json".casefold()
            )
    if not required_indexes <= indexes.keys():
        return False
    for index_name, path in indexes.items():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            weight_map = value.get("weight_map") if isinstance(value, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                return False
            index_directory = PurePosixPath(index_name).parent
            for raw in weight_map.values():
                candidate = PurePosixPath(str(raw or ""))
                if (
                    not str(raw or "") or candidate.is_absolute()
                    or ".." in candidate.parts
                ):
                    return False
                candidates = {candidate.as_posix().casefold()}
                if index_directory != PurePosixPath("."):
                    candidates.add((index_directory / candidate).as_posix().casefold())
                if not candidates & files.keys():
                    return False
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
    return True
