"""Secure, durable model-cache transfers between SparkDeck nodes."""

from __future__ import annotations

import asyncio
import json
import os
import posixpath
import queue
import re
import shutil
import tarfile
import tempfile
import threading
import time
import uuid
import weakref
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator, Awaitable, Callable
from urllib.parse import quote


LOCAL_NODE_ID = "local"
_MODEL_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
VIRTUAL_NAS_DOWNLOAD_CAPABILITY = "virtual-nas-download-v1"
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
    ):
        self.data_dir = Path(data_dir)
        self._hub_path_provider = hub_path_provider
        self.node_registry = node_registry
        self._enabled_provider = enabled_provider
        self._token_provider = token_provider or (lambda: "")
        self.path = self.data_dir / "virtual_nas_transfers.json"
        self.jobs = self._load_jobs()
        self._wake = asyncio.Event()
        self._dispatcher: asyncio.Task | None = None
        self._active: dict[str, asyncio.Task] = {}
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

    async def download_model_checked(
        self, model_id: str, revision: str = "main",
        explicit_token: str | None = None,
        requested_revision: str | None = None,
    ) -> dict[str, Any]:
        """Agent-side capacity gate using only this cache filesystem."""
        model_id = validate_model_id(model_id)
        revision = validate_revision(revision)
        if not _is_commit_sha(revision):
            raise ValueError("download revision must be an immutable Hugging Face commit SHA")
        requested_revision = validate_revision(requested_revision or revision)
        existing = next((
            item for item in await asyncio.to_thread(self.inventory)
            if item.get("model_id") == model_id
            and self._has_revision(item, revision, requested_revision)
        ), None)
        if existing is None:
            expected_bytes = await self.estimate_download_size(
                model_id, revision, explicit_token, force_refresh=True,
            )
            free_bytes = await asyncio.to_thread(self.free_bytes)
            required = expected_bytes * 2 + DOWNLOAD_STAGING_RESERVE_BYTES
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
            return []
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
            size_bytes = 0
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
                    file_count += 1
                    last_modified = max(last_modified, stat.st_mtime)
            revisions = set(snapshot_revisions)
            revision_refs: dict[str, str] = {}
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
                    except (OSError, UnicodeError, ValueError):
                        continue
            models.append({
                "model_id": model_id,
                "size_bytes": size_bytes,
                "file_count": file_count,
                "partial": partial,
                "revisions": sorted(revisions),
                "revision_refs": revision_refs,
                "last_modified": (
                    datetime.fromtimestamp(last_modified, timezone.utc).isoformat()
                    if last_modified else None
                ),
            })
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
        if (
            not repository.is_dir() or repository.is_symlink()
            or not _is_complete_repository(repository)
        ):
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
                        archive.add(repository, arcname=repository.name, recursive=True)
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

    def delete_model(self, model_id: str) -> dict[str, Any]:
        model_id = validate_model_id(model_id)
        if self.model_in_transfer(model_id, LOCAL_NODE_ID):
            raise RuntimeError("model is in use by a virtual NAS transfer")
        repository = self._model_path(model_id)
        if not repository.exists():
            raise LookupError("cached model not found")
        if repository.is_symlink() or not repository.is_dir():
            raise ValueError("cached model repository is not a safe directory")
        hub = self._hub()
        if repository.resolve().parent != hub:
            raise ValueError("model cache path escapes the Hugging Face hub")
        shutil.rmtree(repository)
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
        model_size = _nonnegative_int(source_model.get("size_bytes"))
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
    ) -> dict[str, Any]:
        async with self._queue_lock:
            return await self._queue_download_and_transfer(
                model_id, revision, download_node_id,
                transfer_target_node_ids, expected_bytes, workflow_id,
                workflow_node_ids, additional_download_node_ids,
                source_node_id, requested_revision,
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

        download_required = expected_bytes * 2 + DOWNLOAD_STAGING_RESERVE_BYTES
        for node_id in download_nodes:
            download_storage = await self._node_storage(node_id)
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
            "workflow_id": workflow_id,
            "workflow_node_ids": list(workflow_node_ids or []),
            "status": "queued", "bytes_total": expected_bytes,
            "bytes_transferred": 0, "created_at": created,
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
        status = await self.node_registry.probe(node, force=True)
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
                free_bytes = _optional_nonnegative_int(storage.get("free_size"))
                required = current_size * 2 + DOWNLOAD_STAGING_RESERVE_BYTES
                if free_bytes is None:
                    raise RuntimeError("download node did not report free cache capacity")
                if free_bytes < required:
                    raise RuntimeError(
                        f"download node has insufficient free cache space "
                        f"({free_bytes} bytes available; {required} bytes required)"
                    )
            if job["target_node_id"] == LOCAL_NODE_ID:
                operation = asyncio.to_thread(
                    self.download_model, job["model_id"],
                    job.get("revision") or "main", token,
                    job.get("requested_revision") or job.get("revision") or "main",
                )
            else:
                await self._validate_download_node(job["target_node_id"])
                operation = self.node_registry.request(
                    job["target_node_id"], "POST",
                    self._model_agent_path(job["model_id"], "download"),
                    json_body={
                        "revision": job.get("revision") or "main",
                        "requested_revision": (
                            job.get("requested_revision")
                            or job.get("revision") or "main"
                        ),
                        "hf_token": token,
                    },
                    timeout=24 * 60 * 60,
                )
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
            actual_size = _nonnegative_int(source_model.get("size_bytes"))
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

            tracked_stream = tracked()
            if job["target_node_id"] == LOCAL_NODE_ID:
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


def _is_complete_repository(repository: Path) -> bool:
    return bool(_complete_snapshot_revisions(repository))


def _complete_snapshot_revisions(repository: Path) -> set[str]:
    """Return only revisions containing a usable, fully resolved model snapshot."""
    try:
        snapshots = repository / "snapshots"
        blobs = repository / "blobs"
        if (
            not snapshots.is_dir() or snapshots.is_symlink()
            or not blobs.is_dir() or blobs.is_symlink()
        ):
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


def _is_complete_snapshot(snapshot: Path, blob_root: Path) -> bool:
    files: dict[str, Path] = {}
    try:
        for item in snapshot.rglob("*"):
            relative = item.relative_to(snapshot).as_posix()
            if item.is_symlink():
                if not item.is_file():
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
    weights = {
        name: path for name, path in lowered.items()
        if name.endswith((".safetensors", ".gguf", ".pt", ".pth", ".ckpt", ".onnx"))
        or (
            name.endswith(".bin")
            and Path(name).name.startswith(("pytorch_model", "model", "adapter_model"))
        )
    }
    if not weights or not _required_files_are_nonempty(weights.values()):
        return False
    if not _weight_shards_are_complete(weights):
        return False
    if not _weight_indexes_are_complete(lowered):
        return False

    # GGUF contains its own tensor metadata and tokenizer vocabulary. Other
    # runtimes need both the Transformers configuration and tokenizer assets.
    if any(name.endswith(".gguf") for name in weights):
        return True
    config = lowered.get("config.json")
    if config is None or not _required_files_are_nonempty([config]):
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
    for path in indexes.values():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            weight_map = value.get("weight_map") if isinstance(value, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                return False
            required = set()
            for raw in weight_map.values():
                candidate = PurePosixPath(str(raw or ""))
                if (
                    not str(raw or "") or candidate.is_absolute()
                    or ".." in candidate.parts
                ):
                    return False
                required.add(candidate.as_posix().casefold())
            if not required <= files.keys():
                return False
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
    return True
