"""Filesystem browsing, sizing, and deletion for the disk manager UI."""

from __future__ import annotations

import os
import shutil
import stat
import threading
import time
import uuid
from pathlib import Path
from typing import Callable


def _directory(path: str) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("a directory path is required")
    directory = Path(path).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")
    return directory


def browse_directories(path: str) -> dict:
    """Return the readable child directories at *path* for the folder picker."""
    directory = _directory(path)
    children: list[dict] = []
    errors: list[str] = []
    try:
        with os.scandir(directory) as iterator:
            for entry in iterator:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        children.append({"name": entry.name, "path": entry.path})
                except OSError as exc:
                    errors.append(f"{entry.path}: {exc.strerror or exc}")
    except PermissionError as exc:
        raise PermissionError(f"permission denied: {directory}") from exc

    children.sort(key=lambda item: item["name"].casefold())
    parent = str(directory.parent) if directory.parent != directory else None
    return {
        "path": str(directory),
        "parent": parent,
        "directories": children,
        "errors": errors,
    }


def scan_directory(
    path: str,
    progress: Callable[[dict | None, dict], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    """Recursively inventory a directory without following symbolic links.

    Directory sizes are the sum of their descendants' logical file sizes.
    Unreadable entries are skipped and returned in ``errors``.
    """
    root = _directory(path)
    entries: list[dict] = []
    errors: list[str] = []
    skipped_mounts: list[str] = []
    file_count = 0
    directory_count = 0
    directories_started = 0
    directories_scanned = 0
    items_processed = 0
    scanned_size = 0
    root_device = root.stat().st_dev

    def emit(item: dict | None = None, current_path: str | None = None) -> None:
        if progress is None:
            return
        progress(dict(item) if item is not None else None, {
            "file_count": file_count,
            "directory_count": directory_count,
            "directories_started": directories_started,
            "directories_scanned": directories_scanned,
            "items_processed": items_processed,
            "scanned_size": scanned_size,
            "current_path": current_path,
            "errors_count": len(errors),
            "skipped_mounts_count": len(skipped_mounts),
        })

    def walk(
        directory: Path,
        relative_parent: Path,
        directory_item: dict | None = None,
    ) -> int:
        nonlocal file_count, directory_count, directories_started
        nonlocal directories_scanned, items_processed, scanned_size
        if cancelled is not None and cancelled():
            return 0
        directories_started += 1
        total = 0
        try:
            with os.scandir(directory) as iterator:
                children = list(iterator)
        except OSError as exc:
            errors.append(f"{directory}: {exc.strerror or exc}")
            directories_scanned += 1
            emit(current_path=str(relative_parent) or ".")
            return 0

        for child in children:
            if cancelled is not None and cancelled():
                break
            relative = relative_parent / child.name
            try:
                info = child.stat(follow_symlinks=False)
                is_directory = stat.S_ISDIR(info.st_mode)
                is_symlink = stat.S_ISLNK(info.st_mode)
                if is_directory:
                    directory_count += 1
                    item = {
                        "name": child.name,
                        "path": str(relative),
                        "type": "mount" if info.st_dev != root_device else "directory",
                        "size": 0,
                        "modified": info.st_mtime,
                    }
                    entries.append(item)
                    items_processed += 1
                    emit(item, str(relative))
                    if info.st_dev == root_device:
                        item["size"] = walk(Path(child.path), relative, item)
                    else:
                        skipped_mounts.append(str(relative))
                    item_size = item["size"]
                else:
                    file_count += 1
                    item_size = info.st_size
                    item = {
                        "name": child.name,
                        "path": str(relative),
                        "type": "symlink" if is_symlink else "file",
                        "size": item_size,
                        "modified": info.st_mtime,
                    }
                    entries.append(item)
                    items_processed += 1
                    scanned_size += item_size
                    emit(item, str(relative))
                total += item_size
                if directory_item is not None:
                    directory_item["size"] = total
                    emit(directory_item, str(relative_parent))
            except OSError as exc:
                errors.append(f"{child.path}: {exc.strerror or exc}")
                emit(current_path=str(relative))
        directories_scanned += 1
        emit(current_path=str(relative_parent) or ".")
        return total

    total_size = walk(root, Path())
    return {
        "root": str(root),
        "total_size": total_size,
        "file_count": file_count,
        "directory_count": directory_count,
        "entries": entries,
        "errors": errors,
        "skipped_mounts": skipped_mounts,
        "cancelled": bool(cancelled is not None and cancelled()),
    }


class DiskScanJobs:
    """Run filesystem scans in background threads and expose incremental deltas."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def start(self, path: str) -> dict:
        root = _directory(path)
        scan_id = uuid.uuid4().hex
        job = {
            "id": scan_id,
            "root": str(root),
            "status": "scanning",
            "created_at": time.time(),
            "started_monotonic": time.monotonic(),
            "changes": [],
            "cancel": threading.Event(),
            "progress_pct": 0.0,
            "eta_seconds": None,
            "rate": 0.0,
            "file_count": 0,
            "directory_count": 0,
            "directories_scanned": 0,
            "items_processed": 0,
            "scanned_size": 0,
            "current_path": ".",
            "errors": [],
            "skipped_mounts": [],
            "total_size": 0,
        }
        with self._lock:
            finished = [
                key for key, old in self._jobs.items()
                if old["status"] != "scanning" and time.time() - old["created_at"] > 3600
            ]
            for key in finished:
                del self._jobs[key]
            self._jobs[scan_id] = job
        threading.Thread(target=self._run, args=(scan_id,), daemon=True).start()
        return {"scan_id": scan_id, "root": str(root)}

    def _run(self, scan_id: str) -> None:
        job = self._jobs[scan_id]

        def on_progress(entry: dict | None, stats: dict) -> None:
            elapsed = max(0.001, time.monotonic() - job["started_monotonic"])
            processed = stats["items_processed"] + stats["directories_scanned"]
            pending_directories = max(
                0, stats["directory_count"] + 1 - stats["directories_scanned"],
            )
            average_entries = stats["items_processed"] / max(1, stats["directories_started"])
            estimated_remaining = pending_directories * max(1.0, average_entries)
            raw_pct = 100.0 * processed / max(1.0, processed + estimated_remaining)
            rate = processed / elapsed
            eta = estimated_remaining / rate if rate > 0 and processed >= 10 else None
            with self._lock:
                if entry is not None:
                    job["changes"].append(entry)
                job.update(stats)
                job["rate"] = rate
                job["eta_seconds"] = eta
                job["progress_pct"] = max(
                    job["progress_pct"], min(95.0, raw_pct),
                )

        try:
            result = scan_directory(
                job["root"], on_progress, job["cancel"].is_set,
            )
            with self._lock:
                job.update({
                    "status": "cancelled" if result["cancelled"] else "complete",
                    "file_count": result["file_count"],
                    "directory_count": result["directory_count"],
                    "total_size": result["total_size"],
                    "scanned_size": result["total_size"],
                    "errors": result["errors"],
                    "skipped_mounts": result["skipped_mounts"],
                    "eta_seconds": 0,
                    "progress_pct": 100.0 if not result["cancelled"] else job["progress_pct"],
                })
        except Exception as exc:
            with self._lock:
                job["status"] = "failed"
                job["errors"] = [str(exc)]
                job["eta_seconds"] = None

    def poll(self, scan_id: str, since: int = 0, limit: int = 5000) -> dict:
        if since < 0:
            raise ValueError("scan cursor cannot be negative")
        with self._lock:
            job = self._jobs.get(scan_id)
            if job is None:
                raise KeyError("scan not found")
            end = min(len(job["changes"]), since + max(1, min(limit, 10000)))
            status = job["status"]
            # A busy directory can receive thousands of size updates between
            # browser polls. Send only its newest value in this window while
            # still advancing the cursor across every recorded change.
            latest_changes: dict[str, dict] = {}
            for item in job["changes"][since:end]:
                latest_changes[item["path"]] = item
            return {
                "scan_id": scan_id,
                "root": job["root"],
                "status": status,
                "complete": status != "scanning" and end == len(job["changes"]),
                "changes": [dict(item) for item in latest_changes.values()],
                "next_cursor": end,
                "file_count": job["file_count"],
                "directory_count": job["directory_count"],
                "directories_scanned": job["directories_scanned"],
                "items_processed": job["items_processed"],
                "scanned_size": job["scanned_size"],
                "total_size": job["total_size"],
                "current_path": job["current_path"],
                "progress_pct": round(job["progress_pct"], 1),
                "eta_seconds": round(job["eta_seconds"], 1) if job["eta_seconds"] is not None else None,
                "rate": round(job["rate"], 1),
                "errors": list(job["errors"]),
                "skipped_mounts": list(job["skipped_mounts"]),
            }

    def cancel(self, scan_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(scan_id)
            if job is None:
                raise KeyError("scan not found")
            job["cancel"].set()
            return {"scan_id": scan_id, "cancelling": job["status"] == "scanning"}


def _deletion_target(root: Path, raw_path: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("each deletion path must be a non-empty string")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise ValueError("deletion paths must be relative to the scanned directory")

    target = Path(os.path.abspath(root / relative))
    if target == root or root not in target.parents:
        raise ValueError(f"path is outside the scanned directory: {raw_path}")

    # Resolve the parent to prevent a path from traversing a symlink out of the
    # selected root. The final component is deliberately not resolved so the
    # symlink itself can still be permanently unlinked.
    resolved_parent = target.parent.resolve()
    if resolved_parent != root and root not in resolved_parent.parents:
        raise ValueError(f"path traverses outside the scanned directory: {raw_path}")
    return target


def delete_entries(root_path: str, paths: list[str]) -> dict:
    """Permanently remove selected items beneath a previously scanned root."""
    root = _directory(root_path)
    if not isinstance(paths, list) or not paths:
        raise ValueError("at least one item must be selected")

    targets = [_deletion_target(root, raw_path) for raw_path in paths]
    unique = sorted(set(targets), key=lambda target: len(target.parts))
    collapsed: list[Path] = []
    for target in unique:
        if any(parent == target or parent in target.parents for parent in collapsed):
            continue
        collapsed.append(target)

    deleted: list[str] = []
    errors: list[dict] = []
    root_device = root.stat().st_dev
    for target in collapsed:
        try:
            mode = target.lstat().st_mode
            if stat.S_ISDIR(mode):
                if target.stat().st_dev != root_device:
                    raise OSError("refusing to delete a mounted filesystem")
                for directory, child_directories, _ in os.walk(target, followlinks=False):
                    kept: list[str] = []
                    for name in child_directories:
                        child = Path(directory) / name
                        child_info = child.stat(follow_symlinks=False)
                        if stat.S_ISLNK(child_info.st_mode):
                            kept.append(name)
                        elif child_info.st_dev != root_device:
                            raise OSError(f"refusing to cross mounted filesystem: {child}")
                        else:
                            kept.append(name)
                    child_directories[:] = kept
                shutil.rmtree(target)
            else:
                target.unlink()
            deleted.append(str(target.relative_to(root)))
        except FileNotFoundError:
            errors.append({"path": str(target.relative_to(root)), "error": "item no longer exists"})
        except OSError as exc:
            errors.append({
                "path": str(target.relative_to(root)),
                "error": exc.strerror or str(exc),
            })

    return {"root": str(root), "deleted": deleted, "errors": errors}
