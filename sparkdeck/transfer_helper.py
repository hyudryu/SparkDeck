"""Detached, narrowly-scoped controller-transfer apply helper.

``promote`` runs on the target node: it installs a staged controller-state
bundle, removes the worker assignment (the commit point), and restarts the
service so it boots as the controller. ``demote`` runs on the outgoing
controller after it wrote its own worker assignment and only restarts the
service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import time
import urllib.request
from pathlib import Path

from .controller_transfer import (
    NODE_LOCAL_SETTING_KEYS,
    TRANSFER_STATE_FILENAME,
    TRANSFER_STAGING_DIRNAME,
)
from .private_json import atomic_private_json_write
from .update_helper import restart_service


def validate_state_path(state_path: Path, root: Path) -> Path:
    """Restrict writes to the service-owned state file in the data dir."""
    data_dir = (root / "data").resolve()
    resolved = state_path.resolve()
    if resolved.parent != data_dir or resolved.name != TRANSFER_STATE_FILENAME:
        raise RuntimeError(
            "Transfer state path must be the service state file inside the"
            " checkout data directory"
        )
    return resolved


def write_state(path: Path, **changes) -> None:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    state.update(changes, updated_at=time.time())
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_staged_bundle(root: Path) -> tuple[dict, bytes, dict]:
    staging = root / "data" / TRANSFER_STAGING_DIRNAME
    try:
        manifest = json.loads(
            (staging / "manifest.json").read_text(encoding="utf-8")
        )
        bundle = json.loads((staging / "bundle.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Staged controller state is missing or unreadable") from exc
    store_path = staging / "store.sqlite3"
    if not store_path.is_file():
        raise RuntimeError("Staged controller state has no store payload")
    bundle_bytes = (staging / "bundle.json").read_bytes()
    if hashlib.sha256(bundle_bytes).hexdigest() != manifest.get("bundle_sha256"):
        raise RuntimeError("Staged bundle failed its checksum")
    store_bytes = store_path.read_bytes()
    if hashlib.sha256(store_bytes).hexdigest() != manifest.get("store_sha256"):
        raise RuntimeError("Staged controller store failed its checksum")
    connection = sqlite3.connect(str(store_path))
    try:
        row = connection.execute("PRAGMA integrity_check").fetchone()
    finally:
        connection.close()
    if not row or row[0] != "ok":
        raise RuntimeError("Staged controller store failed its integrity check")
    return bundle, store_bytes, manifest


def _wait_for_controller_role(node_id: str, timeout: int = 180) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:7878/api/v1/onboarding", timeout=5
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("role") == "controller" and str(
                (payload.get("node") or {}).get("id") or ""
            ) == node_id:
                return True
        except (OSError, ValueError):
            pass
        time.sleep(2)
    return False


def _install_private_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.urandom(4).hex()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        # Without O_BINARY, Windows translates newlines and corrupts SQLite.
        flags |= os.O_BINARY
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def promote(root: Path, state_path: Path) -> None:
    root = root.resolve()
    state_path = validate_state_path(state_path, root)
    data_dir = root / "data"
    time.sleep(1.0)  # Let the accepting HTTP response leave the process first.
    bundle, store_bytes, _manifest = _load_staged_bundle(root)
    new_controller_node_id = str(bundle.get("new_controller_node_id") or "")
    if not new_controller_node_id:
        raise RuntimeError("Staged bundle does not name the promoted controller")
    json_files = bundle.get("json_files") or {}
    overlay = bundle.get("settings_overlay") or {}
    unknown = set(overlay) & NODE_LOCAL_SETTING_KEYS
    if unknown:
        raise RuntimeError(
            "Staged settings overlay carries node-local keys: "
            + ", ".join(sorted(unknown))
        )

    backup = data_dir / f"pre-transfer-backup-{int(time.time())}"
    backup.mkdir(parents=True, exist_ok=False)
    replaced: list[str] = []
    committed = False
    try:
        # Copy settings.json before any install step can fail: it is merged
        # in place, so the backup is the only way to restore the machine's
        # original content.
        settings_path = data_dir / "settings.json"
        if settings_path.exists():
            shutil.copy2(str(settings_path), str(backup / "settings.json"))
        # Move aside everything the bundle replaces so a failed apply can put
        # this worker back exactly as it was.
        for name in (
            "sparkdeck.sqlite3",
            "sparkdeck.sqlite3-wal",
            "sparkdeck.sqlite3-shm",
            "nodes.json",
            "controller.json",
            *json_files.keys(),
        ):
            source = data_dir / name
            if source.exists():
                shutil.move(str(source), str(backup / name))
                replaced.append(name)

        _install_private_bytes(
            data_dir / "sparkdeck.sqlite3", store_bytes
        )
        for name, value in json_files.items():
            atomic_private_json_write(data_dir / name, value)
        # settings.json is merged in place, not replaced: the node keeps its
        # machine-local values and receives every cluster-wide setting.
        settings: dict = {}
        if settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                settings = {}
        settings.update(overlay)
        atomic_private_json_write(settings_path, settings)
        atomic_private_json_write(data_dir / "nodes.json", bundle.get("nodes") or [])

        # Commit: record the promoted state, then drop the worker assignment.
        atomic_private_json_write(state_path, bundle["transfer_state"])
        (data_dir / "controller.json").unlink(missing_ok=True)
        committed = True

        restart_service(root)
        if not _wait_for_controller_role(new_controller_node_id):
            raise RuntimeError(
                "SparkDeck did not come back as the promoted controller"
            )
        # The service's TransferService.start() transitions the durable
        # record to "completing" once it observes this boot.
    except Exception as exc:
        if committed:
            try:
                write_state(
                    state_path,
                    phase="recovery_required",
                    error=str(exc)[:500],
                    message=(
                        "Promotion failed after commit; restoring the worker"
                        " assignment and restarting"
                    ),
                )
                (data_dir / "controller.json").unlink(missing_ok=True)
                for name in replaced:
                    source = backup / name
                    if name == "settings.json":
                        continue
                    if source.exists():
                        shutil.move(str(source), str(data_dir / name))
                restored_settings = backup / "settings.json"
                if restored_settings.exists():
                    shutil.copy2(str(restored_settings), str(data_dir / "settings.json"))
                restart_service(root)
            except Exception as rollback_exc:
                write_state(
                    state_path,
                    phase="recovery_required",
                    error=(
                        f"Promotion failed: {str(exc)[:220]}; rollback failed:"
                        f" {str(rollback_exc)[:220]}"
                    ),
                    message="Automatic recovery failed; manual recovery is required",
                )
                raise
            raise
        # Pre-commit: the assignment was never deleted, but the backup loop
        # already moved the worker's originals aside. Restore every original,
        # remove only files the bundle created where the worker had none
        # (existence is tracked through ``replaced``, not the backup dir,
        # because restoring moves the backup copy away), and put settings.json
        # back from its copy.
        try:
            for name in replaced:
                source = backup / name
                if source.exists():
                    shutil.move(str(source), str(data_dir / name))
            for name in json_files:
                if name not in replaced:
                    (data_dir / name).unlink(missing_ok=True)
            settings_backup = backup / "settings.json"
            if settings_backup.exists():
                shutil.copy2(str(settings_backup), str(data_dir / "settings.json"))
            else:
                (data_dir / "settings.json").unlink(missing_ok=True)
            if "sparkdeck.sqlite3" not in replaced:
                # A store was installed where none existed before; remove the
                # fresh copy and any sidecar files it may have produced.
                (data_dir / "sparkdeck.sqlite3").unlink(missing_ok=True)
                (data_dir / "sparkdeck.sqlite3-wal").unlink(missing_ok=True)
                (data_dir / "sparkdeck.sqlite3-shm").unlink(missing_ok=True)
        except Exception as restore_exc:
            write_state(
                state_path,
                phase="recovery_required",
                error=(
                    f"Promotion failed: {str(exc)[:220]}; restore failed:"
                    f" {str(restore_exc)[:220]}"
                ),
                message="Automatic recovery failed; manual recovery is required",
            )
            raise
        write_state(
            state_path,
            phase="failed",
            error=str(exc)[:500],
            message="Promotion failed before the worker assignment changed",
        )
        raise
    finally:
        # Keep the backup directory for inspection; staging stays for retry
        # only when the apply never committed.
        if committed and (data_dir / TRANSFER_STAGING_DIRNAME).exists():
            shutil.rmtree(data_dir / TRANSFER_STAGING_DIRNAME, ignore_errors=True)


def demote(root: Path, state_path: Path) -> None:
    root = root.resolve()
    validate_state_path(state_path, root)
    time.sleep(1.0)  # Let the accepting HTTP response leave the process first.
    restart_service(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument(
        "--action", required=True, choices=["promote", "demote"]
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    state = Path(args.state).resolve()
    if args.action == "promote":
        promote(root, state)
    else:
        demote(root, state)


if __name__ == "__main__":
    main()
