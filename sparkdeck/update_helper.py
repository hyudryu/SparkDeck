"""Detached, narrowly-scoped release apply helper."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from .updater import CAPABILITY, TRUSTED_ORIGINS


def write_state(path: Path, **changes) -> None:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    state.update(changes, updated_at=time.time())
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def run(root: Path, *args: str, timeout: int = 600) -> str:
    result = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[:500] or f"{' '.join(args)} failed")
    return result.stdout.strip()


def install_revision(root: Path, revision: str) -> str:
    """Move only along one release chain without resetting a local branch."""
    forward = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "HEAD", revision], cwd=root,
        capture_output=True, text=True, check=False,
    ).returncode == 0
    backward = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "HEAD"], cwd=root,
        capture_output=True, text=True, check=False,
    ).returncode == 0
    if not forward and not backward:
        raise RuntimeError("Selected release is not in the installed release history")
    if forward:
        run(root, "git", "merge", "--ff-only", revision)
        return "upgrade"
    run(root, "git", "checkout", "--detach", revision)
    return "downgrade"


def wait_for_revision(revision: str, timeout: int = 90) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:7878/api/agent/info", timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("app_revision") == revision:
                return True
        except (OSError, ValueError):
            pass
        time.sleep(2)
    return False


def apply(root: Path, state_path: Path, tag: str, revision: str) -> None:
    time.sleep(1.0)  # Let the accepting HTTP response leave the process first.
    stage_dir: Path | None = None
    previous_revision: str | None = None
    applied = False
    try:
        if run(root, "git", "remote", "get-url", "origin") not in TRUSTED_ORIGINS:
            raise RuntimeError("Git origin is not the official SparkDeck repository")
        if run(root, "git", "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError("Tracked files changed after preflight")
        previous_revision = run(root, "git", "rev-parse", "HEAD").lower()
        write_state(state_path, phase="staging", message=f"Fetching and validating {tag}")
        run(root, "git", "fetch", "--force", "origin", f"refs/tags/{tag}:refs/tags/{tag}")
        fetched = run(root, "git", "rev-parse", f"{tag}^{{commit}}")
        if fetched.lower() != revision:
            raise RuntimeError("Fetched release tag does not match the approved commit")
        stage_dir = Path(tempfile.mkdtemp(prefix="sparkdeck-update-"))
        run(root, "git", "worktree", "add", "--detach", str(stage_dir), revision)
        try:
            manifest = json.loads((stage_dir / "sparkdeck-update.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("Selected release has no valid update compatibility manifest") from exc
        if manifest.get("update_protocol") != 1 or manifest.get("data_schema") != 1:
            raise RuntimeError("Selected release is not compatible with this updater or data schema")
        update_source = stage_dir / "sparkdeck" / "updater.py"
        if not update_source.exists() or CAPABILITY not in update_source.read_text(encoding="utf-8"):
            raise RuntimeError("Selected release does not support safe cluster updates")
        run(stage_dir, os.fspath(Path(os.sys.executable)), "-m", "compileall", "-q", "server.py", "sparkdeck")
        if (stage_dir / "frontend" / "package-lock.json").exists():
            run(stage_dir, "npm", "--prefix", "frontend", "ci", "--ignore-scripts")
            run(stage_dir, "npm", "--prefix", "frontend", "run", "build")
        run(root, "git", "worktree", "remove", "--force", str(stage_dir))
        shutil.rmtree(stage_dir, ignore_errors=True)
        stage_dir = None
        install_revision(root, revision)
        applied = True
        write_state(state_path, phase="restarting", message="Release installed; restarting SparkDeck")
        run(root, "systemctl", "--user", "restart", "sparkdeck.service", timeout=60)
        if not wait_for_revision(revision):
            raise RuntimeError("SparkDeck did not become healthy on the selected release")
        write_state(state_path, phase="succeeded", message="Release installed and verified", error=None)
    except Exception as exc:
        if applied and previous_revision:
            try:
                write_state(state_path, phase="rolling_back", error=str(exc)[:500], message="Health check failed; restoring the previous release")
                run(root, "git", "checkout", "--detach", previous_revision)
                run(root, "systemctl", "--user", "restart", "sparkdeck.service", timeout=60)
                if not wait_for_revision(previous_revision):
                    raise RuntimeError("previous release did not become healthy")
                write_state(state_path, phase="rolled_back", error=str(exc)[:500], message="Selected release failed; previous release restored")
            except Exception as rollback_exc:
                write_state(
                    state_path, phase="recovery_required",
                    error=f"Update failed: {str(exc)[:220]}; rollback failed: {str(rollback_exc)[:220]}",
                    message="Automatic recovery failed; manual service recovery is required",
                )
        else:
            write_state(state_path, phase="failed", error=str(exc)[:500], message="Update failed before the live checkout changed")
    finally:
        if stage_dir:
            subprocess.run(["git", "worktree", "remove", "--force", str(stage_dir)], cwd=root, capture_output=True)
            shutil.rmtree(stage_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    apply(Path(args.root).resolve(), Path(args.state).resolve(), args.tag, args.revision.lower())


if __name__ == "__main__":
    main()
