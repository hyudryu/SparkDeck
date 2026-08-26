"""Detached, narrowly-scoped release apply helper."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .updater import TRUSTED_ORIGINS


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


def apply(root: Path, state_path: Path, tag: str, revision: str) -> None:
    time.sleep(1.0)  # Let the accepting HTTP response leave the process first.
    stage_dir: Path | None = None
    try:
        if run(root, "git", "remote", "get-url", "origin") not in TRUSTED_ORIGINS:
            raise RuntimeError("Git origin is not the official SparkDeck repository")
        if run(root, "git", "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError("Tracked files changed after preflight")
        write_state(state_path, phase="staging", message=f"Fetching and validating {tag}")
        run(root, "git", "fetch", "--force", "origin", f"refs/tags/{tag}:refs/tags/{tag}")
        fetched = run(root, "git", "rev-parse", f"{tag}^{{commit}}")
        if fetched.lower() != revision:
            raise RuntimeError("Fetched release tag does not match the approved commit")
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", "HEAD", revision], cwd=root,
            capture_output=True, text=True, check=False,
        )
        if ancestor.returncode:
            raise RuntimeError("Release is not a fast-forward from the installed revision")
        stage_dir = Path(tempfile.mkdtemp(prefix="sparkdeck-update-"))
        run(root, "git", "worktree", "add", "--detach", str(stage_dir), revision)
        run(stage_dir, os.fspath(Path(os.sys.executable)), "-m", "compileall", "-q", "server.py", "sparkdeck")
        if (stage_dir / "frontend" / "package-lock.json").exists():
            run(stage_dir, "npm", "--prefix", "frontend", "ci", "--ignore-scripts")
            run(stage_dir, "npm", "--prefix", "frontend", "run", "build")
        run(root, "git", "worktree", "remove", "--force", str(stage_dir))
        shutil.rmtree(stage_dir, ignore_errors=True)
        stage_dir = None
        run(root, "git", "merge", "--ff-only", revision)
        write_state(state_path, phase="restarting", message="Release installed; restarting SparkDeck")
        run(root, "systemctl", "--user", "restart", "sparkdeck.service", timeout=60)
    except Exception as exc:
        write_state(state_path, phase="failed", error=str(exc)[:500], message="Update failed before restart")
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
