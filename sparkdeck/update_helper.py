"""Detached, narrowly-scoped update apply helper."""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from .node_toolchain import frontend_build_environment, resolve_node_toolchain

from .updater import (
    CAPABILITY,
    MAIN_BRANCH,
    TRUSTED_ORIGINS,
    UPDATE_STATE_FILENAME,
)


def validate_state_path(state_path: Path, root: Path) -> Path:
    """Restrict writes to the service-owned state file in the checkout's data dir.

    The helper receives --state from the process that spawned it, so the path
    is untrusted input at this trust boundary.
    """
    data_dir = (root / "data").resolve()
    resolved = state_path.resolve()
    if resolved.parent != data_dir or resolved.name != UPDATE_STATE_FILENAME:
        raise RuntimeError(
            "Update state path must be the service state file inside the checkout data directory"
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


def run(root: Path, *args: str, timeout: int = 600, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=timeout, check=False, env=env)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[:500] or f"{' '.join(args)} failed")
    return result.stdout.strip()


def npm_executable() -> str:
    """Resolve npm's platform-specific executable for shell-free subprocesses."""
    return str(resolve_node_toolchain().npm)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _prepare_frontend_bundle(staged_dist: Path, live_dist: Path) -> Path:
    """Copy a verified build beside the live bundle for a same-filesystem swap."""
    swap_root = Path(tempfile.mkdtemp(prefix=".sparkdeck-dist-swap-", dir=live_dist.parent))
    try:
        shutil.copytree(staged_dist, swap_root / "next")
    except Exception:
        shutil.rmtree(swap_root, ignore_errors=True)
        raise
    return swap_root


def _publish_frontend_bundle(live_dist: Path, swap_root: Path) -> bool:
    replacement = swap_root / "next"
    previous = swap_root / "previous"
    if not (replacement / "index.html").is_file():
        raise RuntimeError("Staged frontend bundle has no index.html")
    had_previous = live_dist.exists() or live_dist.is_symlink()
    if had_previous:
        os.replace(live_dist, previous)
    try:
        os.replace(replacement, live_dist)
        # run.sh uses index.html as its freshness marker. Publishing happens
        # after the target checkout, so refresh it to prevent a second build.
        os.utime(live_dist / "index.html", None)
    except Exception:
        _remove_path(live_dist)
        if had_previous:
            os.replace(previous, live_dist)
        raise
    return had_previous


def _restore_frontend_bundle(live_dist: Path, swap_root: Path, had_previous: bool) -> None:
    _remove_path(live_dist)
    if had_previous:
        previous = swap_root / "previous"
        if not previous.exists() and not previous.is_symlink():
            raise RuntimeError("Previous frontend bundle is unavailable for rollback")
        os.replace(previous, live_dist)


def publish_windows_frontend_stamp(root: Path, env: dict[str, str]) -> None:
    """Mark the published bundle with the Windows launcher's exact cache key."""
    if platform.system() != "Windows":
        return
    module = root / "scripts" / "windows" / "SparkDeck.Windows.psm1"
    if not module.is_file():
        raise RuntimeError("The bundled Windows launcher module was not found")
    fingerprint_environment = env.copy()
    fingerprint_environment["SPARKDECK_FINGERPRINT_ROOT"] = str(root)
    command = (
        "$ErrorActionPreference='Stop'; "
        "$root=[IO.Path]::GetFullPath($env:SPARKDECK_FINGERPRINT_ROOT); "
        "Import-Module (Join-Path $root 'scripts\\windows\\SparkDeck.Windows.psm1') -Force; "
        "$paths=Get-SparkDeckPaths -Root $root; "
        "Get-SparkDeckFrontendFingerprint -Paths $paths"
    )
    fingerprint = run(
        root,
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        command,
        timeout=60,
        env=fingerprint_environment,
    ).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise RuntimeError("The Windows launcher returned an invalid frontend fingerprint")
    (root / "frontend" / "dist" / ".sparkdeck-source.stamp").write_text(
        fingerprint, encoding="utf-8",
    )


def install_release_revision(root: Path, revision: str) -> str:
    """Dormant release-mode install retained for restoring release updates."""
    forward = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "HEAD", revision], cwd=root,
        capture_output=True, text=True, check=False,
    ).returncode == 0
    backward = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "HEAD"], cwd=root,
        capture_output=True, text=True, check=False,
    ).returncode == 0
    if not forward and not backward:
        raise RuntimeError("Selected revision is not in the installed update history")
    if forward:
        run(root, "git", "merge", "--ff-only", revision)
        return "upgrade"
    run(root, "git", "checkout", "--detach", revision)
    return "downgrade"


def install_revision(root: Path, revision: str) -> str:
    """Install approved main without moving any local branch pointer.

    A node can legitimately run a clean commit whose changes reached main
    under another hash. Detaching also lets Git's no-overwrite-ignore safety
    apply uniformly to main, feature, and already-detached installations.
    """
    run(
        root, "git", "checkout", "--no-overwrite-ignore", "--detach", revision,
    )
    return "detached"


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


def restart_service(root: Path) -> None:
    """Restart SparkDeck through the launcher that owns this installation."""
    system = platform.system()
    if system == "Linux":
        run(root, "systemctl", "--user", "restart", "sparkdeck.service", timeout=60)
        return
    if system == "Windows":
        launcher = root / "scripts" / "windows" / "sparkdeck.ps1"
        if not launcher.is_file():
            raise RuntimeError("The bundled Windows launcher was not found")
        run(
            root,
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            "restart",
            timeout=180,
        )
        return
    raise RuntimeError("Self-update supports only the bundled Linux and Windows launchers")


def fetch_release_target(root: Path, tag: str, revision: str) -> None:
    """Dormant release-mode fetch retained for restoring release updates."""
    run(root, "git", "fetch", "--force", "origin", f"refs/tags/{tag}:refs/tags/{tag}")
    fetched = run(root, "git", "rev-parse", f"{tag}^{{commit}}")
    if fetched.lower() != revision:
        raise RuntimeError("Fetched release tag does not match the approved commit")


def fetch_update_target(root: Path, branch: str, revision: str) -> None:
    if branch != MAIN_BRANCH:
        raise RuntimeError("The active update target is origin/main")
    remote_ref = f"refs/remotes/origin/{MAIN_BRANCH}"
    run(
        root, "git", "fetch", "--force", "origin",
        f"refs/heads/{MAIN_BRANCH}:{remote_ref}",
    )
    fetched = run(root, "git", "rev-parse", f"{remote_ref}^{{commit}}")
    contains_target = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, fetched], cwd=root,
        capture_output=True, text=True, check=False,
    ).returncode == 0
    if not contains_target:
        raise RuntimeError("The approved commit is no longer in origin/main history")


def apply(root: Path, state_path: Path, branch: str, revision: str) -> None:
    root = root.resolve()
    state_path = validate_state_path(state_path, root)
    time.sleep(1.0)  # Let the accepting HTTP response leave the process first.
    stage_dir: Path | None = None
    frontend_swap: Path | None = None
    frontend_published = False
    had_previous_frontend = False
    previous_revision: str | None = None
    applied = False
    try:
        if run(root, "git", "remote", "get-url", "origin") not in TRUSTED_ORIGINS:
            raise RuntimeError("Git origin is not the official SparkDeck repository")
        if run(root, "git", "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError("Tracked files changed after preflight")
        previous_revision = run(root, "git", "rev-parse", "HEAD").lower()
        write_state(state_path, phase="staging", message=f"Fetching and validating origin/{branch}")
        fetch_update_target(root, branch, revision)
        stage_dir = Path(tempfile.mkdtemp(prefix="sparkdeck-update-"))
        run(root, "git", "worktree", "add", "--detach", str(stage_dir), revision)
        try:
            manifest = json.loads((stage_dir / "sparkdeck-update.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("Selected origin/main revision has no valid update compatibility manifest") from exc
        if manifest.get("update_protocol") != 1 or manifest.get("data_schema") != 1:
            raise RuntimeError("Selected origin/main revision is not compatible with this updater or data schema")
        update_source = stage_dir / "sparkdeck" / "updater.py"
        if not update_source.exists() or CAPABILITY not in update_source.read_text(encoding="utf-8"):
            raise RuntimeError("Selected origin/main revision does not support safe cluster updates")
        run(stage_dir, os.fspath(Path(os.sys.executable)), "-m", "compileall", "-q", ".")
        build_environment = os.environ.copy()
        if (stage_dir / "frontend" / "package-lock.json").exists():
            toolchain = resolve_node_toolchain()
            npm = str(toolchain.npm)
            build_environment = frontend_build_environment(toolchain)
            run(
                stage_dir, npm, "--prefix", "frontend", "ci", "--ignore-scripts",
                env=build_environment,
            )
            if platform.system() != "Windows":
                build_environment["SPARKDECK_VERSION"] = f"{branch}-{revision[:8]}"
            run(stage_dir, npm, "--prefix", "frontend", "run", "build", env=build_environment)
            frontend_swap = _prepare_frontend_bundle(
                stage_dir / "frontend" / "dist", root / "frontend" / "dist",
            )
        run(root, "git", "worktree", "remove", "--force", str(stage_dir))
        shutil.rmtree(stage_dir, ignore_errors=True)
        stage_dir = None
        install_revision(root, revision)
        applied = True
        if frontend_swap:
            had_previous_frontend = _publish_frontend_bundle(root / "frontend" / "dist", frontend_swap)
            frontend_published = True
            publish_windows_frontend_stamp(root, build_environment)
        write_state(state_path, phase="restarting", message="Update installed; restarting SparkDeck")
        restart_service(root)
        if not wait_for_revision(revision):
            raise RuntimeError("SparkDeck did not become healthy on the selected revision")
        write_state(state_path, phase="succeeded", message="Update installed and verified", error=None)
    except Exception as exc:
        if applied and previous_revision:
            try:
                write_state(state_path, phase="rolling_back", error=str(exc)[:500], message="Health check failed; restoring the previous revision")
                run(root, "git", "checkout", "--detach", previous_revision)
                if frontend_published and frontend_swap:
                    _restore_frontend_bundle(
                        root / "frontend" / "dist", frontend_swap, had_previous_frontend,
                    )
                restart_service(root)
                if not wait_for_revision(previous_revision):
                    raise RuntimeError("previous revision did not become healthy")
                write_state(state_path, phase="rolled_back", error=str(exc)[:500], message="Selected update failed; previous revision restored")
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
        if frontend_swap:
            shutil.rmtree(frontend_swap, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    apply(Path(args.root).resolve(), Path(args.state).resolve(), args.branch, args.revision.lower())


if __name__ == "__main__":
    main()
