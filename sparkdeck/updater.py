"""Controller-orchestrated SparkDeck self updates."""
from __future__ import annotations

import asyncio
import ctypes
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx


REPOSITORY = "hyudryu/SparkDeck"
RELEASES_API = f"https://api.github.com/repos/{REPOSITORY}/releases?per_page=100"
MAIN_BRANCH = "main"
MAIN_COMMIT_API = f"https://api.github.com/repos/{REPOSITORY}/commits/{MAIN_BRANCH}"
MAIN_URL = f"https://github.com/{REPOSITORY}/tree/{MAIN_BRANCH}"
TRUSTED_ORIGINS = {
    "https://github.com/hyudryu/SparkDeck.git",
    "https://github.com/hyudryu/SparkDeck",
    "git@github.com:hyudryu/SparkDeck.git",
    "ssh://git@github.com/hyudryu/SparkDeck.git",
}
CAPABILITY = "cluster_update_main_v1"
CONFIRMATION = "update-entire-cluster"
UPDATE_STATE_FILENAME = "system-update-agent.json"
FAILED_NODE_PHASES = {"failed", "rolled_back", "recovery_required"}

_WINDOWS_HELPER_BOOTSTRAP = r"""
import json
import subprocess
import sys

command = json.loads(sys.argv[1])
flags = (
    getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
    | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
)
process = subprocess.Popen(
    command,
    cwd=sys.argv[2],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    creationflags=flags,
    close_fds=True,
)
print(process.pid, flush=True)
"""


def _valid_release_tag(tag: str) -> bool:
    return bool(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}", tag)
        and ".." not in tag and "//" not in tag
        and not tag.endswith(("/", ".", ".lock"))
    )


def _run(root: Path, *args: str, timeout: int = 20) -> str:
    result = subprocess.run(
        args, cwd=root, capture_output=True, text=True, timeout=timeout, check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[:500]
        raise RuntimeError(detail or f"{' '.join(args)} failed")
    return result.stdout.strip()


def current_revision(root: Path) -> str | None:
    try:
        return _run(root, "git", "rev-parse", "HEAD")
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return None


def assert_checkout_safe(root: Path, revision: str) -> None:
    """Dry-run Git's live tree transition before any cluster node changes."""
    result = subprocess.run(
        ["git", "read-tree", "--dry-run", "-m", "-u", "HEAD", revision],
        cwd=root, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[:400]
        raise RuntimeError(
            "origin/main cannot be installed without overwriting local files"
            + (f": {detail}" if detail else "")
        )

    # unpack-trees deliberately permits ignored files to be replaced. The
    # helper uses --no-overwrite-ignore, so mirror that stricter behavior in
    # preflight by checking newly tracked target paths that already exist.
    additions = _run(
        root, "git", "diff", "--name-only", "--diff-filter=A", "-z",
        "HEAD", revision,
    ).split("\0")
    candidates: set[str] = set()
    for relative in (item for item in additions if item):
        candidate = root / relative
        if os.path.lexists(candidate):
            candidates.add(relative)
        parent = candidate.parent
        while parent != root and root in parent.parents:
            if os.path.lexists(parent) and not parent.is_dir():
                candidates.add(parent.relative_to(root).as_posix())
                break
            parent = parent.parent
    if not candidates:
        return
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "-z", "--stdin"],
        cwd=root, input="\0".join(sorted(candidates)) + "\0",
        capture_output=True, text=True, check=False,
    )
    if ignored.returncode not in {0, 1}:
        detail = (ignored.stderr or ignored.stdout).strip()[:400]
        raise RuntimeError(detail or "Could not verify ignored-file checkout safety")
    collisions = [item for item in ignored.stdout.split("\0") if item]
    if collisions:
        raise RuntimeError(
            "origin/main cannot be installed without overwriting ignored local files: "
            + ", ".join(collisions[:10])
        )


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _windows_process_started(pid: int) -> int | None:
    """Return a Windows process creation timestamp without signaling it."""
    if platform.system() != "Windows":
        return None
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    )
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    # PROCESS_QUERY_LIMITED_INFORMATION is sufficient for GetProcessTimes and
    # cannot be used to terminate or otherwise modify the helper process.
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        created = wintypes.FILETIME()
        exited = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return None
        return (created.dwHighDateTime << 32) | created.dwLowDateTime
    finally:
        kernel32.CloseHandle(handle)


def _helper_alive(state: dict) -> bool:
    pid = state.get("helper_pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    if platform.system() == "Windows":
        expected = state.get("helper_started_at")
        return isinstance(expected, int) and _windows_process_started(pid) == expected
    if state.get("boot_id") != _boot_id():
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _spawn_update_helper(root: Path, command: list[str]) -> int:
    """Start the updater outside the service process tree on Windows."""
    if platform.system() == "Windows":
        bootstrap = subprocess.run(
            [sys.executable, "-c", _WINDOWS_HELPER_BOOTSTRAP, json.dumps(command), str(root)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if bootstrap.returncode:
            detail = (bootstrap.stderr or bootstrap.stdout).strip()[:500]
            raise RuntimeError(detail or "Could not detach the Windows update helper")
        try:
            pid = int(bootstrap.stdout.strip())
        except ValueError as exc:
            raise RuntimeError("Windows update helper did not report a process ID") from exc
        if pid <= 0:
            raise RuntimeError("Windows update helper reported an invalid process ID")
        return pid

    process = subprocess.Popen(
        command,
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process.pid


def _service_preflight(root: Path) -> None:
    system = platform.system()
    if system == "Linux":
        _run(root, "systemctl", "--user", "is-active", "--quiet", "sparkdeck.service")
        return
    if system == "Windows":
        launcher = root / "scripts" / "windows" / "sparkdeck.ps1"
        if not launcher.is_file():
            raise RuntimeError("The bundled Windows launcher was not found")
        _run(
            root,
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            "process-status",
            timeout=30,
        )
        return
    raise RuntimeError("Self-update supports only the bundled Linux and Windows launchers")


def local_blockers(root: Path) -> list[str]:
    blockers: list[str] = []
    try:
        origin = _run(root, "git", "remote", "get-url", "origin")
        if origin not in TRUSTED_ORIGINS:
            blockers.append("Git origin is not the official SparkDeck repository")
        if _run(root, "git", "status", "--porcelain", "--untracked-files=no"):
            blockers.append("Tracked files have local changes")
        _service_preflight(root)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        blockers.append(f"Installation preflight failed: {str(exc)[:240]}")
    return blockers


class UpdateService:
    """Owns one durable cluster rollout and one durable local agent operation."""

    def __init__(self, manager: Any, root: Path, data_dir: Path):
        self.manager = manager
        self.root = Path(root).resolve()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cluster_path = self.data_dir / "system-update.json"
        self.agent_path = self.data_dir / UPDATE_STATE_FILENAME
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._agent_lock = asyncio.Lock()
        self._release_cache: tuple[float, list[dict], str | None] | None = None
        self._resolved_releases: dict[str, dict] = {}
        self._main_cache: tuple[float, dict | None, str | None] | None = None

    @staticmethod
    def _read(path: Path) -> dict:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _write(path: Path, value: dict) -> None:
        value["updated_at"] = time.time()
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass

    def _finish_cluster_state(self, state: dict) -> None:
        failures = [
            node for node in state.get("nodes", [])
            if node.get("phase") in FAILED_NODE_PHASES or node.get("error")
        ]
        completed = [
            node for node in state.get("nodes", [])
            if node.get("phase") in {"succeeded", "up_to_date"}
        ]
        state["active"] = False
        if failures:
            state["phase"] = "partial" if completed else "failed"
            state["message"] = (
                f"Updated or verified {len(completed)} node(s); "
                f"{len(failures)} node(s) could not be updated"
                if completed else "No cluster nodes could be updated"
            )
            state["error"] = "; ".join(
                f"{node.get('name')}: {node.get('error') or 'update failed'}"
                for node in failures
            )[:500]
        else:
            state["phase"] = "succeeded"
            state["message"] = "Cluster update completed"
            state.pop("error", None)
        self._write(self.cluster_path, state)

    def agent_status(self) -> dict:
        revision = current_revision(self.root)
        state = self._read(self.agent_path)
        if state.get("phase") in {"accepted", "staging", "restarting"}:
            if not _helper_alive(state):
                state["phase"] = "failed"
                state["error"] = "The local update helper was interrupted"
                state["message"] = "Interrupted update can be retried"
                self._write(self.agent_path, state)
        return {
            "capability": CAPABILITY,
            "current_revision": revision,
            "phase": state.get("phase", "idle"),
            "target_branch": state.get("target_branch"),
            "target_revision": state.get("target_revision"),
            "message": state.get("message"),
            "error": state.get("error"),
            "blockers": local_blockers(self.root),
        }

    async def published_releases(self, *, force: bool = False) -> tuple[list[dict], str | None]:
        if not force and self._release_cache and time.monotonic() - self._release_cache[0] < 300:
            return self._release_cache[1], self._release_cache[2]
        try:
            response = await self.manager.http.get(
                RELEASES_API,
                headers={"Accept": "application/vnd.github+json", "User-Agent": "SparkDeck"},
                timeout=10,
            )
            if response.status_code == 404:
                result = ([], "No published GitHub release is available")
                self._release_cache = (time.monotonic(), *result)
                return result
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                return [], "GitHub returned an invalid releases response"
            releases = []
            for release in payload:
                if not isinstance(release, dict) or release.get("draft"):
                    continue
                tag = str(release.get("tag_name") or "").strip()
                if not _valid_release_tag(tag):
                    continue
                releases.append({
                    "tag": tag,
                    "name": release.get("name") or tag,
                    "url": release.get("html_url"),
                    "published_at": release.get("published_at"),
                    "prerelease": bool(release.get("prerelease")),
                })
            error = None if releases else "No published GitHub release is available"
            result = (releases, error)
            self._release_cache = (time.monotonic(), *result)
            return result
        except (httpx.HTTPError, ValueError) as exc:
            return [], f"Could not check GitHub releases: {str(exc)[:240]}"

    async def resolve_release(self, tag: str, *, force: bool = False) -> tuple[dict | None, str | None]:
        releases, error = await self.published_releases(force=force)
        release = next((item for item in releases if item["tag"] == tag), None)
        if not release:
            return None, error or "Selected version is not a published GitHub release"
        if not force and tag in self._resolved_releases:
            return self._resolved_releases[tag], None
        try:
            commit_response = await self.manager.http.get(
                f"https://api.github.com/repos/{REPOSITORY}/commits/{tag}",
                headers={"Accept": "application/vnd.github+json", "User-Agent": "SparkDeck"},
                timeout=10,
            )
            commit_response.raise_for_status()
            revision = str(commit_response.json().get("sha") or "").lower()
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                return None, "GitHub did not return an immutable release commit"
            resolved = {**release, "revision": revision}
            self._resolved_releases[tag] = resolved
            return resolved, None
        except (httpx.HTTPError, ValueError) as exc:
            return None, f"Could not resolve GitHub release {tag}: {str(exc)[:240]}"

    async def latest_release(self) -> tuple[dict | None, str | None]:
        releases, error = await self.published_releases()
        if not releases:
            return None, error
        return await self.resolve_release(releases[0]["tag"])

    # Release discovery and resolution above are intentionally retained while
    # SparkDeck temporarily follows origin/main for self-updates.
    async def resolve_main(self, *, force: bool = False) -> tuple[dict | None, str | None]:
        if not force and self._main_cache and time.monotonic() - self._main_cache[0] < 300:
            return self._main_cache[1], self._main_cache[2]
        try:
            response = await self.manager.http.get(
                MAIN_COMMIT_API,
                headers={"Accept": "application/vnd.github+json", "User-Agent": "SparkDeck"},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            revision = str(payload.get("sha") or "").lower() if isinstance(payload, dict) else ""
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                result = (None, "GitHub did not return an immutable origin/main commit")
            else:
                result = ({
                    "branch": MAIN_BRANCH,
                    "url": MAIN_URL,
                    "revision": revision,
                }, None)
        except (httpx.HTTPError, ValueError) as exc:
            result = (None, f"Could not check origin/main: {str(exc)[:240]}")
        self._main_cache = (time.monotonic(), *result)
        return result

    async def overview(self) -> dict:
        revision = current_revision(self.root)
        state = self._read(self.cluster_path)
        task_live = self._task is not None and not self._task.done()
        if state.get("active") and not task_live:
            agent_state = self._read(self.agent_path)
            if state.get("phase") == "updating_controller":
                local = next(
                    (node for node in state.get("nodes", []) if node.get("local")),
                    None,
                )
                if (
                    agent_state.get("phase") == "succeeded"
                    and revision == state.get("target_revision")
                    and agent_state.get("target_revision") == state.get("target_revision")
                ):
                    if local:
                        local.update(phase="succeeded", current_revision=revision)
                        local.pop("error", None)
                    self._finish_cluster_state(state)
                elif agent_state.get("phase") in FAILED_NODE_PHASES:
                    if local:
                        local.update(
                            phase=agent_state["phase"],
                            error=agent_state.get("error") or "Controller update failed",
                            current_revision=revision,
                        )
                    self._finish_cluster_state(state)
                elif _helper_alive(agent_state):
                    pass
                else:
                    if local:
                        local.update(
                            phase="failed",
                            error="The local update helper was interrupted",
                            current_revision=revision,
                        )
                    self._finish_cluster_state(state)
            else:
                if not state.get("nodes"):
                    state.update(
                        active=False,
                        phase="failed",
                        error="The controller rollout task was interrupted",
                        message="Interrupted rollout can be retried",
                    )
                    self._write(self.cluster_path, state)
                for node in state.get("nodes", []):
                    if node.get("phase") not in FAILED_NODE_PHASES | {"succeeded", "up_to_date"}:
                        node.update(
                            phase="failed",
                            error=(
                                "The local update helper was interrupted"
                                if node.get("local") and state.get("phase") == "updating_controller"
                                else "The controller rollout task was interrupted"
                            ),
                        )
                if state.get("nodes"):
                    self._finish_cluster_state(state)
        main_target, main_error = await self.resolve_main()
        nodes = await self.manager.cluster_nodes()
        public_nodes = []
        blockers = local_blockers(self.root)
        for node in nodes:
            node_blockers: list[str] = []
            if not node.get("enabled", True):
                continue
            if not node.get("online"):
                node_blockers.append("Node is offline")
            if not node.get("local") and CAPABILITY not in (node.get("capabilities") or []):
                node_blockers.append("One-time manual update required")
            if node.get("local"):
                node_blockers.extend(blockers)
            public_nodes.append({
                "id": node.get("id"), "name": node.get("name"),
                "local": bool(node.get("local")), "online": bool(node.get("online")),
                "current_revision": node.get("app_revision") or (revision if node.get("local") else None),
                "blockers": node_blockers,
            })
        all_blockers = ([main_error] if main_error else []) + [
            f"{node['name']}: {item}" for node in public_nodes for item in node["blockers"]
        ]
        up_to_date = bool(main_target and public_nodes) and all(
            node.get("current_revision") == main_target["revision"] for node in public_nodes
        )
        eligible_nodes = [
            node for node in public_nodes
            if main_target
            and node.get("current_revision") != main_target["revision"]
            and not node["blockers"]
        ]
        return {
            "repository": REPOSITORY,
            "current_revision": revision,
            "target": main_target,
            "up_to_date": up_to_date,
            "can_update": bool(eligible_nodes) and not state.get("active", False),
            "blockers": all_blockers,
            "nodes": public_nodes,
            "job": state or None,
        }

    async def start_cluster(self, confirmation: str, revision: str) -> dict:
        if confirmation != CONFIRMATION:
            raise ValueError("Explicit cluster update confirmation is required")
        revision = revision.lower()
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("An immutable origin/main commit is required")
        async with self._lock:
            existing = self._read(self.cluster_path)
            if existing.get("active"):
                raise RuntimeError("A cluster update is already running")
            overview = await self.overview()
            if not overview["can_update"]:
                raise RuntimeError("; ".join(overview["blockers"]) or "No update is available")
            release, release_error = await self.resolve_main(force=True)
            if release_error or not release:
                raise ValueError(release_error or "origin/main is unavailable")
            if release["revision"] != revision:
                raise RuntimeError("origin/main changed; refresh update status and confirm the new commit")
            if overview["nodes"] and all(
                node.get("current_revision") == release["revision"]
                for node in overview["nodes"]
            ):
                raise RuntimeError("Latest origin/main is already installed on every node")
            nodes = [{
                "id": node["id"], "name": node["name"], "local": node["local"],
                "online": node["online"], "blockers": list(node["blockers"]),
                "phase": "pending", "current_revision": node.get("current_revision"),
            } for node in overview["nodes"]]
            state = {
                "id": uuid.uuid4().hex, "active": True, "phase": "preflight",
                "message": "Checking each cluster node before making changes",
                "target_branch": release["branch"], "target_revision": release["revision"],
                "source_url": release.get("url"), "started_at": time.time(), "nodes": nodes,
            }
            self._write(self.cluster_path, state)
            self._task = asyncio.create_task(self._run_cluster(state))
            return state

    async def _run_cluster(self, state: dict) -> None:
        try:
            # Probe every node independently before mutating any of them. A node
            # failure is durable state for that node, not a cluster-wide abort.
            for node in state["nodes"]:
                if node.get("current_revision") == state["target_revision"]:
                    node["phase"] = "up_to_date"
                    continue
                try:
                    if node.get("blockers"):
                        raise RuntimeError("; ".join(node["blockers"]))
                    if node["local"]:
                        await self.preflight_local(state["target_branch"], state["target_revision"])
                    else:
                        status = await self.manager.node_registry.request(
                            node["id"], "POST", "/api/agent/system-update/preflight",
                            json_body={"branch": state["target_branch"], "revision": state["target_revision"]},
                            timeout=60,
                        )
                        if status.get("blockers"):
                            raise RuntimeError("; ".join(status["blockers"]))
                        if status.get("capability") != CAPABILITY:
                            raise RuntimeError("One-time manual update required")
                    node["phase"] = "ready"
                    node.pop("error", None)
                except Exception as exc:
                    node.update(phase="failed", error=str(exc)[:500])
                self._write(self.cluster_path, state)

            eligible_workers = [
                node for node in state["nodes"]
                if not node["local"] and node.get("phase") == "ready"
            ]
            state.update(
                phase="updating_workers",
                message=f"Updating {len(eligible_workers)} eligible worker(s) one at a time",
            )
            self._write(self.cluster_path, state)
            for node in eligible_workers:
                node["phase"] = "updating"
                self._write(self.cluster_path, state)
                try:
                    await self.manager.node_registry.request(
                        node["id"], "POST", "/api/agent/system-update",
                        json_body={"branch": state["target_branch"], "revision": state["target_revision"]},
                        timeout=120,
                    )
                    deadline = time.monotonic() + 600
                    while time.monotonic() < deadline:
                        await asyncio.sleep(3)
                        try:
                            status = await self.manager.node_registry.request(
                                node["id"], "GET", "/api/agent/system-update", timeout=10,
                            )
                        except RuntimeError:
                            continue
                        node["phase"] = status.get("phase", "updating")
                        node["current_revision"] = status.get("current_revision")
                        node["error"] = status.get("error")
                        self._write(self.cluster_path, state)
                        if status.get("phase") == "succeeded" and status.get("current_revision") == state["target_revision"]:
                            node.pop("error", None)
                            break
                        if status.get("phase") in FAILED_NODE_PHASES:
                            raise RuntimeError(status.get("error") or "Update failed")
                    else:
                        raise RuntimeError("Timed out waiting for restart")
                except Exception as exc:
                    node.update(phase="failed", error=str(exc)[:500])
                    self._write(self.cluster_path, state)

            local = next((node for node in state["nodes"] if node["local"]), None)
            if local and local.get("phase") == "ready":
                state.update(phase="updating_controller", message="Restarting the eligible controller last")
                local["phase"] = "updating"
                self._write(self.cluster_path, state)
                try:
                    await self.start_local(state["target_branch"], state["target_revision"])
                    return
                except Exception as exc:
                    local.update(phase="failed", error=str(exc)[:500])
            self._finish_cluster_state(state)
        except Exception as exc:
            for node in state.get("nodes", []):
                if node.get("phase") not in FAILED_NODE_PHASES | {"succeeded", "up_to_date"}:
                    node.update(phase="failed", error=f"Rollout interrupted: {str(exc)[:440]}")
            self._finish_cluster_state(state)

    async def preflight_local(self, branch: str, revision: str) -> dict:
        if branch != MAIN_BRANCH:
            raise ValueError("The active update target is origin/main")
        blockers = local_blockers(self.root)
        if blockers:
            raise RuntimeError("; ".join(blockers))
        remote_ref = f"refs/remotes/origin/{MAIN_BRANCH}"
        _run(
            self.root, "git", "fetch", "--force", "origin",
            f"refs/heads/{MAIN_BRANCH}:{remote_ref}", timeout=60,
        )
        fetched = _run(self.root, "git", "rev-parse", f"{remote_ref}^{{commit}}")
        still_on_main = subprocess.run(
            ["git", "merge-base", "--is-ancestor", revision, fetched], cwd=self.root,
            capture_output=True, text=True, check=False,
        ).returncode == 0
        if not still_on_main:
            raise RuntimeError("The approved commit is no longer in origin/main history")
        try:
            manifest = json.loads(_run(self.root, "git", "show", f"{revision}:sparkdeck-update.json"))
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError("origin/main has no valid update compatibility manifest") from exc
        if manifest.get("update_protocol") != 1 or manifest.get("data_schema") != 1:
            raise RuntimeError("origin/main is not compatible with this updater or data schema")
        assert_checkout_safe(self.root, revision)
        return {"ok": True, "capability": CAPABILITY, "target_revision": revision.lower()}

    async def preflight_release_local(self, tag: str, revision: str) -> dict:
        """Dormant release-mode preflight retained for restoring release updates."""
        release, error = await self.resolve_release(tag, force=True)
        if error or not release or release["revision"] != revision.lower():
            raise ValueError("Update target is not an official published GitHub release")
        blockers = local_blockers(self.root)
        if blockers:
            raise RuntimeError("; ".join(blockers))
        _run(self.root, "git", "fetch", "--force", "origin", f"refs/tags/{tag}:refs/tags/{tag}", timeout=60)
        fetched = _run(self.root, "git", "rev-parse", f"{tag}^{{commit}}")
        if fetched.lower() != revision.lower():
            raise ValueError("Fetched release tag does not match the approved commit")
        forward = subprocess.run(
            ["git", "merge-base", "--is-ancestor", "HEAD", revision], cwd=self.root,
            capture_output=True, text=True, check=False,
        ).returncode == 0
        backward = subprocess.run(
            ["git", "merge-base", "--is-ancestor", revision, "HEAD"], cwd=self.root,
            capture_output=True, text=True, check=False,
        ).returncode == 0
        if not forward and not backward:
            raise RuntimeError("Selected release is not in the installed release history")
        try:
            manifest = json.loads(_run(self.root, "git", "show", f"{revision}:sparkdeck-update.json"))
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError("Selected release has no valid update compatibility manifest") from exc
        if manifest.get("update_protocol") != 1 or manifest.get("data_schema") != 1:
            raise RuntimeError("Selected release is not compatible with this updater or data schema")
        return {"ok": True, "capability": CAPABILITY, "target_revision": revision.lower()}

    async def start_local(self, branch: str, revision: str) -> dict:
        async with self._agent_lock:
            if not branch or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
                raise ValueError("A valid update target and immutable commit are required")
            await self.preflight_local(branch, revision)
            state = self._read(self.agent_path)
            if state.get("phase") in {"accepted", "staging", "restarting"}:
                raise RuntimeError("This node is already updating")
            if current_revision(self.root) == revision.lower():
                state = {
                    "phase": "succeeded", "target_branch": branch,
                    "target_revision": revision.lower(), "message": "Already up to date",
                }
                self._write(self.agent_path, state)
                return state
            state = {"phase": "accepted", "target_branch": branch, "target_revision": revision.lower(), "message": "Update accepted"}
            self._write(self.agent_path, state)
            command = [
                sys.executable, "-m", "sparkdeck.update_helper", "--root", str(self.root),
                "--state", str(self.agent_path), "--branch", branch,
                "--revision", revision.lower(),
            ]
            helper_pid = _spawn_update_helper(self.root, command)
            state.update(helper_pid=helper_pid, boot_id=_boot_id())
            if platform.system() == "Windows":
                helper_started_at = _windows_process_started(helper_pid)
                if helper_started_at is None:
                    raise RuntimeError("Could not verify the detached Windows update helper")
                state["helper_started_at"] = helper_started_at
            self._write(self.agent_path, state)
            return state
