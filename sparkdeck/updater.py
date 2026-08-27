"""Release-only, controller-orchestrated SparkDeck self updates."""
from __future__ import annotations

import asyncio
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
TRUSTED_ORIGINS = {
    "https://github.com/hyudryu/SparkDeck.git",
    "https://github.com/hyudryu/SparkDeck",
    "git@github.com:hyudryu/SparkDeck.git",
    "ssh://git@github.com/hyudryu/SparkDeck.git",
}
CAPABILITY = "cluster_update_v1"
CONFIRMATION = "update-entire-cluster"
UPDATE_STATE_FILENAME = "system-update-agent.json"


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


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _helper_alive(state: dict) -> bool:
    pid = state.get("helper_pid")
    if not isinstance(pid, int) or pid <= 0 or state.get("boot_id") != _boot_id():
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def local_blockers(root: Path) -> list[str]:
    blockers: list[str] = []
    if platform.system() != "Linux":
        blockers.append("Self-update requires the bundled Linux systemd service")
        return blockers
    try:
        origin = _run(root, "git", "remote", "get-url", "origin")
        if origin not in TRUSTED_ORIGINS:
            blockers.append("Git origin is not the official SparkDeck repository")
        if _run(root, "git", "status", "--porcelain", "--untracked-files=no"):
            blockers.append("Tracked files have local changes")
        _run(root, "systemctl", "--user", "is-active", "--quiet", "sparkdeck.service")
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

    def agent_status(self) -> dict:
        revision = current_revision(self.root)
        state = self._read(self.agent_path)
        if state.get("phase") in {"accepted", "staging", "restarting"}:
            if revision and revision == state.get("target_revision"):
                state["phase"] = "succeeded"
                state["message"] = "Updated and restarted successfully"
                self._write(self.agent_path, state)
            elif not _helper_alive(state):
                state["phase"] = "failed"
                state["error"] = "The local update helper was interrupted"
                state["message"] = "Interrupted update can be retried"
                self._write(self.agent_path, state)
        return {
            "capability": CAPABILITY,
            "current_revision": revision,
            "phase": state.get("phase", "idle"),
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

    async def overview(self) -> dict:
        revision = current_revision(self.root)
        state = self._read(self.cluster_path)
        task_live = self._task is not None and not self._task.done()
        if state.get("active") and not task_live:
            if state.get("phase") == "updating_controller" and revision == state.get("target_revision"):
                state.update(active=False, phase="succeeded", message="Cluster update completed")
            elif state.get("phase") == "updating_controller" and _helper_alive(self._read(self.agent_path)):
                pass
            else:
                changed = any(
                    node.get("phase") == "succeeded"
                    or node.get("current_revision") == state.get("target_revision")
                    for node in state.get("nodes", []) if not node.get("local")
                )
                state.update(
                    active=False,
                    phase="partial" if changed else "failed",
                    error="The controller rollout task was interrupted",
                    message="Interrupted rollout can be retried",
                )
            self._write(self.cluster_path, state)
        releases, release_error = await self.published_releases()
        release_tags = {item["tag"] for item in releases}
        try:
            current_tags = _run(self.root, "git", "tag", "--points-at", "HEAD").splitlines()
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            current_tags = []
        current_release_tag = next((tag for tag in current_tags if tag in release_tags), None)
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
        all_blockers = ([release_error] if release_error else []) + [
            f"{node['name']}: {item}" for node in public_nodes for item in node["blockers"]
        ]
        return {
            "repository": REPOSITORY,
            "current_revision": revision,
            "current_release_tag": current_release_tag,
            "releases": releases,
            "latest_release": releases[0] if releases else None,
            "can_update": bool(releases) and not all_blockers and not state.get("active", False),
            "blockers": all_blockers,
            "nodes": public_nodes,
            "job": state or None,
        }

    async def start_cluster(self, confirmation: str, tag: str) -> dict:
        if confirmation != CONFIRMATION:
            raise ValueError("Explicit cluster update confirmation is required")
        async with self._lock:
            existing = self._read(self.cluster_path)
            if existing.get("active"):
                raise RuntimeError("A cluster update is already running")
            overview = await self.overview()
            if not overview["can_update"]:
                raise RuntimeError("; ".join(overview["blockers"]) or "No update is available")
            release, release_error = await self.resolve_release(tag, force=True)
            if release_error or not release:
                raise ValueError(release_error or "Selected release is unavailable")
            if overview["nodes"] and all(
                node.get("current_revision") == release["revision"]
                for node in overview["nodes"]
            ):
                raise RuntimeError("Selected release is already installed on every node")
            nodes = [{
                "id": node["id"], "name": node["name"], "local": node["local"],
                "phase": "pending", "current_revision": node.get("current_revision"),
            } for node in overview["nodes"]]
            state = {
                "id": uuid.uuid4().hex, "active": True, "phase": "preflight",
                "message": "Checking every cluster node before making changes",
                "target_tag": release["tag"], "target_revision": release["revision"],
                "release_url": release.get("url"), "started_at": time.time(), "nodes": nodes,
            }
            self._write(self.cluster_path, state)
            self._task = asyncio.create_task(self._run_cluster(state))
            return state

    async def _run_cluster(self, state: dict) -> None:
        changed_workers = 0
        try:
            # Re-probe every worker before mutating the first one.
            await self.preflight_local(state["target_tag"], state["target_revision"])
            for node in state["nodes"]:
                if node["local"]:
                    continue
                status = await self.manager.node_registry.request(
                    node["id"], "POST", "/api/agent/system-update/preflight",
                    json_body={"tag": state["target_tag"], "revision": state["target_revision"]},
                    timeout=60,
                )
                if status.get("blockers"):
                    raise RuntimeError(f"{node['name']}: {'; '.join(status['blockers'])}")
                if status.get("capability") != CAPABILITY:
                    raise RuntimeError(f"{node['name']}: one-time manual update required")
            state.update(phase="updating_workers", message="Updating workers one at a time")
            self._write(self.cluster_path, state)
            for node in state["nodes"]:
                if node["local"]:
                    continue
                node["phase"] = "updating"
                self._write(self.cluster_path, state)
                await self.manager.node_registry.request(
                    node["id"], "POST", "/api/agent/system-update",
                    json_body={"tag": state["target_tag"], "revision": state["target_revision"]},
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
                        changed_workers += 1
                        break
                    if status.get("phase") in {"failed", "rolled_back", "recovery_required"}:
                        raise RuntimeError(f"{node['name']}: {status.get('error') or 'update failed'}")
                else:
                    raise RuntimeError(f"{node['name']}: timed out waiting for restart")
            state.update(phase="updating_controller", message="Workers updated; restarting controller last")
            local = next(node for node in state["nodes"] if node["local"])
            local["phase"] = "updating"
            self._write(self.cluster_path, state)
            await self.start_local(state["target_tag"], state["target_revision"])
        except Exception as exc:
            state.update(
                active=False,
                phase="partial" if changed_workers else "failed",
                error=str(exc)[:500],
                message=(
                    "Update stopped after one or more workers changed"
                    if changed_workers else "Update stopped before any node changed"
                ),
            )
            self._write(self.cluster_path, state)

    async def preflight_local(self, tag: str, revision: str) -> dict:
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

    async def start_local(self, tag: str, revision: str) -> dict:
        async with self._agent_lock:
            if not tag or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
                raise ValueError("A valid release tag and immutable commit are required")
            await self.preflight_local(tag, revision)
            state = self._read(self.agent_path)
            if state.get("phase") in {"accepted", "staging", "restarting"}:
                raise RuntimeError("This node is already updating")
            if current_revision(self.root) == revision.lower():
                state = {"phase": "succeeded", "target_revision": revision.lower(), "message": "Already up to date"}
                self._write(self.agent_path, state)
                return state
            state = {"phase": "accepted", "target_tag": tag, "target_revision": revision.lower(), "message": "Update accepted"}
            self._write(self.agent_path, state)
            process = subprocess.Popen(
                [sys.executable, "-m", "sparkdeck.update_helper", "--root", str(self.root),
                 "--state", str(self.agent_path), "--tag", tag, "--revision", revision.lower()],
                cwd=self.root, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True,
            )
            state.update(helper_pid=process.pid, boot_id=_boot_id())
            self._write(self.agent_path, state)
            return state
