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
RELEASE_API = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
TRUSTED_ORIGINS = {
    "https://github.com/hyudryu/SparkDeck.git",
    "https://github.com/hyudryu/SparkDeck",
    "git@github.com:hyudryu/SparkDeck.git",
    "ssh://git@github.com/hyudryu/SparkDeck.git",
}
CAPABILITY = "cluster_update_v1"
CONFIRMATION = "update-entire-cluster"


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
        self.agent_path = self.data_dir / "system-update-agent.json"
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._release_cache: tuple[float, dict | None, str | None] | None = None

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
        return {
            "capability": CAPABILITY,
            "current_revision": revision,
            "phase": state.get("phase", "idle"),
            "target_revision": state.get("target_revision"),
            "message": state.get("message"),
            "error": state.get("error"),
            "blockers": local_blockers(self.root),
        }

    async def latest_release(self) -> tuple[dict | None, str | None]:
        if self._release_cache and time.monotonic() - self._release_cache[0] < 300:
            return self._release_cache[1], self._release_cache[2]
        try:
            response = await self.manager.http.get(
                RELEASE_API,
                headers={"Accept": "application/vnd.github+json", "User-Agent": "SparkDeck"},
                timeout=10,
            )
            if response.status_code == 404:
                result = (None, "No published GitHub release is available")
                self._release_cache = (time.monotonic(), *result)
                return result
            response.raise_for_status()
            release = response.json()
            tag = str(release.get("tag_name") or "").strip()
            if not tag:
                return None, "The latest GitHub release has no tag"
            if not _valid_release_tag(tag):
                return None, "The latest GitHub release has an invalid tag"
            commit_response = await self.manager.http.get(
                f"https://api.github.com/repos/{REPOSITORY}/commits/{tag}",
                headers={"Accept": "application/vnd.github+json", "User-Agent": "SparkDeck"},
                timeout=10,
            )
            commit_response.raise_for_status()
            revision = str(commit_response.json().get("sha") or "").lower()
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                return None, "GitHub did not return an immutable release commit"
            result = ({
                "tag": tag,
                "revision": revision,
                "name": release.get("name") or tag,
                "url": release.get("html_url"),
                "published_at": release.get("published_at"),
            }, None)
            self._release_cache = (time.monotonic(), *result)
            return result
        except (httpx.HTTPError, ValueError) as exc:
            return None, f"Could not check GitHub releases: {str(exc)[:240]}"

    async def overview(self) -> dict:
        revision = current_revision(self.root)
        state = self._read(self.cluster_path)
        if state.get("active") and state.get("phase") == "updating_controller":
            if revision == state.get("target_revision"):
                state.update(active=False, phase="succeeded", message="Cluster update completed")
                self._write(self.cluster_path, state)
        release, release_error = await self.latest_release()
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
            "latest_release": release,
            "can_update": bool(release) and not all_blockers and not state.get("active", False),
            "blockers": all_blockers,
            "nodes": public_nodes,
            "job": state or None,
        }

    async def start_cluster(self, confirmation: str) -> dict:
        if confirmation != CONFIRMATION:
            raise ValueError("Explicit cluster update confirmation is required")
        async with self._lock:
            existing = self._read(self.cluster_path)
            if existing.get("active"):
                raise RuntimeError("A cluster update is already running")
            overview = await self.overview()
            if not overview["can_update"]:
                raise RuntimeError("; ".join(overview["blockers"]) or "No update is available")
            release = overview["latest_release"]
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
        try:
            # Re-probe every worker before mutating the first one.
            for node in state["nodes"]:
                if node["local"]:
                    continue
                status = await self.manager.node_registry.request(
                    node["id"], "GET", "/api/agent/system-update", timeout=10,
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
                    timeout=20,
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
                        break
                    if status.get("phase") == "failed":
                        raise RuntimeError(f"{node['name']}: {status.get('error') or 'update failed'}")
                else:
                    raise RuntimeError(f"{node['name']}: timed out waiting for restart")
            state.update(phase="updating_controller", message="Workers updated; restarting controller last")
            local = next(node for node in state["nodes"] if node["local"])
            local["phase"] = "updating"
            self._write(self.cluster_path, state)
            await self.start_local(state["target_tag"], state["target_revision"])
        except Exception as exc:
            state.update(active=False, phase="failed", error=str(exc)[:500], message="Update stopped safely")
            self._write(self.cluster_path, state)

    async def start_local(self, tag: str, revision: str) -> dict:
        if not tag or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
            raise ValueError("A valid release tag and immutable commit are required")
        latest, error = await self.latest_release()
        if error or not latest or latest["tag"] != tag or latest["revision"] != revision.lower():
            raise ValueError("Update target is not the current official GitHub release")
        blockers = local_blockers(self.root)
        if blockers:
            raise RuntimeError("; ".join(blockers))
        state = self._read(self.agent_path)
        if state.get("phase") in {"accepted", "staging", "restarting"}:
            raise RuntimeError("This node is already updating")
        if current_revision(self.root) == revision.lower():
            state = {"phase": "succeeded", "target_revision": revision.lower(), "message": "Already up to date"}
            self._write(self.agent_path, state)
            return state
        state = {"phase": "accepted", "target_tag": tag, "target_revision": revision.lower(), "message": "Update accepted"}
        self._write(self.agent_path, state)
        subprocess.Popen(
            [sys.executable, "-m", "sparkdeck.update_helper", "--root", str(self.root),
             "--state", str(self.agent_path), "--tag", tag, "--revision", revision.lower()],
            cwd=self.root, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
        return state
