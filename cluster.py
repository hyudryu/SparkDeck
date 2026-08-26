"""Cluster node persistence, pairing credentials, and remote-agent client.

The browser only talks to the coordinator.  This module keeps remote agent
tokens server-side and exposes sanitized node state for the UI.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import httpx


LOCAL_NODE_ID = "local"
AGENT_PROTOCOL_VERSION = 1


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(path)


def normalize_agent_url(value: str) -> str:
    url = (value or "").strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("agent URL must be an http(s) URL with a hostname")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("agent URL cannot contain credentials, query, or fragment")
    return url


class AgentCredentials:
    """Credentials used when this controller is contacted as a node agent."""

    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "agent.json"
        self.data = self._load_or_create()

    def _load_or_create(self) -> dict:
        if self.path.exists():
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
                if value.get("agent_token") and value.get("pairing_code"):
                    if not value.get("cluster_join_code"):
                        value["cluster_join_code"] = f"{secrets.randbelow(1_000_000):06d}"
                        value["cluster_join_code_issued_at"] = time.time()
                        _atomic_json_write(self.path, value)
                    elif not value.get("cluster_join_code_issued_at"):
                        value["cluster_join_code_issued_at"] = time.time()
                        _atomic_json_write(self.path, value)
                    return value
            except Exception:
                pass
        value = {
            "node_id": uuid.uuid4().hex,
            "agent_token": secrets.token_urlsafe(32),
            "pairing_code": f"{secrets.randbelow(1_000_000):06d}",
            "cluster_join_code": f"{secrets.randbelow(1_000_000):06d}",
            "cluster_join_code_issued_at": time.time(),
            "created_at": time.time(),
        }
        _atomic_json_write(self.path, value)
        try:
            self.path.chmod(0o600)
        except OSError:
            pass
        return value

    @property
    def node_id(self) -> str:
        return self.data["node_id"]

    def accepts_token(self, token: str) -> bool:
        return bool(token) and secrets.compare_digest(
            token, self.data.get("agent_token", "")
        )

    @property
    def cluster_join_code(self) -> str:
        return str(self.data["cluster_join_code"])

    def current_cluster_join_code(self, ttl_seconds: float = 600.0) -> str:
        issued_at = float(self.data.get("cluster_join_code_issued_at") or 0)
        if time.time() - issued_at > ttl_seconds:
            self._rotate_cluster_join_code()
        return self.cluster_join_code

    def _rotate_cluster_join_code(self) -> None:
        self.data["cluster_join_code"] = f"{secrets.randbelow(1_000_000):06d}"
        self.data["cluster_join_code_issued_at"] = time.time()
        _atomic_json_write(self.path, self.data)

    def consume_cluster_join_code(self, join_code: str) -> None:
        """Consume the controller's one-time cluster join code and rotate it."""
        if not join_code or not secrets.compare_digest(
            str(join_code), self.current_cluster_join_code()
        ):
            raise ValueError("invalid or expired cluster join code")
        self._rotate_cluster_join_code()

    def revoke_remote_access(self) -> None:
        """Durably invalidate every credential shared during cluster pairing.

        Keep the stable node ID, but replace the agent bearer and both pairing
        codes in one atomic write.  Updating ``self.data`` only after the write
        succeeds ensures callers never clear a controller assignment based on
        an in-memory-only revocation that would disappear after a restart.
        """
        value = {
            **self.data,
            "agent_token": secrets.token_urlsafe(32),
            "pairing_code": f"{secrets.randbelow(1_000_000):06d}",
            "cluster_join_code": f"{secrets.randbelow(1_000_000):06d}",
            "cluster_join_code_issued_at": time.time(),
            "credentials_rotated_at": time.time(),
        }
        value.pop("paired_at", None)
        _atomic_json_write(self.path, value)
        self.data = value

    def validate_pairing_code(self, pairing_code: str) -> None:
        if not pairing_code or not secrets.compare_digest(
            str(pairing_code), str(self.data.get("pairing_code", ""))
        ):
            raise ValueError("invalid pairing code")

    def pair(self, pairing_code: str) -> dict:
        self.validate_pairing_code(pairing_code)
        # One-time pairing codes prevent another controller from replaying a
        # code seen in logs. Rotating the bearer makes coordinator ownership
        # exclusive, so a previously paired cluster cannot keep reading or
        # relaying this node's usage after it joins a different cluster.
        self.data["agent_token"] = secrets.token_urlsafe(32)
        self.data["pairing_code"] = f"{secrets.randbelow(1_000_000):06d}"
        self.data["paired_at"] = time.time()
        _atomic_json_write(self.path, self.data)
        return {
            "node_id": self.node_id,
            "agent_token": self.data["agent_token"],
            "protocol_version": AGENT_PROTOCOL_VERSION,
        }


class NodeRegistry:
    def __init__(self, data_dir: Path, http: httpx.AsyncClient):
        self.path = Path(data_dir) / "nodes.json"
        self.http = http
        self.nodes = self._load()
        self._status_cache: dict[str, tuple[float, dict]] = {}

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, list) else []
        except Exception:
            return []

    def _save(self) -> None:
        _atomic_json_write(self.path, self.nodes)
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def get(self, node_id: str) -> dict | None:
        if node_id == LOCAL_NODE_ID:
            return {"id": LOCAL_NODE_ID, "name": "This node", "local": True}
        return next((n for n in self.nodes if n.get("id") == node_id), None)

    async def pair_remote(
        self,
        agent_url: str,
        pairing_code: str,
        name: str | None = None,
        fabric_ip: str | None = None,
        fabric_interface: str | None = None,
        usage_epoch: Any = None,
        usage_model_epochs: dict | None = None,
    ) -> dict:
        url = normalize_agent_url(agent_url)
        try:
            response = await self.http.post(
                f"{url}/api/agent/pair",
                json={
                    "pairing_code": str(pairing_code or ""),
                    "usage_epoch": usage_epoch,
                    "usage_model_epochs": usage_model_epochs or {},
                },
                timeout=10,
            )
            response.raise_for_status()
            paired = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            raise ValueError(f"agent rejected pairing: {detail}") from exc
        except httpx.HTTPError as exc:
            raise ValueError(f"could not contact node agent: {exc}") from exc
        if not paired.get("agent_token") or not paired.get("node_id"):
            raise ValueError("node agent returned an invalid pairing response")
        node_id = paired["node_id"]
        node = {
            "id": node_id,
            "name": (name or paired.get("name") or "Remote node").strip(),
            "agent_url": url,
            "agent_token": paired["agent_token"],
            "fabric_ip": (fabric_ip or "").strip() or None,
            "fabric_interface": (fabric_interface or "").strip() or None,
            "enabled": True,
            "protocol_version": paired.get("protocol_version"),
            "paired_at": time.time(),
        }
        self.nodes = [n for n in self.nodes if n.get("id") != node_id]
        self.nodes.append(node)
        self._save()
        self._status_cache.pop(node_id, None)
        return self.public_config(node)

    def update(self, node_id: str, changes: dict) -> dict:
        node = self.get(node_id)
        if not node or node_id == LOCAL_NODE_ID:
            raise ValueError("remote node not found")
        for key in ("name", "fabric_ip", "fabric_interface", "enabled"):
            if key in changes:
                node[key] = changes[key]
        if "agent_url" in changes:
            node["agent_url"] = normalize_agent_url(changes["agent_url"])
        self._save()
        self._status_cache.pop(node_id, None)
        return self.public_config(node)

    def remove(self, node_id: str) -> bool:
        before = len(self.nodes)
        self.nodes = [n for n in self.nodes if n.get("id") != node_id]
        if len(self.nodes) == before:
            return False
        self._save()
        self._status_cache.pop(node_id, None)
        return True

    @staticmethod
    def public_config(node: dict) -> dict:
        return {
            k: v for k, v in node.items()
            if k not in {"agent_token", "forward_token", "forward_token_hash"}
        }

    def set_forward_token(self, node_id: str, token: str) -> None:
        node = self.get(node_id)
        if not node or node_id == LOCAL_NODE_ID:
            raise ValueError("remote node not found")
        node.pop("forward_token", None)
        node["forward_token_hash"] = hashlib.sha256(str(token).encode("utf-8")).hexdigest()
        self._save()

    def accepts_forward_token(self, node_id: str, token: str) -> bool:
        node = self.get(node_id)
        candidate = hashlib.sha256(str(token).encode("utf-8")).hexdigest() if token else ""
        return bool(
            node
            and node_id != LOCAL_NODE_ID
            and node.get("enabled", True)
            and token
            and secrets.compare_digest(candidate, str(node.get("forward_token_hash") or ""))
        )

    async def request(
        self,
        node_id: str,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        timeout: float = 30,
    ) -> Any:
        node = self.get(node_id)
        if not node or node_id == LOCAL_NODE_ID:
            raise ValueError("remote node not found")
        if not node.get("enabled", True):
            raise ValueError(f"node {node.get('name', node_id)} is disabled")
        try:
            response = await self.http.request(
                method,
                f"{node['agent_url']}{path}",
                headers={"Authorization": f"Bearer {node['agent_token']}"},
                json=json_body,
                timeout=timeout,
            )
            response.raise_for_status()
            if not response.content:
                return None
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise RuntimeError(
                f"{node.get('name', node_id)} agent error: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"could not contact {node.get('name', node_id)}: {exc}"
            ) from exc

    async def open_stream(
        self,
        node_id: str,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        content: AsyncIterator[bytes] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 600,
    ) -> httpx.Response:
        """Open an authenticated agent stream; the caller must close it."""
        node = self.get(node_id)
        if not node or node_id == LOCAL_NODE_ID:
            raise ValueError("remote node not found")
        if not node.get("enabled", True):
            raise ValueError(f"node {node.get('name', node_id)} is disabled")
        request_headers = dict(headers or {})
        request_headers["Authorization"] = f"Bearer {node['agent_token']}"
        payload: dict[str, Any] = {}
        if content is not None:
            payload["content"] = content
        elif json_body is not None:
            payload["json"] = json_body
        request = self.http.build_request(
            method, f"{node['agent_url']}{path}", headers=request_headers,
            timeout=timeout, **payload,
        )
        try:
            return await self.http.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"could not contact {node.get('name', node_id)}: {exc}"
            ) from exc

    async def probe(self, node: dict, *, force: bool = False) -> dict:
        node_id = node["id"]
        cached = self._status_cache.get(node_id)
        if not force and cached and time.monotonic() - cached[0] < 4.0:
            return cached[1]
        started = time.monotonic()
        public = self.public_config(node)
        if not node.get("enabled", True):
            result = {**public, "status": "disabled", "online": False}
        else:
            try:
                status = await self.request(
                    node_id, "GET", "/api/agent/status", timeout=3
                )
                compatible = status.get("protocol_version") == AGENT_PROTOCOL_VERSION
                issues = []
                if not compatible:
                    issues.append("agent protocol version mismatch")
                if not status.get("docker_ready"):
                    issues.append("Docker is unavailable")
                authoritative_name = str(public.get("name") or "").strip()
                if (
                    authoritative_name
                    and str(status.get("name") or "").strip() != authoritative_name
                ):
                    try:
                        await self.request(
                            node_id, "PATCH", "/api/agent/node",
                            json_body={"name": authoritative_name}, timeout=10,
                        )
                        status["name"] = authoritative_name
                    except Exception:
                        # The registry alias remains authoritative in the
                        # controller UI and a later probe retries convergence.
                        issues.append("node name synchronization pending")
                result = {
                    **public,
                    **status,
                    "status": "online" if not issues else "degraded",
                    "online": True,
                    "latency_ms": round((time.monotonic() - started) * 1000),
                    "last_seen": time.time(),
                }
                # The controller's durable registry is authoritative for the
                # user-assigned display name. Agent status may briefly retain
                # the prior local name after an offline rename.
                result["name"] = public.get("name") or status.get("name")
                result["status_message"] = "; ".join(issues) or None
            except Exception as exc:
                previous = cached[1] if cached else {}
                result = {
                    **public,
                    "status": "unreachable",
                    "online": False,
                    "last_seen": previous.get("last_seen"),
                    "status_message": str(exc),
                }
        self._status_cache[node_id] = (time.monotonic(), result)
        return result

    async def public_nodes(self, local_status: dict) -> list[dict]:
        import asyncio

        remote = await asyncio.gather(
            *[self.probe(n) for n in self.nodes], return_exceptions=True
        )
        result = [{
            "id": LOCAL_NODE_ID,
            "name": local_status.get("name") or "This node",
            "local": True,
            "enabled": True,
            "status": "online",
            "online": True,
            "last_seen": time.time(),
            **local_status,
        }]
        for node, status in zip(self.nodes, remote):
            if isinstance(status, Exception):
                result.append({
                    **self.public_config(node),
                    "status": "unreachable",
                    "online": False,
                    "status_message": str(status),
                })
            else:
                result.append(status)
        return result
