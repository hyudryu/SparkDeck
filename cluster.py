"""Cluster node persistence, pairing credentials, and remote-agent client.

The browser only talks to the coordinator.  This module keeps remote agent
tokens server-side and exposes sanitized node state for the UI.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable
from urllib.parse import urlparse, urlunparse

import httpx

from sparkdeck.private_json import atomic_private_json_write as _atomic_json_write


LOCAL_NODE_ID = "local"
AGENT_PROTOCOL_VERSION = 1
COORDINATOR_ID_HEADER = "X-SparkDeck-Coordinator-ID"


class NodeAgentResponseError(RuntimeError):
    """A non-success response returned by an authenticated cluster agent."""

    def __init__(self, node_name: str, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{node_name} agent error: HTTP {status_code}: {detail}")


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

    def authorize_controller(self, token: str, controller_id: str) -> bool:
        """Authorize and durably claim one coordinator for legacy pairings."""
        if not self.accepts_token(token) or not str(controller_id or "").strip():
            return False
        controller_id = str(controller_id).strip()
        owner = str(self.data.get("paired_controller_id") or "")
        if owner:
            return secrets.compare_digest(owner, controller_id)
        value = {
            **self.data,
            "paired_controller_id": controller_id,
            "controller_claimed_at": time.time(),
        }
        _atomic_json_write(self.path, value)
        self.data = value
        return True

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
        value.pop("paired_controller_id", None)
        value.pop("controller_claimed_at", None)
        _atomic_json_write(self.path, value)
        self.data = value

    def validate_pairing_code(self, pairing_code: str) -> None:
        if not pairing_code or not secrets.compare_digest(
            str(pairing_code), str(self.data.get("pairing_code", ""))
        ):
            raise ValueError("invalid pairing code")

    def pair(self, pairing_code: str, controller_id: str) -> dict:
        self.validate_pairing_code(pairing_code)
        controller_id = str(controller_id or "").strip()
        if not controller_id:
            raise ValueError("controller id is required")
        # One-time pairing codes prevent another controller from replaying a
        # code seen in logs. Rotating the bearer makes coordinator ownership
        # exclusive, so a previously paired cluster cannot keep reading or
        # relaying this node's usage after it joins a different cluster.
        value = {
            **self.data,
            "agent_token": secrets.token_urlsafe(32),
            "pairing_code": f"{secrets.randbelow(1_000_000):06d}",
            "paired_controller_id": controller_id,
            "paired_at": time.time(),
        }
        _atomic_json_write(self.path, value)
        self.data = value
        return {
            "node_id": self.node_id,
            "agent_token": value["agent_token"],
            "protocol_version": AGENT_PROTOCOL_VERSION,
        }


class NodeRegistry:
    def __init__(
        self, data_dir: Path, http: httpx.AsyncClient,
        controller_id: str = "",
        connection_resolver: Callable[[Any], Awaitable[Any]] | None = None,
    ):
        self.path = Path(data_dir) / "nodes.json"
        self.http = http
        self.controller_id = str(controller_id or "")
        self.connection_resolver = connection_resolver
        self.nodes = self._load()
        self._status_cache: dict[str, tuple[float, dict]] = {}

    async def _connection_targets(
        self, agent_url: str, path: str,
    ) -> list[tuple[str, dict[str, str], dict[str, Any]]]:
        if self.connection_resolver is None:
            return [(f"{agent_url}{path}", {}, {})]
        connection = await self.connection_resolver(agent_url)
        return [
            (
                f"{connect_url}{path}",
                {"Host": connection.host_header},
                {"sni_hostname": connection.sni_hostname},
            )
            for connect_url in connection.connect_urls
        ]

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
    ) -> dict:
        url = normalize_agent_url(agent_url)
        try:
            targets = await self._connection_targets(
                url, "/api/agent/pair",
            )
            last_error: httpx.HTTPError | None = None
            for target, pinned_headers, extensions in targets:
                try:
                    response = await self.http.post(
                        target,
                        headers=pinned_headers,
                        json={
                            "pairing_code": str(pairing_code or ""),
                            "controller_id": self.controller_id,
                        },
                        timeout=10,
                        extensions=extensions,
                        follow_redirects=False,
                    )
                    response.raise_for_status()
                    paired = response.json()
                    break
                except httpx.HTTPStatusError:
                    raise
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    last_error = exc
            else:
                if last_error is None:
                    raise RuntimeError("node agent has no connectable endpoints")
                raise last_error
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
            "hidden_from_dashboard": False,
            "protocol_version": paired.get("protocol_version"),
            "paired_at": time.time(),
            "usage_reconciled": False,
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
        if (
            "hidden_from_dashboard" in changes
            and not isinstance(changes["hidden_from_dashboard"], bool)
        ):
            raise ValueError("hidden_from_dashboard must be a boolean")
        for key in (
            "name", "fabric_ip", "fabric_interface", "enabled",
            "hidden_from_dashboard",
        ):
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

    def mark_usage_reconciled(self, node_id: str) -> None:
        node = self.get(node_id)
        if not node or node_id == LOCAL_NODE_ID:
            raise ValueError("remote node not found")
        if node.get("usage_reconciled") is True:
            return
        node["usage_reconciled"] = True
        self._save()

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

    def direct_transfer_source(self, node_id: str) -> str | None:
        """Return a fabric-reachable source endpoint for a direct archive pull."""
        node = self.get(node_id)
        if not node or node_id == LOCAL_NODE_ID or not node.get("enabled", True):
            return None
        try:
            fabric_ip = str(ipaddress.ip_address(str(node.get("fabric_ip") or "")))
            parsed = urlparse(str(node["agent_url"]))
        except (KeyError, TypeError, ValueError):
            return None
        if parsed.scheme != "http" or not parsed.port:
            return None
        host = f"[{fabric_ip}]" if ":" in fabric_ip else fabric_ip
        return urlunparse((
            "http", f"{host}:{parsed.port}", parsed.path.rstrip("/"), "", "", "",
        ))

    async def request(
        self,
        node_id: str,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        timeout: float = 30,
        allow_disabled: bool = False,
    ) -> Any:
        node = self.get(node_id)
        if not node or node_id == LOCAL_NODE_ID:
            raise ValueError("remote node not found")
        if not allow_disabled and not node.get("enabled", True):
            raise ValueError(f"node {node.get('name', node_id)} is disabled")
        try:
            targets = await self._connection_targets(
                node["agent_url"], path,
            )
            last_error: httpx.HTTPError | None = None
            for target, pinned_headers, extensions in targets:
                try:
                    response = await self.http.request(
                        method,
                        target,
                        headers={
                            **pinned_headers,
                            "Authorization": f"Bearer {node['agent_token']}",
                            COORDINATOR_ID_HEADER: self.controller_id,
                        },
                        json=json_body,
                        timeout=timeout,
                        extensions=extensions,
                        follow_redirects=False,
                    )
                    response.raise_for_status()
                    if not response.content:
                        return None
                    return response.json()
                except httpx.HTTPStatusError:
                    raise
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    last_error = exc
            if last_error is None:
                raise RuntimeError(
                    f"{node.get('name', node_id)} agent has no connectable endpoints"
                )
            raise last_error
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise NodeAgentResponseError(
                str(node.get("name", node_id)), exc.response.status_code, detail,
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
        request_headers[COORDINATOR_ID_HEADER] = self.controller_id
        targets = await self._connection_targets(
            node["agent_url"], path,
        )
        payload: dict[str, Any] = {}
        if content is not None:
            payload["content"] = content
        elif json_body is not None:
            payload["json"] = json_body
        last_error: httpx.HTTPError | None = None
        for target, pinned_headers, extensions in targets:
            candidate_headers = {**request_headers, **pinned_headers}
            request = self.http.build_request(
                method, target, headers=candidate_headers,
                timeout=timeout, extensions=extensions, **payload,
            )
            try:
                return await self.http.send(
                    request, stream=True, follow_redirects=False,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = exc
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"could not contact {node.get('name', node_id)}: {exc}"
                ) from exc
        if last_error is None:
            raise RuntimeError(
                f"could not contact {node.get('name', node_id)}: no connectable endpoints"
            )
        raise RuntimeError(
            f"could not contact {node.get('name', node_id)}: {last_error}"
        ) from last_error

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
                    issues.append(
                        str(status.get("status_message") or "").strip()
                        or "Docker is unavailable"
                    )
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
                # the prior local name after an offline rename. Dashboard
                # visibility is also controller-owned presentation state.
                result["name"] = public.get("name") or status.get("name")
                result["hidden_from_dashboard"] = bool(
                    public.get("hidden_from_dashboard", False)
                )
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
