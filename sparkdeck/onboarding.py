"""Secure controller/worker onboarding and one-hop management forwarding."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import secrets
import socket
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from cluster import AGENT_PROTOCOL_VERSION


FORWARD_NODE_HEADER = "x-sparkdeck-forward-node"
FORWARD_HOP_HEADER = "x-sparkdeck-forward-hop"
FORWARD_TOKEN_HEADER = "x-sparkdeck-forward-token"
FORWARD_HEADERS = {FORWARD_NODE_HEADER, FORWARD_HOP_HEADER, FORWARD_TOKEN_HEADER}
JOIN_CODE_TTL_SECONDS = 600.0
JOIN_RATE_LIMIT = 5
JOIN_RATE_WINDOW_SECONDS = 60.0
PROXY_TIMEOUT_SECONDS = 600.0

_TAILSCALE_V4 = ipaddress.ip_network("100.64.0.0/10")
_TAILSCALE_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")


def _atomic_private_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    try:
        temporary.chmod(0o600)
    except OSError:
        pass
    temporary.replace(path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def normalize_control_url(value: Any) -> str:
    text = str(value or "").strip().rstrip("/")
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must be an http(s) URL with a hostname")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("URL cannot contain credentials, query parameters, or fragments")
    if parsed.path not in {"", "/"}:
        raise ValueError("URL must point to the SparkDeck server root")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _allowed_control_ip(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return address.is_loopback or address in _TAILSCALE_V4 or address in _TAILSCALE_V6


async def validate_control_url(value: Any) -> str:
    """Allow only loopback or Tailscale-reachable controller/agent URLs."""
    url = normalize_control_url(value)
    host = urlsplit(url).hostname or ""
    if host.casefold() == "localhost":
        return url
    try:
        if _allowed_control_ip(host):
            return url
        raise ValueError("URL must resolve only to Tailscale or loopback addresses")
    except ValueError as literal_error:
        try:
            rows = await asyncio.to_thread(
                socket.getaddrinfo, host, None, type=socket.SOCK_STREAM
            )
        except OSError as exc:
            raise ValueError("URL hostname could not be resolved") from exc
        addresses = {row[4][0].split("%", 1)[0] for row in rows}
        if not addresses or not all(_allowed_control_ip(address) for address in addresses):
            raise ValueError("URL must resolve only to Tailscale or loopback addresses") from literal_error
        return url


class ControllerAssignment:
    """Worker-only controller identity and forwarding credential."""

    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "controller.json"

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not all(value.get(key) for key in ("controller_url", "forward_token", "node_id")):
            return None
        return value

    def save(self, value: dict[str, Any]) -> None:
        _atomic_private_json(self.path, value)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


def is_forwardable_path(path: str) -> bool:
    if path == "/api/agent" or path.startswith("/api/agent/"):
        return False
    if path == "/api/v1/onboarding" or path.startswith("/api/v1/onboarding/"):
        return False
    return (
        path == "/api" or path.startswith("/api/")
        or path == "/v1" or path.startswith("/v1/")
        or path == "/mcp" or path.startswith("/mcp/")
    )


class OnboardingService:
    def __init__(self, manager: Any, data_dir: Path, port: int = 7878):
        self.manager = manager
        self.port = int(port)
        self.assignment = ControllerAssignment(data_dir)
        self._join_attempts: dict[str, deque[float]] = defaultdict(deque)

    @property
    def role(self) -> str:
        return "worker" if self.assignment.load() else "controller"

    @staticmethod
    def _instructions() -> list[str]:
        return [
            "Workers use their assigned controller while it is reachable.",
            "There is no automatic controller failover or leader election.",
        ]

    def _access_urls(self, request_origin: str) -> list[str]:
        parsed_origin = urlsplit(normalize_control_url(request_origin))
        port = parsed_origin.port or (443 if parsed_origin.scheme == "https" else 80)
        urls = []
        for interface in self.manager._network_interfaces():
            name = str(interface.get("name") or "").casefold()
            for address in interface.get("ipv4") or []:
                try:
                    tailscale = ipaddress.ip_address(address) in _TAILSCALE_V4
                except ValueError:
                    continue
                if tailscale or "tailscale" in name:
                    urls.append(f"{parsed_origin.scheme}://{address}:{port}")
        urls.append(normalize_control_url(request_origin))
        return list(dict.fromkeys(urls))

    async def status(self, request_origin: str) -> dict[str, Any]:
        parsed_origin = urlsplit(normalize_control_url(request_origin))
        port = parsed_origin.port or (443 if parsed_origin.scheme == "https" else 80)
        assignment = self.assignment.load()
        role = "worker" if assignment else "controller"
        controller_url = assignment.get("controller_url") if assignment else None
        reachable = role == "controller"
        if controller_url:
            try:
                response = await self.manager.http.get(
                    f"{controller_url}/api/v1/onboarding", timeout=3,
                )
                response.raise_for_status()
                remote = response.json()
                reachable = (
                    remote.get("role") == "controller"
                    and int(remote.get("node", {}).get("protocol_version") or 0)
                    == AGENT_PROTOCOL_VERSION
                )
            except Exception:
                reachable = False
        result = {
            "role": role,
            "node": {
                "id": self.manager.agent_credentials.node_id,
                "name": self.manager.settings.get("cluster_node_name"),
                "port": port,
                "protocol_version": AGENT_PROTOCOL_VERSION,
                "access_urls": self._access_urls(request_origin),
            },
            "controller_url": controller_url,
            "controller_reachable": reachable,
            "automatic_failover": False,
            "instructions": self._instructions(),
        }
        if role == "controller":
            result["join_code"] = self.manager.agent_credentials.current_cluster_join_code(
                JOIN_CODE_TTL_SECONDS
            )
        return result

    def _check_join_rate(self, client_id: str) -> None:
        now = time.monotonic()
        attempts = self._join_attempts[client_id or "unknown"]
        while attempts and now - attempts[0] > JOIN_RATE_WINDOW_SECONDS:
            attempts.popleft()
        if len(attempts) >= JOIN_RATE_LIMIT:
            raise ValueError("too many join attempts; wait before trying again")
        attempts.append(now)

    async def register(
        self, body: dict[str, Any], request_origin: str, client_id: str,
    ) -> dict[str, Any]:
        if self.role != "controller":
            raise ValueError("only a controller can register workers")
        self._check_join_rate(client_id)
        advertise_url = await validate_control_url(body.get("advertise_url"))
        pairing_code = str(body.get("pairing_code") or "").strip()
        join_code = str(body.get("join_code") or "").strip()
        if not pairing_code:
            raise ValueError("pairing_code is required")
        self.manager.agent_credentials.consume_cluster_join_code(join_code)
        paired = await self.manager.pair_node({
            "agent_url": advertise_url,
            "pairing_code": pairing_code,
            "name": str(body.get("name") or "Worker").strip(),
        })
        if int(paired.get("protocol_version") or 0) != AGENT_PROTOCOL_VERSION:
            raise ValueError("worker agent protocol version mismatch")
        forward_token = secrets.token_urlsafe(32)
        self.manager.node_registry.set_forward_token(paired["id"], forward_token)
        nodes = await self.manager.cluster_nodes()
        selected = next((node for node in nodes if node.get("id") == paired["id"]), paired)
        return {
            "ok": True,
            "role": "controller",
            "protocol_version": AGENT_PROTOCOL_VERSION,
            "controller": await self.status(request_origin),
            "node": self.manager.public_target_node(selected),
            "cluster": {
                "nodes": [self.manager.public_target_node(node) for node in nodes],
            },
            # Returned exactly once to the joining worker and never included
            # in public status, settings, node inventory, or logs.
            "forward_token": forward_token,
        }

    async def join(self, body: dict[str, Any], request_origin: str) -> dict[str, Any]:
        controller_url = await validate_control_url(body.get("controller_url"))
        advertise_url = await validate_control_url(body.get("advertise_url"))
        if controller_url == advertise_url:
            raise ValueError("controller_url and advertise_url must identify different nodes")
        try:
            identity_response = await self.manager.http.get(
                f"{controller_url}/api/v1/onboarding", timeout=5,
            )
            identity_response.raise_for_status()
            identity = identity_response.json()
        except httpx.HTTPError as exc:
            raise ValueError(f"controller is unreachable: {exc}") from exc
        if identity.get("role") != "controller":
            raise ValueError("controller_url points to a worker")
        if int(identity.get("node", {}).get("protocol_version") or 0) != AGENT_PROTOCOL_VERSION:
            raise ValueError("controller protocol version mismatch")

        response = await self.manager.http.post(
            f"{controller_url}/api/v1/onboarding/register",
            json={
                "join_code": str(body.get("join_code") or "").strip(),
                "advertise_url": advertise_url,
                "name": str(body.get("name") or self.manager.settings.get("cluster_node_name") or "Worker").strip(),
                "pairing_code": self.manager.agent_credentials.data["pairing_code"],
            },
            timeout=30,
        )
        if response.status_code >= 400:
            raise ValueError(f"controller rejected join: {response.text[:500]}")
        joined = response.json()
        if (
            joined.get("role") != "controller"
            or int(joined.get("protocol_version") or 0) != AGENT_PROTOCOL_VERSION
            or not joined.get("forward_token")
        ):
            raise ValueError("controller returned an invalid join response")
        controller_node_id = str(identity.get("node", {}).get("id") or "")
        if not controller_node_id:
            raise ValueError("controller identity is missing a node ID")
        self.assignment.save({
            "controller_url": controller_url,
            "forward_token": str(joined["forward_token"]),
            "node_id": self.manager.agent_credentials.node_id,
            "controller_node_id": controller_node_id,
            "protocol_version": AGENT_PROTOCOL_VERSION,
            "joined_at": time.time(),
        })
        adopt_worker = getattr(self.manager, "adopt_worker_role", None)
        if adopt_worker is not None:
            await adopt_worker()
        result = await self.status(request_origin)
        result.update({"ok": True, "cluster": joined.get("cluster")})
        return result

    async def leave(self, request_origin: str) -> dict[str, Any]:
        self.assignment.clear()
        adopt_controller = getattr(self.manager, "adopt_controller_role", None)
        if adopt_controller is not None:
            adopt_controller()
        result = await self.status(request_origin)
        result["ok"] = True
        return result

    def validate_forward_headers(self, headers: Any) -> tuple[bool, str]:
        hop = str(headers.get(FORWARD_HOP_HEADER) or "")
        node_id = str(headers.get(FORWARD_NODE_HEADER) or "")
        token = str(headers.get(FORWARD_TOKEN_HEADER) or "")
        if hop != "1":
            return False, "forward hop must be exactly 1"
        if not self.manager.node_registry.accepts_forward_token(node_id, token):
            return False, "invalid worker forwarding credential"
        return True, ""


async def forward_management_request(
    request: Request, manager: Any, assignment: dict[str, Any],
):
    if any(request.headers.get(name) for name in FORWARD_HEADERS):
        return JSONResponse({"detail": "forwarded request chains are not allowed"}, status_code=508)
    controller_url = normalize_control_url(assignment["controller_url"])
    request_origin = normalize_control_url(str(request.base_url))
    if controller_url == request_origin:
        return JSONResponse({"detail": "controller URL points back to this worker"}, status_code=508)
    target = f"{controller_url}{request.url.path}"
    if request.url.query:
        target = f"{target}?{request.url.query}"
    headers = {
        key: value for key, value in request.headers.items()
        if key.casefold() not in FORWARD_HEADERS | {"host", "content-length", "connection"}
    }
    headers.update({
        FORWARD_NODE_HEADER: assignment["node_id"],
        FORWARD_HOP_HEADER: "1",
        FORWARD_TOKEN_HEADER: assignment["forward_token"],
    })
    try:
        upstream_request = manager.http.build_request(
            request.method, target, headers=headers, content=await request.body(),
            timeout=PROXY_TIMEOUT_SECONDS,
        )
        upstream = await manager.http.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        return JSONResponse(
            {
                "detail": f"controller is unreachable: {exc}",
                "controller_url": controller_url,
                "automatic_failover": False,
            },
            status_code=502,
        )

    async def content():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    response_headers = {
        key: value for key, value in upstream.headers.items()
        if key.casefold() in {
            "content-type", "content-encoding", "cache-control",
            "content-disposition", "vary",
        }
    }
    return StreamingResponse(
        content(), status_code=upstream.status_code, headers=response_headers,
    )
