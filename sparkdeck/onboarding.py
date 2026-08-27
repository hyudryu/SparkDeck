"""Secure controller/worker onboarding and one-hop management forwarding."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import secrets
import socket
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from docker.errors import DockerException
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from cluster import AGENT_PROTOCOL_VERSION, normalize_agent_url
from sparkdeck.private_json import atomic_private_json_write as _atomic_private_json


FORWARD_NODE_HEADER = "x-sparkdeck-forward-node"
FORWARD_HOP_HEADER = "x-sparkdeck-forward-hop"
FORWARD_TOKEN_HEADER = "x-sparkdeck-forward-token"
FORWARD_SCHEME_HEADER = "x-sparkdeck-forward-scheme"
FORWARD_HEADERS = {
    FORWARD_NODE_HEADER,
    FORWARD_HOP_HEADER,
    FORWARD_TOKEN_HEADER,
    FORWARD_SCHEME_HEADER,
}
JOIN_CODE_TTL_SECONDS = 600.0
JOIN_RATE_LIMIT = 5
JOIN_RATE_WINDOW_SECONDS = 60.0
PROXY_TIMEOUT_SECONDS = 600.0
PROXY_DISCONNECT_POLL_SECONDS = 0.1

_TAILSCALE_V4 = ipaddress.ip_network("100.64.0.0/10")
_TAILSCALE_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")


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


@dataclass(frozen=True)
class ControlConnection:
    """Logical SparkDeck URL plus its IP-pinned transport destinations."""

    url: str
    connect_urls: tuple[str, ...]
    host_header: str
    sni_hostname: str

    @property
    def connect_url(self) -> str:
        """First candidate retained for callers that need one preferred URL."""
        return self.connect_urls[0]


async def _resolve_pinned_connection(url: str) -> ControlConnection:
    parsed = httpx.URL(url)
    host = parsed.raw_host.decode("ascii")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        literal = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError as literal_error:
        try:
            rows = await asyncio.to_thread(
                socket.getaddrinfo, host, port, type=socket.SOCK_STREAM
            )
        except OSError as exc:
            raise ValueError("URL hostname could not be resolved") from exc
        addresses = []
        for row in rows:
            address = row[4][0].split("%", 1)[0]
            if address not in addresses:
                addresses.append(address)
        if not addresses or not all(_allowed_control_ip(address) for address in addresses):
            raise ValueError(
                "URL must resolve only to Tailscale or loopback addresses"
            ) from literal_error
    else:
        if not _allowed_control_ip(str(literal)):
            raise ValueError("URL must resolve only to Tailscale or loopback addresses")
        addresses = [str(literal)]

    # The transport connects to this numeric address, so it cannot repeat DNS
    # after validation. Host and SNI retain the logical paired-node identity.
    return ControlConnection(
        url=url,
        connect_urls=tuple(
            str(parsed.copy_with(host=address)) for address in addresses
        ),
        host_header=parsed.netloc.decode("ascii"),
        sni_hostname=host,
    )


async def resolve_control_connection(value: Any) -> ControlConnection:
    """Resolve, validate, and pin one root-level controller connection."""
    return await _resolve_pinned_connection(normalize_control_url(value))


async def resolve_agent_connection(value: Any) -> ControlConnection:
    """Resolve and pin an agent URL while preserving its accepted base path."""
    return await _resolve_pinned_connection(normalize_agent_url(str(value or "")))


async def validate_control_url(value: Any) -> str:
    """Allow only root-level, loopback or Tailscale-reachable controller URLs."""
    return (await resolve_control_connection(value)).url


class _ControlClientDisconnected(Exception):
    pass


async def _send_pinned_control_request(
    http: httpx.AsyncClient,
    connection: ControlConnection,
    method: str,
    path: str,
    *,
    query: str = "",
    headers: dict[str, str] | None = None,
    content: bytes | None = None,
    json_body: Any = None,
    timeout: float,
    stream: bool = False,
    disconnect_task: asyncio.Task | None = None,
) -> httpx.Response:
    """Send to prevalidated addresses, retrying only connection failures."""
    last_connect_error: httpx.HTTPError | None = None
    for connect_url in connection.connect_urls:
        target = f"{connect_url}{path}"
        if query:
            target = f"{target}?{query}"
        payload: dict[str, Any] = {}
        if content is not None:
            payload["content"] = content
        elif json_body is not None:
            payload["json"] = json_body
        request = http.build_request(
            method,
            target,
            headers={**(headers or {}), "Host": connection.host_header},
            timeout=timeout,
            extensions={"sni_hostname": connection.sni_hostname},
            **payload,
        )
        send_task = asyncio.create_task(http.send(
            request, stream=stream, follow_redirects=False,
        ))
        if disconnect_task is not None:
            done, _ = await asyncio.wait(
                {send_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done and disconnect_task.result():
                if send_task.done() and not send_task.cancelled():
                    try:
                        response = send_task.result()
                    except httpx.HTTPError:
                        pass
                    else:
                        await response.aclose()
                else:
                    send_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await send_task
                raise _ControlClientDisconnected
        try:
            return await send_task
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            last_connect_error = exc
    if last_connect_error is None:
        raise RuntimeError("controller has no connectable endpoints")
    raise last_connect_error


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
        self.data_dir = Path(data_dir)
        self.port = int(port)
        self.assignment = ControllerAssignment(data_dir)
        self._join_attempts: dict[str, deque[float]] = defaultdict(deque)
        self._membership_lock = asyncio.Lock()

    @property
    def role(self) -> str:
        return "worker" if self.assignment.load() else "controller"

    @staticmethod
    def _instructions() -> list[str]:
        return [
            "Connect the joining system and this entry node to the same Tailscale tailnet.",
            "Open Cluster onboarding on the system you want to join.",
            "Enter one of this node's access URLs and the one-time pairing code.",
            "Repeat this for every machine; joining one node never imports its former cluster members.",
        ]

    def _access_urls(self, request_origin: str) -> list[str]:
        normalized_origin = normalize_control_url(request_origin)
        parsed_origin = urlsplit(normalized_origin)
        urls = []

        # A Tailscale Serve hostname is the preferred cluster address because
        # it is a browser-secure origin. Preserve it exactly as received.
        try:
            origin_ip = ipaddress.ip_address(parsed_origin.hostname or "")
            origin_is_loopback = origin_ip.is_loopback
        except ValueError:
            origin_is_loopback = (parsed_origin.hostname or "").casefold() == "localhost"
        if not origin_is_loopback and parsed_origin.scheme == "https":
            urls.append(normalized_origin)

        for interface in self.manager._network_interfaces():
            name = str(interface.get("name") or "").casefold()
            for address in interface.get("ipv4") or []:
                try:
                    tailscale = ipaddress.ip_address(address) in _TAILSCALE_V4
                except ValueError:
                    continue
                if tailscale or "tailscale" in name:
                    # SparkDeck's built-in server is plain HTTP. TLS is
                    # terminated separately by Tailscale Serve.
                    urls.append(f"http://{address}:{self.port}")

        # A user may open SparkDeck through its raw Tailscale IP instead of a
        # Serve hostname. Keep that origin, but never advertise loopback to a
        # different node as if it were a cluster-reachable address.
        if not origin_is_loopback and normalized_origin not in urls:
            urls.append(normalized_origin)
        return list(dict.fromkeys(urls))

    async def status(self, request_origin: str) -> dict[str, Any]:
        assignment = self.assignment.load()
        role = "worker" if assignment else "controller"
        controller_url = assignment.get("controller_url") if assignment else None
        reachable = role == "controller"
        controller_node_id = assignment.get("controller_node_id") if assignment else None
        join_code = None
        if controller_url:
            try:
                connection = await resolve_control_connection(controller_url)
                response = await _send_pinned_control_request(
                    self.manager.http,
                    connection,
                    "GET",
                    "/api/v1/onboarding",
                    timeout=3,
                )
                response.raise_for_status()
                remote = response.json()
                remote_node_id = str(remote.get("node", {}).get("id") or "")
                reachable = (
                    remote.get("role") == "controller"
                    and int(remote.get("node", {}).get("protocol_version") or 0)
                    == AGENT_PROTOCOL_VERSION
                    and bool(controller_node_id)
                    and secrets.compare_digest(remote_node_id, str(controller_node_id))
                )
                if reachable:
                    join_code = remote.get("join_code")
            except Exception:
                reachable = False
        result = {
            "role": role,
            "node": {
                "id": self.manager.agent_credentials.node_id,
                "name": self.manager.settings.get("cluster_node_name"),
                "port": self.port,
                "protocol_version": AGENT_PROTOCOL_VERSION,
                "access_urls": self._access_urls(request_origin),
            },
            "controller_url": controller_url,
            "controller_node_id": controller_node_id,
            "controller_reachable": reachable,
            "automatic_failover": False,
            "instructions": self._instructions(),
        }
        if role == "controller":
            join_code = self.manager.agent_credentials.current_cluster_join_code(
                JOIN_CODE_TTL_SECONDS
            )
        if join_code:
            result["join_code"] = join_code
        return result

    def _check_join_rate(self, client_id: str) -> None:
        now = time.monotonic()
        attempts = self._join_attempts[client_id or "unknown"]
        while attempts and now - attempts[0] > JOIN_RATE_WINDOW_SECONDS:
            attempts.popleft()
        if len(attempts) >= JOIN_RATE_LIMIT:
            raise ValueError("too many join attempts; wait before trying again")
        attempts.append(now)

    def _saved_deployment_count(self) -> int:
        """Count local application deployment records without mutating SQLite."""
        path = self.data_dir / "sparkdeck.sqlite3"
        if not path.exists():
            return 0
        try:
            uri = f"file:{path.resolve().as_posix()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=2.0)
            try:
                exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'deployments'"
                ).fetchone()
                if not exists:
                    return 0
                row = connection.execute("SELECT COUNT(*) FROM deployments").fetchone()
                return int(row[0] if row else 0)
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise ValueError(
                "cannot verify that saved deployments were migrated; "
                "repair sparkdeck.sqlite3 and retry"
            ) from exc

    async def _assert_no_owned_workloads(self, action: str) -> None:
        saved = self._saved_deployment_count()
        legacy = len([
            deployment for deployment in getattr(self.manager, "deployments", [])
            if isinstance(deployment, dict)
        ])
        ledger = getattr(self.manager, "managed_workload_ledger", None)
        claims = ledger.snapshot() if ledger is not None else {}
        managed = []
        if not (saved or legacy):
            try:
                containers = await self.manager.list_containers()
            except DockerException as exc:
                # Joining changes only SparkDeck's controller assignment. An
                # otherwise-empty controller-only installation does not need
                # Docker to become a worker. Keep leave fail-closed because an
                # offline worker's inventory is its last ownership check.
                if action == "join" and not claims:
                    return
                if action == "join":
                    raise ValueError(
                        "cannot join while this node has durable managed workload "
                        "ownership claims and Docker is unavailable to verify them; "
                        "restore Docker access, remove the workloads, and retry"
                    ) from exc
                raise ValueError(
                    "cannot verify that managed workloads were migrated because Docker "
                    "is unavailable; restore Docker access and retry"
                ) from exc
            managed = [container for container in containers if container.get("managed")]
            claims = ledger.snapshot() if ledger is not None else {}
        managed_names = {str(container.get("name") or "") for container in managed}
        pending_claims = [name for name in claims if name not in managed_names]
        if not (saved or legacy or managed or pending_claims):
            return
        details = []
        if saved:
            details.append(f"{saved} saved deployment record{'s' if saved != 1 else ''}")
        if legacy:
            details.append(f"{legacy} legacy deployment record{'s' if legacy != 1 else ''}")
        if managed:
            details.append(f"{len(managed)} managed container{'s' if len(managed) != 1 else ''}")
        if pending_claims:
            details.append(
                f"{len(pending_claims)} pending managed workload "
                f"claim{'s' if len(pending_claims) != 1 else ''}"
            )
        if action == "leave":
            raise ValueError(
                "cannot leave while this node still hosts " + ", ".join(details) + ". "
                "Stop and remove them from the current controller, then retry so "
                "the deployment remains manageable."
            )
        raise ValueError(
            "cannot join because this node is currently the workload controller for "
            + ", ".join(details) + ". Keep this node as the controller and join the "
            "other SparkDeck nodes to this node. If you intentionally want another "
            "controller, migrate these deployment records first; model weights and the "
            "Hugging Face cache do not need to be deleted."
        )

    async def _assert_joinable(self) -> None:
        await self._assert_no_owned_workloads("join")
        registered = getattr(self.manager.node_registry, "nodes", [])
        registered = registered if isinstance(registered, list) else []
        children = [
            node for node in registered
            if isinstance(node, dict) and str(node.get("id") or "").strip()
        ]
        if children:
            names = ", ".join(
                str(node.get("name") or node.get("id")) for node in children[:5]
            )
            suffix = "" if len(children) <= 5 else f" and {len(children) - 5} more"
            raise ValueError(
                f"cannot join while this node still controls {len(children)} joined "
                f"node{'s' if len(children) != 1 else ''}: {names}{suffix}. Joining "
                "moves only this machine and never transfers its former cluster members. "
                "On each member, use Leave cluster and then join it directly to the "
                "destination controller; remove stale member records before retrying."
            )

    async def register(
        self, body: dict[str, Any], request_origin: str, client_id: str,
    ) -> dict[str, Any]:
        async with self._membership_lock:
            return await self._register(body, request_origin, client_id)

    async def _register(
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
        async with self._membership_lock:
            return await self._join(body, request_origin)

    async def _join(self, body: dict[str, Any], request_origin: str) -> dict[str, Any]:
        if self.assignment.load():
            raise ValueError("this node is already joined to a controller")
        await self._assert_joinable()
        entry_connection = await resolve_control_connection(body.get("controller_url"))
        entry_url = entry_connection.url
        advertise_url = await validate_control_url(body.get("advertise_url"))
        if entry_url == advertise_url:
            raise ValueError("controller_url and advertise_url must identify different nodes")
        try:
            identity_response = await _send_pinned_control_request(
                self.manager.http,
                entry_connection,
                "GET",
                "/api/v1/onboarding",
                timeout=5,
            )
            identity_response.raise_for_status()
            identity = identity_response.json()
        except httpx.HTTPError as exc:
            raise ValueError(f"controller is unreachable: {exc}") from exc
        entry_node_id = str(identity.get("node", {}).get("id") or "")
        if not entry_node_id:
            raise ValueError("cluster entry identity is missing a node ID")
        if entry_node_id == self.manager.agent_credentials.node_id:
            raise ValueError(
                "controller_url resolves to this node; enter another controller's "
                "Tailscale URL"
            )
        if int(identity.get("node", {}).get("protocol_version") or 0) != AGENT_PROTOCOL_VERSION:
            raise ValueError("cluster entry protocol version mismatch")
        if identity.get("role") == "controller":
            controller_url = entry_url
            controller_connection = entry_connection
            controller_node_id = entry_node_id
        elif identity.get("role") == "worker":
            if not identity.get("controller_reachable"):
                raise ValueError("worker entry point cannot reach its controller")
            controller_connection = await resolve_control_connection(
                identity.get("controller_url")
            )
            controller_url = controller_connection.url
            controller_node_id = str(identity.get("controller_node_id") or "")
            if not controller_node_id:
                raise ValueError("worker entry point is missing its controller identity")
            if controller_url == advertise_url:
                raise ValueError("worker entry point refers back to this joining node")
            controller_host = (urlsplit(controller_url).hostname or "").casefold()
            try:
                controller_is_loopback = ipaddress.ip_address(controller_host).is_loopback
            except ValueError:
                controller_is_loopback = controller_host == "localhost"
            if controller_is_loopback:
                raise ValueError("worker entry point advertised a loopback controller URL")
            try:
                controller_response = await _send_pinned_control_request(
                    self.manager.http,
                    controller_connection,
                    "GET",
                    "/api/v1/onboarding",
                    timeout=5,
                )
                controller_response.raise_for_status()
                controller_identity = controller_response.json()
            except httpx.HTTPError as exc:
                raise ValueError(f"worker's controller is unreachable: {exc}") from exc
            actual_controller_node_id = str(
                controller_identity.get("node", {}).get("id") or ""
            )
            if (
                controller_identity.get("role") != "controller"
                or int(controller_identity.get("node", {}).get("protocol_version") or 0)
                != AGENT_PROTOCOL_VERSION
                or not actual_controller_node_id
                or not secrets.compare_digest(
                    actual_controller_node_id, controller_node_id,
                )
            ):
                raise ValueError("worker entry point returned an invalid controller referral")
        else:
            raise ValueError("controller_url is not a SparkDeck cluster entry point")

        response = await _send_pinned_control_request(
            self.manager.http,
            controller_connection,
            "POST",
            "/api/v1/onboarding/register",
            json_body={
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
        assignment = self.assignment.load()
        if not assignment:
            result = await self.status(request_origin)
            result["ok"] = True
            return result

        # A worker must remain paired while it hosts any managed member. This
        # local check protects offline-controller recovery; unregister below
        # performs the controller's authoritative deployment check.
        await self._assert_no_owned_workloads("leave")

        try:
            connection = await resolve_control_connection(assignment["controller_url"])
            response = await _send_pinned_control_request(
                self.manager.http,
                connection,
                "POST",
                "/api/v1/onboarding/unregister",
                headers={
                    FORWARD_NODE_HEADER: assignment["node_id"],
                    FORWARD_HOP_HEADER: "1",
                    FORWARD_TOKEN_HEADER: assignment["forward_token"],
                },
                timeout=5,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = str(exc.response.json().get("detail") or "")
            except (ValueError, AttributeError):
                detail = exc.response.text[:300]
            if exc.response.status_code == 404 and detail == "worker is not registered":
                pass
            elif exc.response.status_code == 409:
                raise ValueError(
                    "cannot leave while the controller still has a deployment "
                    f"on this node: {detail or 'remove the deployment first'}"
                ) from exc
            else:
                raise ValueError(
                    "controller could not authorize this node to leave; "
                    "restore controller access and retry"
                ) from exc
        except httpx.RequestError:
            # An offline controller cannot acknowledge removal. The local
            # workload check and durable rotation still allow safe recovery.
            pass
        except (ValueError, KeyError):
            # A no-longer-resolvable saved URL is equivalent to an offline
            # controller. Local revocation remains authoritative on this node.
            pass

        # Revoke the credential the former controller uses to reach this
        # worker before deleting the only durable record of that controller.
        # If the atomic credential write fails, leave controller.json intact
        # so the user can retry and the process cannot report a false leave.
        self.manager.agent_credentials.revoke_remote_access()
        self.assignment.clear()
        adopt_controller = getattr(self.manager, "adopt_controller_role", None)
        if adopt_controller is not None:
            adopt_controller()
        result = await self.status(request_origin)
        result["ok"] = True
        return result

    async def detach(self) -> dict[str, Any]:
        """Honor an authenticated controller request to remove this worker."""
        await self._assert_no_owned_workloads("leave")
        self.manager.agent_credentials.revoke_remote_access()
        self.assignment.clear()
        adopt_controller = getattr(self.manager, "adopt_controller_role", None)
        if adopt_controller is not None:
            adopt_controller()
        return {"ok": True, "role": "controller", "revoked": True}

    def unregister(self, headers: Any) -> dict[str, Any]:
        """Revoke a leaving worker's one-hop grant and inventory record."""
        node_id = str(headers.get(FORWARD_NODE_HEADER) or "")
        if self.is_already_unregistered_worker(headers):
            # Force-forget deliberately deletes the worker and its token hash.
            # Report that state before token validation so the forgotten worker
            # can treat its later explicit leave as already unregistered.
            raise ValueError("worker is not registered")
        valid, detail = self.validate_forward_headers(headers)
        if not valid:
            raise PermissionError(detail)
        if not self.manager.remove_cluster_node(node_id):
            raise ValueError("worker is not registered")
        return {"ok": True, "node_id": node_id, "revoked": True}

    def is_already_unregistered_worker(self, headers: Any) -> bool:
        """Recognize only a one-hop claim for a registry record already absent."""
        hop = str(headers.get(FORWARD_HOP_HEADER) or "")
        node_id = str(headers.get(FORWARD_NODE_HEADER) or "")
        return bool(
            hop == "1"
            and node_id
            and self.manager.node_registry.get(node_id) is None
        )

    def validate_forward_headers(self, headers: Any) -> tuple[bool, str]:
        hop = str(headers.get(FORWARD_HOP_HEADER) or "")
        node_id = str(headers.get(FORWARD_NODE_HEADER) or "")
        token = str(headers.get(FORWARD_TOKEN_HEADER) or "")
        scheme = str(headers.get(FORWARD_SCHEME_HEADER) or "")
        if hop != "1":
            return False, "forward hop must be exactly 1"
        if scheme and scheme not in {"http", "https"}:
            return False, "forward scheme must be http or https"
        if not self.manager.node_registry.accepts_forward_token(node_id, token):
            return False, "invalid worker forwarding credential"
        return True, ""


async def forward_management_request(
    request: Request, manager: Any, assignment: dict[str, Any],
):
    if any(request.headers.get(name) for name in FORWARD_HEADERS):
        return JSONResponse({"detail": "forwarded request chains are not allowed"}, status_code=508)
    try:
        # Revalidate the saved destination on every forwarded request. A
        # hostname that was safe during onboarding must not be able to rebind
        # before a write-only cluster credential is forwarded.
        connection = await resolve_control_connection(assignment["controller_url"])
        controller_url = connection.url
    except ValueError as exc:
        return JSONResponse(
            {"detail": f"controller URL is no longer safe: {exc}"},
            status_code=502,
        )
    request_origin = normalize_control_url(str(request.base_url))
    if controller_url == request_origin:
        return JSONResponse({"detail": "controller URL points back to this worker"}, status_code=508)
    headers = {
        key: value for key, value in request.headers.items()
        if key.casefold() not in FORWARD_HEADERS | {"host", "content-length", "connection"}
    }
    headers.update({
        FORWARD_NODE_HEADER: assignment["node_id"],
        FORWARD_HOP_HEADER: "1",
        FORWARD_TOKEN_HEADER: assignment["forward_token"],
        FORWARD_SCHEME_HEADER: request.url.scheme,
    })
    try:
        body = await request.body()

        async def wait_for_disconnect() -> bool:
            try:
                while not await request.is_disconnected():
                    await asyncio.sleep(PROXY_DISCONNECT_POLL_SECONDS)
                return True
            except Exception:
                # A broken disconnect probe must not abort an otherwise valid
                # controller request. The configured proxy timeout still bounds it.
                return False

        disconnect_task = asyncio.create_task(wait_for_disconnect())
        try:
            upstream = await _send_pinned_control_request(
                manager.http,
                connection,
                request.method,
                request.url.path,
                query=request.url.query,
                headers=headers,
                content=body,
                timeout=PROXY_TIMEOUT_SECONDS,
                stream=True,
                disconnect_task=disconnect_task,
            )
        except _ControlClientDisconnected:
            return JSONResponse(
                {"detail": "client disconnected"}, status_code=499,
            )
        finally:
            disconnect_task.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect_task
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
            "content-disposition", "vary", "set-cookie",
        }
    }
    return StreamingResponse(
        content(), status_code=upstream.status_code, headers=response_headers,
    )
