"""RouterOS discovery, telemetry, and narrowly-scoped fan configuration."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx

from .private_json import atomic_private_json_write as _atomic_private_json


MNDP_PORT = 5678
MNDP_TTL_SECONDS = 150.0
MNDP_MAX_DISCOVERED_DEVICES = 256
MNDP_MAX_FIELD_LENGTH = 256
ROUTEROS_TIMEOUT_SECONDS = 5.0
_TAILSCALE_V4 = ipaddress.ip_network("100.64.0.0/10")
_TAILSCALE_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")

_MNDP_FIELDS = {
    0x0001: "mac",
    0x0005: "identity",
    0x0007: "version",
    0x0008: "platform",
    0x000B: "software_id",
    0x000C: "board",
    0x000F: "ipv6",
    0x0010: "interface",
    0x0011: "ipv4",
}

FAN_SETTING_RANGES = {
    "fan-full-speed-temp": (-273, 65),
    "fan-target-temp": (-273, 65),
    "fan-min-speed-percent": (0, 100),
    "fan-control-interval": (5, 30),
    "cpu-overtemp-threshold": (0, 105),
}
FAN_SETTING_KEYS = frozenset({
    *FAN_SETTING_RANGES,
    "cpu-overtemp-check",
    "cpu-overtemp-startup-delay",
})
_ROUTEROS_TIME = re.compile(
    r"^(?:\d{1,3}:\d{2}:\d{2}|(?:\d+(?:\.\d+)?(?:ms|[smhdw]))+)$",
    re.IGNORECASE,
)

_INTERFACE_FIELDS = frozenset({
    ".id", "name", "default-name", "type", "running", "disabled",
    "actual-mtu", "l2mtu", "mac-address", "last-link-up-time",
    "last-link-down-time", "link-downs", "rx-byte", "tx-byte",
    "rx-packet", "tx-packet", "rx-drop", "tx-drop", "tx-queue-drop",
    "rx-error", "tx-error", "fp-rx-byte", "fp-tx-byte", "fp-rx-packet",
    "fp-tx-packet",
})
_TRAFFIC_FIELDS = frozenset({
    "rx-bits-per-second", "tx-bits-per-second",
    "rx-packets-per-second", "tx-packets-per-second",
    "rx-drops-per-second", "tx-drops-per-second",
    "rx-errors-per-second", "tx-errors-per-second",
    "fp-rx-bits-per-second", "fp-tx-bits-per-second",
    "fp-rx-packets-per-second", "fp-tx-packets-per-second",
    "tx-queue-drops-per-second",
})

_MNDP_PUBLIC_FIELDS = frozenset({*_MNDP_FIELDS.values(), "address"})
_MNDP_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


def _format_mac(value: bytes) -> str:
    return ":".join(f"{byte:02X}" for byte in value)


def parse_mndp_packet(data: bytes, source_address: str = "") -> dict[str, Any] | None:
    """Decode the stable MNDP TLVs SparkDeck needs for RouterOS discovery."""
    if len(data) < 8:
        return None
    offset = 4
    result: dict[str, Any] = {}
    while offset + 4 <= len(data):
        field_type = int.from_bytes(data[offset:offset + 2], "big")
        length = int.from_bytes(data[offset + 2:offset + 4], "big")
        offset += 4
        if length < 0 or offset + length > len(data):
            return None
        raw = data[offset:offset + length]
        offset += length
        name = _MNDP_FIELDS.get(field_type)
        if not name:
            continue
        try:
            if name == "mac" and len(raw) == 6:
                result[name] = _format_mac(raw)
            elif name == "ipv4" and len(raw) == 4:
                result[name] = str(ipaddress.ip_address(raw))
            elif name == "ipv6" and len(raw) == 16:
                result[name] = str(ipaddress.ip_address(raw))
            else:
                result[name] = raw.decode("utf-8", errors="replace").strip("\x00")
        except ValueError:
            continue
    if str(result.get("platform") or "").casefold() != "mikrotik":
        return None
    if not (result.get("version") or result.get("software_id") or result.get("board")):
        return None
    address = result.get("ipv4") or source_address
    if address:
        result["address"] = address
    return result


def normalize_routeros_url(value: Any) -> str:
    text = str(value or "").strip().rstrip("/")
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("RouterOS URL must be an http(s) URL with a hostname")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("RouterOS URL cannot contain credentials, query parameters, or fragments")
    path = parsed.path.rstrip("/")
    if path not in {"", "/rest"}:
        raise ValueError("RouterOS URL must point to the device root or /rest")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _allowed_routeros_ip(value: str) -> bool:
    address = ipaddress.ip_address(value.split("%", 1)[0])
    return bool(
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address in _TAILSCALE_V4
        or address in _TAILSCALE_V6
    ) and not (address.is_multicast or address.is_unspecified)


async def validate_routeros_url(value: Any) -> str:
    """Restrict the management target to cluster-private network addresses."""
    return (await _resolve_routeros_connection(value)).url


@dataclass(frozen=True)
class _RouterOSConnection:
    url: str
    connect_urls: tuple[str, ...]
    host_header: str
    sni_hostname: str


async def _resolve_routeros_connection(value: Any) -> _RouterOSConnection:
    """Resolve once and pin every permitted RouterOS management address."""
    url = normalize_routeros_url(value)
    parsed = httpx.URL(url)
    host = parsed.raw_host.decode("ascii")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    literal: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    try:
        literal = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError as literal_error:
        try:
            rows = await asyncio.to_thread(
                socket.getaddrinfo, host, port, type=socket.SOCK_STREAM,
            )
        except OSError as exc:
            raise ValueError("RouterOS hostname could not be resolved") from exc
        addresses = []
        for row in rows:
            address = str(row[4][0])
            if (
                row[0] == socket.AF_INET6
                and "%" not in address
                and len(row[4]) >= 4
                and row[4][3]
            ):
                address = f"{address}%{row[4][3]}"
            if address not in addresses:
                addresses.append(address)
        if not addresses or not all(_allowed_routeros_ip(address) for address in addresses):
            raise ValueError(
                "RouterOS URL must resolve only to private network addresses"
            ) from literal_error
    else:
        if not _allowed_routeros_ip(str(literal)):
            raise ValueError(
                "RouterOS URL must resolve only to private network addresses"
            )
        # Keep a link-local literal's scope identifier. RFC 6874 encodes the
        # separating percent sign as %25 in a URL, while the socket layer needs
        # the decoded ``address%zone`` form.
        addresses = [unquote(host)]
    return _RouterOSConnection(
        url=url,
        connect_urls=tuple(
            str(parsed.copy_with(host=address)) for address in addresses
        ),
        host_header=parsed.netloc.decode("ascii"),
        # A zone is routing metadata, not part of a TLS server identity.
        sni_hostname=str(literal) if literal is not None else host,
    )


def _single(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        return [value]
    return []


class _MNDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, service: "RouterOSService") -> None:
        self.service = service

    def datagram_received(self, data: bytes, address: tuple[str, int]) -> None:
        candidate = parse_mndp_packet(data, address[0])
        if candidate:
            self.service.record_discovery(candidate)

    def error_received(self, exc: Exception) -> None:
        self.service.discovery_error = str(exc)


class RouterOSService:
    """One RouterOS management connection and passive discovery per node."""

    def __init__(self, data_dir: Path, *, transport: httpx.AsyncBaseTransport | None = None):
        self.config_path = Path(data_dir) / "routeros_switch.json"
        self._transport = transport
        self._discovered: dict[str, dict[str, Any]] = {}
        self._discovery_transport: asyncio.DatagramTransport | None = None
        self.discovery_error: str | None = None

    def _load_config(self) -> dict[str, Any] | None:
        if not self.config_path.exists():
            return None
        try:
            value = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict) or not all(
            value.get(key) for key in ("base_url", "username")
        ):
            return None
        return value

    async def start(self) -> None:
        if self._discovery_transport is not None:
            return
        try:
            loop = asyncio.get_running_loop()
            transport, _ = await loop.create_datagram_endpoint(
                lambda: _MNDPProtocol(self),
                local_addr=("0.0.0.0", MNDP_PORT),
                allow_broadcast=True,
            )
            self._discovery_transport = transport
            self.discovery_error = None
        except (OSError, RuntimeError) as exc:
            # Manual configuration remains available when another MNDP listener
            # owns the port or the runtime does not expose datagram sockets.
            self.discovery_error = str(exc)

    async def stop(self) -> None:
        if self._discovery_transport is not None:
            self._discovery_transport.close()
            self._discovery_transport = None

    def record_discovery(self, candidate: dict[str, Any]) -> None:
        now = time.time()
        self._prune_discovery(now)
        sanitized: dict[str, str] = {}
        for field in _MNDP_PUBLIC_FIELDS:
            if field not in candidate:
                continue
            value = _MNDP_CONTROL_CHARACTERS.sub("", str(candidate[field])).strip()
            if not value:
                continue
            value = value[:MNDP_MAX_FIELD_LENGTH]
            if field in {"address", "ipv4", "ipv6"}:
                try:
                    ipaddress.ip_address(value.split("%", 1)[0])
                except ValueError:
                    continue
            sanitized[field] = value
        if sanitized.get("platform", "").casefold() != "mikrotik":
            return
        if not any(sanitized.get(field) for field in ("version", "software_id", "board")):
            return
        key = str(
            sanitized.get("mac") or sanitized.get("address")
            or sanitized.get("identity") or ""
        ).casefold()
        if not key:
            return
        if key not in self._discovered and len(self._discovered) >= MNDP_MAX_DISCOVERED_DEVICES:
            oldest = min(
                self._discovered,
                key=lambda item: float(self._discovered[item].get("last_seen") or 0),
            )
            self._discovered.pop(oldest, None)
        self._discovered[key] = {**sanitized, "last_seen": now}

    def _prune_discovery(self, now: float | None = None) -> None:
        cutoff = (time.time() if now is None else now) - MNDP_TTL_SECONDS
        self._discovered = {
            key: value for key, value in self._discovered.items()
            if float(value.get("last_seen") or 0) >= cutoff
        }

    def _active_discovery(self) -> list[dict[str, Any]]:
        self._prune_discovery()
        return sorted(
            (dict(value) for value in self._discovered.values()),
            key=lambda item: str(item.get("identity") or item.get("address") or ""),
        )

    def presence(self) -> dict[str, Any]:
        config = self._load_config()
        discovery = self._active_discovery()
        return {
            "detected": bool(config or discovery),
            "configured": bool(config),
            "discovery": discovery,
            "discovery_error": self.discovery_error,
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        config: dict[str, Any] | None = None,
        json_body: dict[str, str] | None = None,
    ) -> Any:
        selected = config or self._load_config()
        if not selected:
            raise RuntimeError("RouterOS switch is not configured")
        try:
            return await asyncio.wait_for(
                self._request_within_deadline(
                    selected, method, path, json_body,
                ),
                timeout=ROUTEROS_TIMEOUT_SECONDS,
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {401, 403}:
                raise RuntimeError("RouterOS rejected the configured credentials") from exc
            detail = ""
            try:
                body = exc.response.json()
                detail = str(body.get("detail") or body.get("message") or "")[:200]
            except (ValueError, AttributeError):
                pass
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"RouterOS returned HTTP {status}{suffix}") from exc
        except (
            httpx.HTTPError, json.JSONDecodeError, ValueError, asyncio.TimeoutError,
        ) as exc:
            raise RuntimeError("Could not communicate with the RouterOS REST API") from exc

    async def _request_within_deadline(
        self,
        selected: dict[str, Any],
        method: str,
        path: str,
        json_body: dict[str, str] | None,
    ) -> Any:
        connection = await _resolve_routeros_connection(selected["base_url"])
        async with httpx.AsyncClient(
            auth=(str(selected["username"]), str(selected.get("password") or "")),
            timeout=ROUTEROS_TIMEOUT_SECONDS,
            verify=bool(selected.get("verify_tls", True)),
            transport=self._transport,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            last_connect_error: httpx.HTTPError | None = None
            for connect_url in connection.connect_urls:
                try:
                    response = await client.request(
                        method,
                        f"{connect_url}/rest/{path.lstrip('/')}",
                        json=json_body,
                        headers={
                            "Accept": "application/json",
                            "Host": connection.host_header,
                        },
                        extensions={"sni_hostname": connection.sni_hostname},
                        follow_redirects=False,
                    )
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    last_connect_error = exc
                    continue
                break
            else:
                assert last_connect_error is not None
                raise last_connect_error
            response.raise_for_status()
            if not response.content:
                return []
            return response.json()

    async def connect(self, body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise ValueError("connection settings must be an object")
        allowed = {"base_url", "username", "password", "verify_tls"}
        unknown = set(body) - allowed
        if unknown:
            raise ValueError(f"unknown connection setting(s): {', '.join(sorted(unknown))}")
        base_url = await validate_routeros_url(body.get("base_url"))
        username = str(body.get("username") or "").strip()
        if not username or len(username) > 64:
            raise ValueError("RouterOS username must be between 1 and 64 characters")
        current = self._load_config() or {}
        password = str(body.get("password") or "")
        if not password and current.get("base_url") == base_url and current.get("username") == username:
            password = str(current.get("password") or "")
        if len(password) > 512:
            raise ValueError("RouterOS password is too long")
        verify_tls = body.get("verify_tls", True)
        if not isinstance(verify_tls, bool):
            raise ValueError("verify_tls must be a boolean")
        config = {
            "base_url": base_url,
            "username": username,
            "password": password,
            "verify_tls": verify_tls,
        }
        resource = _single(await self._request("GET", "system/resource", config=config))
        if str(resource.get("platform") or "").casefold() != "mikrotik":
            raise ValueError("The configured endpoint did not identify as MikroTik RouterOS")
        if not (resource.get("version") and resource.get("board-name")):
            raise ValueError("The configured endpoint did not return RouterOS device details")
        _atomic_private_json(self.config_path, config)
        return await self.overview()

    def disconnect(self) -> dict[str, Any]:
        try:
            self.config_path.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError("Could not remove the RouterOS connection") from exc
        return {**self.presence(), "connected": False}

    async def overview(self) -> dict[str, Any]:
        presence = self.presence()
        config = self._load_config()
        if not config:
            return {**presence, "connected": False, "health": [], "interfaces": []}
        try:
            resource = _single(await self._request("GET", "system/resource", config=config))
            if str(resource.get("platform") or "").casefold() != "mikrotik":
                raise RuntimeError("Configured endpoint is not a MikroTik RouterOS device")

            async def optional(method: str, path: str, body: dict[str, str] | None = None) -> Any:
                try:
                    return await self._request(method, path, config=config, json_body=body)
                except RuntimeError:
                    return None

            identity, cpus, health, fan_settings, interfaces = await asyncio.gather(
                optional("GET", "system/identity"),
                optional("GET", "system/resource/cpu"),
                optional("GET", "system/health"),
                optional("GET", "system/health/settings"),
                optional("POST", "interface/print", {"stats-detail": ""}),
            )
            identity_row = _single(identity)
            device = dict(resource)
            if identity_row.get("name"):
                device["identity"] = identity_row["name"]
            interface_rows = [
                {key: value for key, value in row.items() if key in _INTERFACE_FIELDS}
                for row in _rows(interfaces)
            ]
            interface_names = [
                str(row.get("name") or "") for row in interface_rows if row.get("name")
            ]
            traffic = None
            if interface_names:
                traffic = await optional(
                    "POST", "interface/monitor-traffic",
                    {"interface": ",".join(interface_names), "once": ""},
                )
            traffic_by_name = {
                str(row.get("name")): {
                    key: value for key, value in row.items() if key in _TRAFFIC_FIELDS
                }
                for row in _rows(traffic) if row.get("name")
            }
            interface_rows = [
                {**row, **traffic_by_name.get(str(row.get("name") or ""), {})}
                for row in interface_rows
            ]
            settings = _single(fan_settings)
            return {
                **presence,
                "detected": True,
                "configured": True,
                "connected": True,
                "base_url": config["base_url"],
                "verify_tls": bool(config.get("verify_tls", True)),
                "device": device,
                "cpus": _rows(cpus),
                "health": _rows(health),
                "fan_settings": settings or None,
                "fan_capabilities": sorted(FAN_SETTING_KEYS.intersection(settings)),
                "interfaces": interface_rows,
            }
        except RuntimeError as exc:
            return {
                **presence,
                "detected": True,
                "configured": True,
                "connected": False,
                "base_url": config["base_url"],
                "verify_tls": bool(config.get("verify_tls", True)),
                "health": [],
                "interfaces": [],
                "error": str(exc),
            }

    @staticmethod
    def validate_fan_settings(body: Any, supported: set[str] | None = None) -> dict[str, str]:
        if not isinstance(body, dict) or not body:
            raise ValueError("fan settings must be a non-empty object")
        unknown = set(body) - FAN_SETTING_KEYS
        if unknown:
            raise ValueError(f"unsupported fan setting(s): {', '.join(sorted(unknown))}")
        if supported is not None:
            unavailable = set(body) - supported
            if unavailable:
                raise ValueError(
                    f"fan setting(s) unavailable on this device: {', '.join(sorted(unavailable))}"
                )
        result: dict[str, str] = {}
        for key, value in body.items():
            if key == "cpu-overtemp-check":
                if not isinstance(value, bool):
                    raise ValueError("cpu-overtemp-check must be a boolean")
                result[key] = "yes" if value else "no"
            elif key == "cpu-overtemp-startup-delay":
                text = str(value or "").strip()
                if not _ROUTEROS_TIME.fullmatch(text):
                    raise ValueError("cpu-overtemp-startup-delay must be a RouterOS time value")
                result[key] = text
            else:
                if isinstance(value, bool):
                    raise ValueError(f"{key} must be an integer")
                try:
                    parsed = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{key} must be an integer") from exc
                if str(parsed) != str(value).strip():
                    raise ValueError(f"{key} must be an integer")
                minimum, maximum = FAN_SETTING_RANGES[key]
                if parsed < minimum or parsed > maximum:
                    raise ValueError(f"{key} must be between {minimum} and {maximum}")
                result[key] = str(parsed)
        return result

    async def update_fan_settings(self, body: Any) -> dict[str, Any]:
        current = await self.overview()
        if not current.get("connected"):
            raise RuntimeError(str(current.get("error") or "RouterOS switch is not connected"))
        supported = set(current.get("fan_capabilities") or [])
        if not supported:
            raise ValueError("manual fan control is unavailable on this RouterOS device")
        validated = self.validate_fan_settings(body, supported)
        await self._request("POST", "system/health/settings/set", json_body=validated)
        return await self.overview()
