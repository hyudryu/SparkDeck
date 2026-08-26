import json
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import httpx

from manager import Manager, ROUTEROS_FAN_UPDATE_TIMEOUT_SECONDS
from sparkdeck.routeros import (
    ROUTEROS_TIMEOUT_SECONDS,
    RouterOSService,
    parse_mndp_packet,
)


def _tlv(kind: int, value: bytes) -> bytes:
    return kind.to_bytes(2, "big") + len(value).to_bytes(2, "big") + value


class RouterOSDiscoveryTests(unittest.TestCase):
    def test_mndp_parser_returns_only_routeros_identity_fields(self) -> None:
        packet = b"\x00\x00\x00\x01" + b"".join((
            _tlv(0x0001, bytes.fromhex("D4CA6D010203")),
            _tlv(0x0005, b"rack-switch"),
            _tlv(0x0007, b"7.19.1"),
            _tlv(0x0008, b"MikroTik"),
            _tlv(0x000C, b"CRS326-24S+2Q+"),
            _tlv(0x0011, bytes((192, 168, 88, 1))),
        ))

        result = parse_mndp_packet(packet, "10.0.0.1")

        self.assertEqual(result["identity"], "rack-switch")
        self.assertEqual(result["address"], "192.168.88.1")
        self.assertEqual(result["mac"], "D4:CA:6D:01:02:03")

    def test_mndp_parser_rejects_other_platforms_and_truncated_tlvs(self) -> None:
        other = b"\0\0\0\1" + _tlv(0x0008, b"Cisco") + _tlv(0x0007, b"1")
        malformed = b"\0\0\0\1\0\x08\0\x20short"
        self.assertIsNone(parse_mndp_packet(other))
        self.assertIsNone(parse_mndp_packet(malformed))

    def test_presence_expires_old_discovery_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = RouterOSService(Path(directory))
            service.record_discovery({
                "platform": "MikroTik", "version": "7.19", "address": "10.0.0.2",
            })
            self.assertTrue(service.presence()["detected"])
            service._discovered["10.0.0.2"]["last_seen"] = time.time() - 1000
            self.assertFalse(service.presence()["detected"])


class RouterOSServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            self.assertTrue(request.headers.get("authorization", "").startswith("Basic "))
            path = request.url.path
            if path == "/rest/system/resource":
                return httpx.Response(200, json=[{
                    "platform": "MikroTik", "version": "7.19.1",
                    "board-name": "CRS326-24S+2Q+", "cpu-load": "12",
                    "total-memory": "1073741824", "free-memory": "536870912",
                }])
            if path == "/rest/system/identity":
                return httpx.Response(200, json=[{"name": "rack-switch"}])
            if path == "/rest/system/resource/cpu":
                return httpx.Response(200, json=[{"cpu": "0", "load": "12"}])
            if path == "/rest/system/health":
                return httpx.Response(200, json=[
                    {"name": "cpu-temperature", "value": "43", "type": "C"},
                    {"name": "fan1-speed", "value": "5654", "type": "RPM"},
                ])
            if path == "/rest/system/health/settings":
                return httpx.Response(200, json=[{
                    "fan-target-temp": "58", "fan-full-speed-temp": "65",
                    "fan-min-speed-percent": "12", "fan-control-interval": "30",
                    "cpu-overtemp-check": "no", "cpu-overtemp-threshold": "105",
                    "cpu-overtemp-startup-delay": "1m",
                }])
            if path == "/rest/interface/print":
                return httpx.Response(200, json=[{
                    ".id": "*1", "name": "sfp-sfpplus1", "running": "true",
                    "rx-byte": "1200", "tx-byte": "3400", "comment": "private note",
                }])
            if path == "/rest/interface/monitor-traffic":
                return httpx.Response(200, json=[{
                    "name": "sfp-sfpplus1", "rx-bits-per-second": "9800000000",
                    "tx-bits-per-second": "8600000000", ".section": "0",
                }])
            if path == "/rest/system/health/settings/set":
                return httpx.Response(200, json=[])
            return httpx.Response(404, json={"message": "not found"})

        self.service = RouterOSService(
            Path(self.directory.name), transport=httpx.MockTransport(handler),
        )

    async def asyncTearDown(self) -> None:
        self.directory.cleanup()

    async def test_connect_confirms_routeros_and_never_returns_credentials(self) -> None:
        result = await self.service.connect({
            "base_url": "https://192.168.88.1/rest/",
            "username": "sparkdeck",
            "password": "very-secret",
            "verify_tls": True,
        })

        self.assertTrue(result["connected"])
        self.assertEqual(result["device"]["identity"], "rack-switch")
        self.assertEqual(result["health"][1]["type"], "RPM")
        self.assertNotIn("comment", result["interfaces"][0])
        self.assertNotIn(".section", result["interfaces"][0])
        self.assertEqual(result["interfaces"][0]["rx-bits-per-second"], "9800000000")
        serialized = json.dumps(result)
        self.assertNotIn("very-secret", serialized)
        self.assertNotIn("sparkdeck", serialized)
        stored = json.loads(self.service.config_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["password"], "very-secret")

    async def test_fan_update_uses_only_documented_settings_command(self) -> None:
        await self.service.connect({
            "base_url": "https://192.168.88.1",
            "username": "sparkdeck", "password": "secret", "verify_tls": True,
        })
        self.requests.clear()

        result = await self.service.update_fan_settings({
            "fan-target-temp": 55,
            "fan-min-speed-percent": 20,
            "cpu-overtemp-check": True,
        })

        write = next(
            request for request in self.requests
            if request.url.path == "/rest/system/health/settings/set"
        )
        self.assertEqual(write.method, "POST")
        self.assertEqual(json.loads(write.content), {
            "fan-target-temp": "55",
            "fan-min-speed-percent": "20",
            "cpu-overtemp-check": "yes",
        })
        self.assertTrue(result["connected"])

    async def test_connection_rejects_non_routeros_endpoint_without_persisting(self) -> None:
        service = RouterOSService(
            Path(self.directory.name),
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json=[{
                    "platform": "Other", "version": "7", "board-name": "switch",
                }]),
            ),
        )
        with self.assertRaisesRegex(ValueError, "did not identify"):
            await service.connect({
                "base_url": "https://10.0.0.2", "username": "user",
                "password": "secret", "verify_tls": True,
            })
        self.assertFalse(service.config_path.exists())

    async def test_authenticated_request_pins_validated_hostname_resolution(self) -> None:
        requests = []
        resolutions = []

        def resolve(host, port, **kwargs):
            resolutions.append((host, port))
            address = "192.168.88.10" if len(resolutions) == 1 else "8.8.8.8"
            return [(
                socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                (address, port),
            )]

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=[{"platform": "MikroTik"}], request=request)

        service = RouterOSService(
            Path(self.directory.name), transport=httpx.MockTransport(handler),
        )
        config = {
            "base_url": "https://switch.private.example:8443",
            "username": "sparkdeck", "password": "secret", "verify_tls": True,
        }
        with mock.patch(
            "sparkdeck.routeros.socket.getaddrinfo", side_effect=resolve,
        ):
            result = await service._request("GET", "system/resource", config=config)

        self.assertEqual(result, [{"platform": "MikroTik"}])
        self.assertEqual(resolutions, [("switch.private.example", 8443)])
        self.assertEqual(
            str(requests[0].url),
            "https://192.168.88.10:8443/rest/system/resource",
        )
        self.assertEqual(requests[0].headers["host"], "switch.private.example:8443")
        self.assertEqual(
            requests[0].extensions["sni_hostname"], "switch.private.example",
        )
        self.assertTrue(requests[0].headers["authorization"].startswith("Basic "))

    async def test_each_request_rejects_hostname_that_rebinds_public(self) -> None:
        requests = []
        resolution_count = 0

        def resolve(host, port, **kwargs):
            nonlocal resolution_count
            resolution_count += 1
            address = "192.168.88.10" if resolution_count == 1 else "8.8.8.8"
            return [(
                socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                (address, port),
            )]

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=[], request=request)

        service = RouterOSService(
            Path(self.directory.name), transport=httpx.MockTransport(handler),
        )
        config = {
            "base_url": "http://switch.private.example:8080",
            "username": "sparkdeck", "password": "secret", "verify_tls": True,
        }
        with mock.patch(
            "sparkdeck.routeros.socket.getaddrinfo", side_effect=resolve,
        ):
            await service._request("GET", "system/resource", config=config)
            with self.assertRaisesRegex(RuntimeError, "Could not communicate"):
                await service._request("GET", "system/resource", config=config)

        self.assertEqual(resolution_count, 2)
        self.assertEqual(len(requests), 1)

    async def test_request_tries_each_pinned_private_address_on_connect_failure(self) -> None:
        requests = []
        resolution_count = 0

        def resolve(host, port, **kwargs):
            nonlocal resolution_count
            resolution_count += 1
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.10", port)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.11", port)),
            ]

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if len(requests) == 1:
                raise httpx.ConnectError("first address unavailable", request=request)
            return httpx.Response(200, json=[], request=request)

        service = RouterOSService(
            Path(self.directory.name), transport=httpx.MockTransport(handler),
        )
        config = {
            "base_url": "https://switch.private.example",
            "username": "sparkdeck", "password": "secret", "verify_tls": True,
        }
        with mock.patch(
            "sparkdeck.routeros.socket.getaddrinfo", side_effect=resolve,
        ):
            result = await service._request("POST", "system/health", config=config)

        self.assertEqual(result, [])
        self.assertEqual(resolution_count, 1)
        self.assertEqual([str(request.url) for request in requests], [
            "https://10.0.0.10/rest/system/health",
            "https://10.0.0.11/rest/system/health",
        ])
        self.assertTrue(all(
            request.headers["host"] == "switch.private.example"
            and request.extensions["sni_hostname"] == "switch.private.example"
            and request.headers["authorization"].startswith("Basic ")
            for request in requests
        ))

    async def test_authenticated_request_does_not_follow_redirects(self) -> None:
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                307,
                headers={"location": "https://attacker.example/collect"},
                request=request,
            )

        service = RouterOSService(
            Path(self.directory.name), transport=httpx.MockTransport(handler),
        )
        with self.assertRaisesRegex(RuntimeError, "HTTP 307"):
            await service._request("GET", "system/resource", config={
                "base_url": "http://10.0.0.10", "username": "sparkdeck",
                "password": "secret", "verify_tls": True,
            })

        self.assertEqual(len(requests), 1)

    def test_fan_validation_rejects_unknown_unavailable_and_out_of_range_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.service.validate_fan_settings({"script": "anything"})
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            self.service.validate_fan_settings({"fan-min-speed-percent": 101})
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.service.validate_fan_settings(
                {"fan-target-temp": 55}, {"fan-control-interval"},
            )


class RouterOSClusterTests(unittest.IsolatedAsyncioTestCase):
    async def test_remote_fan_update_timeout_covers_worker_request_budget(self) -> None:
        manager = Manager.__new__(Manager)
        manager._routeros_target = mock.AsyncMock(return_value={"id": "worker-1"})
        manager.node_registry = mock.Mock()
        manager.node_registry.request = mock.AsyncMock(return_value={"connected": True})
        body = {"fan-target-temp": 55}

        result = await manager.update_routeros_fan_settings("worker-1", body)

        self.assertTrue(result["connected"])
        manager.node_registry.request.assert_awaited_once_with(
            "worker-1",
            "PATCH",
            "/api/agent/routeros/fan-settings",
            json_body=body,
            timeout=ROUTEROS_FAN_UPDATE_TIMEOUT_SECONDS,
        )
        # update_fan_settings has two three-phase overviews plus one write.
        self.assertGreater(
            ROUTEROS_FAN_UPDATE_TIMEOUT_SECONDS,
            ROUTEROS_TIMEOUT_SECONDS * 7,
        )

    async def test_remote_only_discovery_enables_cluster_presence(self) -> None:
        manager = Manager.__new__(Manager)
        manager.cluster_nodes = mock.AsyncMock(return_value=[
            {"id": "local", "name": "Controller", "online": True, "routeros": {"detected": False}},
            {"id": "worker-1", "name": "Rack node", "online": True,
             "routeros": {"detected": True, "configured": False}},
        ])

        result = await manager.routeros_cluster_presence()

        self.assertTrue(result["detected"])
        self.assertTrue(result["nodes"][1]["detected"])

    async def test_remote_failure_does_not_hide_other_switch_nodes(self) -> None:
        manager = Manager.__new__(Manager)
        manager.cluster_nodes = mock.AsyncMock(return_value=[
            {"id": "local", "name": "Controller", "online": True, "routeros": {"detected": False}},
            {"id": "worker-1", "name": "Rack node", "online": True,
             "routeros": {"detected": True, "configured": True}},
        ])
        manager.routeros = mock.Mock()
        manager.node_registry = mock.Mock()
        manager.node_registry.request = mock.AsyncMock(side_effect=RuntimeError("agent unavailable"))

        result = await manager.routeros_cluster_overview()

        self.assertTrue(result["detected"])
        self.assertFalse(result["nodes"][1]["connected"])
        self.assertIn("agent unavailable", result["nodes"][1]["error"])
