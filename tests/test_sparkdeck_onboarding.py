import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi import Request

from cluster import AGENT_PROTOCOL_VERSION, AgentCredentials, NodeRegistry
from manager import Manager
from sparkdeck.onboarding import (
    FORWARD_HOP_HEADER,
    FORWARD_NODE_HEADER,
    FORWARD_TOKEN_HEADER,
    ControllerAssignment,
    OnboardingService,
    forward_management_request,
    is_forwardable_path,
    validate_control_url,
)


class FakeManager:
    def __init__(self, data_dir: Path, http: httpx.AsyncClient | None = None):
        self.data_dir = data_dir
        self.http = http or httpx.AsyncClient()
        self.agent_credentials = AgentCredentials(data_dir)
        self.settings = {"cluster_node_name": "Spark Worker"}
        self.node_registry = Mock()
        self.node_registry.set_forward_token = Mock()
        self.node_registry.accepts_forward_token = Mock(return_value=True)
        self.pair_node = AsyncMock(return_value={
            "id": "worker-id", "name": "Worker",
            "protocol_version": AGENT_PROTOCOL_VERSION,
        })
        self.cluster_nodes = AsyncMock(return_value=[{
            "id": "worker-id", "name": "Worker", "online": True,
            "docker_ready": True, "enabled": True,
        }])
        self.adopt_worker_role = AsyncMock()
        self.adopt_controller_role = Mock()

    @staticmethod
    def _network_interfaces():
        return [
            {"name": "eth0", "ipv4": ["192.168.1.5"]},
            {"name": "tailscale0", "ipv4": ["100.100.20.30"]},
        ]

    @staticmethod
    def public_target_node(node):
        return Manager.public_target_node(node)


def request_for(
    path: str,
    *,
    method: str = "POST",
    query: str = "",
    body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> Request:
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    raw_headers = [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]
    return Request({
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "headers": raw_headers,
        "server": ("worker.test", 7878),
        "client": ("100.100.20.40", 50000),
        "root_path": "",
    }, receive)


class JoinCredentialTests(unittest.TestCase):
    def test_join_code_is_one_time_and_rotates(self):
        with tempfile.TemporaryDirectory() as directory:
            credentials = AgentCredentials(Path(directory))
            code = credentials.current_cluster_join_code()

            credentials.consume_cluster_join_code(code)

            self.assertNotEqual(credentials.current_cluster_join_code(), code)
            with self.assertRaisesRegex(ValueError, "invalid or expired"):
                credentials.consume_cluster_join_code(code)

    def test_expired_join_code_rotates_before_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            credentials = AgentCredentials(Path(directory))
            expired = credentials.cluster_join_code
            credentials.data["cluster_join_code_issued_at"] = 0

            with self.assertRaisesRegex(ValueError, "invalid or expired"):
                credentials.consume_cluster_join_code(expired)
            self.assertNotEqual(credentials.cluster_join_code, expired)

    def test_forward_token_is_hashed_at_rest_and_not_public(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = NodeRegistry(root, Mock())
            registry.nodes = [{"id": "worker-id", "enabled": True}]

            registry.set_forward_token("worker-id", "plain-secret")

            persisted = (root / "nodes.json").read_text(encoding="utf-8")
            self.assertNotIn("plain-secret", persisted)
            self.assertIn("forward_token_hash", persisted)
            self.assertTrue(registry.accepts_forward_token("worker-id", "plain-secret"))
            self.assertFalse(registry.accepts_forward_token("worker-id", "wrong"))
            self.assertNotIn(
                "forward_token_hash", NodeRegistry.public_config(registry.nodes[0])
            )

    def test_worker_assignment_is_private_and_clearable(self):
        with tempfile.TemporaryDirectory() as directory:
            assignment = ControllerAssignment(Path(directory))
            assignment.save({
                "controller_url": "http://100.100.20.30:7878",
                "forward_token": "secret-forward-token",
                "node_id": "worker-id",
            })

            self.assertEqual(assignment.load()["forward_token"], "secret-forward-token")
            assignment.clear()
            self.assertIsNone(assignment.load())


class UrlGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_allows_tailscale_and_loopback_but_rejects_lan_ssrf(self):
        self.assertEqual(
            await validate_control_url("http://100.100.20.30:7878/"),
            "http://100.100.20.30:7878",
        )
        self.assertEqual(
            await validate_control_url("http://127.0.0.1:7878"),
            "http://127.0.0.1:7878",
        )
        with self.assertRaisesRegex(ValueError, "Tailscale or loopback"):
            await validate_control_url("http://192.168.1.25:7878")


class OnboardingFlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    async def asyncTearDown(self):
        self.temp.cleanup()

    def assert_status_shape(self, value, role):
        self.assertEqual(value["role"], role)
        self.assertIsInstance(value["instructions"], list)
        self.assertIn("controller_reachable", value)
        self.assertIn("controller_url", value)
        self.assertEqual(
            {"id", "name", "port", "access_urls"} - set(value["node"]), set()
        )

    async def test_controller_register_pairs_back_and_returns_separate_forward_token(self):
        manager = FakeManager(self.root)
        service = OnboardingService(manager, self.root)
        join_code = manager.agent_credentials.current_cluster_join_code()

        registered = await service.register({
            "join_code": join_code,
            "pairing_code": "123456",
            "advertise_url": "http://127.0.0.1:9001",
            "name": "Worker",
        }, "http://127.0.0.1:7878", "127.0.0.1")

        manager.pair_node.assert_awaited_once_with({
            "agent_url": "http://127.0.0.1:9001",
            "pairing_code": "123456",
            "name": "Worker",
        })
        forward_token = registered["forward_token"]
        self.assertTrue(forward_token)
        manager.node_registry.set_forward_token.assert_called_once_with("worker-id", forward_token)
        self.assertNotEqual(forward_token, manager.agent_credentials.data["agent_token"])
        self.assertNotIn("forward_token", registered["node"])

        with self.assertRaisesRegex(ValueError, "invalid or expired"):
            await service.register({
                "join_code": join_code,
                "pairing_code": "123456",
                "advertise_url": "http://127.0.0.1:9001",
            }, "http://127.0.0.1:7878", "127.0.0.2")
        await manager.http.aclose()

    async def test_worker_verifies_controller_then_persists_assignment(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.method == "GET":
                return httpx.Response(200, json={
                    "role": "controller",
                    "node": {"id": "controller-id", "protocol_version": AGENT_PROTOCOL_VERSION},
                })
            return httpx.Response(200, json={
                "ok": True,
                "role": "controller",
                "protocol_version": AGENT_PROTOCOL_VERSION,
                "forward_token": "forward-secret",
                "node": {"id": "worker-id", "name": "Worker"},
                "cluster": {"nodes": []},
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        manager = FakeManager(self.root, http)
        service = OnboardingService(manager, self.root)

        joined = await service.join({
            "controller_url": "http://127.0.0.1:9000",
            "join_code": "654321",
            "advertise_url": "http://127.0.0.1:9001",
            "name": "Worker",
        }, "http://127.0.0.1:7878")

        assignment = service.assignment.load()
        self.assertEqual(joined["role"], "worker")
        self.assert_status_shape(joined, "worker")
        self.assertEqual(joined["node"]["id"], manager.agent_credentials.node_id)
        self.assertNotIn("forward_token", joined)
        self.assertEqual(assignment["forward_token"], "forward-secret")
        self.assertEqual(assignment["controller_node_id"], "controller-id")
        manager.adopt_worker_role.assert_awaited_once()
        posted = json.loads(requests[1].content)
        self.assertEqual(posted["pairing_code"], manager.agent_credentials.data["pairing_code"])
        await http.aclose()

    async def test_status_prefers_tailscale_and_never_exposes_credentials(self):
        manager = FakeManager(self.root)
        service = OnboardingService(manager, self.root)

        status = await service.status("http://spark.local:7878")

        self.assertEqual(status["role"], "controller")
        self.assertEqual(status["node"]["access_urls"][0], "http://100.100.20.30:7878")
        self.assertFalse(status["automatic_failover"])
        self.assert_status_shape(status, "controller")
        self.assertNotIn("agent_token", json.dumps(status))
        self.assertNotIn("forward_token", json.dumps(status))
        await manager.http.aclose()

    async def test_leave_returns_complete_controller_status(self):
        manager = FakeManager(self.root)
        service = OnboardingService(manager, self.root)
        service.assignment.save({
            "controller_url": "http://127.0.0.1:9000",
            "forward_token": "secret",
            "node_id": manager.agent_credentials.node_id,
        })

        left = await service.leave("http://127.0.0.1:7878")

        self.assert_status_shape(left, "controller")
        self.assertIn("join_code", left)
        self.assertIsNone(service.assignment.load())
        manager.adopt_controller_role.assert_called_once()
        await manager.http.aclose()

    async def test_join_registration_is_rate_limited(self):
        manager = FakeManager(self.root)
        service = OnboardingService(manager, self.root)
        body = {
            "join_code": "invalid",
            "pairing_code": "123456",
            "advertise_url": "http://127.0.0.1:9001",
        }
        for _ in range(5):
            with self.assertRaisesRegex(ValueError, "invalid or expired"):
                await service.register(body, "http://127.0.0.1:7878", "client")
        with self.assertRaisesRegex(ValueError, "too many join attempts"):
            await service.register(body, "http://127.0.0.1:7878", "client")
        await manager.http.aclose()


class ForwardingTests(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_preserves_query_body_status_content_type_and_authenticates_worker(self):
        captured = []

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'{"partial":true}'

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(
                207, stream=Body(),
                headers={"content-type": "application/problem+json"},
            )

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        manager = Mock(http=http)
        request = request_for(
            "/api/v1/images/pull", query="force=1", body=b'{"image":"x"}',
            headers={"content-type": "application/json", FORWARD_TOKEN_HEADER: "client-spoof"},
        )
        # Client-supplied forwarding headers are rejected instead of relayed.
        rejected = await forward_management_request(request, manager, {
            "controller_url": "http://127.0.0.1:9000",
            "forward_token": "worker-secret", "node_id": "worker-id",
        })
        self.assertEqual(rejected.status_code, 508)

        request = request_for(
            "/api/v1/images/pull", query="force=1", body=b'{"image":"x"}',
            headers={"content-type": "application/json"},
        )
        response = await forward_management_request(request, manager, {
            "controller_url": "http://127.0.0.1:9000",
            "forward_token": "worker-secret", "node_id": "worker-id",
        })
        content = b"".join([chunk async for chunk in response.body_iterator])

        self.assertEqual(response.status_code, 207)
        self.assertEqual(response.headers["content-type"], "application/problem+json")
        self.assertEqual(content, b'{"partial":true}')
        self.assertEqual(str(captured[0].url), "http://127.0.0.1:9000/api/v1/images/pull?force=1")
        self.assertEqual(captured[0].content, b'{"image":"x"}')
        self.assertEqual(captured[0].headers[FORWARD_HOP_HEADER], "1")
        self.assertEqual(captured[0].headers[FORWARD_NODE_HEADER], "worker-id")
        self.assertEqual(captured[0].headers[FORWARD_TOKEN_HEADER], "worker-secret")
        await http.aclose()

    def test_exclusions_and_controller_token_validation(self):
        self.assertFalse(is_forwardable_path("/api/agent/status"))
        self.assertFalse(is_forwardable_path("/api/v1/onboarding"))
        self.assertTrue(is_forwardable_path("/api/state"))
        self.assertTrue(is_forwardable_path("/v1/chat/completions"))
        self.assertTrue(is_forwardable_path("/mcp"))

        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager(Path(directory))
            service = OnboardingService(manager, Path(directory))
            valid, _ = service.validate_forward_headers({
                FORWARD_HOP_HEADER: "1",
                FORWARD_NODE_HEADER: "worker-id",
                FORWARD_TOKEN_HEADER: "secret",
            })
            self.assertTrue(valid)
            manager.node_registry.accepts_forward_token.assert_called_once_with(
                "worker-id", "secret"
            )
            valid, detail = service.validate_forward_headers({
                FORWARD_HOP_HEADER: "2",
                FORWARD_NODE_HEADER: "worker-id",
                FORWARD_TOKEN_HEADER: "secret",
            })
            self.assertFalse(valid)
            self.assertIn("exactly 1", detail)


class WorkerSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_start_skips_controller_only_schedulers(self):
        manager = Manager.__new__(Manager)
        manager.data_dir = Path("unused")
        manager.is_joined_worker = Mock(return_value=True)
        manager._start_controller_tasks = Mock()
        blocker = asyncio.Event()

        async def temperature_loop():
            await blocker.wait()

        manager._temperature_history_monitor_loop = temperature_loop
        manager._start_mem_bw_monitor = Mock()

        await manager.start()

        manager._start_controller_tasks.assert_not_called()
        manager._start_mem_bw_monitor.assert_called_once()
        manager.temperature_history_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await manager.temperature_history_task


if __name__ == "__main__":
    unittest.main()
