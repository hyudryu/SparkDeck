import asyncio
import json
import sqlite3
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
        self.node_registry.remove = Mock(return_value=True)
        self.remove_cluster_node = Mock(return_value=True)
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
        self.deployments = []
        self.list_containers = AsyncMock(return_value=[])

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

    def test_leave_rotation_revokes_agent_token_and_both_pairing_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            credentials = AgentCredentials(root)
            old_token = credentials.data["agent_token"]
            old_pairing = credentials.data["pairing_code"]
            old_join = credentials.cluster_join_code

            credentials.revoke_remote_access()

            self.assertFalse(credentials.accepts_token(old_token))
            self.assertNotEqual(credentials.data["pairing_code"], old_pairing)
            self.assertNotEqual(credentials.cluster_join_code, old_join)
            persisted = AgentCredentials(root)
            self.assertEqual(persisted.data["agent_token"], credentials.data["agent_token"])
            self.assertEqual(persisted.data["pairing_code"], credentials.data["pairing_code"])


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

    async def test_normal_remote_join_verifies_controller_then_persists_assignment(self):
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

    async def test_join_rejects_localhost_alias_when_identity_is_this_node(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={
                "role": "controller",
                "node": {
                    "id": manager.agent_credentials.node_id,
                    "protocol_version": AGENT_PROTOCOL_VERSION,
                },
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        manager = FakeManager(self.root, http)
        service = OnboardingService(manager, self.root)

        with self.assertRaisesRegex(ValueError, "resolves to this node"):
            await service.join({
                "controller_url": "http://localhost:9000",
                "join_code": "654321",
                "advertise_url": "http://127.0.0.1:9001",
                "name": "Worker",
            }, "http://127.0.0.1:7878")

        self.assertEqual([request.method for request in requests], ["GET"])
        self.assertIsNone(service.assignment.load())
        manager.adopt_worker_role.assert_not_awaited()
        await http.aclose()

    async def test_join_rejects_saved_sqlite_deployments_before_contacting_controller(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(500)

        database = self.root / "sparkdeck.sqlite3"
        connection = sqlite3.connect(database)
        try:
            connection.execute("CREATE TABLE deployments (id TEXT PRIMARY KEY)")
            connection.execute("INSERT INTO deployments(id) VALUES ('existing')")
            connection.commit()
        finally:
            connection.close()
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        manager = FakeManager(self.root, http)
        service = OnboardingService(manager, self.root)

        with self.assertRaisesRegex(ValueError, "1 saved deployment record"):
            await service.join({
                "controller_url": "http://127.0.0.1:9000",
                "join_code": "654321",
                "advertise_url": "http://127.0.0.1:9001",
            }, "http://127.0.0.1:7878")

        self.assertEqual(requests, [])
        manager.list_containers.assert_awaited_once()
        await http.aclose()

    async def test_join_rejects_managed_containers_before_contacting_controller(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(500)

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        manager = FakeManager(self.root, http)
        manager.list_containers.return_value = [
            {"name": "sparkdeck-model", "managed": True, "status": "exited"},
            {"name": "unrelated", "managed": False, "status": "running"},
        ]
        service = OnboardingService(manager, self.root)

        with self.assertRaisesRegex(ValueError, "1 managed container"):
            await service.join({
                "controller_url": "http://127.0.0.1:9000",
                "join_code": "654321",
                "advertise_url": "http://127.0.0.1:9001",
            }, "http://127.0.0.1:7878")

        self.assertEqual(requests, [])
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

    async def test_status_never_advertises_loopback_to_cluster_nodes(self):
        manager = FakeManager(self.root)
        service = OnboardingService(manager, self.root)

        status = await service.status("http://127.0.0.1:7878")

        self.assertEqual(status["node"]["port"], 7878)
        self.assertEqual(status["node"]["access_urls"], ["http://100.100.20.30:7878"])
        self.assertNotIn("127.0.0.1", json.dumps(status["node"]["access_urls"]))
        await manager.http.aclose()

    async def test_status_preserves_https_serve_origin_and_raw_tailscale_fallback(self):
        manager = FakeManager(self.root)
        service = OnboardingService(manager, self.root)

        status = await service.status("https://spark-one.example-tailnet.ts.net")

        self.assertEqual(status["node"]["port"], 7878)
        self.assertEqual(status["node"]["access_urls"], [
            "https://spark-one.example-tailnet.ts.net",
            "http://100.100.20.30:7878",
        ])
        await manager.http.aclose()

    async def test_leave_unregisters_then_revokes_before_clearing_assignment(self):
        requests = []
        observed = {}

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            observed["assignment_present"] = service.assignment.load() is not None
            observed["old_agent_revoked"] = not manager.agent_credentials.accepts_token(old_token)
            return httpx.Response(200, json={"ok": True})

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        manager = FakeManager(self.root, http)
        service = OnboardingService(manager, self.root)
        service.assignment.save({
            "controller_url": "http://127.0.0.1:9000",
            "forward_token": "secret",
            "node_id": manager.agent_credentials.node_id,
        })
        old_token = manager.agent_credentials.data["agent_token"]
        old_pairing = manager.agent_credentials.data["pairing_code"]
        old_join = manager.agent_credentials.cluster_join_code

        left = await service.leave("http://127.0.0.1:7878")

        self.assert_status_shape(left, "controller")
        self.assertIn("join_code", left)
        self.assertIsNone(service.assignment.load())
        manager.adopt_controller_role.assert_called_once()
        self.assertFalse(manager.agent_credentials.accepts_token(old_token))
        self.assertNotEqual(manager.agent_credentials.data["pairing_code"], old_pairing)
        self.assertNotEqual(manager.agent_credentials.cluster_join_code, old_join)
        self.assertEqual(requests[0].url.path, "/api/v1/onboarding/unregister")
        self.assertEqual(requests[0].headers[FORWARD_NODE_HEADER], manager.agent_credentials.node_id)
        self.assertEqual(requests[0].headers[FORWARD_HOP_HEADER], "1")
        self.assertEqual(requests[0].headers[FORWARD_TOKEN_HEADER], "secret")
        self.assertEqual(observed, {
            "assignment_present": True,
            "old_agent_revoked": False,
        })
        await http.aclose()

    async def test_offline_controller_cannot_prevent_durable_local_leave(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline", request=request)

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        manager = FakeManager(self.root, http)
        service = OnboardingService(manager, self.root)
        service.assignment.save({
            "controller_url": "http://127.0.0.1:9000",
            "forward_token": "secret",
            "node_id": manager.agent_credentials.node_id,
        })
        old_token = manager.agent_credentials.data["agent_token"]

        await service.leave("http://127.0.0.1:7878")

        self.assertFalse(manager.agent_credentials.accepts_token(old_token))
        self.assertIsNone(service.assignment.load())
        manager.adopt_controller_role.assert_called_once()
        await http.aclose()

    async def test_leave_refuses_local_managed_member_without_rotating_credentials(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"ok": True})

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        manager = FakeManager(self.root, http)
        manager.list_containers.return_value = [{
            "name": "cluster-member", "managed": True,
            "deployment_id": "deployment-1", "status": "running",
        }]
        service = OnboardingService(manager, self.root)
        service.assignment.save({
            "controller_url": "http://127.0.0.1:9000",
            "forward_token": "secret",
            "node_id": manager.agent_credentials.node_id,
        })
        old_token = manager.agent_credentials.data["agent_token"]

        with self.assertRaisesRegex(ValueError, "cannot leave.*1 managed container"):
            await service.leave("http://127.0.0.1:7878")

        self.assertTrue(manager.agent_credentials.accepts_token(old_token))
        self.assertIsNotNone(service.assignment.load())
        self.assertEqual(requests, [])
        manager.adopt_controller_role.assert_not_called()
        await http.aclose()

    async def test_leave_honors_authoritative_controller_deployment_guard(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(409, json={
                "detail": "node is still used by production; remove that deployment first",
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        manager = FakeManager(self.root, http)
        service = OnboardingService(manager, self.root)
        service.assignment.save({
            "controller_url": "http://127.0.0.1:9000",
            "forward_token": "secret",
            "node_id": manager.agent_credentials.node_id,
        })
        old_token = manager.agent_credentials.data["agent_token"]

        with self.assertRaisesRegex(ValueError, "controller still has a deployment"):
            await service.leave("http://127.0.0.1:7878")

        self.assertTrue(manager.agent_credentials.accepts_token(old_token))
        self.assertIsNotNone(service.assignment.load())
        manager.adopt_controller_role.assert_not_called()
        await http.aclose()

    async def test_leave_keeps_assignment_when_credential_rotation_is_not_durable(self):
        manager = FakeManager(self.root)
        service = OnboardingService(manager, self.root)
        service.assignment.save({
            "controller_url": "http://127.0.0.1:9000",
            "forward_token": "secret",
            "node_id": manager.agent_credentials.node_id,
        })

        with patch("cluster._atomic_json_write", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                await service.leave("http://127.0.0.1:7878")

        self.assertIsNotNone(service.assignment.load())
        manager.adopt_controller_role.assert_not_called()
        await manager.http.aclose()

    def test_controller_unregister_requires_one_hop_credential_and_removes_worker(self):
        manager = FakeManager(self.root)
        service = OnboardingService(manager, self.root)

        result = service.unregister({
            FORWARD_HOP_HEADER: "1",
            FORWARD_NODE_HEADER: "worker-id",
            FORWARD_TOKEN_HEADER: "secret",
        })

        self.assertTrue(result["revoked"])
        manager.remove_cluster_node.assert_called_once_with("worker-id")
        manager.remove_cluster_node.reset_mock()
        manager.remove_cluster_node.side_effect = ValueError(
            "node is still used by production; remove that deployment first"
        )
        with self.assertRaisesRegex(ValueError, "still used by production"):
            service.unregister({
                FORWARD_HOP_HEADER: "1",
                FORWARD_NODE_HEADER: "worker-id",
                FORWARD_TOKEN_HEADER: "secret",
            })
        with self.assertRaises(PermissionError):
            service.unregister({FORWARD_HOP_HEADER: "2"})

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
    async def test_proxy_cancels_stalled_controller_headers_on_browser_disconnect(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def send(*_args, **_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        http = Mock(
            build_request=Mock(return_value=object()),
            send=send,
        )
        request = request_for("/api/state")
        request.is_disconnected = AsyncMock(side_effect=[False, True])

        response = await asyncio.wait_for(
            forward_management_request(request, Mock(http=http), {
                "controller_url": "http://127.0.0.1:9000",
                "forward_token": "worker-secret", "node_id": "worker-id",
            }),
            timeout=1,
        )

        self.assertTrue(started.is_set())
        self.assertTrue(cancelled.is_set())
        self.assertEqual(response.status_code, 499)

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
