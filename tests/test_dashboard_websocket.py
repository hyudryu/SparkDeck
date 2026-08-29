import asyncio
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, Mock, patch

from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


with patch("docker.from_env", return_value=Mock()):
    import server


STATS = {
    "cpu_pct": 12.5,
    "mem": {"used": 8, "total": 16, "pct": 50},
    "gpus": [],
    "active_requests": {},
    "ts": 1_777_000_000,
}
ADMISSION = {"chat": {"running": 1, "queued": 0}}
DEPLOYMENTS = [{"id": "dep-1", "alias": "Chat", "status": "running"}]
COMMUNITY_SYNC = {"consent": True, "outbox": {"pending": 2}}
NODES = [{"id": "local", "name": "Spark Four"}]


def _patch_sources(stack: ExitStack, stats_override=None):
    mocks = Mock()
    mocks.get_stats = stats_override or AsyncMock(return_value=STATS)
    mocks.inference_admission = Mock(return_value=ADMISSION)
    mocks.deployments = AsyncMock(return_value=DEPLOYMENTS)
    mocks.community_sync = Mock(return_value=COMMUNITY_SYNC)
    mocks.cluster_nodes = AsyncMock(return_value=NODES)
    stack.enter_context(patch.object(server.manager, "get_stats", mocks.get_stats))
    stack.enter_context(patch.object(
        server.manager, "inference_admission", mocks.inference_admission))
    stack.enter_context(patch.object(
        server.sparkdeck, "deployments", mocks.deployments))
    stack.enter_context(patch.object(
        server, "_community_sync_status", mocks.community_sync))
    stack.enter_context(patch.object(
        server.manager, "cluster_nodes", mocks.cluster_nodes))
    stack.enter_context(patch.object(
        server.manager, "public_target_node",
        Mock(side_effect=lambda node: dict(node, public=True))))
    return mocks


class DashboardWebSocketTests(unittest.TestCase):
    def test_first_snapshot_has_all_five_sources(self):
        client = TestClient(server.app)
        with ExitStack() as stack:
            _patch_sources(stack)
            with client.websocket_connect("/api/ws/dashboard") as websocket:
                snapshot = websocket.receive_json()

        self.assertEqual(snapshot["type"], "snapshot")
        self.assertEqual(snapshot["stats"], STATS)
        self.assertEqual(snapshot["admission"], ADMISSION)
        self.assertEqual(snapshot["deployments"], {"items": DEPLOYMENTS})
        self.assertEqual(snapshot["community_sync"], COMMUNITY_SYNC)
        self.assertEqual(
            snapshot["nodes"],
            {"items": [dict(NODES[0], public=True)]},
        )

    def test_failed_source_streams_none_without_closing(self):
        client = TestClient(server.app)
        with ExitStack() as stack:
            _patch_sources(
                stack,
                stats_override=AsyncMock(
                    side_effect=RuntimeError("telemetry down")))
            with client.websocket_connect("/api/ws/dashboard") as websocket:
                first = websocket.receive_json()
                second = websocket.receive_json()

        for snapshot in (first, second):
            self.assertEqual(snapshot["type"], "snapshot")
            self.assertIsNone(snapshot["stats"])
            self.assertEqual(snapshot["admission"], ADMISSION)

    def test_slow_sources_repeat_between_refresh_ticks(self):
        client = TestClient(server.app)
        with ExitStack() as stack:
            mocks = _patch_sources(stack)
            with client.websocket_connect("/api/ws/dashboard") as websocket:
                first = websocket.receive_json()
                second = websocket.receive_json()

        # The second tick re-reads the fast sources but reuses the slow ones.
        self.assertEqual(mocks.get_stats.await_count, 2)
        self.assertEqual(mocks.inference_admission.call_count, 2)
        self.assertEqual(mocks.deployments.await_count, 1)
        self.assertEqual(mocks.community_sync.call_count, 1)
        self.assertEqual(mocks.cluster_nodes.await_count, 1)
        self.assertEqual(second["deployments"], first["deployments"])
        self.assertEqual(second["community_sync"], first["community_sync"])
        self.assertEqual(second["nodes"], first["nodes"])

    def test_cross_origin_socket_is_rejected(self):
        client = TestClient(server.app)
        with self.assertRaises(WebSocketDisconnect) as raised:
            with client.websocket_connect(
                "/api/ws/dashboard",
                headers={"origin": "https://malicious.example"},
            ):
                pass
        self.assertEqual(raised.exception.code, 1008)

    def test_origin_scheme_must_match_the_socket_scheme(self):
        # The test client speaks ws://, so an https origin is a downgrade
        # attempt against a plaintext socket and must be refused.
        client = TestClient(server.app)
        with self.assertRaises(WebSocketDisconnect) as raised:
            with client.websocket_connect(
                "/api/ws/dashboard",
                headers={"origin": "https://testserver"},
            ):
                pass
        self.assertEqual(raised.exception.code, 1008)

    def test_same_origin_socket_is_accepted(self):
        client = TestClient(server.app)
        with ExitStack() as stack:
            _patch_sources(stack)
            with client.websocket_connect(
                "/api/ws/dashboard",
                headers={"origin": "http://testserver"},
            ) as websocket:
                snapshot = websocket.receive_json()
        self.assertEqual(snapshot["type"], "snapshot")

    def test_joined_worker_refuses_the_stream(self):
        client = TestClient(server.app)
        with patch.object(
            server.onboarding.assignment, "load",
            Mock(return_value={"controller_url": "http://controller:8080"}),
        ):
            with self.assertRaises(WebSocketDisconnect) as raised:
                with client.websocket_connect("/api/ws/dashboard"):
                    pass
        self.assertEqual(raised.exception.code, 1008)

    def test_hung_source_streams_none_after_the_timeout(self):
        async def hang_forever():
            await asyncio.sleep(60)

        client = TestClient(server.app)
        with ExitStack() as stack:
            _patch_sources(stack, stats_override=hang_forever)
            stack.enter_context(patch.object(
                server, "DASHBOARD_SOURCE_TIMEOUT_SECONDS", 0.1))
            with client.websocket_connect("/api/ws/dashboard") as websocket:
                snapshot = websocket.receive_json()

        self.assertEqual(snapshot["type"], "snapshot")
        self.assertIsNone(snapshot["stats"])
        self.assertEqual(snapshot["admission"], ADMISSION)

    def test_timed_out_source_is_not_relaunched(self):
        calls = 0

        async def hang_forever():
            nonlocal calls
            calls += 1
            await asyncio.sleep(60)

        client = TestClient(server.app)
        with ExitStack() as stack:
            _patch_sources(stack, stats_override=hang_forever)
            stack.enter_context(patch.object(
                server, "DASHBOARD_SOURCE_TIMEOUT_SECONDS", 0.1))
            stack.enter_context(patch.object(
                server, "DASHBOARD_STREAM_INTERVAL_SECONDS", 0.05))
            with client.websocket_connect("/api/ws/dashboard") as websocket:
                for _ in range(3):
                    self.assertIsNone(websocket.receive_json()["stats"])

        # The stuck read is reused across ticks, never stacked.
        self.assertEqual(calls, 1)

    def test_stream_closes_when_node_joins_mid_stream(self):
        client = TestClient(server.app)
        with ExitStack() as stack:
            _patch_sources(stack)
            # Handshake allows, the first tick allows, then the assignment
            # appears and the stream must close so REST forwarding wins.
            stack.enter_context(patch.object(
                server.onboarding.assignment, "load",
                Mock(side_effect=[
                    None, None, {"controller_url": "http://controller:8080"},
                ]),
            ))
            stack.enter_context(patch.object(
                server, "DASHBOARD_STREAM_INTERVAL_SECONDS", 0.05))
            with self.assertRaises(WebSocketDisconnect) as raised:
                with client.websocket_connect("/api/ws/dashboard") as websocket:
                    self.assertEqual(
                        websocket.receive_json()["type"], "snapshot")
                    websocket.receive_json()
        self.assertEqual(raised.exception.code, 1008)

    def test_csp_permits_same_host_websockets(self):
        client = TestClient(server.app)
        with patch.object(
            server.manager, "get_stats", AsyncMock(return_value=STATS),
        ):
            response = client.get("/api/stats")
        csp = response.headers["Content-Security-Policy"]
        self.assertIn("connect-src 'self'", csp)
        self.assertIn("ws://testserver", csp)
        self.assertIn("wss://testserver", csp)


if __name__ == "__main__":
    unittest.main()
