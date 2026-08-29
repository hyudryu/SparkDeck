import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, Mock, patch

from starlette.testclient import TestClient


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
    stack.enter_context(patch.object(
        server.manager, "get_stats",
        stats_override or AsyncMock(return_value=STATS)))
    stack.enter_context(patch.object(
        server.manager, "inference_admission", Mock(return_value=ADMISSION)))
    stack.enter_context(patch.object(
        server.sparkdeck, "deployments", AsyncMock(return_value=DEPLOYMENTS)))
    stack.enter_context(patch.object(
        server, "_community_sync_status", Mock(return_value=COMMUNITY_SYNC)))
    stack.enter_context(patch.object(
        server.manager, "cluster_nodes", AsyncMock(return_value=NODES)))
    stack.enter_context(patch.object(
        server.manager, "public_target_node",
        Mock(side_effect=lambda node: dict(node, public=True))))


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


if __name__ == "__main__":
    unittest.main()
