import asyncio
import logging
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

with patch("docker.from_env", return_value=Mock()):
    import server


class ActivityLogTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.handler = server._DequeHandler()
        self.handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        self.activity = patch.object(server, "_activity_buffer", server.deque(maxlen=5))
        self.raw = patch.object(server, "_log_buffer", server.deque(maxlen=5))
        self.activity.start()
        self.raw.start()
        self.token = patch.object(server.manager, "_resolved_hf_token", return_value="test-secret-token")
        self.token.start()
        self.addCleanup(self.activity.stop)
        self.addCleanup(self.raw.stop)
        self.addCleanup(self.token.stop)

    def emit(self, message, level=logging.INFO, event=None, name="test", args=()):
        record = logging.LogRecord(name, level, __file__, 1, message, args, None)
        if event:
            record.deployment_event = event
        self.handler.emit(record)
        return record

    async def test_noise_does_not_evict_events_and_duplicate_capture_is_ignored(self):
        record = self.emit("Model launched", event="launched")
        self.handler.emit(record)
        for index in range(20):
            self.emit(f"Health check {index}")
        self.emit("A warning", logging.WARNING)
        self.assertEqual(len(server._activity_buffer), 1)
        self.assertEqual(server._activity_buffer[0]["event"], "launched")

    async def test_errors_are_redacted_and_http_failures_are_included(self):
        self.emit("Failed using test-secret-token api_key=secret", logging.ERROR)
        self.emit("Fatal error", logging.CRITICAL)
        self.emit('%s - "%s %s HTTP/%s" %d', name="uvicorn.access",
                  args=("127.0.0.1", "GET", "/api/state", "1.1", 200))
        self.emit('%s - "%s %s HTTP/%s" %d', name="uvicorn.access",
                  args=("127.0.0.1", "POST", "/api/deployments", "1.1", 500))
        entries = list(server._activity_buffer)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["message"], "Failed using [REDACTED] api_key=[REDACTED]")
        self.assertEqual(entries[-1]["level"], "error")

    async def test_endpoint_returns_events_without_waiting_for_inventory(self):
        self.emit("Model launched", event="launched")
        self.emit("Model stopped", event="stopped")
        with patch.object(server.sparkdeck, "deployments", new=AsyncMock()) as refresh:
            result = await server.get_activity_logs(tail=1)
        refresh.assert_not_awaited()
        self.assertEqual([entry["event"] for entry in result["entries"]], ["stopped"])

    async def test_background_monitor_survives_outages_without_repeating_errors(self):
        refresh = AsyncMock(side_effect=[
            RuntimeError("agent unavailable"), RuntimeError("agent unavailable"),
            [], RuntimeError("agent unavailable"), asyncio.CancelledError(),
        ])
        logger = Mock()
        with patch.object(server.sparkdeck, "deployments", refresh), \
                patch.object(server.asyncio, "sleep", new=AsyncMock()) as sleep, \
                patch.object(server.logging, "getLogger", return_value=logger):
            with self.assertRaises(asyncio.CancelledError):
                await server._deployment_log_loop()
        self.assertEqual(refresh.await_count, 5)
        self.assertEqual(sleep.await_count, 4)
        self.assertEqual(logger.error.call_count, 2)

    async def test_observed_lifecycle_reaches_the_logs_api_once(self):
        row = {"id": "model-1", "alias": "My model", "status": "running"}
        snapshots = iter([
            [row], [row], [{**row, "status": "stopped", "desired_state": "stopped"}],
        ])

        async def refresh(*, observe_events=False):
            try:
                rows = next(snapshots)
            except StopIteration:
                raise asyncio.CancelledError()
            if observe_events:
                server.sparkdeck._observe_deployment_events(rows, inventory_complete=True)
            return rows
        with patch.object(server.sparkdeck, "deployments", refresh), \
                patch.object(server.sparkdeck, "_deployment_log_states", {}), \
                patch.object(server.sparkdeck, "_deployment_log_errors", {}), \
                patch.object(server.asyncio, "sleep", new=AsyncMock()):
            with self.assertRaises(asyncio.CancelledError):
                await server._deployment_log_loop()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test",
        ) as client:
            response = await client.get("/api/v1/logs")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [entry["event"] for entry in response.json()["entries"]],
            ["launched", "stopped"],
        )


    async def test_production_server_reinstalls_access_capture_after_uvicorn_config(self):
        import uvicorn

        names = ("", "uvicorn", "uvicorn.access", "uvicorn.error")
        previous = {
            name: (list(logging.getLogger(name).handlers), logging.getLogger(name).level,
                   logging.getLogger(name).propagate)
            for name in names
        }

        async def serve(instance):
            access = logging.getLogger("uvicorn.access")
            for status in (200, 404, 500):
                access.info('%s - "%s %s HTTP/%s" %d',
                            "127.0.0.1", "GET", "/api/missing", "1.1", status)
            server._install_log_capture()
            self.assertEqual(sum(isinstance(h, server._DequeHandler)
                                 for h in access.handlers), 1)

        try:
            with patch.object(uvicorn.Server, "serve", serve), \
                    patch.object(server, "_discard_shutdown_request"), \
                    patch.object(server, "_shutdown_request_process_ids", return_value=set()):
                await server._serve_application()
            entries = list(server._activity_buffer)
            self.assertEqual(len(entries), 2)
            self.assertTrue(all(entry["level"] == "error" for entry in entries))
            self.assertIn("404", entries[0]["message"])
            self.assertIn("500", entries[1]["message"])
        finally:
            for name, (handlers, level, propagate) in previous.items():
                logger = logging.getLogger(name)
                logger.handlers = handlers
                logger.setLevel(level)
                logger.propagate = propagate
