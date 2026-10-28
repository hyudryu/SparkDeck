import asyncio
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, Mock, patch

from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


with patch("docker.from_env", return_value=Mock()):
    import server


def _frames(*contents):
    async def gen():
        for content in contents:
            yield f'data: {{"choices":[{{"delta":{{"content":"{content}"}}}}]}}\n\n'
        yield "data: [DONE]\n\n"
    return gen()


class ChatWebSocketMuxTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.proxy_calls = []

    def _patch_proxy(self, proxy):
        self.stack.enter_context(patch.object(server.sparkdeck, "proxy", proxy))

    def _client(self):
        return TestClient(server.app)

    def test_concurrent_streams_interleave_over_one_socket(self):
        async def proxy(body, suffix, cancel, caller_ip=None):
            self.proxy_calls.append((body["model"], suffix, caller_ip))
            return _frames(f"{body['id']}-a", f"{body['id']}-b")

        self._patch_proxy(AsyncMock(side_effect=proxy))
        with self._client().websocket_connect("/api/v1/chat/ws") as socket:
            for stream_id in ("alpha", "beta"):
                socket.send_json({
                    "op": "start", "id": stream_id,
                    "body": {"model": "reasoner", "stream": True, "id": stream_id,
                             "messages": [{"role": "user", "content": "hi"}]},
                })
            frames = [socket.receive_json() for _ in range(8)]

        self.assertEqual({call[0] for call in self.proxy_calls}, {"reasoner"})
        self.assertTrue(all(call[1] == "chat/completions" for call in self.proxy_calls))
        by_id = {}
        for frame in frames:
            by_id.setdefault(frame["id"], []).append(frame)
        for stream_id in ("alpha", "beta"):
            # Raw SSE chunks are forwarded verbatim (the [DONE] sentinel
            # included), followed by the mux's own terminal done frame.
            self.assertEqual(len(by_id[stream_id]), 4)
            self.assertIn(f"{stream_id}-a", by_id[stream_id][0]["chunk"])
            self.assertIn(f"{stream_id}-b", by_id[stream_id][1]["chunk"])
            self.assertIn("[DONE]", by_id[stream_id][2]["chunk"])
            self.assertTrue(by_id[stream_id][-1]["done"])

    def test_stop_aborts_upstream_and_reports_cancelled(self):
        async def proxy(body, suffix, cancel, caller_ip=None):
            async def gen():
                yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                while not cancel.is_set():
                    await asyncio.sleep(0.01)
                raise server.ClientAbort()
            return gen()

        self._patch_proxy(AsyncMock(side_effect=proxy))
        with self._client().websocket_connect("/api/v1/chat/ws") as socket:
            socket.send_json({"op": "start", "id": "slow",
                              "body": {"model": "reasoner", "stream": True}})
            first = socket.receive_json()
            self.assertEqual(first["id"], "slow")
            self.assertIn("partial", first["chunk"])
            socket.send_json({"op": "stop", "id": "slow"})
            final = socket.receive_json()

        self.assertEqual(final, {"id": "slow", "cancelled": True})

    def test_routing_lookup_error_becomes_error_frame(self):
        async def proxy(body, suffix, cancel, caller_ip=None):
            raise LookupError("no deployment serves model 'ghost'")

        self._patch_proxy(AsyncMock(side_effect=proxy))
        with self._client().websocket_connect("/api/v1/chat/ws") as socket:
            socket.send_json({"op": "start", "id": "ghost",
                              "body": {"model": "ghost", "stream": True}})
            frame = socket.receive_json()

        self.assertEqual(frame["id"], "ghost")
        self.assertIn("no deployment serves", frame["error"])

    def test_start_without_model_is_rejected_without_calling_proxy(self):
        proxy = AsyncMock()
        self._patch_proxy(proxy)
        with self._client().websocket_connect("/api/v1/chat/ws") as socket:
            socket.send_json({"op": "start", "id": "bad", "body": {"stream": True}})
            socket.send_json({"op": "start", "id": "x" * 65,
                              "body": {"model": "reasoner"}})
            bad = socket.receive_json()
            long_id = socket.receive_json()

        self.assertEqual(bad, {"id": "bad", "error": "invalid start frame"})
        self.assertEqual(long_id, {"id": "x" * 65, "error": "invalid start frame"})
        proxy.assert_not_awaited()

    def test_cross_origin_socket_is_refused(self):
        with self.assertRaises(WebSocketDisconnect) as caught:
            with self._client().websocket_connect(
                    "/api/v1/chat/ws", headers={"origin": "http://evil.example"}):
                pass
        self.assertEqual(caught.exception.code, 1008)

    def test_joined_worker_refuses_the_socket(self):
        self.stack.enter_context(patch.object(
            server.onboarding.assignment, "load",
            Mock(return_value={"controller": "10.0.0.1:7878"})))
        with self.assertRaises(WebSocketDisconnect) as caught:
            with self._client().websocket_connect("/api/v1/chat/ws"):
                pass
        self.assertEqual(caught.exception.code, 1008)

    def test_disconnect_cancels_running_streams(self):
        stopped = asyncio.Event()

        async def proxy(body, suffix, cancel, caller_ip=None):
            async def gen():
                yield 'data: {"choices":[{"delta":{"content":"one"}}]}\n\n'
                try:
                    while not cancel.is_set():
                        await asyncio.sleep(0.01)
                finally:
                    stopped.set()  # reached on abort and on task cancel
                raise server.ClientAbort()
            return gen()

        self._patch_proxy(AsyncMock(side_effect=proxy))
        client = self._client()
        with client.websocket_connect("/api/v1/chat/ws") as socket:
            socket.send_json({"op": "start", "id": "orphan",
                              "body": {"model": "reasoner", "stream": True}})
            self.assertIn("one", socket.receive_json()["chunk"])
        # After the socket closes, the mux must have cancelled the still
        # running stream so the upstream does not burn its token budget.
        for _ in range(200):
            if stopped.is_set():
                break
            import time
            time.sleep(0.01)
        self.assertTrue(stopped.is_set(), "orphaned stream was not cancelled")


if __name__ == "__main__":
    unittest.main()
