import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import server
from starlette.requests import Request
from manager import ClientAbort, SourceRoutingUnavailable
from sparkdeck.request_limits import is_inference_request_path
from sparkdeck.onboarding import forward_management_request


class ResponsesEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.assignment = patch.object(server.onboarding.assignment, "load", return_value=None)
        self.assignment.start()
        self.addCleanup(self.assignment.stop)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test")
        self.addAsyncCleanup(self.client.aclose)

    async def test_text_request_uses_existing_router_and_caller_identity(self):
        proxy = AsyncMock(return_value={"model": "served", "choices": [{
            "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop",
        }], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})
        with patch.object(server.sparkdeck, "proxy", proxy):
            response = await self.client.post("/v1/responses", json={
                "model": "alias", "instructions": "Be concise", "input": "Hi", "store": False,
            })
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["object"], "response")
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["output"][0]["content"][0]["text"], "Hello")
        self.assertEqual(data["usage"]["input_tokens"], 3)
        body, endpoint, cancel = proxy.await_args.args
        self.assertEqual(endpoint, "chat/completions")
        self.assertEqual(body["model"], "alias")
        self.assertEqual(body["messages"][-1], {"role": "user", "content": "Hi"})
        self.assertEqual(proxy.await_args.kwargs["caller_ip"], "127.0.0.1")
        self.assertIsInstance(cancel, asyncio.Event)

    async def test_stream_emits_responses_events_and_closes_upstream(self):
        closed = []
        async def upstream():
            try:
                for chunk in [
                    {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
                    {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}},
                ]:
                    yield "data: " + json.dumps(chunk) + "\n\n"
                yield "data: [DONE]\n\n"
            finally:
                closed.append(True)
        with patch.object(server.sparkdeck, "proxy", AsyncMock(return_value=upstream())):
            response = await self.client.post("/v1/responses", json={"model": "alias", "input": "Hi", "stream": True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("text/event-stream", response.headers["content-type"])
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual(events[0]["type"], "response.created")
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(events[-1]["response"]["usage"]["output_tokens"], 1)
        self.assertTrue(any(event["type"] == "response.output_text.delta" and event["delta"] == "Hello" for event in events))
        self.assertEqual(closed, [True])

    async def test_invalid_and_stateful_requests_fail_before_inference(self):
        for body in [[], {}, {"model": "x", "input": "hello", "previous_response_id": "resp_unknown"},
                     {"model": "x", "input": [{"type": "item_reference", "id": "secret"}]}]:
            with self.subTest(body=body), patch.object(server.sparkdeck, "proxy", AsyncMock()) as proxy:
                response = await self.client.post("/v1/responses", json=body)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("error", response.json())
                proxy.assert_not_awaited()

    async def test_body_limits_and_invalid_json(self):
        self.assertTrue(is_inference_request_path("/v1/responses"))
        with patch.object(server, "MAX_INFERENCE_REQUEST_BYTES", 64), patch.object(server.sparkdeck, "proxy", AsyncMock()) as proxy:
            response = await self.client.post("/v1/responses", json={"model": "x", "input": "x" * 80})
            self.assertEqual(response.status_code, 413)
            proxy.assert_not_awaited()
        response = await self.client.post("/v1/responses", content="{")
        self.assertEqual(response.status_code, 400)

    async def test_existing_routing_errors_and_abort_status_are_preserved(self):
        for error, status in [(SourceRoutingUnavailable("no target"), 503), (LookupError("missing"), 404),
                              (TimeoutError("queue timed out"), 504), (ClientAbort(), 499)]:
            with self.subTest(status=status), patch.object(server.sparkdeck, "proxy", AsyncMock(side_effect=error)):
                response = await self.client.post("/v1/responses", json={"model": "x", "input": "Hi"})
                self.assertEqual(response.status_code, status, response.text)

    async def test_malformed_upstream_response_returns_502_and_stops_watcher(self):
        for upstream in [None, [], {}, {"choices": [None]}, {"choices": [{"message": None}]}]:
            with self.subTest(upstream=upstream):
                watcher = asyncio.get_running_loop().create_future()
                with patch.object(server.sparkdeck, "proxy", AsyncMock(return_value=upstream)), patch.object(
                    server, "_watch_disconnect", return_value=watcher,
                ):
                    response = await self.client.post("/v1/responses", json={"model": "x", "input": "Hi"})
                self.assertEqual(response.status_code, 502, response.text)
                self.assertEqual(response.json()["detail"], "invalid upstream chat response")
                self.assertTrue(watcher.cancelled())

    async def test_early_stream_close_releases_upstream_and_disconnect_watcher(self):
        # Stop at the first Responses event, before consuming any upstream
        # tokens. Returning the HTTP response must not cancel its watcher.
        class Upstream:
            closed = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise AssertionError("upstream must not be consumed yet")

            async def aclose(self):
                self.closed = True

        upstream = Upstream()
        raw_body = json.dumps({"model": "x", "input": "Hi", "stream": True}).encode()
        req = Request({"type": "http", "method": "POST", "path": "/v1/responses",
                       "headers": [], "client": ("127.0.0.1", 1234)}, receive=AsyncMock(
                           return_value={"type": "http.request", "body": raw_body, "more_body": False},
                       ))
        watcher = asyncio.get_running_loop().create_future()
        with patch.object(server.sparkdeck, "proxy", AsyncMock(return_value=upstream)), patch.object(
            server, "_watch_disconnect", return_value=watcher,
        ):
            response = await server.v1_responses(req)
        self.assertFalse(watcher.cancelled())
        try:
            first = await anext(response.body_iterator)
            self.assertIn("response.created", first)
            self.assertFalse(watcher.cancelled())
        finally:
            await response.body_iterator.aclose()
        self.assertTrue(upstream.closed)
        self.assertTrue(watcher.cancelled())

    async def test_worker_forwarding_bounds_responses_before_sending_to_controller(self):
        req = Request({"type": "http", "method": "POST", "path": "/v1/responses",
                       "scheme": "http", "server": ("worker.test", 7878), "root_path": "",
                       "query_string": b"", "headers": [(b"content-length", b"65")],
                       "client": ("127.0.0.1", 1234)}, receive=AsyncMock())
        with patch("sparkdeck.onboarding.MAX_INFERENCE_REQUEST_BYTES", 64), patch(
            "sparkdeck.onboarding.resolve_control_connection", AsyncMock(
                return_value=SimpleNamespace(url="http://100.64.0.30:7878"),
            ),
        ), patch("sparkdeck.onboarding._send_pinned_control_request", AsyncMock()) as send:
            response = await forward_management_request(req, SimpleNamespace(http=None), {
                "controller_url": "http://100.64.0.30:7878", "node_id": "worker",
                "forward_token": "test-token",
            })
        self.assertEqual(response.status_code, 413)
        send.assert_not_awaited()
