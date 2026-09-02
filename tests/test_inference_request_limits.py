import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

import server
from sparkdeck.request_limits import (
    MAX_CLUSTER_ROUTING_ENVELOPE_BYTES,
    RequestBodyTooLarge,
    read_limited_json,
    read_limited_request_body,
)


class FakeRequest:
    def __init__(self, chunks, content_length=None):
        self.chunks = chunks
        self.headers = {}
        if content_length is not None:
            self.headers["content-length"] = str(content_length)
        self.streamed = False

    async def stream(self):
        self.streamed = True
        for chunk in self.chunks:
            yield chunk


class InferenceRequestLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_content_length_rejects_before_streaming(self):
        request = FakeRequest([b"ignored"], content_length=101)

        with self.assertRaises(RequestBodyTooLarge):
            await read_limited_request_body(request, 100)

        self.assertFalse(request.streamed)

    async def test_chunked_body_is_bounded_and_exact_limit_is_allowed(self):
        exact = FakeRequest([b"12345", b"67890"])
        self.assertEqual(await read_limited_request_body(exact, 10), b"1234567890")

        oversized = FakeRequest([b"12345", b"678901"])
        with self.assertRaises(RequestBodyTooLarge):
            await read_limited_request_body(oversized, 10)

    async def test_limited_json_parses_multimodal_content(self):
        body = {
            "model": "vision",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64,iVBOZw==",
                    }},
                ],
            }],
        }
        encoded = json.dumps(body).encode()

        self.assertEqual(
            await read_limited_json(FakeRequest([encoded[:20], encoded[20:]]), 4096),
            body,
        )


class InferenceRequestEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.assignment_patch = patch.object(
            server.onboarding.assignment, "load", return_value=None,
        )
        self.assignment_patch.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app),
            base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        self.assignment_patch.stop()

    async def test_chat_endpoint_preserves_multimodal_content(self):
        body = {
            "model": "vision",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBOZw=="},
                }],
            }],
            "stream": False,
        }
        proxy = AsyncMock(return_value={"choices": []})

        with patch.object(server.sparkdeck, "proxy", proxy):
            response = await self.client.post("/v1/chat/completions", json=body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(proxy.await_args.args[0], body)

    async def test_chat_endpoint_returns_413_before_proxying_oversized_json(self):
        proxy = AsyncMock()
        with (
            patch.object(server, "MAX_INFERENCE_REQUEST_BYTES", 128),
            patch.object(server.sparkdeck, "proxy", proxy),
        ):
            response = await self.client.post("/v1/chat/completions", json={
                "model": "vision",
                "messages": [{"role": "user", "content": "x" * 256}],
            })

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"], "inference request exceeds the 32 MB limit")
        proxy.assert_not_awaited()

    async def test_agent_limit_reserves_space_for_cluster_routing_envelope(self):
        request = FakeRequest([b"{}"])

        with (
            patch.object(server, "MAX_INFERENCE_REQUEST_BYTES", 128),
            patch.object(server, "read_limited_json", AsyncMock(return_value={
                "model": "vision",
                "_sparkdeck_container_name": "replica-0",
            })) as read_json,
            patch.object(server.manager, "_vllm_chat", AsyncMock(return_value={
                "choices": [],
            })),
            patch.object(server, "_require_agent"),
        ):
            await server.agent_inference("chat/completions", request)

        self.assertEqual(
            read_json.await_args.args[1],
            128 + MAX_CLUSTER_ROUTING_ENVELOPE_BYTES,
        )


if __name__ == "__main__":
    unittest.main()
