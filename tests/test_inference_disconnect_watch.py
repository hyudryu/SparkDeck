"""Client disconnects must reach in-flight inference handlers.

Inference routes start ``_watch_disconnect`` and rely on its ``cancel`` event:
the admission reaper, the prompt gate's eager stream reader, and every
transport race treat that event as the only proof that a client went away. A
middleware that hides disconnects from ``Request.is_disconnected()`` therefore
strands abandoned streams, their admission slots, and their live-session
records forever (shown on the dashboard as an ever-growing "Prefilling" age).
"""
import asyncio
import json
import socket
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

with patch("docker.from_env", return_value=Mock()):
    import server

from sparkdeck.onboarding import FORWARD_HOP_HEADER


SECURITY_HEADERS = (
    "Content-Security-Policy", "X-Content-Type-Options",
    "Referrer-Policy", "Permissions-Policy",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class DisconnectWatchTests(unittest.TestCase):
    def test_server_middleware_stack_does_not_hide_disconnects(self):
        # BaseHTTPMiddleware (``@app.middleware("http")``) answers every
        # in-flight is_disconnected() probe with False, so it must not wrap
        # the inference routes.
        offenders = [
            middleware for middleware in server.app.user_middleware
            if middleware.cls is BaseHTTPMiddleware
        ]
        self.assertEqual(offenders, [])

    def test_inflight_handler_observes_client_disconnect(self):
        async def run():
            handler_started = asyncio.Event()
            outcome: dict[str, bool] = {}

            async def handler(req: Request):
                # Same shape as the inference routes: read the body, then
                # watch for the client leaving while upstream work runs.
                await req.body()
                cancel = asyncio.Event()
                watcher = server._watch_disconnect(req, cancel)
                handler_started.set()
                try:
                    await asyncio.wait_for(cancel.wait(), 5)
                    outcome["cancelled"] = True
                except asyncio.TimeoutError:
                    outcome["cancelled"] = False
                finally:
                    watcher.cancel()
                return {}

            # Not a worker-forwardable prefix, so no assignment is consulted.
            path = "/__sparkdeck-test__/disconnect-watch"
            server.app.add_api_route(path, handler, methods=["POST"])
            route = server.app.router.routes.pop()
            server.app.router.routes.insert(0, route)
            port = _free_port()
            uv = uvicorn.Server(uvicorn.Config(
                server.app, host="127.0.0.1", port=port, loop="asyncio",
                lifespan="off", log_level="critical",
            ))
            serving = asyncio.create_task(uv.serve())
            try:
                while not uv.started:
                    await asyncio.sleep(0.01)
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                body = json.dumps({"model": "m", "stream": True})
                writer.write((
                    f"POST {path} HTTP/1.1\r\nHost: test\r\n"
                    "Content-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\n\r\n{body}"
                ).encode())
                await writer.drain()
                await asyncio.wait_for(handler_started.wait(), 5)
                writer.close()
                for _ in range(100):
                    if "cancelled" in outcome:
                        break
                    await asyncio.sleep(0.1)
            finally:
                uv.should_exit = True
                await asyncio.wait_for(serving, 10)
                server.app.router.routes.remove(route)
            return outcome.get("cancelled")

        self.assertTrue(asyncio.run(run()))


class SecurityMiddlewareBehaviorTests(unittest.IsolatedAsyncioTestCase):
    def client(self):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app),
            base_url="http://worker.test",
        )

    async def test_assigned_worker_forwards_with_security_headers(self):
        forwarded = AsyncMock(return_value=JSONResponse({"forwarded": True}))
        with (
            patch.object(server.onboarding.assignment, "load",
                         return_value={"controller_url": "http://controller"}),
            patch.object(server, "forward_management_request", forwarded),
        ):
            async with self.client() as client:
                response = await client.get("/api/stats")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"forwarded": True})
        forwarded.assert_awaited_once()
        request, _, assignment = forwarded.await_args.args
        self.assertEqual(request.url.path, "/api/stats")
        self.assertEqual(assignment, {"controller_url": "http://controller"})
        for name in SECURITY_HEADERS:
            self.assertIn(name, response.headers)
        self.assertIn("ws://worker.test", response.headers["Content-Security-Policy"])

    async def test_invalid_forward_headers_are_rejected_with_security_headers(self):
        with (
            patch.object(server.onboarding.assignment, "load", return_value=None),
            patch.object(server.onboarding, "validate_forward_headers",
                         return_value=(False, "bad forwarding token")),
        ):
            async with self.client() as client:
                response = await client.get(
                    "/api/stats", headers={FORWARD_HOP_HEADER: "1"},
                )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"detail": "bad forwarding token"})
        for name in SECURITY_HEADERS:
            self.assertIn(name, response.headers)

    async def test_route_headers_win_over_security_defaults(self):
        async def handler():
            return JSONResponse({}, headers={"Referrer-Policy": "same-origin"})

        path = "/__sparkdeck-test__/header-precedence"
        server.app.add_api_route(path, handler, methods=["GET"])
        route = server.app.router.routes.pop()
        server.app.router.routes.insert(0, route)
        try:
            async with self.client() as client:
                response = await client.get(path)
        finally:
            server.app.router.routes.remove(route)

        self.assertEqual(response.headers.get_list("Referrer-Policy"), ["same-origin"])
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")


if __name__ == "__main__":
    unittest.main()
