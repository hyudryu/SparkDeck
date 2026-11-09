"""Cancelled requests must not strand httpx connection-pool slots."""

import asyncio
import os
import unittest
from unittest import mock

import httpcore
import httpx

from sparkdeck.http_transport import (
    CancellationSafeTransport,
    cancellation_safe_client,
    reap_orphaned_connections,
)


class _KeepAliveServer:
    """Minimal HTTP/1.1 server answering every request with ``200 ok``."""

    def __init__(self, response_delay: float = 0.0):
        self.response_delay = response_delay
        self.server = None
        self.url = ""

    async def __aenter__(self):
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/"
        return self

    async def __aexit__(self, *exc):
        self.server.close()

    async def _handle(self, reader, writer):
        try:
            while True:
                await reader.readuntil(b"\r\n\r\n")
                if self.response_delay:
                    await asyncio.sleep(self.response_delay)
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()


async def _cancel_after_connect(event, info):
    # A client disconnect landing right after the TCP connect completes, before
    # the HTTP/1.1 connection leaves its NEW state.
    if event == "connection.connect_tcp.complete":
        asyncio.current_task().cancel()


def _pool(client: httpx.AsyncClient):
    return client._transport._pool


class StrandedConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def _strand(self, client, url, count):
        for _ in range(count):
            task = asyncio.create_task(
                client.get(url, extensions={"trace": _cancel_after_connect})
            )
            results = await asyncio.gather(task, return_exceptions=True)
            self.assertIsInstance(results[0], asyncio.CancelledError)

    async def test_plain_httpx_pool_is_exhausted_by_cancellation(self):
        """Documents the upstream defect this transport exists to contain."""
        async with _KeepAliveServer() as server:
            client = httpx.AsyncClient(limits=httpx.Limits(max_connections=3))
            try:
                await self._strand(client, server.url, 3)
                states = [c.info() for c in _pool(client).connections]
                if not all("NEW" in state for state in states) or len(states) != 3:
                    self.skipTest(f"httpcore no longer strands connections: {states}")
                with self.assertRaises(httpx.PoolTimeout):
                    await client.get(server.url, timeout=httpx.Timeout(5, pool=0.5))
            finally:
                await client.aclose()

    async def test_cancelled_requests_do_not_exhaust_the_pool(self):
        async with _KeepAliveServer() as server:
            client = cancellation_safe_client(
                limits=httpx.Limits(max_connections=3), timeout=5,
            )
            try:
                await self._strand(client, server.url, 20)
                self.assertFalse(
                    [c.info() for c in _pool(client).connections if "NEW" in c.info()]
                )
                response = await client.get(
                    server.url, timeout=httpx.Timeout(5, pool=0.5),
                )
                self.assertEqual(response.status_code, 200)
            finally:
                await client.aclose()

    async def test_reaping_spares_connections_owned_by_in_flight_requests(self):
        async with _KeepAliveServer(response_delay=0.3) as server:
            client = cancellation_safe_client(
                limits=httpx.Limits(max_connections=4), timeout=5,
            )
            try:
                in_flight = [
                    asyncio.create_task(client.get(server.url)) for _ in range(2)
                ]
                await asyncio.sleep(0.1)  # both are now waiting for responses
                await self._strand(client, server.url, 3)
                responses = await asyncio.gather(*in_flight)
                self.assertEqual([r.status_code for r in responses], [200, 200])
                self.assertFalse(
                    [c.info() for c in _pool(client).connections if "NEW" in c.info()]
                )
            finally:
                await client.aclose()

    async def test_unowned_connecting_connection_is_reaped_and_queue_woken(self):
        async with _KeepAliveServer() as server:
            transport = CancellationSafeTransport(
                limits=httpx.Limits(max_connections=1),
            )
            client = httpx.AsyncClient(transport=transport, timeout=5)
            try:
                pool = transport._pool
                # Simulate the queued-request race: the pool created a
                # connection for a request that was cancelled before dialing.
                origin = httpx.URL(server.url)
                stranded = httpcore.AsyncHTTPConnection(
                    origin=httpcore.Origin(b"http", origin.host.encode(), origin.port),
                )
                pool._connections.append(stranded)
                self.assertEqual(stranded.info(), "CONNECTING")
                self.assertEqual(await reap_orphaned_connections(pool), 1)
                self.assertNotIn(stranded, pool.connections)
                response = await client.get(
                    server.url, timeout=httpx.Timeout(5, pool=0.5),
                )
                self.assertEqual(response.status_code, 200)
            finally:
                await client.aclose()

    async def test_client_settings_are_preserved(self):
        client = cancellation_safe_client(timeout=600, trust_env=False)
        try:
            self.assertIsInstance(client._transport, CancellationSafeTransport)
            self.assertFalse(client.trust_env)
            self.assertEqual(client.timeout, httpx.Timeout(600))
        finally:
            await client.aclose()

    async def test_transport_options_reach_the_pool(self):
        client = cancellation_safe_client(limits=httpx.Limits(max_connections=3))
        try:
            self.assertEqual(_pool(client)._max_connections, 3)
        finally:
            await client.aclose()

    async def test_environment_proxies_are_still_honoured(self):
        proxy_env = {
            "HTTP_PROXY": "http://proxy.invalid:3128",
            "HTTPS_PROXY": "http://proxy.invalid:3128",
            "NO_PROXY": "",
        }
        with mock.patch.dict(os.environ, proxy_env):
            client = cancellation_safe_client(timeout=5)
            no_env = cancellation_safe_client(timeout=5, trust_env=False)
        try:
            for url in ("http://example.com/", "https://example.com/"):
                transport = client._transport_for_url(httpx.URL(url))
                self.assertIsNot(transport, client._transport)
                self.assertIsInstance(transport, CancellationSafeTransport)
                self.assertIsInstance(transport._pool, httpcore.AsyncHTTPProxy)
                self.assertIs(no_env._transport_for_url(httpx.URL(url)), no_env._transport)
        finally:
            await client.aclose()
            await no_env.aclose()


if __name__ == "__main__":
    unittest.main()
