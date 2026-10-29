"""Node-probe caching regressions.

A registered node that is offline and black-holes TCP (a Tailscale peer that
went away) used to burn its full connect timeout on *every* request that
fanned out through ``Manager.get_state()``: the unreachable snapshot carried
no ``docker_ready`` key, so ``probe(details=True)`` refused to reuse it and
re-dialed forever. These tests lock in the negative cache, the background
refresh, and the shared in-flight dial.
"""

import asyncio
import tempfile
import time
import unittest
from pathlib import Path

import httpx

from cluster import (
    AGENT_PROTOCOL_VERSION,
    NODE_NEGATIVE_CACHE_SERVE_SECONDS,
    NodeRegistry,
)
from manager import Manager


def _node() -> dict:
    return {
        "id": "remote-1", "name": "Spark 2",
        "agent_url": "http://spark-2:7878", "agent_token": "secret",
        "enabled": True,
    }


def _healthy_status() -> dict:
    return {
        "node_id": "remote-1",
        "protocol_version": AGENT_PROTOCOL_VERSION,
        "docker_ready": True,
        "status_message": None,
        "fabric_ready": False,
        "interfaces": [],
        "stats": {},
        "containers": [],
    }


class NodeProbeCacheTests(unittest.IsolatedAsyncioTestCase):
    def _registry(self, directory: str) -> NodeRegistry:
        return NodeRegistry(
            Path(directory), httpx.AsyncClient(), "controller",
        )

    async def test_unreachable_snapshot_is_reused_without_reprobing(self) -> None:
        requests = []

        async def request(node_id, method, path, **_kwargs):
            requests.append(path)
            raise httpx.ConnectTimeout("timed out")

        with tempfile.TemporaryDirectory() as directory:
            registry = self._registry(directory)
            registry.request = request
            node = _node()

            first = await registry.probe(node, details=True)
            self.assertEqual(first["status"], "unreachable")
            self.assertFalse(first["online"])
            after_first = len(requests)
            self.assertGreater(after_first, 0)

            # details=True on an unreachable snapshot used to bypass the
            # cache because the entry lacks docker_ready and re-dial every
            # time. It must now be served from the cache instead.
            second = await registry.probe(node, details=True)
            self.assertEqual(second["status"], "unreachable")
            self.assertEqual(len(requests), after_first)

            # An explicit force still re-dials.
            third = await registry.probe(node, force=True, details=True)
            self.assertEqual(third["status"], "unreachable")
            self.assertGreater(len(requests), after_first)
            await registry.http.aclose()

    async def test_stale_unreachable_snapshot_serves_cache_while_refreshing(self) -> None:
        requests = []
        block: asyncio.Event = asyncio.Event()
        recovered = False

        async def request(node_id, method, path, **_kwargs):
            nonlocal recovered
            requests.append(path)
            await block.wait()
            if not recovered:
                raise httpx.ConnectTimeout("timed out")
            if path == "/api/agent/health":
                return {"protocol_version": AGENT_PROTOCOL_VERSION}
            return _healthy_status()

        with tempfile.TemporaryDirectory() as directory:
            registry = self._registry(directory)
            registry.request = request
            node = _node()
            stale_at = time.monotonic() - (NODE_NEGATIVE_CACHE_SERVE_SECONDS - 5)
            registry._status_cache["remote-1"] = (
                stale_at,
                {**registry.public_config(node), "status": "unreachable",
                 "online": False, "status_message": "timed out"},
            )

            # Expired negative snapshot: the caller gets it immediately even
            # though the background refresh is stuck on the connect timeout.
            deadline = time.monotonic() + 2.0
            result = await asyncio.wait_for(
                registry.probe(node, details=True), timeout=1.0,
            )
            self.assertLess(time.monotonic(), deadline)
            self.assertEqual(result["status"], "unreachable")
            await asyncio.sleep(0)  # let the background dial start
            self.assertEqual(len(requests), 1)

            # Release the blocked dial and let the background refresh finish.
            pending = registry._probe_inflight["remote-1"]["task"]
            recovered = True
            block.set()
            await asyncio.wait_for(pending, timeout=2.0)
            refreshed = registry._status_cache["remote-1"][1]
            self.assertEqual(refreshed["status"], "online")
            self.assertTrue(refreshed["online"])
            await registry.http.aclose()

    async def test_concurrent_probes_share_one_dial(self) -> None:
        requests = []
        gate: asyncio.Event = asyncio.Event()

        async def request(node_id, method, path, **_kwargs):
            requests.append(path)
            await gate.wait()
            if path == "/api/agent/health":
                return {"protocol_version": AGENT_PROTOCOL_VERSION}
            return _healthy_status()

        with tempfile.TemporaryDirectory() as directory:
            registry = self._registry(directory)
            registry.request = request
            node = _node()

            rich = asyncio.create_task(registry.probe(node, details=True))
            lean = asyncio.create_task(registry.probe(node, details=False))
            # Give both a chance to reach the shared in-flight entry.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            gate.set()
            first, second = await asyncio.gather(rich, lean)

            self.assertEqual(first["status"], "online")
            # The liveness-only caller must be allowed to share the richer
            # details dial, not stack a second one behind it.
            self.assertEqual(requests.count("/api/agent/health"), 1)
            self.assertTrue(second["online"])
            await registry.http.aclose()

    async def test_background_refresh_is_deduplicated_per_node(self) -> None:
        requests = []
        block: asyncio.Event = asyncio.Event()

        async def request(node_id, method, path, **_kwargs):
            requests.append(path)
            await block.wait()
            raise httpx.ConnectTimeout("timed out")

        with tempfile.TemporaryDirectory() as directory:
            registry = self._registry(directory)
            registry.request = request
            node = _node()
            stale_at = time.monotonic() - (NODE_NEGATIVE_CACHE_SERVE_SECONDS - 5)
            registry._status_cache["remote-1"] = (
                stale_at,
                {**registry.public_config(node), "status": "unreachable",
                 "online": False},
            )

            for _ in range(5):
                result = await asyncio.wait_for(
                    registry.probe(node, details=True), timeout=1.0,
                )
                self.assertEqual(result["status"], "unreachable")
            await asyncio.sleep(0)  # let the single shared dial start
            self.assertEqual(len(requests), 1)

            pending = registry._probe_inflight["remote-1"]["task"]
            block.set()
            await asyncio.wait_for(pending, timeout=2.0)
            await registry.http.aclose()


class CachedStateTests(unittest.IsolatedAsyncioTestCase):
    class _FakeManager(Manager):
        def __init__(self):  # deliberately skips Manager.__init__
            self.calls = 0
            self.fail = False
            self.gate: asyncio.Event | None = None

        async def get_state(self) -> dict:
            self.calls += 1
            if self.gate is not None:
                await self.gate.wait()
            if self.fail:
                raise RuntimeError("inventory exploded")
            return {"snapshot": self.calls}

    async def test_concurrent_callers_share_one_snapshot(self) -> None:
        manager = self._FakeManager()
        manager.gate = asyncio.Event()
        first = asyncio.create_task(manager.cached_state())
        second = asyncio.create_task(manager.cached_state())
        await asyncio.sleep(0)
        manager.gate.set()
        results = await asyncio.gather(first, second)
        self.assertEqual(manager.calls, 1)
        self.assertEqual(results[0], results[1])

        # Repeat calls inside the freshness window reuse the snapshot.
        third = await manager.cached_state()
        self.assertEqual(third, results[0])
        self.assertEqual(manager.calls, 1)

    async def test_stale_snapshot_served_while_refreshing(self) -> None:
        manager = self._FakeManager()
        await manager.cached_state()  # seeds the cache with snapshot 1
        manager._state_cache = (time.monotonic() - 5, {"snapshot": "cached"})
        manager.gate = asyncio.Event()

        result = await asyncio.wait_for(manager.cached_state(), timeout=1.0)
        self.assertEqual(result, {"snapshot": "cached"})  # caller never blocks
        await asyncio.sleep(0)  # let the background refresh reach the gate
        self.assertEqual(manager.calls, 2)

        pending = manager._state_inflight
        manager.gate.set()
        await asyncio.wait_for(pending, timeout=1.0)
        fresh = await manager.cached_state()
        self.assertEqual(fresh, {"snapshot": 2})

    async def test_expired_snapshot_refetches(self) -> None:
        manager = self._FakeManager()
        await manager.cached_state()
        self.assertEqual(manager.calls, 1)
        manager._state_cache = (time.monotonic() - 30, {"snapshot": "stale"})
        result = await manager.cached_state(ttl_seconds=1.5)
        self.assertEqual(result, {"snapshot": 2})
        self.assertEqual(manager.calls, 2)

    async def test_failed_snapshot_is_not_cached(self) -> None:
        manager = self._FakeManager()
        manager.fail = True
        with self.assertRaises(RuntimeError):
            await manager.cached_state()
        self.assertIsNone(manager._state_cache)
        self.assertIsNone(manager._state_inflight)
        manager.fail = False
        result = await manager.cached_state()
        self.assertEqual(result, {"snapshot": 2})
        self.assertEqual(manager.calls, 2)


if __name__ == "__main__":
    unittest.main()
