import asyncio
import unittest
from unittest.mock import AsyncMock

from manager import ClientAbort
from test_cluster_load_balancing import build_manager, replicated_deployment, member_loads


class NonstreamCleanupOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnected_remote_request_keeps_capacity_until_transport_unwinds(self):
        deployment = replicated_deployment()
        deployment["launch_settings"]["extra_args"] = ["--max-num-seqs", "1"]
        manager = build_manager(deployment)
        del manager._acquire_inference_slot
        del manager._release_inference_slot
        manager.list_containers = AsyncMock(return_value=[])
        opened, closing, finish_close = asyncio.Event(), asyncio.Event(), asyncio.Event()
        cancel = asyncio.Event()

        async def request(*args, **kwargs):
            opened.set()
            try:
                await asyncio.Event().wait()
            finally:
                closing.set()
                await finish_close.wait()

        manager.node_registry.request = request
        owner = asyncio.create_task(manager._proxy_cluster_member(
            deployment, deployment["members"][0], "org/model",
            {"model": "org/model", "stream": False}, "chat/completions", cancel,
        ))
        await asyncio.wait_for(opened.wait(), 1)
        rec = next(iter(manager._active_reqs.values()))
        lease = rec["admission_target"]
        target = {"name": "repl-1-r0", "stats_key": "org/model",
                  "load_settings": {"max_concurrency": 1}}
        queued = asyncio.create_task(manager._acquire_inference_slot(target, "org/model", None))
        cancel.set()
        await asyncio.wait_for(closing.wait(), 1)
        try:
            self.assertFalse(owner.done())
            self.assertEqual(await manager.reap_stale_admission_targets(), [])
            self.assertFalse(lease.released)
            self.assertFalse(queued.done())
            self.assertEqual(member_loads(manager, deployment), [1, 0])
        finally:
            finish_close.set()
        with self.assertRaises(ClientAbort):
            await owner
        replacement = await asyncio.wait_for(queued, 1)
        self.assertTrue(lease.released)
        self.assertEqual(member_loads(manager, deployment), [0, 0])
        self.assertFalse(manager._active_reqs)
        manager._release_inference_slot(replacement)
