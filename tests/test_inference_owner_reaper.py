import asyncio
import unittest
from unittest.mock import AsyncMock, Mock

from manager import Manager


def container():
    return {"name": "engine", "deployment_id": "target", "stats_key": "model",
            "load_settings": {"max_concurrency": 1}}


class InferenceOwnerReaperTests(unittest.IsolatedAsyncioTestCase):
    def manager(self, *, local_running=False):
        manager = Manager.__new__(Manager)
        manager.list_containers = AsyncMock(return_value=[{**container(), "status": "running"}] if local_running else [])
        return manager

    async def abandoned(self, manager, *, detach=False, callback=None):
        cancel = asyncio.Event()
        lease = await manager._acquire_inference_slot(container(), "model", cancel)
        rid = manager._track_start("model", streaming=True, admission_target=lease)
        manager._transfer_inference_ownership(lease, rid, cancel=cancel, release_callback=callback)
        if detach:
            manager._transfer_inference_ownership(lease, rid, owner=None)
        return lease, rid, cancel

    async def test_dead_owner_clears_remote_and_running_local_ghosts_and_wakes_queue(self):
        for local_running in (False, True):
            with self.subTest(local_running=local_running):
                manager = self.manager(local_running=local_running)
                lease, rid, cancel = await asyncio.create_task(self.abandoned(manager))
                queued = asyncio.create_task(manager._acquire_inference_slot(container(), "model", None))
                await asyncio.sleep(0)
                self.assertFalse(queued.done())
                self.assertEqual(await manager.reap_stale_admission_targets(), ["target"])
                replacement = await queued
                manager._transfer_inference_ownership(replacement)
                self.assertNotIn(rid, manager._active_reqs)
                self.assertTrue(cancel.is_set())
                self.assertEqual(manager.inference_admission()["target"]["running"], 1)
                # The old generator's eventual finally must not release its successor.
                manager._release_inference_slot(lease)
                self.assertEqual(manager.inference_admission()["target"]["running"], 1)
                manager._release_inference_slot(replacement)

    async def test_live_prefill_is_not_reaped_for_age_or_zero_output(self):
        manager = self.manager(local_running=True)
        ready, finish = asyncio.Event(), asyncio.Event()
        async def prefill():
            lease, rid, _ = await self.abandoned(manager)
            manager._active_reqs[rid]["started_at"] = -100000000
            ready.set()
            await finish.wait()
            manager._track_end(rid)
            manager._release_inference_slot(lease)
        task = asyncio.create_task(prefill())
        await ready.wait()
        try:
            self.assertEqual(await manager.reap_stale_admission_targets(), [])
            self.assertEqual(manager.active_requests()["model"]["connections"], 1)
        finally:
            finish.set()
            await task

    async def test_detached_prepared_stream_survives_endpoint_exit_until_cancel(self):
        manager = self.manager()
        release = Mock()
        lease, rid, cancel = await asyncio.create_task(self.abandoned(manager, detach=True, callback=release))
        self.assertEqual(await manager.reap_stale_admission_targets(), [])
        self.assertIn(rid, manager._active_reqs)
        release.assert_not_called()
        cancel.set()
        self.assertEqual(await manager.reap_stale_admission_targets(), ["target"])
        release.assert_called_once()
        self.assertEqual(manager._active_reqs, {})
        self.assertEqual(manager.inference_admission(), {})
        self.assertEqual(await manager.reap_stale_admission_targets(), [])
        manager._release_inference_slot(lease)
        release.assert_called_once()

    async def test_transfer_preserves_callback_and_moves_to_live_consumer(self):
        manager = self.manager()
        release = Mock()
        lease, rid, _ = await asyncio.create_task(self.abandoned(manager, detach=True, callback=release))
        manager._transfer_inference_ownership(lease, rid)
        self.assertEqual(await manager.reap_stale_admission_targets(), [])
        self.assertIs(manager._active_reqs[rid]["release_callback"], release)
        self.assertIs(manager._active_reqs[rid]["owner_task"], asyncio.current_task())
        manager._track_end(rid)
        manager._release_inference_slot(lease)

    async def test_reset_generation_and_duplicate_cleanup_do_not_release_new_grant(self):
        manager = self.manager()
        old = await manager._acquire_inference_slot(container(), "model", None)
        manager.reset_inference_admission()
        replacement = await manager._acquire_inference_slot(container(), "model", None)
        manager._release_inference_slot(old)
        manager._release_inference_slot(old)
        self.assertEqual(manager.inference_admission()["target"]["running"], 1)
        manager._release_inference_slot(replacement)
        manager._release_inference_slot(replacement)
        self.assertEqual(manager.inference_admission(), {})

    async def test_explicit_disconnect_reaps_even_if_owner_task_remains_alive(self):
        manager = self.manager(local_running=True)
        lease, rid, cancel = await self.abandoned(manager)
        self.assertFalse(asyncio.current_task().done())
        cancel.set()
        self.assertEqual(await manager.reap_stale_admission_targets(), ["target"])
        self.assertNotIn(rid, manager._active_reqs)
        self.assertEqual(manager.inference_admission(), {})
        manager._release_inference_slot(lease)

    async def test_transfer_attaches_admission_to_existing_record(self):
        manager = self.manager()
        lease = await manager._acquire_inference_slot(container(), "model", None)
        rid = manager._track_start("model", streaming=True)
        manager._transfer_inference_ownership(lease, rid)
        self.assertIs(manager._active_reqs[rid]["admission_target"], lease)
        manager._track_end(rid)
        manager._release_inference_slot(lease)

    async def test_dead_unlimited_request_owner_is_removed_without_admission(self):
        manager = self.manager()
        async def abandoned():
            return manager._track_start("model", streaming=True)
        rid = await asyncio.create_task(abandoned())
        self.assertEqual(await manager.reap_stale_admission_targets(), [])
        self.assertNotIn(rid, manager._active_reqs)
