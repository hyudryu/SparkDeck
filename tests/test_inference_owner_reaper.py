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


    async def test_prepared_local_transport_closes_before_queued_request_is_admitted(self):
        from types import SimpleNamespace
        import httpx
        manager = self.manager(local_running=True)
        close_started, allow_close = asyncio.Event(), asyncio.Event()

        class BlockingTransport(httpx.AsyncByteStream):
            async def aclose(self):
                close_started.set()
                await allow_close.wait()

        transport = BlockingTransport()
        transport.aclose = AsyncMock(wraps=transport.aclose)
        response = httpx.Response(200, stream=transport)

        async def exit_context(*args):
            await response.aclose()

        context = SimpleNamespace(__aenter__=AsyncMock(return_value=response),
                                  __aexit__=AsyncMock(side_effect=exit_context))
        manager.http = SimpleNamespace(stream=Mock(return_value=context))
        target = {**container(), "port": 8000, "model": "model"}
        cancel = asyncio.Event()
        lease = await manager._acquire_inference_slot(target, "model", cancel)
        manager._admit_vllm_target = AsyncMock(return_value=(target, lease))
        stream = manager._vllm_stream("http://localhost/v1/chat/completions",
                                     {"model": "model", "stream": True}, "model",
                                     cancel, target)
        await stream.prepare()
        rec = next(iter(manager._active_reqs.values()))
        self.assertIsNone(rec["owner_task"])
        queued = asyncio.create_task(manager._acquire_inference_slot(target, "model", None))
        cancel.set()
        reaping = asyncio.create_task(manager.reap_stale_admission_targets())
        await asyncio.wait_for(close_started.wait(), 1)
        closing = asyncio.create_task(stream.aclose())
        try:
            # HTTPX marks the response closed before transport shutdown ends.
            # A second closer must wait for that original shutdown anyway.
            self.assertTrue(response.is_closed)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertFalse(closing.done())
            self.assertFalse(queued.done())
            self.assertFalse(lease.released)
            self.assertEqual(manager.inference_admission()["target"]["running"], 1)
        finally:
            allow_close.set()
            await closing
        self.assertEqual(await reaping, ["target"])
        replacement = await queued
        transport.aclose.assert_awaited_once()
        self.assertFalse(manager._active_reqs)
        await stream.aclose()
        self.assertFalse(replacement.released)
        manager._release_inference_slot(replacement)

    async def test_new_active_owner_during_reaper_cleanup_is_not_an_orphan(self):
        from types import SimpleNamespace
        manager = self.manager(local_running=True)
        close_started, allow_close = asyncio.Event(), asyncio.Event()

        async def close():
            close_started.set()
            await allow_close.wait()

        abandoned_lease, rid, cancel = await self.abandoned(manager, detach=True)
        manager._transfer_inference_ownership(
            abandoned_lease, rid, owner=None,
            cleanup_stream=SimpleNamespace(aclose=AsyncMock(side_effect=close)),
        )
        cancel.set()
        reaping = asyncio.create_task(manager.reap_stale_admission_targets())
        await asyncio.wait_for(close_started.wait(), 1)
        # This request was not present when the reaper took its snapshot.
        target = {**container(), "name": "second-engine", "deployment_id": "second-target"}
        active_cancel = asyncio.Event()
        active_lease = await manager._acquire_inference_slot(target, "model", active_cancel)
        active_rid = manager._track_start("model", streaming=True, admission_target=active_lease)
        active_response = SimpleNamespace(aclose=AsyncMock())
        manager._transfer_inference_ownership(
            active_lease, active_rid, cancel=active_cancel, cleanup_stream=active_response,
        )
        active_cancel.set()
        allow_close.set()
        try:
            self.assertEqual(await reaping, ["target"])
            self.assertFalse(active_lease.released)
            self.assertIn(active_rid, manager._active_reqs)
            active_response.aclose.assert_not_awaited()
        finally:
            manager._track_end(active_rid)
            manager._release_inference_slot(active_lease)

    async def test_reaper_leaves_active_transport_cleanup_to_consumer(self):
        from types import SimpleNamespace
        manager = self.manager(local_running=True)
        lease, rid, cancel = await self.abandoned(manager)
        response = SimpleNamespace(aclose=AsyncMock())
        manager._transfer_inference_ownership(lease, rid, cleanup_stream=response)
        cancel.set()
        self.assertEqual(await manager.reap_stale_admission_targets(), [])
        response.aclose.assert_not_awaited()
        self.assertFalse(lease.released)
        manager._track_end(rid)
        manager._release_inference_slot(lease)

    async def test_abandoned_transport_close_failure_still_releases_reservation(self):
        from types import SimpleNamespace
        manager = self.manager()
        release = Mock()
        lease, rid, cancel = await self.abandoned(manager, detach=True, callback=release)
        response = SimpleNamespace(aclose=AsyncMock(side_effect=RuntimeError("close failed")))
        manager._transfer_inference_ownership(lease, rid, owner=None, cleanup_stream=response)
        cancel.set()
        with self.assertLogs("manager", level="ERROR"):
            self.assertEqual(await manager.reap_stale_admission_targets(), ["target"])
        response.aclose.assert_awaited_once()
        release.assert_called_once()
        self.assertTrue(lease.released)
        self.assertFalse(manager._active_reqs)
