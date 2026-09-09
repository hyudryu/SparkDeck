import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from manager import ClientAbort
from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import SparkDeckService


class ServicePromptWaiterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.service = SparkDeckService(SimpleNamespace(http=None), Path(self.directory.name))
        self.addAsyncCleanup(self.service.close)
        self.service.store.add_deployment(Deployment(
            id="target", alias="target", runtime=RuntimeKind.LLAMA_CPP,
            kind=DeploymentKind.EXTERNAL, model=ModelIdentity("org/model"),
        ), base_url="http://localhost:8000")
        self.deployment = self.service.store.deployment("target", include_private=True)
        self.service._proxy_registered_unlimited = AsyncMock(return_value={"ok": True})

    async def queued(self):
        gate = self.service.prompt_gate.get(("deployment", "target"))
        lease = await gate.acquire()
        self.addCleanup(lease.release)
        cancel = asyncio.Event()
        task = asyncio.create_task(self.service._proxy_registered(
            self.deployment, {"model": "target"}, "completions", cancel,
        ))
        self.addCleanup(task.cancel)
        async def wait():
            while not gate._waiting:
                await asyncio.sleep(0)
        await asyncio.wait_for(wait(), 2)
        return task, lease, cancel

    async def test_deleted_deployment_never_dispatches_after_wait(self):
        task, lease, _ = await self.queued()
        self.service.store.delete_deployment("target")
        lease.release()
        with self.assertRaisesRegex(LookupError, "changed"):
            await asyncio.wait_for(task, 2)
        self.service._proxy_registered_unlimited.assert_not_awaited()
        self.assertEqual(self.service.manager._prompt_waiting_requests, {})

    async def test_changed_endpoint_never_dispatches_after_wait(self):
        task, lease, _ = await self.queued()
        self.service.store._connection.execute(
            "UPDATE deployments SET base_url = ? WHERE id = ?", ("http://localhost:9000", "target")
        )
        lease.release()
        with self.assertRaisesRegex(LookupError, "changed"):
            await asyncio.wait_for(task, 2)
        self.service._proxy_registered_unlimited.assert_not_awaited()

    async def test_waiter_counter_is_removed_on_cancel(self):
        task, _, cancel = await self.queued()
        waiters = self.service.manager._prompt_waiting_requests
        self.assertEqual(len(waiters), 1)
        self.assertEqual(next(iter(waiters.values()))["group"]["group_id"], "target")
        cancel.set()
        with self.assertRaises(ClientAbort):
            await asyncio.wait_for(task, 2)
        self.assertEqual(waiters, {})
        self.service._proxy_registered_unlimited.assert_not_awaited()

    async def test_waiter_counter_is_removed_before_dispatch(self):
        task, lease, _ = await self.queued()
        async def dispatch(*args, **kwargs):
            self.assertEqual(self.service.manager._prompt_waiting_requests, {})
            return {"ok": True}
        self.service._proxy_registered_unlimited.side_effect = dispatch
        lease.release()
        self.assertEqual(await asyncio.wait_for(task, 2), {"ok": True})

    async def test_removed_discovered_container_never_dispatches_after_wait(self):
        live = {"name": "live", "id": "original", "runtime": "vllm",
                "status": "running", "model": "org/model", "port": 8000}
        self.service.manager.list_containers = AsyncMock(return_value=[live])
        deployment = self.service._discovered_deployment(live, "vllm", "org/model")
        gate = self.service.prompt_gate.get(("deployment", "container:live"))
        lease = await gate.acquire()
        self.addCleanup(lease.release)
        task = asyncio.create_task(self.service._proxy_registered(
            deployment, {"model": "org/model"}, "completions", None,
        ))
        self.addCleanup(task.cancel)
        async def wait():
            while not gate._waiting:
                await asyncio.sleep(0)
        await asyncio.wait_for(wait(), 2)
        self.service.manager.list_containers.return_value = []
        lease.release()
        with self.assertRaises(LookupError):
            await asyncio.wait_for(task, 2)
        self.service._proxy_registered_unlimited.assert_not_awaited()
        self.assertEqual(self.service.manager._prompt_waiting_requests, {})

    async def test_legacy_waiter_is_visible_and_task_cancel_cleans_it(self):
        gate = self.service.prompt_gate.get(("legacy", "org/model"))
        lease = await gate.acquire()
        self.addCleanup(lease.release)
        upstream = AsyncMock()
        task = asyncio.create_task(self.service._run_service_prompt_gate(
            ("legacy", "org/model"), upstream, cancel=None, model="org/model",
        ))
        self.addCleanup(task.cancel)
        async def wait():
            while not gate._waiting:
                await asyncio.sleep(0)
        await asyncio.wait_for(wait(), 2)
        self.assertEqual(len(self.service.manager._prompt_waiting_requests), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.service.manager._prompt_waiting_requests, {})
        upstream.assert_not_awaited()

    async def test_replaced_record_never_dispatches_after_wait(self):
        task, lease, _ = await self.queued()
        self.service.store.delete_deployment("target")
        self.service.store.add_deployment(Deployment(
            id="target", alias="target", runtime=RuntimeKind.LLAMA_CPP,
            kind=DeploymentKind.EXTERNAL, model=ModelIdentity("org/model"),
        ), base_url="http://localhost:8000")
        lease.release()
        with self.assertRaisesRegex(LookupError, "changed"):
            await asyncio.wait_for(task, 2)
        self.service._proxy_registered_unlimited.assert_not_awaited()

    async def test_changed_credentials_never_dispatch_after_wait(self):
        from unittest.mock import Mock
        self.service._get_credential = Mock(return_value="original")
        task, lease, _ = await self.queued()
        self.service._get_credential.return_value = "replacement"
        lease.release()
        with self.assertRaisesRegex(LookupError, "credentials changed"):
            await asyncio.wait_for(task, 2)
        self.service._proxy_registered_unlimited.assert_not_awaited()

    async def test_canceled_queued_proxy_does_not_contaminate_active_startup(self):
        self.service._live_deployment_for_model_id = AsyncMock(return_value=self.deployment)
        scopes = self.service._community_observation_scopes(self.deployment, "target")
        startup = self.service._community_observation_start(scopes, deferred=True)
        startup["startup_benchmark"] = True
        self.service._community_observation_activate(startup)
        self.addCleanup(self.service._community_observation_end, startup)
        gate = self.service.prompt_gate.get(("deployment", "target"))
        lease = await gate.acquire()
        self.addCleanup(lease.release)
        cancel = asyncio.Event()
        task = asyncio.create_task(self.service.proxy(
            {"model": "target"}, "completions", cancel,
        ))
        self.addCleanup(task.cancel)
        async def wait():
            while not gate._waiting:
                await asyncio.sleep(0)
        await asyncio.wait_for(wait(), 2)
        self.assertFalse(startup["contaminated"])
        self.assertEqual(list(self.service._community_active_observations), [startup["id"]])
        cancel.set()
        with self.assertRaises(ClientAbort):
            await asyncio.wait_for(task, 2)
        self.assertFalse(startup["contaminated"])
        self.assertEqual(list(self.service._community_active_observations), [startup["id"]])
        self.service._proxy_registered_unlimited.assert_not_awaited()
        self.assertEqual(self.service.manager._prompt_waiting_requests, {})

    async def test_manager_admission_reports_standalone_service_waiter(self):
        from manager import Manager
        manager = object.__new__(Manager)
        manager.http = None
        manager.deployments = []
        self.service.manager = manager
        group = manager._request_group("target", "target")
        manager._active_reqs = {
            1: {"key": "target", "group": group, "paused": False},
            2: {"key": "target", "group": group, "paused": False},
            3: {"key": "target", "group": group, "paused": False},
        }
        task, _, cancel = await self.queued()
        snapshots = list(manager.inference_admission().values())
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["deployment_id"], "target")
        self.assertEqual(snapshots[0]["running"], 3)
        self.assertEqual(snapshots[0]["queued"], 1)
        cancel.set()
        with self.assertRaises(ClientAbort):
            await asyncio.wait_for(task, 2)
        self.assertFalse(any(snapshot["queued"] for snapshot in manager.inference_admission().values()))
