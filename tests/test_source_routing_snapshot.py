import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from cluster import NodeAgentResponseError
from manager import Manager, SourceRoutingUnavailable
from sparkdeck.service import SparkDeckService


class RoutingSnapshotTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stored = {"id": "saved", "alias": "model", "runtime": "vllm", "kind": "managed",
                       "model": {"repository": "org/model"}, "settings": {"manager_deployment_id": "runtime"}}
        self.rule = {"deployment_id": "saved", "instance_id": 0, "node_ids": ["local", "worker"]}
        self.manager = Manager.__new__(Manager)
        self.manager.deployments = [{
            "id": "runtime", "sparkdeck_record_id": "saved", "model": "org/model",
            "mode": "grouped_sharded", "desired_state": "running",
            "members": [
                {"node_id": "local", "container_name": "coordinator", "rank": 0, "instance_id": 0},
                {"node_id": "worker", "container_name": "rank-1", "rank": 1, "instance_id": 0},
                {"node_id": "offline", "container_name": "other-group", "rank": 0, "instance_id": 1},
            ],
        }]
        self.manager.get_state = AsyncMock(side_effect=AssertionError("full inventory forbidden"))
        self.manager._container_by_name = AsyncMock(return_value={"name": "coordinator", "status": "running"})
        self.manager._check_ready = AsyncMock(return_value=True)

        async def request(node, method, path, **kwargs):
            self.assertEqual(node, "worker", "offline sibling must not be contacted")
            return {"name": "rank-1", "status": "running", "ready": None}

        self.manager.node_registry = SimpleNamespace(request=AsyncMock(side_effect=request))
        self.service = SparkDeckService.__new__(SparkDeckService)
        self.service.manager = self.manager
        self.service.store = SimpleNamespace(deployment=Mock(side_effect=lambda *a, **k: self.stored))
        self.service.deployments = AsyncMock(side_effect=AssertionError("full deployment inventory forbidden"))

    async def snapshot(self):
        return await self.service._source_routing_snapshot(self.stored, self.rule)

    async def test_concurrent_requests_share_only_selected_group_health_and_warm_cache(self):
        snapshots = await asyncio.gather(*(self.snapshot() for _ in range(20)))
        self.assertTrue(all(rows[0]["instances"][0]["status"] == "running" for rows in snapshots))
        self.manager._container_by_name.assert_awaited_once_with("coordinator")
        self.manager.node_registry.request.assert_awaited_once_with(
            "worker", "GET", "/api/agent/containers/rank-1/state?check_ready=false", timeout=8,
        )
        await self.snapshot()
        self.assertEqual(self.manager._container_by_name.await_count, 1)
        self.manager.get_state.assert_not_awaited()
        self.service.deployments.assert_not_awaited()
        snapshots[0][0]["alias"] = "mutated"
        self.assertEqual((await self.snapshot())[0]["alias"], "model")

    async def test_expired_success_never_masks_failed_refresh_and_failure_is_cached(self):
        await self.snapshot()
        key = next(iter(self.service._source_routing_cache))
        _, rows, signature = self.service._source_routing_cache[key]
        self.service._source_routing_cache[key] = (0, rows, signature)
        self.manager.node_registry.request.side_effect = RuntimeError("worker offline")
        for _ in range(2):
            with self.assertRaises(SourceRoutingUnavailable):
                await self.snapshot()
        self.assertEqual(self.manager.node_registry.request.await_count, 2)

    async def test_alias_and_stop_intent_changes_invalidate_warm_snapshot(self):
        await self.snapshot()
        self.stored = {**self.stored, "alias": "renamed"}
        self.assertEqual((await self.snapshot())[0]["alias"], "renamed")
        self.assertEqual(self.manager._container_by_name.await_count, 2)
        self.manager.deployments[0]["members"][0]["desired_state"] = "stopped"
        with self.assertRaises(SourceRoutingUnavailable):
            await self.snapshot()
        self.assertEqual(self.manager._container_by_name.await_count, 2)

    async def test_cancelled_caller_does_not_cancel_shared_health_refresh(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def lookup(name):
            entered.set()
            await release.wait()
            return {"name": name, "status": "running"}

        self.manager._container_by_name.side_effect = lookup
        first = asyncio.create_task(self.snapshot())
        await entered.wait()
        second = asyncio.create_task(self.snapshot())
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        release.set()
        self.assertEqual((await second)[0]["status"], "running")
        self.assertEqual(self.manager._container_by_name.await_count, 1)
        self.assertEqual(self.service._source_routing_refresh_tasks, {})

    async def test_legacy_agent_fallback_is_selected_node_only_and_only_on_404(self):
        self.manager.node_registry.request.side_effect = [
            NodeAgentResponseError("worker", 404, "endpoint missing"),
            {"docker_ready": True, "containers": [{"name": "rank-1", "status": "running"}]},
        ]
        self.assertEqual((await self.snapshot())[0]["status"], "running")
        self.assertEqual([call.args[0] for call in self.manager.node_registry.request.await_args_list], ["worker", "worker"])
        self.assertEqual(self.manager.node_registry.request.await_args_list[-1].args[2], "/api/agent/status")

    async def test_config_change_during_probe_is_discarded(self):
        async def ready(*args, **kwargs):
            self.stored = {**self.stored, "alias": "changed-during-probe"}
            return True

        self.manager._check_ready.side_effect = ready
        with self.assertRaises(SourceRoutingUnavailable):
            await self.snapshot()

    async def test_healthy_group_ignores_stale_parent_status_and_same_node_sibling(self):
        self.manager.deployments[0]["status"] = "stopped"
        self.manager.deployments[0]["members"].append(
            {"node_id": "local", "container_name": "stopped-sibling", "rank": 0,
             "instance_id": 1, "desired_state": "stopped"},
        )
        result = await self.snapshot()
        self.assertEqual(result[0]["instances"][0]["status"], "running")
        self.manager._container_by_name.assert_awaited_once_with("coordinator")

    async def test_model_change_and_moved_fingerprint_cannot_reuse_warm_cache(self):
        await self.snapshot()
        self.manager.deployments[0]["model"] = "org/replaced"
        await self.snapshot()
        self.assertEqual(self.manager._container_by_name.await_count, 2)
        self.manager.deployments[0]["members"][1]["node_id"] = "moved"
        with self.assertRaises(SourceRoutingUnavailable):
            await self.snapshot()
        self.assertEqual(self.manager._container_by_name.await_count, 2)

    async def test_failed_rank_cancels_other_pending_probes(self):
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def local(name):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def remote(*args, **kwargs):
            await started.wait()
            raise RuntimeError("selected worker failed")

        self.manager._container_by_name.side_effect = local
        self.manager.node_registry.request.side_effect = remote
        with self.assertRaises(SourceRoutingUnavailable):
            await self.snapshot()
        self.assertTrue(cancelled.is_set())
        self.assertEqual(self.service._source_routing_refresh_tasks, {})
