import asyncio
import copy
import unittest
from unittest.mock import AsyncMock, Mock

from test_grouped_sharded_mode import four_node_manager


class GroupRecreateTests(unittest.IsolatedAsyncioTestCase):
    def manager(self, *, dirty=False, sibling_running=True):
        manager = four_node_manager()
        manager._save_deployments = Mock()
        manager._resolved_hf_token = Mock(return_value=None)
        manager._validate_available_port = AsyncMock(return_value=8000)
        manager._member_action = AsyncMock(return_value={"ok": True})
        manager._create_member = AsyncMock(return_value={"id": "new", "status": "starting", "port": 8000})
        manager._deployment_environment_drift = AsyncMock(return_value={"old-2": ["changed:NCCL_DEBUG"]})
        members = [{
            "node_id": node, "node_name": node, "instance_id": i // 2,
            "rank": i % 2, "container_name": f"old-{i}", "port": 8000,
            "status": "running" if i < 2 and sibling_running else "stopped",
            "desired_state": "running" if i < 2 and sibling_running else "stopped",
        } for i, node in enumerate(["local", "remote-1", "remote-2", "remote-3"])]
        deployment = {
            "id": "group-test", "name": "Model", "model": "org/model", "engine": "vllm",
            "mode": "grouped_sharded", "instances": 2, "members": members,
            "node_ids": [m["node_id"] for m in members], "api_port": 8000,
            "desired_state": "running" if sibling_running else "stopped",
            "status": "degraded" if sibling_running else "stopped", "settings_dirty": dirty,
            "launch_settings": {
                "model": "org/model", "engine": "vllm", "deployment_mode": "grouped_sharded",
                "tensor_parallel_size": 2, "instances": 2,
                "node_ids": [m["node_id"] for m in members], "extra_args": ["--max-model-len", "400000"],
                "environment": {"NCCL_DEBUG": "WARN"},
            },
        }
        manager.deployments = [deployment]
        return manager, deployment

    async def test_drift_recreates_selected_pair_preserving_identity_and_running_sibling(self):
        manager, deployment = self.manager()
        sibling = copy.deepcopy(deployment["members"][:2])
        launch = copy.deepcopy(deployment["launch_settings"])
        result = await manager.deployment_action("group-test", "start", instance=1)
        self.assertTrue(result["ok"])
        self.assertEqual(deployment["members"][:2], sibling)
        self.assertEqual(deployment["launch_settings"], launch)
        self.assertEqual(len(manager.deployments), 1)
        self.assertEqual(deployment["instances"], 2)
        self.assertEqual([call.args[0]["instance_id"] for call in manager._member_action.await_args_list], [1, 1])
        payloads = [call.args[1] for call in manager._create_member.await_args_list]
        self.assertEqual([call.args[0] for call in manager._create_member.await_args_list], ["remote-2", "remote-3"])
        self.assertEqual([p["name"] for p in payloads], ["old-2", "old-3"])
        self.assertEqual([p["cluster_member"]["instance_id"] for p in payloads], [1, 1])
        self.assertTrue(all(p["environment"]["NCCL_DEBUG"] == "WARN" for p in payloads))
        self.assertTrue(all(p["port"] == 8000 for p in payloads))
        inspected = manager._deployment_environment_drift.await_args.args[0]
        self.assertEqual([m["instance_id"] for m in inspected["members"]], [1, 1])

    async def test_recreate_does_not_start_other_stopped_pair(self):
        manager, deployment = self.manager(sibling_running=False)
        result = await manager.deployment_action("group-test", "start", instance=1)
        self.assertTrue(result["ok"])
        self.assertTrue(all(m["desired_state"] == "stopped" for m in deployment["members"][:2]))
        self.assertTrue(all(m["desired_state"] == "running" for m in deployment["members"][2:]))

    async def test_invalid_selected_topology_rejected_before_removing_containers(self):
        manager, deployment = self.manager(dirty=True)
        deployment["launch_settings"]["tensor_parallel_size"] = 4
        with self.assertRaises(ValueError):
            await manager.deployment_action("group-test", "start", instance=1)
        manager._member_action.assert_not_awaited()
        manager._create_member.assert_not_awaited()

    async def test_failed_creation_cleans_only_selected_group_and_keeps_sibling_running(self):
        manager, deployment = self.manager()
        sibling = copy.deepcopy(deployment["members"][:2])
        manager._create_member.side_effect = [{"id": "new", "status": "starting", "port": 8000}, RuntimeError("agent failed")]
        result = await manager.deployment_action("group-test", "start", instance=1)
        self.assertFalse(result["ok"])
        self.assertEqual(deployment["members"][:2], sibling)
        self.assertTrue(all(call.args[0]["instance_id"] == 1 for call in manager._member_action.await_args_list))
        self.assertTrue(all(m["desired_state"] == "stopped" for m in deployment["members"][2:]))

    async def test_dirty_settings_apply_independently_and_clear_only_after_both_groups(self):
        manager, deployment = self.manager(dirty=True)
        manager._deployment_environment_drift.return_value = None
        await manager.deployment_action("group-test", "start", instance=1)
        self.assertTrue(deployment["settings_dirty"])
        self.assertEqual(manager._create_member.await_count, 2)
        await manager.deployment_action("group-test", "stop", instance=1)
        await manager.deployment_action("group-test", "start", instance=1)
        self.assertEqual(manager._create_member.await_count, 2)
        await manager.deployment_action("group-test", "start", instance=0)
        self.assertFalse(deployment["settings_dirty"])
        self.assertEqual(manager._create_member.await_count, 4)
        deployment["launch_settings"]["environment"]["NCCL_DEBUG"] = "INFO"
        deployment["settings_dirty"] = True
        await manager.deployment_action("group-test", "start", instance=1)
        self.assertTrue(deployment["settings_dirty"])
        self.assertEqual(manager._create_member.await_count, 6)

    async def test_unreachable_sibling_does_not_block_selected_group_preflight(self):
        manager, _ = self.manager()
        original = manager.cluster_nodes
        async def nodes(local_stats=None):
            return [node for node in await original() if node["id"] in {"remote-2", "remote-3"}]
        manager.cluster_nodes = nodes
        self.assertTrue((await manager.deployment_action("group-test", "start", instance=1))["ok"])

    async def test_remove_failure_never_creates_over_unclean_group(self):
        manager, deployment = self.manager()
        sibling = copy.deepcopy(deployment["members"][:2])
        manager._member_action.side_effect = [{"ok": True}, RuntimeError("node disconnected")]
        result = await manager.deployment_action("group-test", "start", instance=1)
        self.assertFalse(result["ok"])
        manager._create_member.assert_not_awaited()
        self.assertEqual(deployment["members"][:2], sibling)
        self.assertTrue(deployment["settings_dirty"])

    async def test_cancelled_creation_cleans_selected_group_and_allows_retry(self):
        manager, deployment = self.manager()
        sibling = copy.deepcopy(deployment["members"][:2])
        entered = asyncio.Event()
        release = asyncio.Event()
        async def block_create(*args):
            entered.set()
            await release.wait()
            return {"id": "finished", "status": "starting", "port": 8000}
        manager._create_member.side_effect = block_create
        task = asyncio.create_task(manager.deployment_action("group-test", "start", instance=1))
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        self.assertEqual(manager._member_action.await_count, 2)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(deployment["members"][:2], sibling)
        self.assertTrue(all(call.args[0]["instance_id"] == 1 for call in manager._member_action.await_args_list))
        self.assertTrue(all(m["desired_state"] == "stopped" for m in deployment["members"][2:]))
        manager._create_member.side_effect = None
        manager._deployment_environment_drift.return_value = None
        self.assertTrue((await manager.deployment_action("group-test", "start", instance=1))["ok"])

    async def test_persisted_interrupted_group_recreates_without_drift_report(self):
        manager, deployment = self.manager()
        manager._deployment_environment_drift.return_value = None
        for member in deployment["members"][2:]:
            member["recreate_pending"] = True
            member["status"] = "queued"
        self.assertTrue((await manager.deployment_action("group-test", "start", instance=1))["ok"])
        self.assertEqual(manager._create_member.await_count, 2)
        self.assertFalse(any(m.get("recreate_pending") for m in deployment["members"][2:]))

    async def test_sglang_recreates_one_tp_group_with_same_rendezvous_identity(self):
        manager, deployment = self.manager(dirty=True)
        deployment["engine"] = deployment["launch_settings"]["engine"] = "sglang"
        deployment["launch_settings"]["sg_tp_size"] = 2
        deployment["launch_settings"]["environment"] = {}
        deployment["launch_settings"]["extra_args"] = ["--context-length", "400000"]
        self.assertTrue((await manager.deployment_action("group-test", "start", instance=1))["ok"])
        payloads = [call.args[1] for call in manager._create_member.await_args_list]
        self.assertTrue(all(p["sg_tp_size"] == 2 for p in payloads))
        self.assertTrue(all(p["extra_args"][p["extra_args"].index("--dist-init-addr") + 1] == "169.254.10.3:29502" for p in payloads))

    async def test_pending_group_keeps_local_port_reserved_without_any_containers(self):
        manager, deployment = self.manager(sibling_running=False)
        manager.client = Mock()
        manager.client.containers.list.return_value = []
        deployment["members"][0]["recreate_pending"] = True
        self.assertIn(8000, await manager._used_host_ports())
        self.assertNotIn(8000, await manager._used_host_ports(exclude_deployment_id="group-test"))

    async def test_changed_saved_port_rejected_before_recording_settings_as_applied(self):
        manager, deployment = self.manager(dirty=True)
        deployment["launch_settings"]["port"] = 9000
        with self.assertRaises(ValueError):
            await manager.deployment_action("group-test", "start", instance=0)
        manager._member_action.assert_not_awaited()
        self.assertTrue(deployment["settings_dirty"])
