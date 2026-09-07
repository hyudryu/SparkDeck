"""Stopped intent survives disconnects without stopping a running peer group."""

import asyncio
import copy
import unittest
from unittest.mock import AsyncMock, Mock

from manager import Manager


class ReconnectStopTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, *, running_peer=False):
        manager = Manager.__new__(Manager)
        manager._save_deployments = Mock()
        manager._member_action = AsyncMock(return_value={"ok": True})
        members = [
            {"node_id": f"node-{i + 1}", "container_name": f"rank-{i}",
             "rank": i % 2, "instance_id": i // 2, "status": "running",
             "desired_state": "running" if running_peer and i >= 2 else "stopped"}
            for i in range(4 if running_peer else 2)
        ]
        for member in members[:2]:
            member["failed_stop_error"] = "agent disconnected"
        deployment = {
            "id": "reconnect", "mode": "grouped_sharded", "members": members,
            "desired_state": "running" if running_peer else "stopped",
            "status": "degraded" if running_peer else "error",
            "error": "agent disconnected",
        }
        manager.deployments = [deployment]
        nodes = [{"id": member["node_id"], "online": True, "status": "online",
                  "docker_ready": True, "containers": [{
                      "name": member["container_name"], "status": "running"}]}
                 for member in members]
        manager.cluster_nodes = AsyncMock(return_value=nodes)
        return manager, deployment, nodes

    async def test_health_tick_retries_failed_stop_when_nodes_reconnect(self):
        manager, deployment, nodes = self.fixture()
        for node in nodes:
            node["online"] = False
        await manager._cluster_health_tick()
        manager._member_action.assert_not_awaited()
        self.assertTrue(all(m.get("failed_stop_error") for m in deployment["members"]))

        for node in nodes:
            node["online"] = True
        await manager._cluster_health_tick()
        self.assertEqual(manager._member_action.await_count, 2)
        self.assertTrue(all(c.args[1] == "stop" for c in manager._member_action.await_args_list))
        self.assertTrue(all(not m.get("failed_stop_error") for m in deployment["members"]))
        self.assertTrue(all(m["status"] in {"stopped", "exited"} for m in deployment["members"]))
        manager._save_deployments.assert_called()

    async def test_stopped_group_reconciles_without_changing_running_peer(self):
        manager, deployment, _ = self.fixture(running_peer=True)
        peer = copy.deepcopy(deployment["members"][2:])
        await manager._reconcile_stopped_members()
        self.assertEqual(manager._member_action.await_count, 2)
        self.assertTrue(all(c.args[0]["instance_id"] == 0 for c in manager._member_action.await_args_list))
        self.assertEqual(deployment["members"][2:], peer)
        self.assertEqual(deployment["desired_state"], "running")

    async def test_new_start_intent_is_rechecked_after_waiting_for_action_lock(self):
        manager, deployment, _ = self.fixture()
        lock = manager._cluster_action_lock()
        await lock.acquire()
        task = asyncio.create_task(manager._reconcile_stopped_members())
        try:
            await asyncio.sleep(0)
            deployment["desired_state"] = "running"
            for member in deployment["members"]:
                member["desired_state"] = "running"
        finally:
            lock.release()
        await task
        manager._member_action.assert_not_awaited()

    async def test_negative_stop_response_keeps_failure_and_running_status(self):
        manager, deployment, _ = self.fixture()
        manager._member_action.return_value = {"ok": False, "error": "stop refused"}
        await manager._reconcile_stopped_members()
        self.assertEqual(manager._member_action.await_count, 2)
        self.assertTrue(all(m.get("failed_stop_error") for m in deployment["members"]))
        self.assertTrue(all(m["status"] == "running" for m in deployment["members"]))

    async def test_unready_docker_does_not_prove_containers_are_missing(self):
        manager, deployment, nodes = self.fixture()
        for node in nodes:
            node.update(docker_ready=False, containers=[])
        await manager._reconcile_stopped_members()
        manager._member_action.assert_not_awaited()
        self.assertTrue(all(m.get("failed_stop_error") for m in deployment["members"]))

    async def test_confirmed_missing_containers_clear_failed_stop_without_rpc(self):
        manager, deployment, nodes = self.fixture()
        for node in nodes:
            node["containers"] = []
        await manager._reconcile_stopped_members()
        manager._member_action.assert_not_awaited()
        self.assertTrue(all(not m.get("failed_stop_error") for m in deployment["members"]))
        self.assertTrue(all(m["status"] in {"missing", "stopped", "exited"} for m in deployment["members"]))

    async def test_incomplete_inventory_does_not_clear_reservations(self):
        for inventory in (None, "unavailable", {}):
            with self.subTest(inventory=inventory):
                manager, deployment, nodes = self.fixture()
                for node in nodes:
                    node["containers"] = inventory
                before = copy.deepcopy(deployment)
                await manager._reconcile_stopped_members()
                manager._member_action.assert_not_awaited()
                manager._save_deployments.assert_not_called()
                self.assertEqual(deployment, before)
        manager, deployment, nodes = self.fixture()
        for node in nodes:
            del node["containers"]
        await manager._reconcile_stopped_members()
        manager._member_action.assert_not_awaited()
        self.assertTrue(all(m.get("failed_stop_error") for m in deployment["members"]))

    async def test_targeted_success_clears_aggregate_stop_error(self):
        manager, deployment, _ = self.fixture(running_peer=True)
        await manager._reconcile_stopped_members()
        self.assertFalse(deployment.get("error"))

    async def test_exited_container_with_failed_stop_still_disarms_restart_policy(self):
        manager, deployment, nodes = self.fixture()
        for node in nodes:
            node["containers"][0]["status"] = "exited"
        await manager._reconcile_stopped_members()
        self.assertEqual(manager._member_action.await_count, 2)
        self.assertTrue(all(c.args[1] == "stop" for c in manager._member_action.await_args_list))
        self.assertTrue(all(not m.get("failed_stop_error") for m in deployment["members"]))

    async def test_confirmed_stop_does_not_persist_again_when_nothing_changed(self):
        manager, _, nodes = self.fixture()
        for node in nodes:
            node["containers"] = []
        await manager._reconcile_stopped_members()
        manager._save_deployments.reset_mock()
        await manager._reconcile_stopped_members()
        manager._save_deployments.assert_not_called()

    async def test_cancelled_member_stop_does_not_confirm_success(self):
        manager, deployment, _ = self.fixture()
        manager._member_action.side_effect = asyncio.CancelledError()
        try:
            await manager._reconcile_stopped_members()
        except asyncio.CancelledError:
            pass
        self.assertTrue(all(m.get("failed_stop_error") for m in deployment["members"]))
        self.assertTrue(all(m["status"] == "running" for m in deployment["members"]))
        self.assertFalse(manager._cluster_action_lock().locked())
