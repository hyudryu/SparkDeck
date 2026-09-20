"""Stopped Docker members must not retain synthetic launch reservations."""

import unittest
from unittest.mock import AsyncMock, Mock, patch

import docker
from fastapi import HTTPException

from manager import Manager
from sparkdeck.service import _observed_occupied_node_ids


class StopLaunchTrackingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = Manager.__new__(Manager)
        self.container = Mock(status="running", labels={})
        self.manager.client = Mock()
        self.manager.client.containers.get.return_value = self.container
        self.manager._reap_container_admission = AsyncMock()
        self.launch = {"phase": "starting", "updated_at": 1}
        self.manager.cluster_member_launches = {"rank-0": self.launch}

        def stopped(*, timeout):
            self.container.status = "exited"

        self.container.stop.side_effect = stopped

    async def test_successful_stop_clears_synthetic_launch(self):
        result = await self.manager.stop_container("rank-0", explicit=True)

        self.assertEqual(result, {"ok": True})
        self.assertNotIn("rank-0", self.manager.cluster_member_launches)
        self.assertIn("rank-0", self.manager._explicitly_stopped_containers)
        self.container.update.assert_called_once_with(restart_policy={"Name": "no"})

    async def test_agent_inventory_and_controller_snapshot_release_phantom_node(self):
        # Reproduce the reported state: a stopped deployment whose worker has
        # no Docker container but still advertises its old launch progress.
        agent = self.manager
        agent.client.containers.get.side_effect = docker.errors.NotFound("gone")
        agent.list_containers = AsyncMock(return_value=[])
        agent.get_disk = AsyncMock(return_value={})
        agent._docker_runtime_status = AsyncMock(return_value=(True, None))
        agent._network_interfaces = Mock(return_value=[])
        agent.settings = {}
        agent.agent_health = Mock(return_value={})
        agent.routeros = Mock()
        agent.routeros.presence.return_value = {}
        agent.llama_rpc_status = Mock(return_value={})

        controller = Manager.__new__(Manager)
        controller.list_containers = AsyncMock(return_value=[])
        controller.list_images = AsyncMock(return_value=[])
        controller.get_stats = AsyncMock(return_value={})
        controller.settings = {}
        for field in (
            "_token_usage_sync_status", "token_stats", "usage_aliases",
            "usage_merge_groups", "usage_routing_rules", "usage_cache_estimates",
            "session_token_stats", "jobs",
        ):
            setattr(controller, field, {})
        for method in (
            "public_settings", "active_requests", "active_request_groups",
            "inference_admission",
        ):
            setattr(controller, method, Mock(return_value={}))
        controller.usage_rows = Mock(return_value=[])
        controller.deployments = [{
            "id": "stopped-model", "model": "org/model", "status": "stopped",
            "desired_state": "stopped", "mode": "single", "node_ids": ["node-3"],
            "launch_settings": {"engine": "vllm"},
            "members": [{"node_id": "node-3", "container_name": "rank-0", "rank": 0,
                         "status": "stopped", "desired_state": "stopped"}],
        }, {
            "id": "serving-peer", "model": "org/peer", "status": "running",
            "desired_state": "running", "mode": "single", "node_ids": ["node-2"],
            "launch_settings": {"engine": "vllm"},
            "members": [{"node_id": "node-2", "container_name": "peer-0", "rank": 0,
                         "status": "running", "desired_state": "running"}],
        }]

        async def cluster_nodes(*_args):
            return [{"id": "node-3", "name": "Node 3", **await agent.agent_status(stats={})}, {
                "id": "node-2", "name": "Node 2", "online": True, "status": "online",
                "docker_ready": True, "inventory_available": True,
                "containers": [{"name": "peer-0", "status": "running", "phase": {"phase": "ready"}}],
            }]

        controller.cluster_nodes = cluster_nodes
        before = await controller.get_state()
        stopped, peer = before["deployments"]
        self.assertEqual(before["nodes"][0]["containers"][0]["status"], "creating")
        self.assertFalse(stopped["members"][0]["has_live_container"])
        self.assertEqual(_observed_occupied_node_ids(stopped), ["node-3"])
        self.assertEqual(_observed_occupied_node_ids(peer), ["node-2"])

        await agent.stop_container("rank-0", explicit=True)

        after = await controller.get_state()
        stopped, peer = after["deployments"]
        self.assertEqual(after["nodes"][0]["containers"], [])
        self.assertEqual(stopped["status"], "stopped")
        self.assertEqual(_observed_occupied_node_ids(stopped), [])
        self.assertEqual(_observed_occupied_node_ids(peer), ["node-2"])
        self.assertEqual(peer["status"], "ready")

    async def test_confirmed_absence_clears_stale_completed_launch(self):
        self.manager.client.containers.get.side_effect = docker.errors.NotFound("gone")

        result = await self.manager.stop_container("rank-0", explicit=True)

        self.assertEqual(result, {"ok": True})
        self.assertNotIn("rank-0", self.manager.cluster_member_launches)

    async def test_failed_stop_retains_launch_reservation(self):
        self.container.stop.side_effect = docker.errors.APIError("Docker unavailable")

        with self.assertRaises(docker.errors.APIError):
            await self.manager.stop_container("rank-0", explicit=True)

        self.assertIs(self.manager.cluster_member_launches["rank-0"], self.launch)

    async def test_absence_during_uncancellable_creation_requires_retry(self):
        self.launch["phase"] = "pulling_image"
        self.manager.client.containers.get.side_effect = docker.errors.NotFound("not created yet")

        with self.assertRaisesRegex(RuntimeError, "launch is still in progress"):
            await self.manager.stop_container("rank-0", explicit=True)

        self.assertIs(self.manager.cluster_member_launches["rank-0"], self.launch)

    async def test_launch_progress_during_stop_is_preserved_for_reconciliation(self):
        def stopped(*, timeout):
            self.container.status = "exited"
            self.launch.update(phase="creating_container", updated_at=2)

        self.container.stop.side_effect = stopped

        await self.manager.stop_container("rank-0", explicit=True)

        self.assertIs(self.manager.cluster_member_launches["rank-0"], self.launch)
        self.assertEqual(self.launch["updated_at"], 2)

    async def test_agent_stop_cleans_confirmed_absent_tracked_launch(self):
        import server

        self.manager.is_managed_container = AsyncMock(return_value=False)
        self.manager.client.containers.get.side_effect = docker.errors.NotFound("gone")
        with patch.object(server, "manager", self.manager), patch.object(server, "_require_agent") as auth:
            result = await server.agent_stop_container("rank-0", Mock(), explicit=True)

        auth.assert_called_once()
        self.assertEqual(result, {"ok": True})
        self.assertNotIn("rank-0", self.manager.cluster_member_launches)

    async def test_agent_stop_does_not_touch_existing_unmanaged_container(self):
        import server

        self.manager.is_managed_container = AsyncMock(return_value=False)
        with patch.object(server, "manager", self.manager), patch.object(server, "_require_agent"):
            with self.assertRaises(HTTPException) as raised:
                await server.agent_stop_container("rank-0", Mock(), explicit=True)

        self.assertEqual(raised.exception.status_code, 404)
        self.container.stop.assert_not_called()
        self.assertIn("rank-0", self.manager.cluster_member_launches)

    async def test_agent_stop_preserves_authentication_for_tracked_launch(self):
        import server

        with (
            patch.object(server, "manager", self.manager),
            patch.object(server, "_require_agent", side_effect=HTTPException(401, "invalid token")),
        ):
            with self.assertRaises(HTTPException) as raised:
                await server.agent_stop_container("rank-0", Mock(), explicit=True)

        self.assertEqual(raised.exception.status_code, 401)
        self.manager.client.containers.get.assert_not_called()

    async def test_agent_stop_does_not_turn_unknown_docker_state_into_absence(self):
        import server

        self.manager.is_managed_container = AsyncMock(return_value=False)
        self.manager.client.containers.get.side_effect = docker.errors.APIError("Docker unavailable")
        with patch.object(server, "manager", self.manager), patch.object(server, "_require_agent"):
            with self.assertRaises(docker.errors.APIError):
                await server.agent_stop_container("rank-0", Mock(), explicit=True)

        self.assertIn("rank-0", self.manager.cluster_member_launches)
