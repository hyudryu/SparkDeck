import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

from cluster import NodeAgentResponseError
from manager import Manager, SourceRoutingUnavailable
from sparkdeck.service import SparkDeckService


class SourceRoutingServedStateTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, node="local"):
        manager = Manager.__new__(Manager)
        manager.deployments = [{
            "id": "runtime", "sparkdeck_record_id": "saved", "model": "org/model",
            "mode": "single", "desired_state": "running", "settings_dirty": True,
            "launch_settings": {"extra_args": ["--served-model-name", "new-name"]},
            "members": [{"node_id": node, "rank": 0, "container_name": "coordinator"}],
        }]
        manager._container_by_name = AsyncMock(return_value={
            "name": "coordinator", "status": "running",
            "served_model": "old-name", "served_models": ["old-name", "old-alias"],
        })
        manager._check_ready = AsyncMock(return_value=True)
        manager.node_registry = SimpleNamespace(request=AsyncMock())
        stored = {
            "id": "saved", "alias": "model", "runtime": "vllm", "kind": "managed",
            "model": {"repository": "org/model"},
            "settings": {"manager_deployment_id": "runtime"},
        }
        service = SparkDeckService.__new__(SparkDeckService)
        service.manager = manager
        service.store = SimpleNamespace(deployment=Mock(return_value=stored))
        rule = {"deployment_id": "saved", "instance_id": None, "node_ids": [node]}
        return manager, service, stored, rule

    async def test_local_snapshot_preserves_running_names_when_launch_settings_are_dirty(self):
        manager, service, stored, rule = self.fixture()
        self.assertEqual(manager._deployment_served_models(manager.deployments[0]), ["new-name"])
        rows = await service._observe_source_routing_target(stored, rule)
        self.assertEqual(rows[0]["served_models"], ["old-name", "old-alias"])
        manager.node_registry.request.assert_not_awaited()

    async def test_local_snapshot_preserves_legacy_singular_observed_name(self):
        manager, service, stored, rule = self.fixture()
        manager._container_by_name.return_value.pop("served_models")
        rows = await service._observe_source_routing_target(stored, rule)
        self.assertEqual(rows[0]["served_models"], ["old-name"])

    async def test_remote_dirty_snapshot_recovers_names_from_older_agent_inventory(self):
        for targeted in (
            {"status": "running", "ready": True},
            NodeAgentResponseError("worker", 404, "endpoint missing"),
        ):
            with self.subTest(targeted=targeted):
                manager, service, stored, rule = self.fixture("worker")
                manager.node_registry.request.side_effect = [
                    targeted,
                    {"docker_ready": True, "inventory_available": True, "containers": [
                        {"name": "unrelated", "status": "running", "served_models": ["wrong-name"]},
                        {"name": "coordinator", "status": "running", "served_models": ["old-name"]},
                    ]},
                    {"ready": True, "health_status": 200},
                ]
                rows = await service._observe_source_routing_target(stored, rule)
                self.assertEqual(rows[0]["served_models"], ["old-name"])
                calls = manager.node_registry.request.await_args_list
                self.assertTrue(all(call.args[0] == "worker" for call in calls))
                self.assertEqual(calls[1].args[1:], ("GET", "/api/agent/status"))

    async def test_dirty_snapshot_without_observed_names_fails_closed(self):
        manager, service, stored, rule = self.fixture()
        manager._container_by_name.return_value = {"name": "coordinator", "status": "running"}
        with self.assertRaises(SourceRoutingUnavailable):
            await service._source_routing_snapshot(stored, rule)

    async def test_agent_reports_running_names_independently_of_pending_settings(self):
        import server

        container = {
            "name": "coordinator", "status": "running",
            "served_model": "old-name", "served_models": ["old-name", "old-alias"],
            "load_settings": {"served_model_name": "new-name"},
            "environment": {"SECRET": "private"},
        }
        pending = [{
            "id": "deployment", "settings_dirty": True,
            "command": ["--served-model-name", "new-name"],
        }]
        with patch.object(server, "_require_agent"), patch.object(
            server.manager, "is_managed_container", AsyncMock(return_value=True),
        ), patch.object(
            server.manager, "_container_by_name", AsyncMock(return_value=container),
        ), patch.object(
            server.manager, "_check_ready", AsyncMock(return_value=True),
        ), patch.object(server.manager, "deployments", pending):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server.app), base_url="http://test",
            ) as client:
                response = await client.get(
                    "/api/agent/containers/coordinator/state?check_ready=true",
                )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "name": "coordinator", "status": "running", "ready": True,
            "served_model": "old-name", "served_models": ["old-name", "old-alias"],
        })
