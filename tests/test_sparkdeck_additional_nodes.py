
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import httpx

from manager import Manager
from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import SparkDeckService


def cluster_node(node_id: str, *, local: bool = False) -> dict:
    return {
        "id": node_id, "name": "Controller" if local else node_id,
        "local": local, "enabled": True, "online": True,
        "docker_ready": True, "agent_url": f"http://{node_id}.private",
        "agent_token": "secret",
    }


def weights_inventory(*node_ids: str) -> list[dict]:
    return [
        {
            "id": node_id,
            "models": [{
                "model_id": "org/model", "partial": False,
                "revisions": ["main"],
            }],
        }
        for node_id in node_ids
    ]


class AdditionalNodeLaunchTests(unittest.IsolatedAsyncioTestCase):
    """Starting a running deployment on additional nodes grows the layout."""

    def build_manager(self, *, mode: str = "single", node_ids: list[str] | None = None):
        node_ids = node_ids if node_ids is not None else ["worker-1"]
        replacement = {
            "id": "manager-2", "api_port": 8010,
            "node_ids": node_ids,
            "members": [{"container_name": "new-rank"}],
        }
        manager = Manager.__new__(Manager)
        manager.http = httpx.AsyncClient()
        manager.deployments_path = Path(self.temp.name) / "manager-deployments.json"
        manager.deployments = [{
            "id": "manager-1",
            "sparkdeck_record_id": "record-1",
            "name": "Chat model",
            "model": "org/model",
            "engine": "vllm",
            "status": "running",
            "desired_state": "running",
            "api_port": 8000,
            "node_ids": node_ids,
            "members": [{
                "rank": 0, "node_id": node_ids[0],
                "container_name": "rank-0",
            }],
            "launch_settings": {
                "deployment_name": "Chat model",
                "model": "org/model",
                "engine": "vllm",
                "deployment_mode": mode,
                "node_ids": node_ids,
                "extra_args": [],
            },
        }]
        manager.cluster_nodes = AsyncMock(return_value=[
            cluster_node("local", local=True), cluster_node("worker-1"),
            cluster_node("worker-2"), cluster_node("worker-3"),
        ])
        manager.list_containers = AsyncMock(return_value=[])
        manager.deployment_action = AsyncMock(return_value={
            "ok": True, "errors": [], "replaced_deployment_id": "manager-1",
            "deployment": replacement,
        })
        manager.model_cache_inventory = AsyncMock(
            return_value=weights_inventory("worker-1", "worker-2", "worker-3"),
        )
        return manager

    def build_service(self, manager: Manager) -> SparkDeckService:
        service = SparkDeckService(manager, Path(self.temp.name))
        service.store.add_deployment(Deployment(
            id="record-1", alias="Chat model", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
            container_name="rank-0",
            settings={
                "context_length": 8192,
                "node_ids": ["worker-1"],
                "manager_deployment_id": "manager-1",
            },
        ), "http://127.0.0.1:8000")
        return service

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def test_additional_nodes_promote_single_to_replicated(self):
        manager = self.build_manager()
        manager.deployment_action.return_value["deployment"]["node_ids"] = [
            "worker-1", "worker-2",
        ]
        service = self.build_service(manager)
        try:
            result = await service.deployment_action(
                "record-1", "start", additional_node_ids=["worker-2"],
            )

            self.assertEqual(result["status"], "running")
            manager.deployment_action.assert_awaited_once_with(
                "manager-1", "start", ["worker-1", "worker-2"], "replicated",
            )
            stored = service.store.deployment("record-1", include_private=True)
            self.assertEqual(stored["settings"]["node_ids"], ["worker-1", "worker-2"])
            self.assertEqual(stored["settings"]["manager_deployment_id"], "manager-2")
            self.assertEqual(stored["desired_state"], "running")
        finally:
            await manager.http.aclose()
            await service.close()

    async def test_additional_nodes_keep_replicated_layout_without_mode_override(self):
        manager = self.build_manager(mode="replicated", node_ids=["worker-1", "worker-2"])
        service = self.build_service(manager)
        try:
            await service.deployment_action(
                "record-1", "start", additional_node_ids=["worker-3"],
            )

            manager.deployment_action.assert_awaited_once_with(
                "manager-1", "start", ["worker-1", "worker-2", "worker-3"],
            )
        finally:
            await manager.http.aclose()
            await service.close()

    async def test_additional_nodes_reject_sharded_layout(self):
        manager = self.build_manager(mode="sharded", node_ids=["local", "worker-1"])
        service = self.build_service(manager)
        try:
            with self.assertRaisesRegex(ValueError, "sharded deployments cannot launch"):
                await service.deployment_action(
                    "record-1", "start", additional_node_ids=["worker-2"],
                )

            manager.deployment_action.assert_not_awaited()
        finally:
            await manager.http.aclose()
            await service.close()

    async def test_additional_nodes_require_cached_weights(self):
        manager = self.build_manager()
        manager.model_cache_inventory = AsyncMock(
            return_value=weights_inventory("worker-1"),
        )
        service = self.build_service(manager)
        try:
            with self.assertRaisesRegex(ValueError, "model weights are not available"):
                await service.deployment_action(
                    "record-1", "start", additional_node_ids=["worker-2"],
                )

            manager.deployment_action.assert_not_awaited()
        finally:
            await manager.http.aclose()
            await service.close()

    async def test_additional_nodes_reject_requests_without_new_nodes(self):
        manager = self.build_manager()
        service = self.build_service(manager)
        try:
            with self.assertRaisesRegex(ValueError, "at least one node"):
                await service.deployment_action(
                    "record-1", "start", additional_node_ids=["worker-1"],
                )

            manager.deployment_action.assert_not_awaited()
        finally:
            await manager.http.aclose()
            await service.close()

    async def test_additional_nodes_reject_standalone_container(self):
        manager = self.build_manager()
        manager.deployments = []
        service = self.build_service(manager)
        service.store.update_managed_routing(
            "record-1",
            {"context_length": 8192, "node_ids": ["worker-1"]},
            "rank-0",
            "http://127.0.0.1:8000",
        )
        try:
            with self.assertRaisesRegex(ValueError, "only available for cluster deployments"):
                await service.deployment_action(
                    "record-1", "start", additional_node_ids=["worker-2"],
                )

            manager.deployment_action.assert_not_awaited()
        finally:
            await manager.http.aclose()
            await service.close()


class ManagerRelaunchModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_relaunch_persists_relaunch_mode_in_launch_body(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = Manager.__new__(Manager)
            manager.deployments_path = Path(directory) / "manager-deployments.json"
            manager.deployments = [{
                "id": "manager-1",
                "name": "Chat model",
                "model": "org/model",
                "engine": "vllm",
                "status": "running",
                "desired_state": "running",
                "api_port": 8000,
                "node_ids": ["worker-1"],
                "members": [{
                    "rank": 0, "node_id": "worker-1",
                    "container_name": "rank-0",
                }],
                "launch_settings": {
                    "deployment_name": "Chat model",
                    "model": "org/model",
                    "engine": "vllm",
                    "deployment_mode": "single",
                    "node_ids": ["worker-1"],
                    "extra_args": [],
                },
            }]
            manager._preflight_deployment_launch = AsyncMock(return_value={})
            manager.create_deployment = AsyncMock(return_value={
                "id": "manager-2", "api_port": 8010,
                "node_ids": ["worker-1", "worker-2"],
                "members": [{"container_name": "new-rank"}],
            })
            manager._member_action = AsyncMock(return_value={})

            result = await manager.deployment_action(
                "manager-1", "start", ["worker-1", "worker-2"], "replicated",
            )

            self.assertTrue(result["ok"])
            launch_body = manager.create_deployment.await_args.args[0]
            self.assertEqual(launch_body["deployment_mode"], "replicated")
            self.assertEqual(launch_body["node_ids"], ["worker-1", "worker-2"])
            # The old record is retired; the replacement persists through the
            # real create_deployment path rather than this mocked one.
            self.assertEqual(
                [item["id"] for item in manager.deployments],
                [],
            )


class ManagerOnlyClusterCardTests(unittest.IsolatedAsyncioTestCase):
    async def test_manager_only_cluster_card_reports_cluster_node_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = Manager.__new__(Manager)
            manager.http = httpx.AsyncClient()
            manager.deployments = [{
                "id": "manager-1",
                "name": "Chat model",
                "model": "org/model",
                "engine": "vllm",
                "status": "running",
                "desired_state": "running",
                "api_port": 8000,
                "node_ids": ["worker-1", "worker-2"],
                "members": [
                    {"rank": 0, "node_id": "worker-1", "container_name": "rank-0"},
                    {"rank": 1, "node_id": "worker-2", "container_name": "rank-1"},
                ],
                "launch_settings": {
                    "deployment_name": "Chat model",
                    "model": "org/model",
                    "engine": "vllm",
                    "deployment_mode": "replicated",
                    "node_ids": ["worker-1", "worker-2"],
                    "extra_args": [],
                },
            }]
            manager.cluster_nodes = AsyncMock(return_value=[
                cluster_node("local", local=True),
                cluster_node("worker-1"), cluster_node("worker-2"),
            ])
            manager.list_containers = AsyncMock(return_value=[{
                "name": "rank-0", "model": "org/model", "runtime": "vllm",
                "managed": True, "status": "running", "port": 8000,
            }])
            service = SparkDeckService(manager, Path(directory))
            try:
                listed = await service.deployments()

                self.assertEqual(len(listed), 1)
                card = listed[0]
                self.assertFalse(str(card["id"]).startswith("container:"))
                # The card must carry the owning cluster's node set so the UI
                # can show the running nodes and lock them in an add-nodes
                # picker instead of treating the card as standalone.
                self.assertEqual(card["node_ids"], ["worker-1", "worker-2"])
                self.assertEqual(
                    [node["id"] for node in card["selected_nodes"]],
                    ["worker-1", "worker-2"],
                )
                self.assertEqual(card["deployment_mode"], "replicated")
            finally:
                await manager.http.aclose()
                await service.close()


if __name__ == "__main__":
    unittest.main()
