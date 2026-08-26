import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

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


class ClusterImageInventoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_inventory_collects_local_and_remote_with_safe_partial_errors(self):
        manager = Manager.__new__(Manager)
        manager.cluster_nodes = AsyncMock(return_value=[
            cluster_node("local", local=True),
            cluster_node("worker-1"),
            cluster_node("worker-2"),
        ])
        manager.list_images = AsyncMock(return_value=[{"id": "sha256:a", "tags": ["org/a:latest"]}])
        manager.list_containers = AsyncMock(return_value=[])
        manager.node_registry = Mock()

        async def remote(node_id, *_args, **_kwargs):
            if node_id == "worker-2":
                raise RuntimeError("offline")
            return {
                "images": [{"id": "sha256:a", "tags": ["org/a:latest"]}],
                "containers": [],
            }

        manager.node_registry.request = AsyncMock(side_effect=remote)

        inventory = await manager.cluster_image_inventory()

        self.assertTrue(inventory["partial"])
        self.assertEqual([row["node"]["id"] for row in inventory["results"]], ["local", "worker-1"])
        self.assertEqual(inventory["errors"][0]["node"]["id"], "worker-2")
        self.assertNotIn("agent_url", str(inventory))
        self.assertNotIn("secret", str(inventory))


class FakeManager:
    def __init__(self):
        self.http = httpx.AsyncClient()
        self.deployments = []
        self.cluster_nodes = AsyncMock(return_value=[cluster_node("worker-1")])
        self.list_containers = AsyncMock(return_value=[])
        self.deployment_action = AsyncMock(return_value={"ok": True, "errors": []})

    @staticmethod
    def public_target_node(node):
        return Manager.public_target_node(node)


class ReplacementReconciliationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = FakeManager()
        self.service = SparkDeckService(self.manager, Path(self.temp.name))
        self.service.store.add_deployment(Deployment(
            id="record-1", alias="friendly", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
            container_name="old-rank",
            settings={
                "context_length": 8192,
                "node_ids": ["worker-1"],
                "manager_deployment_id": "old-manager",
            },
        ), "http://127.0.0.1:8000")

    async def asyncTearDown(self):
        await self.manager.http.aclose()
        await self.service.close()
        self.temp.cleanup()

    async def test_computed_ready_replacement_updates_sqlite_and_model_visibility(self):
        replacement = {
            "id": "new-manager", "sparkdeck_record_id": "record-1",
            "status": "ready", "api_port": 8010, "node_ids": ["worker-1"],
            "members": [{"rank": 0, "node_id": "worker-1", "container_name": "new-rank"}],
        }
        self.manager.get_state = AsyncMock(return_value={"deployments": [replacement]})

        listed = await self.service.deployments()
        models = await self.service.models()

        self.assertEqual(listed[0]["status"], "running")
        self.assertEqual(models["data"][0]["id"], "friendly")
        stored = self.service.store.deployment("record-1", include_private=True)
        self.assertEqual(stored["settings"]["manager_deployment_id"], "new-manager")
        self.assertEqual(stored["container_name"], "new-rank")
        self.assertEqual(stored["_base_url"], "http://127.0.0.1:8010")

        await self.service.deployment_action("record-1", "stop")
        self.manager.deployment_action.assert_awaited_once_with("new-manager", "stop")

    async def test_settings_dirty_start_result_immediately_reconciles_replacement(self):
        self.manager.deployment_action.return_value = {
            "ok": True,
            "errors": [],
            "replaced_deployment_id": "old-manager",
            "deployment": {
                "id": "replacement-manager", "api_port": 8020,
                "node_ids": ["worker-1"],
                "members": [{"container_name": "replacement-rank"}],
            },
        }

        started = await self.service.deployment_action("record-1", "start")

        self.assertEqual(started["status"], "running")
        stored = self.service.store.deployment("record-1", include_private=True)
        self.assertEqual(
            stored["settings"]["manager_deployment_id"], "replacement-manager"
        )
        self.assertEqual(stored["container_name"], "replacement-rank")
        self.assertEqual(stored["_base_url"], "http://127.0.0.1:8020")


class WorkerSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_joined_worker_keeps_local_inference_nudger(self):
        manager = Manager.__new__(Manager)
        manager.data_dir = Path("unused")
        manager.is_joined_worker = Mock(return_value=True)
        manager._start_controller_tasks = Mock()
        blocker = asyncio.Event()

        async def loop():
            await blocker.wait()

        manager._temperature_history_monitor_loop = loop
        manager._inference_nudger_loop = loop
        manager._start_mem_bw_monitor = Mock()

        await manager.start()

        manager._start_controller_tasks.assert_not_called()
        self.assertFalse(manager.inference_nudger_task.done())
        for task in (manager.inference_nudger_task, manager.temperature_history_task):
            task.cancel()
        await asyncio.gather(
            manager.inference_nudger_task, manager.temperature_history_task,
            return_exceptions=True,
        )


if __name__ == "__main__":
    unittest.main()
