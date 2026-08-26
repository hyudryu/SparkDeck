import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx

from manager import Manager
from sparkdeck.service import SparkDeckService


def node(node_id: str, name: str, *, local: bool = False) -> dict:
    return {
        "id": node_id,
        "name": name,
        "local": local,
        "enabled": True,
        "status": "online",
        "online": True,
        "docker_ready": True,
        "fabric_ready": True,
        "agent_url": f"https://{node_id}.private.example",
        "fabric_ip": "169.254.10.2",
        "stats": {"gpus": [{"name": "GB10"}]},
        "disk": {"free_bytes": 1000},
    }


class NodePullTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = Manager.__new__(Manager)
        self.nodes = [node("local", "Coordinator", local=True), node("remote-1", "Worker")]
        self.manager.cluster_nodes = AsyncMock(return_value=self.nodes)
        self.manager.pull_image_result = AsyncMock(return_value={"ok": True})
        self.manager.node_registry = Mock()
        self.manager.node_registry.request = AsyncMock(return_value={"ok": True})

    async def test_pull_dispatches_only_to_selected_nodes_and_returns_safe_metadata(self):
        result = await self.manager.pull_image_on_nodes(
            "ghcr.io/example/vllm:latest", ["remote-1", "local", "remote-1"],
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["node_ids"], ["remote-1", "local"])
        self.manager.pull_image_result.assert_awaited_once_with("ghcr.io/example/vllm:latest")
        self.manager.node_registry.request.assert_awaited_once_with(
            "remote-1", "POST", "/api/agent/images/pull",
            json_body={"image": "ghcr.io/example/vllm:latest"}, timeout=1800,
        )
        self.assertNotIn("agent_url", result["selected_nodes"][0])
        self.assertNotIn("fabric_ip", result["selected_nodes"][0])

    async def test_pull_reports_per_node_partial_failure(self):
        self.manager.node_registry.request.side_effect = RuntimeError("agent unavailable")

        result = await self.manager.pull_image_on_nodes("example/image", ["local", "remote-1"])

        self.assertFalse(result["ok"])
        self.assertTrue(result["results"][0]["ok"])
        self.assertEqual(result["results"][1]["error"], "agent unavailable")

    async def test_unknown_node_is_rejected_before_pull(self):
        with self.assertRaisesRegex(ValueError, "unknown cluster node"):
            await self.manager.pull_image_on_nodes("example/image", ["missing"])
        self.manager.pull_image_result.assert_not_awaited()


class FakeServiceManager:
    def __init__(self):
        self.http = httpx.AsyncClient()
        self.nodes = [node("local", "Coordinator", local=True), node("remote-1", "Worker")]
        self.selected_cluster_nodes = AsyncMock(side_effect=self._selected)
        self.cluster_nodes = AsyncMock(return_value=self.nodes)
        self.create_deployment = AsyncMock(return_value={
            "id": "cluster-1",
            "status": "starting",
            "api_port": 8008,
            "node_ids": ["local", "remote-1"],
            "members": [
                {"node_id": "local", "node_name": "Coordinator", "container_name": "rank-0"},
                {"node_id": "remote-1", "node_name": "Worker", "container_name": "rank-1"},
            ],
        })
        self.deployment_action = AsyncMock(return_value={"ok": True, "errors": []})
        self.list_containers = AsyncMock(return_value=[])
        self.remove_container = AsyncMock(return_value={"ok": True})

    async def _selected(self, node_ids):
        by_id = {item["id"]: item for item in self.nodes}
        return [by_id[node_id] for node_id in node_ids]

    @staticmethod
    def public_target_node(value):
        return Manager.public_target_node(value)


class SelectedDeploymentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = FakeServiceManager()
        self.service = SparkDeckService(self.manager, Path(self.temp.name))

    async def asyncTearDown(self):
        await self.manager.http.aclose()
        await self.service.close()
        self.temp.cleanup()

    async def test_selected_nodes_use_cluster_launch_and_persist_safe_configuration(self):
        result = await self.service.create_deployment({
            "model": "org/model",
            "alias": "selected-model",
            "runtime": "vllm",
            "node_ids": ["local", "remote-1"],
            "revision": "release-2",
            "settings": {"context_length": 8192, "image": "private/image"},
        })

        launch = self.manager.create_deployment.await_args.args[0]
        self.assertEqual(launch["node_ids"], ["local", "remote-1"])
        self.assertEqual(launch["deployment_mode"], "replicated")
        self.assertEqual(launch["extra_args"][-2:], ["--revision", "release-2"])
        self.assertEqual(result["node_ids"], ["local", "remote-1"])
        self.assertEqual([item["name"] for item in result["selected_nodes"]], ["Coordinator", "Worker"])

        stored = self.service.store.deployment("selected-model")
        self.assertEqual(stored["settings"]["node_ids"], ["local", "remote-1"])
        self.assertEqual(stored["settings"]["deployment_mode"], "replicated")
        self.assertEqual(stored["settings"]["manager_deployment_id"], "cluster-1")
        self.assertEqual(stored["settings"]["context_length"], 8192)
        self.assertNotIn("image", stored["settings"])

        self.manager.deployments = [self.manager.create_deployment.return_value]
        listed = (await self.service.deployments())[0]
        self.assertEqual(listed["node_ids"], ["local", "remote-1"])
        self.assertEqual([item["id"] for item in listed["selected_nodes"]], ["local", "remote-1"])

    async def test_remote_only_run_is_rejected_until_coordinator_proxy_exists(self):
        with self.assertRaisesRegex(ValueError, "coordinator node must be included"):
            await self.service.create_deployment({
                "model": "org/model", "runtime": "sglang", "node_id": "remote-1",
            })
        self.manager.create_deployment.assert_not_awaited()

    async def test_cluster_lifecycle_dispatches_to_owning_manager_deployment(self):
        await self.service.create_deployment({
            "model": "org/model", "alias": "clustered", "runtime": "vllm",
            "node_ids": ["local", "remote-1"],
        })

        stopped = await self.service.deployment_action(result_id := result_id_for(self.service, "clustered"), "stop")
        removed = await self.service.delete_deployment(result_id)

        self.assertEqual(stopped["status"], "stopped")
        self.assertEqual(removed, {"ok": True, "id": result_id})
        self.assertEqual(
            [call.args for call in self.manager.deployment_action.await_args_list],
            [("cluster-1", "stop"), ("cluster-1", "remove")],
        )


def result_id_for(service: SparkDeckService, alias: str) -> str:
    return service.store.deployment(alias)["id"]


class RemovedOllamaTests(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_models_ignores_legacy_ollama_state(self):
        manager = Manager.__new__(Manager)
        manager.get_state = AsyncMock(return_value={
            "containers": [], "ollama": {"models": [{"name": "legacy/model"}]},
            "unsloth": {}, "sparkrun_targets": {},
        })

        result = await manager.proxy_models()

        self.assertEqual(result["data"], [])

    async def test_removed_ollama_prefix_is_not_routed_to_ollama(self):
        manager = Manager.__new__(Manager)
        manager._unsloth_loaded_model = AsyncMock(return_value=None)
        manager._vllm_chat = AsyncMock(return_value={"ok": True})

        await manager.proxy_chat_completions({"model": "CLOUD legacy", "messages": []})

        manager._vllm_chat.assert_awaited_once()

    def test_legacy_ollama_setting_is_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = Manager.__new__(Manager)
            manager.settings_path = Path(directory) / "settings.json"
            manager.settings_path.write_text(json.dumps({"ollama_base_url": "http://localhost:11434"}))

            settings = manager._load_settings()

        self.assertNotIn("ollama_base_url", settings)

    def test_legacy_ollama_deployment_is_retained_but_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = Manager.__new__(Manager)
            manager.deployments_path = Path(directory) / "deployments.json"
            manager.deployments_path.write_text(json.dumps([
                {"id": "legacy", "engine": "ollama", "status": "running"},
            ]))

            deployments = manager._load_deployments()

        self.assertEqual(deployments[0]["status"], "error")
        self.assertIn("unsupported persisted runtime", deployments[0]["error"])

    async def test_ollama_container_launch_is_rejected(self):
        manager = Manager.__new__(Manager)
        with self.assertRaisesRegex(ValueError, "engine must be vllm or sglang"):
            await manager.create_container("legacy/model", engine="ollama")


if __name__ == "__main__":
    unittest.main()
