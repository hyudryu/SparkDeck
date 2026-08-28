"""Saved-deployment bookmarks: save, prepare weights via Virtual NAS, launch."""

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


class FakeBookmarkManager:
    def __init__(self, nodes):
        self.http = httpx.AsyncClient()
        self.nodes = nodes
        self.deployments = []
        self.create_deployment = AsyncMock(return_value={
            "id": "cluster-1",
            "status": "starting",
            "api_port": 8008,
            "node_ids": ["remote-1"],
            "members": [{
                "node_id": "remote-1", "node_name": "Worker",
                "container_name": "rank-0",
            }],
        })
        self.deployment_action = AsyncMock(return_value={"ok": True, "errors": []})
        self.list_containers = AsyncMock(return_value=[])
        self.remove_container = AsyncMock(return_value={"ok": True})
        self.selected_cluster_nodes = AsyncMock(side_effect=self._selected)
        self.cluster_nodes = AsyncMock(return_value=self.nodes)
        self.model_cache_inventory = AsyncMock(return_value=[
            {"id": "remote-1", "models": [
                {"model_id": "org/model", "partial": False, "revisions": ["main"]},
            ]},
            {"id": "local", "models": []},
        ])
        self.recipe_model_preparation_preflight = AsyncMock(return_value={
            "enabled": True, "model_id": "org/model", "revision": "main",
            "targets": [], "node_ids": ["remote-1"], "eligible": True,
            "action": "ready", "download_node_ids": [],
            "transfer_target_node_ids": [], "reason": None,
        })
        self.queue_recipe_model_preparation = AsyncMock(return_value={
            "workflow_id": None, "job_ids": [], "jobs": [],
        })

    async def _selected(self, node_ids):
        by_id = {item["id"]: item for item in self.nodes}
        return [by_id[node_id] for node_id in node_ids]

    @staticmethod
    def _without_sensitive_cli_credentials(args):
        return [str(item) for item in args or []]

    @staticmethod
    def _deployment_launch_controls(settings):
        return {}

    @staticmethod
    def public_target_node(value):
        return Manager.public_target_node(value)


class DeploymentBookmarkTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = FakeBookmarkManager(
            [node("local", "Coordinator", local=True), node("remote-1", "Worker")],
        )
        self.service = SparkDeckService(self.manager, Path(self.temp.name))

    async def asyncTearDown(self):
        await self.manager.http.aclose()
        await self.service.close()
        self.temp.cleanup()

    async def test_create_saves_bookmark_without_launching(self):
        created = await self.service.create_deployment({
            "model": "org/model",
            "alias": "bookmark",
            "runtime": "vllm",
            "node_ids": ["remote-1"],
            "deployment_mode": "single",
            "settings": {"context_length": 8192},
        })

        self.assertEqual(created["status"], "saved")
        self.assertEqual(created["node_ids"], ["remote-1"])
        self.assertEqual(created["deployment_mode"], "single")
        self.manager.create_deployment.assert_not_awaited()
        stored = self.service.store.deployment("bookmark", include_private=True)
        self.assertEqual(stored["desired_state"], "stopped")
        self.assertIsNone(stored["container_name"])
        self.assertEqual(stored["settings"]["node_ids"], ["remote-1"])

    async def test_bookmark_is_reported_as_saved_in_the_deployments_list(self):
        await self.service.create_deployment({
            "model": "org/model", "alias": "bookmark", "runtime": "vllm",
            "node_ids": ["remote-1"], "deployment_mode": "single",
        })

        listed = await self.service.deployments()
        self.assertEqual(listed[0]["status"], "saved")
        self.assertEqual(listed[0]["node_ids"], ["remote-1"])
        self.assertNotIn("last_error", listed[0])

    async def test_start_launches_saved_bookmark_on_requested_nodes(self):
        await self.service.create_deployment({
            "model": "org/model", "alias": "bookmark", "runtime": "vllm",
            "node_ids": ["local"], "deployment_mode": "single",
            "settings": {"context_length": 8192},
        })
        self.manager.create_deployment.assert_not_awaited()

        started = await self.service.deployment_action(
            "bookmark", "start", ["remote-1"],
        )

        launch = self.manager.create_deployment.await_args.args[0]
        self.assertEqual(launch["engine"], "vllm")
        self.assertEqual(launch["node_ids"], ["remote-1"])
        self.assertEqual(launch["deployment_mode"], "single")
        self.assertEqual(launch["extra_args"], ["--max-model-len", "8192"])
        self.assertEqual(started["node_ids"], ["remote-1"])
        stored = self.service.store.deployment("bookmark", include_private=True)
        self.assertEqual(stored["desired_state"], "running")
        self.assertEqual(stored["settings"]["manager_deployment_id"], "cluster-1")

    async def test_start_without_nodes_falls_back_to_saved_preferences(self):
        await self.service.create_deployment({
            "model": "org/model", "alias": "bookmark", "runtime": "vllm",
            "node_ids": ["remote-1"], "deployment_mode": "single",
        })

        await self.service.deployment_action("bookmark", "start")

        launch = self.manager.create_deployment.await_args.args[0]
        self.assertEqual(launch["node_ids"], ["remote-1"])

    async def test_llama_bookmark_launches_cluster_members_with_cache_relative_artifact(self):
        revision = "a" * 40
        virtual_nas = Mock()
        virtual_nas.resolve_download_revision = AsyncMock(return_value={
            "resolved_revision": revision,
        })
        self.manager.virtual_nas = virtual_nas
        await self.service.create_deployment({
            "model": "org/model", "alias": "llama-bookmark",
            "runtime": "llama.cpp",
            "node_ids": ["remote-1"], "deployment_mode": "single",
            "settings": {"artifact": "FP16/model-F16.gguf", "context_length": 4096},
        })

        await self.service.deployment_action("llama-bookmark", "start")

        launch = self.manager.create_deployment.await_args.args[0]
        self.assertEqual(launch["engine"], "llama.cpp")
        self.assertEqual(
            launch["llama_artifact"],
            f"models--org--model/snapshots/{revision}/FP16/model-F16.gguf",
        )
        self.assertEqual(launch["llama_context_length"], 4096)
        self.assertNotIn("--revision", launch["extra_args"])

    async def test_llama_bookmark_without_nodes_prepares_controller_gguf_at_start(self):
        revision = "b" * 40
        model_root = Path(self.temp.name) / "models--org--model"
        artifact = model_root / "snapshots" / revision / "model-F16.gguf"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"gguf")
        virtual_nas = Mock()
        virtual_nas.resolve_download_revision = AsyncMock(return_value={
            "resolved_revision": revision,
        })
        virtual_nas.download_model_files_checked = AsyncMock(return_value={"ok": True})
        virtual_nas._model_path = Mock(return_value=model_root)
        self.manager.virtual_nas = virtual_nas
        launch = AsyncMock(return_value={
            "name": "sparkdeck-llama-local", "port": 8080, "status": "running",
            "model_source": "public_repository",
        })

        await self.service.create_deployment({
            "model": "org/model", "alias": "llama-local",
            "runtime": "llama.cpp",
            "settings": {"artifact": "model-F16.gguf"},
        })
        with patch("sparkdeck.service.launch_managed_container", launch):
            started = await self.service.deployment_action("llama-local", "start")

        virtual_nas.download_model_files_checked.assert_awaited_once()
        self.assertEqual(started["status"], "running")
        stored = self.service.store.deployment("llama-local", include_private=True)
        self.assertEqual(stored["desired_state"], "running")
        self.assertTrue(stored["container_name"])

    async def test_prepare_endpoints_delegate_to_manager_model_preparation(self):
        await self.service.create_deployment({
            "model": "org/model", "alias": "bookmark", "runtime": "vllm",
            "node_ids": ["remote-1"], "deployment_mode": "single",
        })

        plan = await self.service.deployment_preparation_preflight(
            "bookmark", ["remote-1"],
        )
        result = await self.service.deployment_prepare("bookmark", ["remote-1"])

        self.manager.recipe_model_preparation_preflight.assert_awaited_once_with(
            "org/model", "main", ["remote-1"],
        )
        self.manager.queue_recipe_model_preparation.assert_awaited_once_with(
            "org/model", "main", ["remote-1"],
        )
        self.assertTrue(plan["eligible"])
        self.assertEqual(result["workflow_id"], None)

    async def test_saved_bookmark_settings_and_node_preferences_are_editable(self):
        await self.service.create_deployment({
            "model": "org/model", "alias": "bookmark", "runtime": "vllm",
            "node_ids": ["local"], "deployment_mode": "single",
        })

        detail = await self.service.update_deployment_settings("bookmark", {
            "context_length": 16384,
            "node_ids": ["remote-1"],
        })

        self.assertTrue(detail["editable"])
        stored = self.service.store.deployment("bookmark", include_private=True)
        self.assertEqual(stored["settings"]["context_length"], 16384)
        self.assertEqual(stored["settings"]["node_ids"], ["remote-1"])

    async def test_external_bookmarks_still_register_without_nodes(self):
        created = await self.service.create_deployment({
            "model": "org/model", "alias": "external", "runtime": "vllm",
            "kind": "external", "base_url": "http://127.0.0.1:9000/v1",
        })

        self.assertEqual(created["alias"], "external")
        self.manager.create_deployment.assert_not_awaited()


class LlamaContainerArtifactTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()

    async def asyncTearDown(self):
        self.temp.cleanup()

    def manager_with_cache(self, root: Path):
        manager = Manager.__new__(Manager)
        manager.settings = {"hf_cache": str(root)}
        manager.client = Mock()
        manager.client.images.get = Mock(
            return_value=Mock(attrs={"Config": {"Env": []}}),
        )
        return manager

    def snapshot(self, root: Path, filename: str) -> Path:
        artifact = (
            root / "hub" / "models--org--model" / "snapshots" / ("a" * 40)
            / filename
        )
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"gguf")
        return artifact

    async def test_resolve_llama_artifact_maps_into_the_container_cache(self):
        root = Path(self.temp.name)
        self.snapshot(root, "model-F16.gguf")
        manager = self.manager_with_cache(root)

        model_path, siblings = manager._resolve_llama_artifact(
            "org/model", "models--org--model/snapshots/" + ("a" * 40) + "/model-F16.gguf",
        )

        self.assertEqual(
            model_path,
            f"/root/.cache/huggingface/hub/models--org--model/snapshots/"
            f"{'a' * 40}/model-F16.gguf",
        )
        self.assertEqual(siblings, [
            "models--org--model/snapshots/" + ("a" * 40) + "/model-F16.gguf",
        ])

    async def test_resolve_llama_artifact_requires_every_shard(self):
        root = Path(self.temp.name)
        revision = "a" * 40
        self.snapshot(root, "model-00001-of-00002.gguf")
        self.snapshot(root, "model-00002-of-00002.gguf")
        manager = self.manager_with_cache(root)
        reference = f"models--org--model/snapshots/{revision}/model-00001-of-00002.gguf"

        model_path, siblings = manager._resolve_llama_artifact("org/model", reference)
        self.assertTrue(model_path.endswith("model-00001-of-00002.gguf"))
        self.assertEqual(len(siblings), 2)

        (root / "hub" / "models--org--model" / "snapshots" / revision
         / "model-00002-of-00002.gguf").unlink()
        with self.assertRaisesRegex(ValueError, "not cached on this node"):
            manager._resolve_llama_artifact("org/model", reference)

    async def test_resolve_llama_artifact_rejects_escape_and_non_gguf(self):
        root = Path(self.temp.name)
        manager = self.manager_with_cache(root)

        with self.assertRaisesRegex(ValueError, "cache-relative GGUF artifact"):
            manager._resolve_llama_artifact(
                "org/model", "models--org--model/snapshots/../../escape.gguf",
            )
        with self.assertRaisesRegex(ValueError, "require a .gguf artifact"):
            manager._resolve_llama_artifact(
                "org/model", "models--org--model/snapshots/" + ("a" * 40) + "/model.bin",
            )


if __name__ == "__main__":
    unittest.main()
