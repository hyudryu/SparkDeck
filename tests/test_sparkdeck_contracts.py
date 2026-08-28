import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx

from sparkdeck.catalog import HuggingFaceCatalog
from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import SparkDeckService


class FakeManager:
    def __init__(self):
        self.http = httpx.AsyncClient()
        self.list_containers = AsyncMock(return_value=[])
        self.remove_container = AsyncMock(return_value={"ok": True})
        self.start_container = AsyncMock(return_value={"status": "running"})
        self.stop_container = AsyncMock(return_value={"status": "exited"})
        self._vllm_chat = AsyncMock()
        self._vllm_completions = AsyncMock()


class SparkDeckContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = FakeManager()
        self.service = SparkDeckService(self.manager, Path(self.temp.name))

    async def asyncTearDown(self):
        await self.manager.http.aclose()
        await self.service.close()
        self.temp.cleanup()

    async def test_catalog_shape_filter_and_local_ids_match_frontend_contract(self):
        remote = HuggingFaceCatalog._public_item({
            "id": "org/model", "author": "org", "tags": ["transformers", "safetensors"],
        })
        self.service.catalog.search = AsyncMock(return_value=[remote])
        self.service.store.add_deployment(Deployment(
            id="dep-1", alias="model", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
            container_name="sparkdeck-model",
        ))

        result = await self.service.catalog_search("model", 24, "vllm")

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["local_deployment_ids"], ["dep-1"])
        compatibility = result["items"][0]["runtime_compatibility"]
        self.assertIsInstance(compatibility, list)
        self.assertTrue(any(item == {"runtime": "vllm", "supported": True} for item in compatibility))
        self.assertEqual(
            (await self.service.catalog_search("model", 24, "llama.cpp"))["items"], []
        )

    async def test_deployment_action_returns_a_wire_deployment(self):
        self.manager.list_containers.return_value = [{
            "name": "sparkdeck-model", "model": "org/model", "engine": "vllm",
            "managed": True, "status": "running", "port": 8000,
        }]
        self.service.store.add_deployment(Deployment(
            id="dep-1", alias="model", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
            container_name="sparkdeck-model",
        ), "http://127.0.0.1:8000")

        result = await self.service.deployment_action("dep-1", "start")

        self.assertEqual(result["id"], "dep-1")
        self.assertEqual(result["model"]["repository"], "org/model")
        self.assertEqual(result["status"], "running")
        self.assertNotIn("_base_url", result)

    async def test_registered_managed_runtimes_keep_manager_admission_proxy(self):
        completion = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 20},
        }
        self.manager._vllm_chat.side_effect = lambda *args, **kwargs: dict(completion)
        self.manager._vllm_completions.side_effect = lambda *args, **kwargs: dict(completion)
        cases = (
            ("vllm-chat", RuntimeKind.VLLM, "chat/completions", self.manager._vllm_chat),
            ("vllm-completion", RuntimeKind.VLLM, "completions", self.manager._vllm_completions),
            ("sglang-chat", RuntimeKind.SGLANG, "chat/completions", self.manager._vllm_chat),
            ("sglang-completion", RuntimeKind.SGLANG, "completions", self.manager._vllm_completions),
        )
        for index, (alias, runtime, endpoint, expected_proxy) in enumerate(cases):
            with self.subTest(alias=alias):
                self.service.store.add_deployment(Deployment(
                    id=f"dep-{index}", alias=alias, runtime=runtime,
                    kind=DeploymentKind.MANAGED,
                    model=ModelIdentity(f"org/model-{index}", revision=f"rev-{index}"),
                    container_name=f"container-{index}",
                ), f"http://127.0.0.1:{8000 + index}")
                result = await self.service.proxy(
                    {"model": alias, "messages": [], "stream": False}, endpoint,
                )
                self.assertEqual(result["model"], alias)
                self.assertEqual(expected_proxy.await_args.args[0], f"org/model-{index}")

    async def test_external_v1_base_url_is_not_duplicated_for_inference(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "model": "org/model",
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
        }
        self.manager.http.post = AsyncMock(return_value=response)
        self.service.store.add_deployment(Deployment(
            id="external-1", alias="external", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.EXTERNAL, model=ModelIdentity("org/model"),
            base_url_set=True,
        ), "https://example.test/openai/v1")

        await self.service.proxy({"model": "external", "messages": []}, "chat/completions")

        self.assertEqual(
            self.manager.http.post.await_args.args[0],
            "https://example.test/openai/v1/chat/completions",
        )

    async def test_managed_revision_reaches_runtime_launch_settings(self):
        launch = AsyncMock(return_value={
            "name": "sparkdeck-model", "port": 8000, "status": "running",
        })
        with patch("sparkdeck.service.launch_managed_container", launch):
            await self.service.create_deployment({
                "model": "org/model", "alias": "revision-model", "runtime": "vllm",
                "revision": "revision-abc",
            })

        self.assertEqual(launch.await_args.args[5]["revision"], "revision-abc")

    async def test_repo_relative_gguf_is_prepared_and_resolved_before_launch(self):
        revision = "a" * 40
        model_root = Path(self.temp.name) / "models--org--model"
        artifact = model_root / "snapshots" / revision / "FP16" / "model-F16.gguf"
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
            "name": "sparkdeck-gguf", "port": 8080, "status": "running",
            "model_source": "public_repository",
        })

        with patch("sparkdeck.service.launch_managed_container", launch):
            created = await self.service.create_deployment({
                "model": "org/model", "alias": "prepared-gguf",
                "runtime": "llama.cpp", "revision": "release-gguf",
                "quantization": "f16",
                "settings": {"artifact": "FP16/model-F16.gguf"},
            })

        virtual_nas.resolve_download_revision.assert_awaited_once_with(
            "org/model", "release-gguf",
        )
        virtual_nas.download_model_files_checked.assert_awaited_once_with(
            "org/model", revision, ["FP16/model-F16.gguf"],
            requested_revision="release-gguf",
        )
        launch_settings = launch.await_args.args[5]
        self.assertEqual(launch_settings["artifact"], str(artifact.resolve()))
        self.assertEqual(created["model"]["repository"], "org/model")
        self.assertEqual(created["model"]["quantization"], "FP16")
        self.assertEqual(created["model"]["artifact"], str(artifact.resolve()))

    @unittest.skipIf(os.name == "nt", "creating cache symlinks requires privileges")
    async def test_prepared_gguf_preserves_logical_snapshot_symlink_name(self):
        revision = "b" * 40
        model_root = Path(self.temp.name) / "models--org--model"
        first_blob = model_root / "blobs" / "first-content-hash"
        second_blob = model_root / "blobs" / "second-content-hash"
        first_blob.parent.mkdir(parents=True)
        first_blob.write_bytes(b"first")
        second_blob.write_bytes(b"second")
        artifact = (
            model_root / "snapshots" / revision
            / "model-00001-of-00002.gguf"
        )
        second_artifact = artifact.with_name("model-00002-of-00002.gguf")
        artifact.parent.mkdir(parents=True)
        artifact.symlink_to(Path("../../blobs/first-content-hash"))
        second_artifact.symlink_to(Path("../../blobs/second-content-hash"))
        virtual_nas = Mock()
        virtual_nas.resolve_download_revision = AsyncMock(return_value={
            "resolved_revision": revision,
        })
        virtual_nas.download_model_files_checked = AsyncMock(return_value={"ok": True})
        virtual_nas._model_path = Mock(return_value=model_root)
        self.manager.virtual_nas = virtual_nas

        prepared = await self.service._prepare_public_gguf_artifact(
            "org/model", "model-00001-of-00002.gguf", "main", None,
        )

        self.assertEqual(prepared, str(artifact))
        self.assertTrue(Path(prepared).is_symlink())

        outside = Path(self.temp.name) / "outside-shard"
        outside.write_bytes(b"outside")
        second_artifact.unlink()
        second_artifact.symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "complete selected GGUF shard set"):
            await self.service._prepare_public_gguf_artifact(
                "org/model", "model-00001-of-00002.gguf", "main", None,
            )

    async def test_prepared_gguf_preserves_uppercase_shard_filename_casing(self):
        revision = "c" * 40
        model_root = Path(self.temp.name) / "models--org--model"
        snapshot = model_root / "snapshots" / revision
        snapshot.mkdir(parents=True)
        first = snapshot / "MODEL-00001-OF-00002.GGUF"
        second = snapshot / "MODEL-00002-OF-00002.GGUF"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        virtual_nas = Mock()
        virtual_nas.resolve_download_revision = AsyncMock(return_value={
            "resolved_revision": revision,
        })
        virtual_nas.download_model_files_checked = AsyncMock(return_value={"ok": True})
        virtual_nas._model_path = Mock(return_value=model_root)
        self.manager.virtual_nas = virtual_nas

        prepared = await self.service._prepare_public_gguf_artifact(
            "org/model", "MODEL-00001-OF-00002.GGUF", "main", None,
        )

        self.assertEqual(prepared, str(first))
        virtual_nas.download_model_files_checked.assert_awaited_once_with(
            "org/model", revision,
            ["MODEL-00001-OF-00002.GGUF", "MODEL-00002-OF-00002.GGUF"],
            requested_revision="main",
        )

    async def test_managed_ownership_is_durable_before_container_launch(self):
        async def fail_launch(*args):
            stored = self.service.store.deployment("durable-launch")
            self.assertIsNotNone(stored)
            self.assertTrue(stored["container_name"].startswith("sparkdeck-durable-launch-"))
            raise RuntimeError("launch failed")

        with patch(
            "sparkdeck.service.launch_managed_container", side_effect=fail_launch,
        ):
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                await self.service.create_deployment({
                    "model": "org/model", "alias": "durable-launch",
                    "runtime": "vllm",
                })

        self.manager.remove_container.assert_awaited_once()
        self.assertIsNone(self.service.store.deployment("durable-launch"))

    async def test_failed_container_cleanup_retains_ownership_record(self):
        self.manager.remove_container.side_effect = RuntimeError("Docker unavailable")
        with patch(
            "sparkdeck.service.launch_managed_container",
            side_effect=RuntimeError("launch failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                await self.service.create_deployment({
                    "model": "org/model", "alias": "retained-launch",
                    "runtime": "vllm",
                })

        stored = self.service.store.deployment("retained-launch")
        self.assertIsNotNone(stored)
        self.assertTrue(stored["container_name"].startswith("sparkdeck-retained-launch-"))



if __name__ == "__main__":
    unittest.main()
