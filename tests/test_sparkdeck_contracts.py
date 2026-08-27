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
