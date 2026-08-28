import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx


with patch("docker.from_env", return_value=Mock()):
    import server

from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.storage import SparkDeckStore


RECIPE = {
    "id": "r1",
    "name": "DS config",
    "model": "deepseek-ai/DeepSeek-V4",
    "engine": "vllm",
    "extra_args": ["--max-model-len", "32768", "--enable-prefix-caching"],
    "deployment_mode": "single",
    "node_ids": ["local"],
}


class ModelsApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test",
        )
        # Keep requests local even if the data dir holds a worker assignment.
        self.assignment = patch.object(
            server.onboarding.assignment, "load", return_value=None,
        )
        self.assignment.start()

    async def asyncTearDown(self):
        await self.client.aclose()
        self.assignment.stop()

    async def test_recipe_detail_returns_editable_args_and_launch_controls(self):
        with patch.object(server.manager, "get_recipe", AsyncMock(return_value=dict(RECIPE))):
            response = await self.client.get("/api/v1/recipes/r1")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["extra_args"], RECIPE["extra_args"])
        self.assertEqual(body["extra_args_count"], 3)
        self.assertEqual(body["launch_controls"]["context_window"], 32768)

    async def test_recipe_detail_maps_missing_recipe(self):
        with patch.object(server.manager, "get_recipe", AsyncMock(return_value=None)):
            response = await self.client.get("/api/v1/recipes/missing")

        self.assertEqual(response.status_code, 404)

    async def test_update_recipe_delegates_and_returns_public_detail(self):
        update = AsyncMock(return_value=dict(RECIPE))
        with patch.object(server.manager, "update_recipe", update):
            response = await self.client.put(
                "/api/v1/recipes/r1",
                json={
                    "launch_controls": {"context_window": 65536},
                    "extra_args": ["--enable-prefix-caching"],
                },
            )

        self.assertEqual(response.status_code, 200)
        update.assert_awaited_once_with("r1", {
            "launch_controls": {"context_window": 65536},
            "extra_args": ["--enable-prefix-caching"],
        })
        self.assertEqual(response.json()["extra_args"], RECIPE["extra_args"])

    async def test_update_recipe_rejects_unknown_fields(self):
        response = await self.client.put("/api/v1/recipes/r1", json={"bogus": 1})

        self.assertEqual(response.status_code, 400)

    async def test_update_recipe_maps_validation_and_missing_recipe(self):
        update = AsyncMock(side_effect=ValueError("recipe not found"))
        with patch.object(server.manager, "update_recipe", update):
            missing = await self.client.put("/api/v1/recipes/r1", json={"name": "x"})
        update.side_effect = ValueError("context_window must be a positive integer")
        with patch.object(server.manager, "update_recipe", update):
            invalid = await self.client.put(
                "/api/v1/recipes/r1", json={"launch_controls": {"context_window": -1}},
            )

        self.assertEqual(missing.status_code, 404)
        self.assertEqual(invalid.status_code, 400)

    async def test_delete_recipe_delegates_and_returns_no_content(self):
        delete = AsyncMock(return_value=True)
        with patch.object(server.manager, "delete_recipe", delete):
            response = await self.client.delete("/api/v1/recipes/r1")

        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.content, b"")
        delete.assert_awaited_once_with("r1")

    async def test_delete_recipe_maps_missing_recipe(self):
        with patch.object(server.manager, "delete_recipe", AsyncMock(return_value=False)):
            response = await self.client.delete("/api/v1/recipes/missing")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "saved configuration not found")

    async def test_legacy_delete_recipe_contract_remains_available(self):
        delete = AsyncMock(return_value=True)
        with patch.object(server.manager, "delete_recipe", delete):
            response = await self.client.delete("/api/recipes/r1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})
        delete.assert_awaited_once_with("r1")

    async def test_rename_deployment_delegates_to_service(self):
        rename = AsyncMock(return_value={"id": "dep-1", "alias": "Renamed"})
        with patch.object(server.sparkdeck, "rename_deployment", rename):
            response = await self.client.patch(
                "/api/v1/deployments/dep-1", json={"alias": "Renamed"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["alias"], "Renamed")
        rename.assert_awaited_once_with("dep-1", "Renamed")

    async def test_rename_deployment_maps_validation_and_missing(self):
        rename = AsyncMock(side_effect=ValueError("alias is required"))
        with patch.object(server.sparkdeck, "rename_deployment", rename):
            invalid = await self.client.patch("/api/v1/deployments/dep-1", json={"alias": " "})
        rename.side_effect = LookupError("deployment not found")
        with patch.object(server.sparkdeck, "rename_deployment", rename):
            missing = await self.client.patch(
                "/api/v1/deployments/missing", json={"alias": "Renamed"},
            )

        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(missing.status_code, 404)

    async def test_deployment_detail_returns_curated_editable_settings(self):
        detail = {
            "id": "dep-1", "alias": "Model", "runtime": "vllm",
            "kind": "managed", "model": {"repository": "org/model"},
            "status": "stopped", "settings": {}, "editable": True,
            "edit_reason": None, "desired_state": "stopped",
            "extra_args": ["--enable-prefix-caching"],
            "launch_controls": {"context_window": 32768},
            "gpu_memory_utilization": 0.9, "gpu_memory_gb": None,
            "image": "example/vllm:test",
        }
        get_detail = AsyncMock(return_value=detail)
        with patch.object(server.sparkdeck, "deployment_detail", get_detail):
            response = await self.client.get("/api/v1/deployments/dep-1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), detail)
        get_detail.assert_awaited_once_with("dep-1")

    async def test_deployment_detail_maps_missing_deployment(self):
        get_detail = AsyncMock(side_effect=LookupError("deployment not found"))
        with patch.object(server.sparkdeck, "deployment_detail", get_detail):
            response = await self.client.get("/api/v1/deployments/missing")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "deployment not found")

    async def test_update_deployment_settings_uses_exact_public_contract(self):
        changes = {
            "extra_args": ["--enable-prefix-caching"],
            "launch_controls": {"context_window": 65536},
            "gpu_memory_utilization": 0.9,
            "gpu_memory_gb": None,
        }
        updated = AsyncMock(return_value={
            "id": "dep-1", "editable": True, "extra_args": changes["extra_args"],
        })
        with patch.object(server.sparkdeck, "update_deployment_settings", updated):
            response = await self.client.put(
                "/api/v1/deployments/dep-1/settings", json=changes,
            )

        self.assertEqual(response.status_code, 200)
        updated.assert_awaited_once_with("dep-1", changes)

        response = await self.client.put(
            "/api/v1/deployments/dep-1/settings",
            json={"desired_state": "running"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported field", response.json()["detail"])

    async def test_update_deployment_settings_maps_missing_and_running_state(self):
        update = AsyncMock(side_effect=LookupError("deployment not found"))
        with patch.object(server.sparkdeck, "update_deployment_settings", update):
            missing = await self.client.put(
                "/api/v1/deployments/missing/settings", json={"extra_args": []},
            )

        update.side_effect = ValueError(
            "stop the cluster before changing its launch settings"
        )
        with patch.object(server.sparkdeck, "update_deployment_settings", update):
            running = await self.client.put(
                "/api/v1/deployments/dep-1/settings", json={"extra_args": []},
            )

        self.assertEqual(missing.status_code, 404)
        self.assertEqual(running.status_code, 409)


class DeploymentRenameStoreTests(unittest.TestCase):
    def test_update_alias_persists_new_name(self):
        with tempfile.TemporaryDirectory() as temp:
            store = SparkDeckStore(Path(temp) / "sparkdeck.sqlite3")
            try:
                store.add_deployment(Deployment(
                    id="dep-1", alias="old-name", runtime=RuntimeKind.VLLM,
                    kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
                ))
                store.update_alias("dep-1", "new-name")
                self.assertEqual(store.deployment("dep-1")["alias"], "new-name")
            finally:
                store.close()


class ContainerRecipeImportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_import_returns_public_detail_with_sglang_scalars(self):
        imported = {
            "id": "r9", "name": "qwen3.8-27b-sglang",
            "model": "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead",
            "engine": "sglang",
            "extra_args": ["--kv-cache-dtype", "fp8_e4m3"],
            "deployment_mode": "single", "node_ids": ["local"],
            "sg_context_length": 262144, "sg_max_running_requests": 10,
        }
        to_recipe = AsyncMock(return_value=dict(imported))
        with patch.object(server.manager, "container_to_recipe", to_recipe):
            response = await self.client.post(
                "/api/containers/qwen3.8-27b-sglang/recipe",
            )

        self.assertEqual(response.status_code, 201)
        to_recipe.assert_awaited_once_with("qwen3.8-27b-sglang")
        body = response.json()
        self.assertEqual(body["extra_args_count"], 4)
        self.assertEqual(body["launch_controls"]["context_window"], 262144)

    async def test_import_maps_container_and_value_errors(self):
        missing = AsyncMock(side_effect=LookupError("Container 'gone' not found"))
        bad = AsyncMock(side_effect=ValueError("could not determine the served model"))
        with patch.object(server.manager, "container_to_recipe", missing):
            not_found = await self.client.post("/api/containers/gone/recipe")
        with patch.object(server.manager, "container_to_recipe", bad):
            invalid = await self.client.post("/api/containers/broken/recipe")

        self.assertEqual(not_found.status_code, 404)
        self.assertEqual(invalid.status_code, 400)


class RecipeLaunchSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_sglang_launch_settings_map_to_recipe_scalars(self):
        add_recipe = AsyncMock(return_value={"id": "r2"})
        payload = {
            "model": "org/model", "engine": "sglang",
            "launch_settings": {
                "context_length": 262144,
                "max_running_requests": 10,
                "mem_fraction_static": 0.9,
                "tensor_parallel_size": 1,
                "kv_cache_dtype": "fp8_e4m3",
                "extra_args": ["--enable-metrics"],
            },
        }
        with patch.object(server.manager, "add_recipe", add_recipe):
            response = await self.client.post("/api/recipes", json=payload)

        self.assertEqual(response.status_code, 200)
        kwargs = add_recipe.await_args.kwargs
        self.assertEqual(kwargs["sg_context_length"], 262144)
        self.assertEqual(kwargs["sg_max_running_requests"], 10)
        self.assertEqual(kwargs["sg_mem_fraction"], 0.9)
        self.assertEqual(kwargs["sg_tp_size"], 1)
        self.assertEqual(kwargs["extra_args"], ["--enable-metrics"])
        self.assertEqual(kwargs["launch_controls"], {"kv_cache_dtype": "fp8_e4m3"})

    async def test_vllm_launch_settings_map_to_launch_controls(self):
        add_recipe = AsyncMock(return_value={"id": "r3"})
        payload = {
            "model": "org/model", "engine": "vllm",
            "launch_settings": {
                "max_model_len": 65536,
                "max_running_requests": 16,
                "gpu_memory_utilization": 0.85,
            },
        }
        with patch.object(server.manager, "add_recipe", add_recipe):
            response = await self.client.post("/api/recipes", json=payload)

        self.assertEqual(response.status_code, 200)
        kwargs = add_recipe.await_args.kwargs
        self.assertEqual(kwargs["gpu_memory_utilization"], 0.85)
        self.assertEqual(kwargs["launch_controls"], {
            "context_window": 65536, "max_concurrency": 16,
        })

    async def test_launch_settings_reject_non_string_extra_args(self):
        add_recipe = AsyncMock(return_value={"id": "r4"})
        payload = {
            "model": "org/model", "engine": "vllm",
            "launch_settings": {"extra_args": "--enable-metrics"},
        }
        with patch.object(server.manager, "add_recipe", add_recipe):
            response = await self.client.post("/api/recipes", json=payload)

        self.assertEqual(response.status_code, 400)
        add_recipe.assert_not_awaited()

    async def test_update_recipe_accepts_sglang_scalars(self):
        update = AsyncMock(return_value=dict(RECIPE))
        with patch.object(server.manager, "update_recipe", update):
            response = await self.client.put(
                "/api/v1/recipes/r1",
                json={"sg_mem_fraction": 0.9, "sg_tp_size": 1},
            )

        self.assertEqual(response.status_code, 200)
        update.assert_awaited_once_with(
            "r1", {"sg_mem_fraction": 0.9, "sg_tp_size": 1},
        )


class DiscoveredDeploymentDetailTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    def _stub_discovered(self, card, container):
        return (
            patch.object(server.sparkdeck, "deployments", AsyncMock(return_value=[card])),
            patch.object(
                server.sparkdeck,
                "_resolve_discovered_container",
                AsyncMock(return_value=container),
            ),
        )

    async def test_discovered_vllm_detail_surfaces_container_flags(self):
        card = {
            "id": "container:vllm-dspark", "alias": "dspark", "runtime": "vllm",
            "kind": "external", "model": {"repository": "org/model"},
            "status": "stopped", "settings": {},
        }
        container = {
            "name": "vllm-dspark", "image": "example/dspark:latest",
            "load_settings": {
                "engine": "vllm", "editable": True,
                "extra_args": [
                    "--tensor-parallel-size", "2",
                    "--speculative-config",
                    '{"method":"mtp","num_speculative_tokens":4}',
                    "--enable-prefix-caching",
                ],
                "context_window": 65536, "max_concurrency": 8,
                "kv_cache_dtype": "fp8", "thinking_mode": "default",
                "gpu_memory_utilization": 0.85, "tensor_parallel_size": 2,
            },
        }
        patches = self._stub_discovered(card, container)
        with patches[0], patches[1]:
            response = await self.client.get("/api/v1/deployments/container:vllm-dspark")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["editable"])
        self.assertIn("read-only", body["edit_reason"])
        self.assertEqual(body["launch_controls"]["context_window"], 65536)
        self.assertEqual(body["launch_controls"]["max_concurrency"], 8)
        self.assertEqual(body["launch_controls"]["kv_cache_dtype"], "fp8")
        self.assertEqual(
            body["launch_controls"]["dspark_num_speculative_tokens"], 4,
        )
        self.assertEqual(body["gpu_memory_utilization"], 0.85)
        self.assertIsNone(body["sg_mem_fraction"])
        self.assertEqual(body["image"], "example/dspark:latest")
        self.assertIn("--enable-prefix-caching", body["extra_args"])
        self.assertIn("--tensor-parallel-size", body["extra_args"])
        self.assertNotIn("--max-model-len", body["extra_args"])

    async def test_discovered_sglang_detail_maps_sglang_scalars(self):
        card = {
            "id": "container:qwen3.8-27b-sglang", "alias": "qwen",
            "runtime": "sglang", "kind": "external",
            "model": {"repository": "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead"},
            "status": "running", "settings": {},
        }
        container = {
            "name": "qwen3.8-27b-sglang", "image": "lmsysorg/sglang:latest",
            "load_settings": {
                "engine": "sglang", "editable": True,
                "extra_args": ["--enable-metrics"],
                "context_window": 262144, "max_concurrency": 10,
                "kv_cache_dtype": "fp8_e4m3", "thinking_mode": "default",
                "gpu_memory_utilization": 0.9, "tensor_parallel_size": 1,
            },
        }
        patches = self._stub_discovered(card, container)
        with patches[0], patches[1]:
            response = await self.client.get(
                "/api/v1/deployments/container:qwen3.8-27b-sglang",
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["launch_controls"]["context_window"], 262144)
        self.assertEqual(body["launch_controls"]["max_concurrency"], 10)
        self.assertEqual(body["launch_controls"]["kv_cache_dtype"], "fp8_e4m3")
        self.assertEqual(body["sg_mem_fraction"], 0.9)
        self.assertEqual(body["sg_tp_size"], 1)
        self.assertIsNone(body["gpu_memory_utilization"])
        self.assertEqual(body["extra_args"], ["--enable-metrics"])

    async def test_discovered_detail_falls_back_when_container_is_gone(self):
        card = {
            "id": "container:vanished", "alias": "vanished", "runtime": "vllm",
            "kind": "external", "model": {"repository": "org/model"},
            "status": "stopped", "settings": {},
        }
        patches = (
            patch.object(server.sparkdeck, "deployments", AsyncMock(return_value=[card])),
            patch.object(
                server.sparkdeck,
                "_resolve_discovered_container",
                AsyncMock(side_effect=LookupError("managed container not found")),
            ),
        )
        with patches[0], patches[1]:
            response = await self.client.get("/api/v1/deployments/container:vanished")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["extra_args"], [])
        self.assertEqual(body["launch_controls"], {})


if __name__ == "__main__":
    unittest.main()
