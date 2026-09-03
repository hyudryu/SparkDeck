import asyncio
import contextlib
import json
import os
import shlex
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx


with patch("docker.from_env", return_value=Mock()):
    import server

from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import SparkDeckService, _EXTERNAL_HOOK_KILL_SIGNAL
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

    async def test_clone_deployment_delegates_to_service(self):
        clone = AsyncMock(return_value={"id": "dep-copy", "alias": "(Copy) Model"})
        with patch.object(server.sparkdeck, "clone_deployment", clone):
            response = await self.client.post("/api/v1/deployments/dep-1/clone")

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["alias"], "(Copy) Model")
        clone.assert_awaited_once_with("dep-1")

    async def test_clone_deployment_maps_missing_and_discovered(self):
        clone = AsyncMock(side_effect=LookupError("deployment not found"))
        with patch.object(server.sparkdeck, "clone_deployment", clone):
            missing = await self.client.post("/api/v1/deployments/missing/clone")
        clone.side_effect = ValueError("discovered containers cannot be cloned")
        with patch.object(server.sparkdeck, "clone_deployment", clone):
            discovered = await self.client.post(
                "/api/v1/deployments/container%3Auntracked/clone",
            )

        self.assertEqual(missing.status_code, 404)
        self.assertEqual(discovered.status_code, 400)

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

    async def test_update_deployment_settings_maps_selector_conflict(self):
        update = AsyncMock(side_effect=ValueError(
            "deployment selector 'vision-public' is already in use"
        ))
        with patch.object(server.sparkdeck, "update_deployment_settings", update):
            response = await self.client.put(
                "/api/v1/deployments/dep-1/settings",
                json={
                    "extra_args": [
                        "--served-model-name", "vision-public",
                    ],
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("already in use", response.json()["detail"])


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
                "environment": {"NCCL_DEBUG": "WARN"},
            },
        }
        with patch.object(server.manager, "add_recipe", add_recipe):
            response = await self.client.post("/api/recipes", json=payload)

        self.assertEqual(response.status_code, 200)
        kwargs = add_recipe.await_args.kwargs
        self.assertEqual(kwargs["gpu_memory_utilization"], 0.85)
        self.assertEqual(kwargs["environment"], {"NCCL_DEBUG": "WARN"})
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

    async def test_launch_settings_reject_non_object_launch_controls(self):
        add_recipe = AsyncMock(return_value={"id": "r5"})
        payload = {
            "model": "org/model", "engine": "sglang",
            "launch_settings": {"context_length": 262144},
            "launch_controls": "max_concurrency=10",
        }
        with patch.object(server.manager, "add_recipe", add_recipe):
            response = await self.client.post("/api/recipes", json=payload)

        self.assertEqual(response.status_code, 400)
        add_recipe.assert_not_awaited()

    async def test_launch_settings_reject_out_of_range_sg_scalars(self):
        add_recipe = AsyncMock(
            side_effect=ValueError("sg_mem_fraction must be between 0 and 1")
        )
        payload = {
            "model": "org/model", "engine": "sglang",
            "launch_settings": {"mem_fraction_static": 1.2},
        }
        with patch.object(server.manager, "add_recipe", add_recipe):
            response = await self.client.post("/api/recipes", json=payload)

        self.assertEqual(response.status_code, 400)
        self.assertIn("sg_mem_fraction", response.json()["detail"])

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
        self.assertTrue(body["editable"])
        self.assertIsNone(body["edit_reason"])
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

    async def test_hook_backed_vllm_uses_normal_settings_and_detaches_hooks(self):
        card = {
            "id": "container:vllm-dspark", "alias": "dspark", "runtime": "vllm",
            "kind": "external", "model": {"repository": "org/model"},
            "status": "running", "settings": {},
        }
        container = {
            "name": "vllm-dspark", "image": "example/dspark:latest",
            "start_command": "/opt/stack/start.sh",
            "stop_command": "/opt/stack/stop.sh",
            "settings_env_file": "/opt/stack/.env.dspark",
            "load_settings": {
                "engine": "vllm", "editable": True,
                "extra_args": ["--enable-prefix-caching"],
                "command_flags": (
                    '--enable-prefix-caching '
                    '--max-cudagraph-capture-size "$(( 6 * (${TOKENS:-5} + 1) ))" '
                    "--speculative-config '${SPECULATIVE_CONFIG}'"
                ),
                "context_window": 65536, "max_concurrency": 8,
                "thinking_mode": "default", "gpu_memory_utilization": 0.85,
                "environment": {"NCCL_DEBUG": "INFO"},
            },
        }
        update = AsyncMock(return_value={"ok": True})
        with (
            patch.object(server.sparkdeck, "deployments", AsyncMock(return_value=[card])),
            patch.object(
                server.sparkdeck, "_resolve_discovered_container",
                AsyncMock(return_value=container),
            ),
            patch.object(server.manager, "update_container_settings", update),
        ):
            response = await self.client.put(
                "/api/v1/deployments/container:vllm-dspark/settings",
                json={
                    "command_flags": container["load_settings"]["command_flags"],
                    "launch_controls": {
                        "context_window": 131072,
                        "max_concurrency": 4,
                        "tensor_parallel_size": 2,
                        "pipeline_parallel_size": 1,
                        "kv_cache_dtype": "fp8",
                        "thinking_mode": "disabled",
                        "speculative_method": "dspark",
                        "draft_sample_method": "greedy",
                        "dspark_num_speculative_tokens": 6,
                        "max_cudagraph_capture_size": None,
                        "max_num_batched_tokens": 8192,
                    },
                    "gpu_memory_utilization": 0.8,
                    "gpu_memory_gb": None,
                    "sg_tp_size": None,
                    "sg_mem_fraction": None,
                    "environment": {"NCCL_DEBUG": "WARN"},
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["command_flags"],
            container["load_settings"]["command_flags"],
        )
        self.assertTrue(update.await_args.kwargs["detach_external_lifecycle"])
        replacement = update.await_args.args[1]
        self.assertEqual(replacement["context_window"], 131072)
        self.assertEqual(replacement["max_concurrency"], 4)
        self.assertEqual(replacement["gpu_memory_utilization"], 0.8)
        self.assertEqual(replacement["environment"], {"NCCL_DEBUG": "WARN"})
        flags = replacement["command_flags"]
        self.assertIn("-tp 2", flags)
        self.assertIn("-pp 1", flags)
        self.assertIn("--max-num-batched-tokens 8192", flags)
        self.assertIn('$(( 6 * (${TOKENS:-5} + 1) ))', flags)
        self.assertNotIn("${SPECULATIVE_CONFIG}", flags)
        self.assertIn("--speculative-config '", flags)
        self.assertIn('"num_speculative_tokens":6', flags)

    async def test_settings_takeover_rejects_running_lifecycle_hook(self):
        container = {
            "name": "vllm-dspark", "start_command": "/opt/stack/start.sh",
            "load_settings": {"engine": "vllm", "editable": True},
        }
        blocker = asyncio.Event()
        task = asyncio.create_task(blocker.wait())
        server.sparkdeck._external_lifecycle_tasks["vllm-dspark"] = {
            "action": "start", "task": task,
        }
        update = AsyncMock()
        try:
            with (
                patch.object(
                    server.sparkdeck, "_resolve_discovered_container",
                    AsyncMock(return_value=container),
                ),
                patch.object(server.manager, "update_container_settings", update),
            ):
                response = await self.client.put(
                    "/api/v1/deployments/container:vllm-dspark/settings",
                    json={"extra_args": [], "launch_controls": {}},
                )
            self.assertEqual(response.status_code, 400)
            self.assertIn("start command is already running", response.json()["detail"])
            update.assert_not_awaited()
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            server.sparkdeck._external_lifecycle_tasks.pop("vllm-dspark", None)

    async def test_discovered_vllm_settings_create_and_inline_speculative_environment(self):
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
                    "--speculative-config", "${SPECULATIVE_CONFIG}",
                    "--enable-prefix-caching",
                ],
                "thinking_mode": "default", "environment": {"NCCL_DEBUG": "WARN"},
            },
        }
        update = AsyncMock(return_value={"ok": True})
        with (
            patch.object(server.sparkdeck, "deployments", AsyncMock(return_value=[card])),
            patch.object(
                server.sparkdeck, "_resolve_discovered_container",
                AsyncMock(return_value=container),
            ),
            patch.object(server.manager, "update_container_settings", update),
        ):
            response = await self.client.put(
                "/api/v1/deployments/container:vllm-dspark/settings",
                json={
                    "extra_args": container["load_settings"]["extra_args"],
                    "environment": {"NCCL_DEBUG": "WARN"},
                    "launch_controls": {
                        "speculative_method": "dspark",
                        "draft_sample_method": "probabilistic",
                        "dspark_num_speculative_tokens": 5,
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        replacement = update.await_args.args[1]
        self.assertEqual(
            json.loads(replacement["environment"]["SPECULATIVE_CONFIG"]),
            {
                "method": "dspark",
                "draft_sample_method": "probabilistic",
                "num_speculative_tokens": 5,
            },
        )
        self.assertNotIn("${SPECULATIVE_CONFIG}", replacement["command_flags"])
        self.assertIn("--enable-prefix-caching", replacement["command_flags"])

    async def test_rename_discovered_container_updates_display_alias(self):
        card = {
            "id": "container:vllm-dspark", "alias": "dspark", "runtime": "vllm",
            "kind": "external", "model": {"repository": "org/model"},
            "status": "stopped", "settings": {},
        }
        container = {"name": "vllm-dspark", "image": "example/dspark:latest"}
        rename = AsyncMock(return_value={"ok": True, "alias": "Vision exp"})
        detail = AsyncMock(return_value={**card, "alias": "Vision exp"})
        with (
            patch.object(server.sparkdeck, "deployments", AsyncMock(return_value=[card])),
            patch.object(
                server.sparkdeck, "_resolve_discovered_container",
                AsyncMock(return_value=container),
            ),
            patch.object(server.sparkdeck, "deployment_detail", detail),
            patch.object(server.sparkdeck, "_owning_cluster_deployment", Mock(return_value=None)),
            patch.object(server.manager, "update_container_alias", rename),
        ):
            response = await self.client.patch(
                "/api/v1/deployments/container:vllm-dspark",
                json={"alias": "Vision exp"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["alias"], "Vision exp")
        rename.assert_awaited_once_with("vllm-dspark", "Vision exp")
        detail.assert_awaited_once_with("container:vllm-dspark")

    async def test_rename_discovered_container_rejects_live_alias_conflict(self):
        card = {
            "id": "container:vllm-dspark", "alias": "dspark", "runtime": "vllm",
            "kind": "external", "model": {"repository": "org/model"},
            "status": "stopped", "settings": {},
        }
        other = {
            "id": "container:vision-exp", "alias": "Vision exp", "runtime": "vllm",
            "kind": "external", "model": {"repository": "org/vision"},
            "status": "running", "settings": {},
        }
        rename = AsyncMock()
        with (
            patch.object(
                server.sparkdeck, "deployments",
                AsyncMock(return_value=[card, other]),
            ),
            patch.object(
                server.sparkdeck, "_resolve_discovered_container",
                AsyncMock(return_value={"name": "vllm-dspark"}),
            ),
            patch.object(server.sparkdeck, "_owning_cluster_deployment", Mock(return_value=None)),
            patch.object(server.manager, "update_container_alias", rename),
        ):
            response = await self.client.patch(
                "/api/v1/deployments/container:vllm-dspark",
                json={"alias": "vision EXP"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("already in use", response.json()["detail"])
        rename.assert_not_awaited()

    async def test_rename_discovered_container_rejects_deployment_id_conflict(self):
        card = {
            "id": "container:vllm-dspark", "alias": "dspark", "runtime": "vllm",
            "kind": "external", "model": {"repository": "org/model"},
            "status": "stopped", "settings": {},
        }
        other = {
            "id": "saved-deployment-id", "alias": "Different alias", "runtime": "vllm",
            "kind": "managed", "model": {"repository": "org/other"},
            "status": "saved", "settings": {},
        }
        rename = AsyncMock()
        with (
            patch.object(
                server.sparkdeck, "deployments",
                AsyncMock(return_value=[card, other]),
            ),
            patch.object(
                server.sparkdeck, "_resolve_discovered_container",
                AsyncMock(return_value={"name": "vllm-dspark"}),
            ),
            patch.object(server.sparkdeck, "_owning_cluster_deployment", Mock(return_value=None)),
            patch.object(server.manager, "update_container_alias", rename),
        ):
            response = await self.client.patch(
                "/api/v1/deployments/container:vllm-dspark",
                json={"alias": "SAVED-DEPLOYMENT-ID"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("already in use", response.json()["detail"])
        rename.assert_not_awaited()

    async def test_rename_discovered_container_rejects_saved_repository_conflict(self):
        card = {
            "id": "container:vllm-dspark", "alias": "dspark", "runtime": "vllm",
            "kind": "external", "model": {"repository": "org/model"},
            "status": "stopped", "settings": {},
        }
        saved = {
            "id": "saved-1", "alias": "Future bookmark", "runtime": "vllm",
            "kind": "managed", "model": {"repository": "org/future-model"},
            "status": "saved",
            "settings": {
                "extra_args": ["--served-model-name", "future-public-name"],
            },
        }
        rename = AsyncMock()
        with (
            patch.object(
                server.sparkdeck, "deployments",
                AsyncMock(return_value=[card, saved]),
            ),
            patch.object(
                server.sparkdeck, "_resolve_discovered_container",
                AsyncMock(return_value={"name": "vllm-dspark"}),
            ),
            patch.object(server.sparkdeck, "_owning_cluster_deployment", Mock(return_value=None)),
            patch.object(server.manager, "update_container_alias", rename),
        ):
            response = await self.client.patch(
                "/api/v1/deployments/container:vllm-dspark",
                json={"alias": "ORG/FUTURE-MODEL"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("already in use", response.json()["detail"])
        rename.assert_not_awaited()

    async def test_rename_discovered_container_rejects_saved_served_name_conflict(self):
        card = {
            "id": "container:vllm-dspark", "alias": "dspark", "runtime": "vllm",
            "kind": "external", "model": {"repository": "org/model"},
            "status": "stopped", "settings": {},
        }
        saved = {
            "id": "saved-1", "alias": "Future bookmark", "runtime": "vllm",
            "kind": "managed", "model": {"repository": "org/future-model"},
            "status": "saved",
            "settings": {
                "extra_args": ["--served-model-name", "future-public-name"],
            },
        }
        rename = AsyncMock()
        with (
            patch.object(
                server.sparkdeck, "deployments",
                AsyncMock(return_value=[card, saved]),
            ),
            patch.object(
                server.sparkdeck, "_resolve_discovered_container",
                AsyncMock(return_value={"name": "vllm-dspark"}),
            ),
            patch.object(server.sparkdeck, "_owning_cluster_deployment", Mock(return_value=None)),
            patch.object(server.manager, "update_container_alias", rename),
        ):
            response = await self.client.patch(
                "/api/v1/deployments/container:vllm-dspark",
                json={"alias": "FUTURE-PUBLIC-NAME"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("already in use", response.json()["detail"])
        rename.assert_not_awaited()

    async def test_rename_discovered_container_rejects_public_model_id_conflict(self):
        card = {
            "id": "container:vllm-dspark", "alias": "dspark", "runtime": "vllm",
            "kind": "external", "model": {"repository": "org/model"},
            "status": "stopped", "settings": {},
        }
        other = {
            "id": "managed-1", "alias": "Different alias", "runtime": "vllm",
            "kind": "managed", "model": {"repository": "org/other"},
            "served_models": ["org/other", "public-vision-model"],
            "status": "running", "settings": {},
        }
        rename = AsyncMock()
        with (
            patch.object(
                server.sparkdeck, "deployments",
                AsyncMock(return_value=[card, other]),
            ),
            patch.object(
                server.sparkdeck, "_resolve_discovered_container",
                AsyncMock(return_value={"name": "vllm-dspark"}),
            ),
            patch.object(server.sparkdeck, "_owning_cluster_deployment", Mock(return_value=None)),
            patch.object(server.manager, "update_container_alias", rename),
        ):
            response = await self.client.patch(
                "/api/v1/deployments/container:vllm-dspark",
                json={"alias": "PUBLIC-VISION-MODEL"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("already in use", response.json()["detail"])
        rename.assert_not_awaited()

    async def test_rename_discovered_container_rejects_cluster_member(self):
        rename = AsyncMock()
        with (
            patch.object(
                server.sparkdeck, "_resolve_discovered_container",
                AsyncMock(return_value={"name": "cluster-rank-0"}),
            ),
            patch.object(
                server.sparkdeck, "_owning_cluster_deployment",
                Mock(return_value={"id": "managed-cluster"}),
            ),
            patch.object(server.manager, "update_container_alias", rename),
        ):
            response = await self.client.patch(
                "/api/v1/deployments/container:cluster-rank-0",
                json={"alias": "Only rank zero"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("rename the cluster deployment", response.json()["detail"])
        rename.assert_not_awaited()

    async def test_rename_discovered_container_rejects_running_hook(self):
        blocker = asyncio.Event()
        task = asyncio.create_task(blocker.wait())
        server.sparkdeck._external_lifecycle_tasks["vllm-dspark"] = {
            "action": "start", "task": task,
        }
        rename = AsyncMock()
        try:
            with (
                patch.object(
                    server.sparkdeck, "_resolve_discovered_container",
                    AsyncMock(return_value={"name": "vllm-dspark"}),
                ),
                patch.object(server.sparkdeck, "_owning_cluster_deployment", Mock(return_value=None)),
                patch.object(server.manager, "update_container_alias", rename),
            ):
                response = await self.client.patch(
                    "/api/v1/deployments/container:vllm-dspark",
                    json={"alias": "Vision exp"},
                )
            self.assertEqual(response.status_code, 400)
            self.assertIn("start command is already running", response.json()["detail"])
            rename.assert_not_awaited()
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            server.sparkdeck._external_lifecycle_tasks.pop("vllm-dspark", None)

    async def test_runtime_flags_preview_resolves_environment_backed_speculation(self):
        response = await self.client.post(
            "/api/v1/runtime-flags/preview",
            json={
                "runtime": "vllm",
                "extra_args": ["--speculative-config", "${SPECULATIVE_CONFIG}"],
                "environment": {"NCCL_DEBUG": "WARN"},
                "launch_controls": {
                    "speculative_method": "dspark",
                    "draft_sample_method": "probabilistic",
                    "dspark_num_speculative_tokens": 5,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertNotIn("${SPECULATIVE_CONFIG}", body["command_flags"])
        self.assertEqual(
            json.loads(body["environment"]["SPECULATIVE_CONFIG"]),
            {
                "method": "dspark",
                "draft_sample_method": "probabilistic",
                "num_speculative_tokens": 5,
            },
        )

    async def test_discovered_partial_update_preserves_memory_utilization(self):
        card = {
            "id": "container:vllm-partial", "alias": "partial", "runtime": "vllm",
            "kind": "external", "model": {"repository": "org/model"},
            "status": "running", "settings": {},
        }
        container = {
            "name": "vllm-partial", "load_settings": {
                "engine": "vllm", "editable": True, "extra_args": [],
                "context_window": 65536, "max_concurrency": 8,
                "thinking_mode": "default", "gpu_memory_utilization": 0.85,
            },
        }
        update = AsyncMock(return_value={"ok": True})
        with (
            patch.object(server.sparkdeck, "deployments", AsyncMock(return_value=[card])),
            patch.object(
                server.sparkdeck, "_resolve_discovered_container",
                AsyncMock(return_value=container),
            ),
            patch.object(server.manager, "update_container_settings", update),
        ):
            response = await self.client.put(
                "/api/v1/deployments/container:vllm-partial/settings",
                json={
                    "environment": {"NCCL_DEBUG": "WARN"},
                    "launch_controls": {"context_window": 131072},
                },
            )

        self.assertEqual(response.status_code, 200)
        replacement = update.await_args.args[1]
        self.assertEqual(replacement["gpu_memory_utilization"], 0.85)
        self.assertEqual(replacement["context_window"], 131072)
        self.assertEqual(replacement["max_concurrency"], 8)

    async def test_unparseable_discovered_settings_remain_read_only(self):
        card = {
            "id": "container:custom", "alias": "custom", "runtime": "vllm",
            "kind": "external", "model": {"repository": "org/model"},
            "status": "running", "settings": {},
        }
        container = {
            "name": "custom", "load_settings": {
                "engine": "vllm", "editable": False, "extra_args": [],
            },
        }
        patches = self._stub_discovered(card, container)
        with patches[0], patches[1]:
            response = await self.client.get("/api/v1/deployments/container:custom")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["editable"])
        self.assertIn("cannot be edited safely", response.json()["edit_reason"])

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

    async def test_discovered_sglang_settings_map_runtime_specific_fields(self):
        card = {
            "id": "container:qwen-sglang", "alias": "qwen", "runtime": "sglang",
            "kind": "external", "model": {"repository": "org/model"},
            "status": "running", "settings": {},
        }
        container = {
            "name": "qwen-sglang", "load_settings": {
                "engine": "sglang", "editable": True,
                "extra_args": [
                    "--max-total-tokens", "999", "--enable-metrics",
                ],
                "context_window": 131072, "max_concurrency": 8,
                "kv_cache_dtype": "fp8_e4m3", "thinking_mode": "default",
                "gpu_memory_utilization": 0.9, "tensor_parallel_size": 1,
            },
        }
        update = AsyncMock(return_value={"ok": True})
        with (
            patch.object(server.sparkdeck, "deployments", AsyncMock(return_value=[card])),
            patch.object(
                server.sparkdeck, "_resolve_discovered_container",
                AsyncMock(return_value=container),
            ),
            patch.object(server.manager, "update_container_settings", update),
        ):
            response = await self.client.put(
                "/api/v1/deployments/container:qwen-sglang/settings",
                json={
                    "launch_controls": {
                        "context_window": 262144, "max_concurrency": 4,
                        "kv_cache_dtype": "fp8_e4m3", "thinking_mode": "default",
                    },
                    "gpu_memory_utilization": None, "gpu_memory_gb": None,
                    "sg_tp_size": 2, "sg_mem_fraction": 0.75,
                    "environment": {},
                },
            )

        self.assertEqual(response.status_code, 200)
        replacement = update.await_args.args[1]
        self.assertEqual(replacement["context_window"], 262144)
        self.assertEqual(replacement["max_concurrency"], 4)
        self.assertEqual(replacement["gpu_memory_utilization"], 0.75)
        self.assertIn("--tp-size 2", replacement["command_flags"])
        self.assertNotIn("--max-total-tokens 999", replacement["command_flags"])
        self.assertIn(
            "--max-total-tokens 2097152", replacement["command_flags"]
        )

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


class HookBackedEnvFileSettingsTests(unittest.IsolatedAsyncioTestCase):
    """Hook-backed cards with an env-file label edit the file, not Docker."""

    ENV_TEXT = (
        "# regenerated by start.sh\n"
        "SERVED_MODEL_NAME='org/old'\n"
        "MAX_MODEL_LEN=262144\n"
        "# MAX_NUM_SEQS=32\n"
        "API_TOKEN=supersecret\n"
    )

    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test",
        )
        self.dir = tempfile.TemporaryDirectory()
        self.env_path = Path(self.dir.name) / ".env.dspark"
        self.env_path.write_text(self.ENV_TEXT, encoding="utf-8")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.dir.cleanup()

    def _card(self):
        return {
            "id": "container:vllm-dspark", "alias": "dspark", "runtime": "vllm",
            "kind": "external", "model": {"repository": "org/model"},
            "status": "stopped", "settings": {},
        }

    def _container(self, env_file=True):
        container = {
            "name": "vllm-dspark", "image": "example/dspark:latest",
            "start_command": "/opt/stack/start.sh",
            "stop_command": "/opt/stack/stop.sh",
            "load_settings": {"engine": "vllm", "editable": False, "extra_args": []},
        }
        if env_file:
            container["settings_env_file"] = str(self.env_path)
        return container

    def _patches(self, container):
        return (
            patch.object(
                server.sparkdeck, "deployments",
                AsyncMock(return_value=[self._card()]),
            ),
            patch.object(
                server.sparkdeck, "_resolve_discovered_container",
                AsyncMock(return_value=container),
            ),
        )

    async def test_detail_is_env_file_editable_with_secrets_redacted(self):
        patches = self._patches(self._container())
        with patches[0], patches[1]:
            response = await self.client.get(
                "/api/v1/deployments/container:vllm-dspark",
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["editable"])
        self.assertIsNone(body["edit_reason"])
        self.assertEqual(body["edit_mode"], "env-file")
        settings_env = body["settings_env"]
        # The absolute host path stays server-side; only the basename leaks.
        self.assertEqual(settings_env["name"], ".env.dspark")
        self.assertNotIn("path", settings_env)
        self.assertNotIn(str(self.env_path), json.dumps(settings_env))
        self.assertIsInstance(settings_env["mtime"], float)
        entries = {entry["key"]: entry for entry in settings_env["entries"]}
        self.assertEqual(entries["SERVED_MODEL_NAME"]["value"], "'org/old'")
        self.assertFalse(entries["MAX_NUM_SEQS"]["enabled"])
        self.assertIsNone(entries["API_TOKEN"]["value"])
        self.assertTrue(entries["API_TOKEN"]["redacted"])
        self.assertEqual(
            settings_env["field_mapping"]["served_model_name"], "SERVED_MODEL_NAME",
        )

    async def test_detail_reports_unreadable_env_file_but_stays_editable(self):
        container = self._container()
        container["settings_env_file"] = str(Path(self.dir.name) / "missing.env")
        patches = self._patches(container)
        with patches[0], patches[1]:
            response = await self.client.get(
                "/api/v1/deployments/container:vllm-dspark",
            )

        body = response.json()
        self.assertTrue(body["editable"])
        self.assertEqual(body["edit_mode"], "env-file")
        self.assertIn("error", body["settings_env"])
        self.assertEqual(body["settings_env"]["name"], "missing.env")

    async def test_clearing_the_served_name_writes_an_empty_assignment(self):
        patches = self._patches(self._container())
        with patches[0], patches[1]:
            response = await self.client.put(
                "/api/v1/deployments/container:vllm-dspark/settings",
                json={"served_model_name": ""},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "SERVED_MODEL_NAME=\n", self.env_path.read_text(encoding="utf-8"),
        )

    async def test_save_writes_the_env_file_and_never_recreates_the_container(self):
        update = AsyncMock(return_value={"ok": True})
        patches = self._patches(self._container())
        with patches[0], patches[1], patch.object(
            server.manager, "update_container_settings", update,
        ):
            response = await self.client.put(
                "/api/v1/deployments/container:vllm-dspark/settings",
                json={
                    "launch_controls": {"context_window": 131072},
                    "served_model_name": "org/new",
                    # The UI sends line-addressed operations so duplicate-key
                    # rows edit the exact line shown.
                    "environment": [
                        {"key": "MAX_NUM_SEQS", "line": 4,
                         "value": {"value": "16", "enabled": True}},
                        {"key": "API_TOKEN", "line": 5, "value": None},
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        update.assert_not_called()
        text = self.env_path.read_text(encoding="utf-8")
        self.assertIn("# regenerated by start.sh", text)
        self.assertIn("SERVED_MODEL_NAME=org/new\n", text)
        self.assertIn("MAX_MODEL_LEN=131072\n", text)
        self.assertIn("MAX_NUM_SEQS=16\n", text)
        self.assertNotIn("API_TOKEN", text)
        self.assertTrue(response.json()["restart_required"])
        # The saved state is reflected by the returned detail payload.
        entries = {
            entry["key"]: entry for entry in response.json()["settings_env"]["entries"]
        }
        self.assertTrue(entries["MAX_NUM_SEQS"]["enabled"])

    async def test_clearing_a_mapped_control_comments_the_variable_out(self):
        patches = self._patches(self._container())
        with patches[0], patches[1]:
            response = await self.client.put(
                "/api/v1/deployments/container:vllm-dspark/settings",
                json={"launch_controls": {"context_window": None}},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "# MAX_MODEL_LEN=262144\n", self.env_path.read_text(encoding="utf-8"),
        )
        entries = {
            entry["key"]: entry for entry in response.json()["settings_env"]["entries"]
        }
        self.assertFalse(entries["MAX_MODEL_LEN"]["enabled"])

    async def test_secret_shaped_values_are_redacted_in_detail(self):
        self.env_path.write_text(
            "DATABASE_URL=postgres://user:pass@db:5432/serving\n"
            "MAX_MODEL_LEN=262144\n",
            encoding="utf-8",
        )
        patches = self._patches(self._container())
        with patches[0], patches[1]:
            response = await self.client.get(
                "/api/v1/deployments/container:vllm-dspark",
            )

        entries = {
            entry["key"]: entry
            for entry in response.json()["settings_env"]["entries"]
        }
        self.assertIsNone(entries["DATABASE_URL"]["value"])
        self.assertTrue(entries["DATABASE_URL"]["redacted"])
        self.assertEqual(entries["MAX_MODEL_LEN"]["value"], "262144")

    async def test_save_rejects_controls_without_a_backing_variable(self):
        container = self._container()
        env_text = "SERVED_MODEL_NAME='org/old'\n"
        self.env_path.write_text(env_text, encoding="utf-8")
        patches = self._patches(container)
        with patches[0], patches[1]:
            response = await self.client.put(
                "/api/v1/deployments/container:vllm-dspark/settings",
                json={"launch_controls": {"max_concurrency": 4}},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("max_concurrency", response.json()["detail"])
        self.assertIn("MAX_NUM_SEQS", response.json()["detail"])
        self.assertEqual(self.env_path.read_text(encoding="utf-8"), env_text)

    async def test_save_rejects_extra_args_edits(self):
        patches = self._patches(self._container())
        with patches[0], patches[1]:
            response = await self.client.put(
                "/api/v1/deployments/container:vllm-dspark/settings",
                json={"extra_args": ["--enforce-eager"]},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("external start script", response.json()["detail"])

    async def test_hook_card_without_env_file_label_is_not_editable(self):
        patches = self._patches(self._container(env_file=False))
        with patches[0], patches[1]:
            response = await self.client.put(
                "/api/v1/deployments/container:vllm-dspark/settings",
                json={"launch_controls": {"context_window": 131072}},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("io.sparkdeck.env-file", response.json()["detail"])

    async def test_save_with_a_stale_mtime_conflicts(self):
        patches = self._patches(self._container())
        with patches[0], patches[1]:
            response = await self.client.put(
                "/api/v1/deployments/container:vllm-dspark/settings",
                json={
                    "launch_controls": {"context_window": 131072},
                    "env_file_mtime": 12345.0,
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("changed on disk", response.json()["detail"])
        self.assertEqual(
            self.env_path.read_text(encoding="utf-8"), self.ENV_TEXT,
        )


class ExternalEndpointProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_loading_discovered_card_keeps_starting_phase_when_port_is_closed(self):
        deployment = {
            "id": "container:loading", "kind": "external",
            "runtime": "vllm", "status": "starting", "port": 8123,
            "launch_phase": "loading",
            "launch_message": "loading checkpoint shards 7/48",
        }
        health = AsyncMock(side_effect=RuntimeError("not listening yet"))
        with patch.object(
            server.sparkdeck.registry, "get",
            Mock(return_value=SimpleNamespace(health=health)),
        ), patch.object(
            server.sparkdeck, "_get_credential", Mock(return_value=None),
        ):
            await server.sparkdeck._probe_external_endpoint(deployment)

        self.assertEqual(deployment["status"], "starting")
        self.assertEqual(deployment["launch_phase"], "loading")
        self.assertNotIn("last_error", deployment)

    async def test_discovered_card_without_port_keeps_docker_status(self):
        deployment = {
            "id": "container:host-net", "kind": "external",
            "runtime": "sglang", "status": "running", "port": None,
        }
        await server.sparkdeck._probe_external_endpoint(deployment)

        self.assertEqual(deployment["status"], "running")
        self.assertNotIn("last_error", deployment)

    async def test_discovered_card_with_port_probes_derived_url(self):
        deployment = {
            "id": "container:bound", "kind": "external",
            "runtime": "vllm", "status": "running", "port": 8123,
        }
        health = AsyncMock()
        with patch.object(
            server.sparkdeck.registry, "get",
            Mock(return_value=SimpleNamespace(health=health)),
        ), patch.object(
            server.sparkdeck, "_get_credential", Mock(return_value=None),
        ):
            await server.sparkdeck._probe_external_endpoint(deployment)

        health.assert_awaited_once()
        self.assertEqual(health.await_args.args[1], "http://127.0.0.1:8123")
        self.assertEqual(deployment["status"], "running")

    async def test_running_discovered_card_keeps_docker_status_when_probe_is_unauthorized(self):
        deployment = {
            "id": "container:protected", "kind": "external",
            "runtime": "vllm", "status": "running", "port": 8123,
        }
        health = AsyncMock(side_effect=RuntimeError("401 Unauthorized"))
        with patch.object(
            server.sparkdeck.registry, "get",
            Mock(return_value=SimpleNamespace(health=health)),
        ), patch.object(
            server.sparkdeck, "_get_credential", Mock(return_value=None),
        ):
            await server.sparkdeck._probe_external_endpoint(deployment)

        health.assert_awaited_once()
        self.assertEqual(deployment["status"], "running")
        self.assertNotIn("last_error", deployment)

    async def test_stored_external_still_probes_saved_base_url(self):
        deployment = {
            "id": "dep-ext", "kind": "external",
            "runtime": "vllm", "status": "unknown", "_base_url": "http://10.0.0.9:8000",
        }
        health = AsyncMock(side_effect=RuntimeError("unreachable"))
        with patch.object(
            server.sparkdeck.registry, "get",
            Mock(return_value=SimpleNamespace(health=health)),
        ), patch.object(
            server.sparkdeck, "_get_credential", Mock(return_value=None),
        ):
            await server.sparkdeck._probe_external_endpoint(deployment)

        self.assertEqual(deployment["status"], "error")
        self.assertEqual(deployment["last_error"], "Endpoint health check failed")


class ExternalLifecycleHookTests(unittest.IsolatedAsyncioTestCase):
    class FakeManager:
        def __init__(self):
            self.http = httpx.AsyncClient()
            self.deployments = []
            self.list_containers = AsyncMock(return_value=[])
            self.create_deployment = AsyncMock()
            self.start_container = AsyncMock(return_value={"ok": True})
            self.stop_container = AsyncMock(return_value={"ok": True})
            self.recipe_model_preparation_preflight = AsyncMock(return_value={
                "eligible": True, "action": "ready", "targets": [],
            })
            self.queue_recipe_model_preparation = AsyncMock(return_value={
                "workflow_id": None, "job_ids": [], "jobs": [],
                "plan": {"eligible": True, "action": "ready"},
            })
            self.pin_container_image_on_nodes = AsyncMock(
                return_value="sha256:source-image",
            )

        @staticmethod
        def _has_unresolvable_shell_tokens(args):
            return type(server.manager)._has_unresolvable_shell_tokens(args)

        @staticmethod
        def _recovered_deployment_launch_settings(deployment, container=None):
            settings = (container or {}).get("load_settings") or {}
            command_flags = settings.get("command_flags")
            recovered_args = (
                shlex.split(command_flags)
                if isinstance(command_flags, str)
                else list(settings.get("extra_args") or [])
            )
            recovered = {
                "model": settings.get("model") or deployment.get("model"),
                "engine": deployment.get("engine", "vllm"),
                "environment": dict(settings.get("environment") or {}),
                "extra_args": recovered_args,
            }
            if recovered["engine"] == "sglang":
                recovered["sg_tp_size"] = settings.get(
                    "tensor_parallel_size"
                )
            return recovered

        @staticmethod
        def recipe_deployment_contract(recipe):
            return server.manager.recipe_deployment_contract(recipe)

        @staticmethod
        def _cli_option(args, names, cast=None):
            return server.manager._cli_option(args, names, cast)

    class FakeStream:
        def __init__(self, chunks=()):
            self._chunks = list(chunks)

        async def read(self, _size=-1):
            return self._chunks.pop(0) if self._chunks else b""

    class FakeProcess:
        def __init__(self, blocker=None, returncode=0, stdout=(b"stack up\n",), stderr=()):
            self.returncode = None
            self._result = returncode
            self.pid = 43210
            self.stdout = ExternalLifecycleHookTests.FakeStream(stdout)
            self.stderr = ExternalLifecycleHookTests.FakeStream(stderr)
            self._blocker = blocker
            self.terminated = False
            self.killed = False

        async def wait(self):
            if self._blocker is not None:
                await self._blocker.wait()
            if self.returncode is None:
                self.returncode = self._result
            return self.returncode

        def terminate(self):
            self.terminated = True
            if self.returncode is None:
                self.returncode = -15
            if self._blocker is not None:
                self._blocker.set()

        def kill(self):
            self.killed = True
            if self.returncode is None:
                self.returncode = -9
            if self._blocker is not None:
                self._blocker.set()

    def _service(self, directory, container):
        if container.get("direct_start"):
            # Hand-built direct-start fixtures model a canonical ``vllm serve``
            # summary unless a test explicitly supplies the opposite.
            container.setdefault("launch_prefix_replayable", True)
            container.setdefault("environment_replayable", True)
            container.setdefault("user_replayable", True)
            container.setdefault("working_dir_replayable", True)
            container.setdefault("gpu_requests_replayable", True)
            container.setdefault("ipc_mode_replayable", True)
            container.setdefault("resource_constraints_replayable", True)
            container.setdefault("image_replayable", True)
        manager = self.FakeManager()
        manager.list_containers.return_value = [container]
        return SparkDeckService(manager, Path(directory)), manager

    async def test_card_advertises_hook_booleans_without_raw_commands(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited",
            "load_settings": {"tensor_parallel_size": 2},
            "start_command": "/opt/stack/start.sh --token secret",
            "stop_command": "/opt/stack/stop.sh",
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            card = service._discovered_deployment(container, "vllm", "org/model")

            self.assertTrue(card["has_start_hook"])
            self.assertTrue(card["has_stop_hook"])
            self.assertFalse(card["promotable"])
            payload = json.dumps(card)
            self.assertNotIn("start_command", payload)
            self.assertNotIn("stop_command", payload)
            self.assertNotIn("/opt/stack", payload)
            await manager.http.aclose()
            await service.close()

    async def test_card_without_hooks_advertises_false_booleans(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "load_settings": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            card = service._discovered_deployment(container, "vllm", "org/model")

            self.assertFalse(card["has_start_hook"])
            self.assertFalse(card["has_stop_hook"])
            self.assertTrue(card["promotable"])
            await manager.http.aclose()
            await service.close()

    async def test_card_with_either_hook_is_not_promotable(self):
        for hook in (
            {"start_command": "/opt/stack/start.sh"},
            {"stop_command": "/opt/stack/stop.sh"},
        ):
            with self.subTest(hook=next(iter(hook))):
                container = {
                    "name": "external-stack", "model": "org/model",
                    "engine": "vllm", "managed": False, "status": "exited",
                    "load_settings": {}, **hook,
                }
                with tempfile.TemporaryDirectory() as directory:
                    service, manager = self._service(directory, container)
                    card = service._discovered_deployment(
                        container, "vllm", "org/model",
                    )

                    self.assertFalse(card["promotable"])
                    await manager.http.aclose()
                    await service.close()

    async def test_start_with_hook_spawns_script_instead_of_docker_start(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited",
            "load_settings": {"tensor_parallel_size": 2},
            "start_command": "/opt/stack/start.sh",
            "stop_command": "/opt/stack/stop.sh",
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            spawn = AsyncMock(return_value=self.FakeProcess())
            with patch("asyncio.create_subprocess_shell", spawn):
                result = await service.deployment_action(
                    "container:external-stack", "start",
                    node_ids=["local", "worker-1"],
                )
                task = service._external_lifecycle_tasks["external-stack"]["task"]
                await task

            spawn.assert_awaited_once_with(
                "/opt/stack/start.sh",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **(
                    {"start_new_session": True}
                    if os.name == "posix"
                    else {
                        "creationflags": getattr(
                            subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200,
                        )
                    }
                    if os.name == "nt"
                    else {}
                ),
            )
            manager.start_container.assert_not_awaited()
            manager.create_deployment.assert_not_awaited()
            self.assertEqual(result["status"], "running")
            self.assertNotIn(
                "external-stack", service._external_lifecycle_tasks,
            )
            await manager.http.aclose()
            await service.close()

    async def test_start_without_hook_falls_back_to_docker_start(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "load_settings": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            spawn = AsyncMock(return_value=self.FakeProcess())
            with patch("asyncio.create_subprocess_shell", spawn):
                result = await service.deployment_action(
                    "container:external-stack", "start",
                )

            spawn.assert_not_awaited()
            manager.start_container.assert_awaited_once_with(
                "external-stack", explicit=True, managed=False,
            )
            self.assertEqual(result["status"], "running")
            await manager.http.aclose()
            await service.close()

    async def test_direct_start_uses_node_picker_to_create_managed_deployment(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "load_settings": {
                "tensor_parallel_size": 2,
                "command_flags": (
                    "--tensor-parallel-size 2 --pipeline-parallel-size 2 "
                    "--nnodes 4 --max-num-seqs 8 --enable-prefix-caching"
                ),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            promoted = AsyncMock(return_value={"id": "managed", "status": "starting"})
            with patch.object(
                service, "_promote_discovered_deployment", promoted,
            ):
                result = await service.deployment_action(
                    "container:external-stack", "start",
                    node_ids=["local", "worker-1", "worker-2", "worker-3"],
                )

            promoted.assert_awaited_once_with(
                unittest.mock.ANY, container,
                ["local", "worker-1", "worker-2", "worker-3"],
            )
            manager.start_container.assert_not_awaited()
            self.assertEqual(result["id"], "managed")
            card = service._discovered_deployment(
                container, "vllm", "org/model",
            )
            self.assertTrue(card["promotable"])
            self.assertEqual(card["deployment_mode"], "sharded")
            self.assertEqual(card["required_node_count"], 4)
            self.assertEqual(card["parallel_rank_count"], 4)
            self.assertFalse(card["flexible_node_count"])
            await manager.http.aclose()
            await service.close()

    async def test_direct_start_allows_parallel_ranks_on_one_gpu_rich_host(self):
        container = {
            "name": "external-tp8", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "load_settings": {
                "tensor_parallel_size": 8,
                "command_flags": "--tensor-parallel-size 8",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            deployment = service._discovered_deployment(
                container, "vllm", "org/model",
            )
            manager.create_deployment.return_value = {
                "id": "managed-cluster", "launch_settings": {},
            }
            adopted = Mock(return_value={"id": "managed", "status": "starting"})

            with patch.object(service, "_adopt_manager_replacement", adopted):
                result = await service._promote_discovered_deployment(
                    deployment, container, ["local"],
                )

            launch = manager.create_deployment.await_args.args[0]
            self.assertEqual(launch["deployment_mode"], "single")
            self.assertEqual(launch["node_ids"], ["local"])
            self.assertEqual(result["id"], "managed")
            await manager.http.aclose()
            await service.close()

    async def test_direct_sglang_tp_is_flexible_and_preserved(self):
        container = {
            "name": "external-sglang-tp8", "model": "org/model",
            "engine": "sglang", "managed": False, "status": "exited",
            "direct_start": True, "mounts_replayable": True,
            "load_settings": {
                "model": "org/model", "tensor_parallel_size": 8,
                "command_flags": "--tp-size 8 --context-length 32768",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            deployment = service._discovered_deployment(
                container, "sglang", "org/model",
            )
            manager.create_deployment.return_value = {
                "id": "managed-cluster", "launch_settings": {},
            }
            adopted = Mock(return_value={"id": "managed", "status": "starting"})

            self.assertTrue(deployment["flexible_node_count"])
            self.assertEqual(deployment["parallel_rank_count"], 8)
            with patch.object(service, "_adopt_manager_replacement", adopted):
                await service._promote_discovered_deployment(
                    deployment, container, ["local"],
                )
                single = manager.create_deployment.await_args.args[0]
                manager.create_deployment.reset_mock()
                await service._promote_discovered_deployment(
                    deployment, container,
                    ["local", "worker-1", "worker-2", "worker-3"],
                )
                sharded = manager.create_deployment.await_args.args[0]

            self.assertEqual(single["deployment_mode"], "single")
            self.assertEqual(single["sg_tp_size"], 8)
            self.assertEqual(sharded["deployment_mode"], "sharded")
            self.assertEqual(sharded["sg_tp_size"], 8)
            with self.assertRaisesRegex(ValueError, "ranks must divide evenly"):
                await service._promote_discovered_deployment(
                    deployment, container, ["local", "worker-1", "worker-2"],
                )

            await manager.http.aclose()
            await service.close()

    async def test_direct_start_rejects_selection_that_does_not_divide_tp_pp(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "load_settings": {
                "tensor_parallel_size": 2,
                "command_flags": (
                    "--tensor-parallel-size 2 --pipeline-parallel-size 2"
                ),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            deployment = service._discovered_deployment(
                container, "vllm", "org/model",
            )

            with self.assertRaisesRegex(ValueError, "ranks must divide evenly"):
                await service._promote_discovered_deployment(
                    deployment, container, ["local", "worker-1", "worker-2"],
                )

            manager.create_deployment.assert_not_awaited()
            manager.start_container.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_direct_start_promotion_carries_recovered_tp_pp_contract(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "image": "example/vllm:latest",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "load_settings": {
                "tensor_parallel_size": 2,
                "command_flags": (
                    "--tensor-parallel-size 2 --pipeline-parallel-size 2 "
                    "--enable-prefix-caching"
                ),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            deployment = service._discovered_deployment(
                container, "vllm", "org/model",
            )
            manager.create_deployment.return_value = {
                "id": "managed-cluster", "launch_settings": {},
            }
            adopted = Mock(return_value={"id": "managed", "status": "starting"})
            selected = ["local", "worker-1", "worker-2", "worker-3"]

            with patch.object(service, "_adopt_manager_replacement", adopted):
                result = await service._promote_discovered_deployment(
                    deployment, container, selected,
                )

            launch = manager.create_deployment.await_args.args[0]
            self.assertEqual(launch["deployment_mode"], "sharded")
            self.assertEqual(launch["node_ids"], selected)
            self.assertEqual(launch["image"], "sha256:source-image")
            self.assertIn("--pipeline-parallel-size", launch["extra_args"])
            manager.pin_container_image_on_nodes.assert_awaited_once_with(
                "external-stack", "example/vllm:latest", selected,
            )
            self.assertEqual(result["id"], "managed")
            await manager.http.aclose()
            await service.close()

    async def test_direct_start_rejects_node_without_source_image_identity(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "image": "example/vllm:latest",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "load_settings": {"command_flags": "--tensor-parallel-size 2"},
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            deployment = service._discovered_deployment(
                container, "vllm", "org/model",
            )
            manager.pin_container_image_on_nodes.side_effect = ValueError(
                "source image identity is unavailable on selected node(s): Worker",
            )

            with self.assertRaisesRegex(ValueError, "image identity.*Worker"):
                await service._promote_discovered_deployment(
                    deployment, container, ["local", "worker-1"],
                )

            manager.create_deployment.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    def test_mount_replayability_allows_only_managed_volume_contracts(self):
        inspect = server.manager._container_mounts_are_replayable
        managed_cache = str(Path("/host/hf").expanduser().resolve())
        other_cache = str(Path("/other/hf").expanduser().resolve())
        with patch.object(
            server.manager, "_image_hf_cache_target", return_value="/opt/hf-cache",
        ), patch.object(
            server.manager, "_resolve_local_path", return_value=None,
        ), patch.dict(server.manager.settings, {"hf_cache": "/host/hf"}):
            self.assertTrue(inspect({
                "Mounts": [{
                    "Type": "bind", "Source": managed_cache,
                    "Destination": "/opt/hf-cache", "Mode": "rw", "RW": True,
                }],
            }, "vision:latest", "org/model"))
            self.assertFalse(inspect({
                "Mounts": [], "HostConfig": {"Binds": []},
            }, "vision:latest", "org/model"))
            self.assertFalse(inspect({
                "Mounts": [{
                    "Type": "bind", "Source": managed_cache,
                    "Destination": "/opt/hf-cache", "Mode": "ro", "RW": False,
                }],
            }, "vision:latest", "org/model"))
            self.assertFalse(inspect({
                "Mounts": [{
                    "Type": "bind", "Source": managed_cache,
                    "Destination": "/opt/hf-cache", "Mode": "rw", "RW": True,
                }],
                "HostConfig": {
                    "Binds": [f"{managed_cache}:/opt/hf-cache:ro"],
                },
            }, "vision:latest", "org/model"))
            self.assertFalse(inspect({
                "Mounts": [{
                    "Type": "bind", "Source": "/host/config",
                    "Destination": "/config", "Mode": "rw", "RW": True,
                }],
            }, "vision:latest", "org/model"))
            self.assertFalse(inspect({
                "Mounts": [{
                    "Type": "volume", "Source": "runtime-config",
                    "Destination": "/config", "Mode": "rw", "RW": True,
                }],
            }, "vision:latest", "org/model"))
            self.assertTrue(inspect({
                "HostConfig": {
                    "Binds": [f"{managed_cache}:/opt/hf-cache:rw"],
                },
            }, "vision:latest", "org/model"))
            self.assertFalse(inspect({
                "HostConfig": {
                    "Binds": [f"{managed_cache}:/opt/hf-cache:ro"],
                },
            }, "vision:latest", "org/model"))
            self.assertFalse(inspect({
                "Mounts": [{
                    "Type": "bind", "Source": other_cache,
                    "Destination": "/opt/hf-cache", "Mode": "rw", "RW": True,
                }],
            }, "vision:latest", "org/model"))
            self.assertFalse(inspect({
                "HostConfig": {
                    "Binds": [f"{other_cache}:/opt/hf-cache:rw"],
                },
            }, "vision:latest", "org/model"))
            self.assertFalse(inspect({
                "HostConfig": {"Binds": ["/host/config:/config:ro"]},
            }, "vision:latest", "org/model"))

    async def test_custom_mount_direct_start_stays_on_fixed_lifecycle(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": False,
            "load_settings": {
                "command_flags": "--chat-template /config/template.jinja",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            card = service._discovered_deployment(container, "vllm", "org/model")
            result = await service.deployment_action(
                "container:external-stack", "start",
            )

            self.assertFalse(card["promotable"])
            manager.start_container.assert_awaited_once_with(
                "external-stack", explicit=True, managed=False,
            )
            manager.create_deployment.assert_not_awaited()
            self.assertEqual(result["status"], "running")
            await manager.http.aclose()
            await service.close()

    async def test_custom_mount_direct_start_rejects_explicit_promotion(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": False,
            "load_settings": {
                "command_flags": "--chat-template /config/template.jinja",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            with self.assertRaisesRegex(
                ValueError, "custom mounts cannot be reproduced",
            ):
                await service.deployment_action(
                    "container:external-stack", "start",
                    node_ids=["local"], promote=True,
                )

            manager.create_deployment.assert_not_awaited()
            manager.start_container.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_shell_dependent_direct_start_stays_on_fixed_lifecycle(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "load_settings": {
                "tensor_parallel_size": 2,
                "command_flags": (
                    "--max-cudagraph-capture-size "
                    "'$(( 6 * (${TOKENS:-5} + 1) ))' --enable-prefix-caching"
                ),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            card = service._discovered_deployment(
                container, "vllm", "org/model",
            )
            result = await service.deployment_action(
                "container:external-stack", "start",
            )

            self.assertFalse(card["promotable"])
            manager.start_container.assert_awaited_once_with(
                "external-stack", explicit=True, managed=False,
            )
            manager.create_deployment.assert_not_awaited()
            self.assertEqual(result["status"], "running")
            await manager.http.aclose()
            await service.close()

    async def test_shell_dependent_direct_start_rejects_explicit_promotion(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "load_settings": {
                "tensor_parallel_size": 2,
                "command_flags": (
                    "--tensor-parallel-size 2 --flag '$(resolve-value)'"
                ),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            with self.assertRaisesRegex(ValueError, "depends on shell expansion"):
                await service.deployment_action(
                    "container:external-stack", "start",
                    node_ids=["local", "worker-1"], promote=True,
                )

            manager.create_deployment.assert_not_awaited()
            manager.start_container.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_shell_path_expansion_blocks_direct_start_promotion(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "load_settings": {
                "command_flags": "--chat-template /templates/*.jinja",
                "_shell_path_expansion": True,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            card = service._discovered_deployment(
                container, "vllm", "org/model",
            )
            with self.assertRaisesRegex(ValueError, "pathname expansion"):
                await service.deployment_action(
                    "container:external-stack", "start",
                    node_ids=["local"], promote=True,
                )

            self.assertFalse(card["promotable"])
            manager.create_deployment.assert_not_awaited()
            manager.start_container.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_bash_brace_expansion_blocks_direct_start_promotion(self):
        load_settings = server.manager._container_load_settings(
            [
                "/bin/bash", "-lc",
                "exec vllm serve org/model "
                "--chat-template /templates/{chat,base}.jinja",
            ],
            "vllm", "org/model",
        )
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True, "load_settings": load_settings,
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            card = service._discovered_deployment(
                container, "vllm", "org/model",
            )
            with self.assertRaisesRegex(ValueError, "Bash brace expansion"):
                await service.deployment_action(
                    "container:external-stack", "start",
                    node_ids=["local"], promote=True,
                )

            self.assertFalse(card["promotable"])
            manager.create_deployment.assert_not_awaited()
            manager.start_container.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_bash_ansi_c_quoting_blocks_direct_start_promotion(self):
        load_settings = server.manager._container_load_settings(
            [
                "/bin/bash", "-lc",
                "exec vllm serve org/model "
                "--served-model-name $'model\\nname'",
            ],
            "vllm", "org/model",
        )
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True, "load_settings": load_settings,
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            card = service._discovered_deployment(
                container, "vllm", "org/model",
            )
            with self.assertRaisesRegex(ValueError, "Bash ANSI-C quoting"):
                await service.deployment_action(
                    "container:external-stack", "start",
                    node_ids=["local"], promote=True,
                )

            self.assertFalse(card["promotable"])
            manager.create_deployment.assert_not_awaited()
            manager.start_container.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_literal_shell_path_metacharacters_remain_promotable(self):
        for value in ("'/templates/*.jinja'", r"\~/templates/\*.jinja"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                load_settings = server.manager._container_load_settings(
                    [
                        "/bin/bash", "-lc",
                        "exec vllm serve org/model "
                        f"--chat-template {value}",
                    ],
                    "vllm", "org/model",
                )
                container = {
                    "name": "external-stack", "model": "org/model",
                    "engine": "vllm", "managed": False, "status": "exited",
                    "direct_start": True, "mounts_replayable": True,
                    "load_settings": load_settings,
                }
                service, manager = self._service(directory, container)

                card = service._discovered_deployment(
                    container, "vllm", "org/model",
                )

                self.assertTrue(card["promotable"])
                await manager.http.aclose()
                await service.close()

    async def test_filtered_speculative_environment_blocks_promotion(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "load_settings": {
                "command_flags": "--speculative-config '${ALT_CONFIG}'",
                # Docker discovery intentionally filtered ALT_CONFIG out.
                "environment": {},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            card = service._discovered_deployment(
                container, "vllm", "org/model",
            )
            with self.assertRaisesRegex(ValueError, "depends on shell expansion"):
                await service.deployment_action(
                    "container:external-stack", "start",
                    node_ids=["local"], promote=True,
                )

            self.assertFalse(card["promotable"])
            manager.create_deployment.assert_not_awaited()
            manager.start_container.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_preserved_speculative_environment_allows_promotion(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "load_settings": {
                "command_flags": (
                    "--speculative-config '${SPECULATIVE_CONFIG}'"
                ),
                "environment": {
                    "SPECULATIVE_CONFIG": '{"method":"dspark"}',
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            card = service._discovered_deployment(
                container, "vllm", "org/model",
            )

            self.assertTrue(card["promotable"])
            await manager.http.aclose()
            await service.close()

    async def test_direct_start_promotion_can_prepare_selected_node_weights(self):
        container = {
            "name": "external-stack", "model": "served-alias", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "load_settings": {
                "model": "org/actual-model",
                "command_flags": "--max-num-seqs 8 --revision pinned-release",
                "environment": {"HF_HUB_OFFLINE": "1"},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            plan = await service.deployment_preparation_preflight(
                "container:external-stack", ["remote-1"],
            )
            prepared = await service.deployment_prepare(
                "container:external-stack", ["remote-1"],
            )

            self.assertTrue(plan["eligible"])
            self.assertEqual(prepared["jobs"], [])
            manager.recipe_model_preparation_preflight.assert_awaited_once_with(
                "org/actual-model", "pinned-release", ["remote-1"],
            )
            manager.queue_recipe_model_preparation.assert_awaited_once_with(
                "org/actual-model", "pinned-release", ["remote-1"],
            )
            await manager.http.aclose()
            await service.close()

    async def test_entrypoint_wrapper_blocks_direct_start_promotion(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "launch_prefix_replayable": False,
            "load_settings": {"command_flags": "--max-num-seqs 8"},
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            card = service._discovered_deployment(
                container, "vllm", "org/model",
            )
            with self.assertRaisesRegex(ValueError, "executable prefix"):
                await service.deployment_action(
                    "container:external-stack", "start",
                    node_ids=["local"], promote=True,
                )

            self.assertFalse(card["promotable"])
            manager.create_deployment.assert_not_awaited()
            manager.start_container.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_unreplayed_environment_blocks_direct_start_promotion(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "environment_replayable": False,
            "load_settings": {"command_flags": "--max-num-seqs 8"},
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            card = service._discovered_deployment(
                container, "vllm", "org/model",
            )
            with self.assertRaisesRegex(ValueError, "environment overrides"):
                await service.deployment_action(
                    "container:external-stack", "start",
                    node_ids=["local"], promote=True,
                )

            self.assertFalse(card["promotable"])
            manager.create_deployment.assert_not_awaited()
            manager.start_container.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_container_user_override_blocks_direct_start_promotion(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "user_replayable": False,
            "load_settings": {"command_flags": "--max-num-seqs 8"},
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            card = service._discovered_deployment(
                container, "vllm", "org/model",
            )
            with self.assertRaisesRegex(ValueError, "container user override"):
                await service.deployment_action(
                    "container:external-stack", "start",
                    node_ids=["local"], promote=True,
                )

            self.assertFalse(card["promotable"])
            manager.create_deployment.assert_not_awaited()
            manager.start_container.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_working_directory_override_blocks_direct_start_promotion(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "working_dir_replayable": False,
            "load_settings": {"command_flags": "--max-num-seqs 8"},
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            card = service._discovered_deployment(
                container, "vllm", "org/model",
            )
            with self.assertRaisesRegex(ValueError, "working directory override"):
                await service.deployment_action(
                    "container:external-stack", "start",
                    node_ids=["local"], promote=True,
                )

            self.assertFalse(card["promotable"])
            manager.create_deployment.assert_not_awaited()
            manager.start_container.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_restricted_gpu_request_blocks_direct_start_promotion(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "gpu_requests_replayable": False,
            "load_settings": {"command_flags": "--max-num-seqs 8"},
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            card = service._discovered_deployment(
                container, "vllm", "org/model",
            )
            with self.assertRaisesRegex(ValueError, "GPU device request"):
                await service.deployment_action(
                    "container:external-stack", "start",
                    node_ids=["local"], promote=True,
                )

            self.assertFalse(card["promotable"])
            manager.create_deployment.assert_not_awaited()
            manager.start_container.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_non_host_ipc_mode_blocks_direct_start_promotion(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "ipc_mode_replayable": False,
            "load_settings": {"command_flags": "--max-num-seqs 8"},
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            card = service._discovered_deployment(
                container, "vllm", "org/model",
            )
            with self.assertRaisesRegex(ValueError, "host IPC contract"):
                await service.deployment_action(
                    "container:external-stack", "start",
                    node_ids=["local"], promote=True,
                )

            self.assertFalse(card["promotable"])
            manager.create_deployment.assert_not_awaited()
            manager.start_container.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_resource_limit_blocks_direct_start_promotion(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "resource_constraints_replayable": False,
            "load_settings": {"command_flags": "--max-num-seqs 8"},
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            card = service._discovered_deployment(
                container, "vllm", "org/model",
            )
            with self.assertRaisesRegex(ValueError, "Docker resource constraints"):
                await service.deployment_action(
                    "container:external-stack", "start",
                    node_ids=["local"], promote=True,
                )

            self.assertFalse(card["promotable"])
            manager.create_deployment.assert_not_awaited()
            manager.start_container.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_repointed_image_blocks_direct_start_promotion(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "direct_start": True,
            "mounts_replayable": True,
            "image_replayable": False,
            "load_settings": {"command_flags": "--max-num-seqs 8"},
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            card = service._discovered_deployment(
                container, "vllm", "org/model",
            )
            with self.assertRaisesRegex(ValueError, "source image"):
                await service.deployment_action(
                    "container:external-stack", "start",
                    node_ids=["local"], promote=True,
                )

            self.assertFalse(card["promotable"])
            manager.create_deployment.assert_not_awaited()
            manager.start_container.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_explicit_hook_promotion_remains_rejected(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited",
            "load_settings": {"tensor_parallel_size": 2},
            "start_command": "/opt/stack/start.sh",
            "stop_command": "/opt/stack/stop.sh",
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)

            with self.assertRaisesRegex(ValueError, "cannot be converted"):
                await service.deployment_action(
                    "container:external-stack", "start",
                    node_ids=["local", "worker-1"], promote=True,
                )

            manager.create_deployment.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_stop_interrupts_running_start_hook_before_stop_hook(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "load_settings": {},
            "start_command": "/opt/stack/start.sh",
            "stop_command": "/opt/stack/stop.sh",
        }
        blocker = asyncio.Event()
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            start_process = self.FakeProcess(blocker=blocker)
            stop_process = self.FakeProcess()
            spawn = AsyncMock(side_effect=[start_process, stop_process])

            def signal_process(process, sig, _group_id=None):
                if sig == signal.SIGTERM:
                    process.terminate()
                else:
                    process.kill()

            with (
                patch("asyncio.create_subprocess_shell", spawn),
                patch.object(
                    service, "_signal_process_group", side_effect=signal_process,
                ),
            ):
                await service.deployment_action(
                    "container:external-stack", "start",
                )
                # Yield so the background hook task reaches the subprocess call.
                await asyncio.sleep(0)
                with self.assertRaisesRegex(ValueError, "start.*already running"):
                    await service.deployment_action(
                        "container:external-stack", "start",
                    )
                result = await service.deployment_action(
                    "container:external-stack", "stop",
                )
                stop_task = service._external_lifecycle_tasks["external-stack"]["task"]
                await stop_task

                self.assertTrue(start_process.terminated)
                self.assertEqual(
                    [call.args[0] for call in spawn.await_args_list],
                    ["/opt/stack/start.sh", "/opt/stack/stop.sh"],
                )
                self.assertEqual(result["status"], "stopped")

            manager.start_container.assert_not_awaited()
            manager.stop_container.assert_not_awaited()
            self.assertNotIn(
                "external-stack", service._external_lifecycle_tasks,
            )
            await manager.http.aclose()
            await service.close()

    async def test_immediate_stop_waits_for_start_process_then_interrupts_it(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "load_settings": {},
            "start_command": "/opt/stack/start.sh",
            "stop_command": "/opt/stack/stop.sh",
        }
        spawn_started = asyncio.Event()
        release_spawn = asyncio.Event()
        start_blocker = asyncio.Event()
        start_process = self.FakeProcess(blocker=start_blocker)
        stop_process = self.FakeProcess()

        async def spawn(command, **_kwargs):
            if command == "/opt/stack/start.sh":
                spawn_started.set()
                await release_spawn.wait()
                return start_process
            return stop_process

        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            with (
                patch("asyncio.create_subprocess_shell", side_effect=spawn),
                patch.object(
                    service, "_signal_process_group",
                    side_effect=lambda process, _sig, _group_id=None: process.terminate(),
                ),
            ):
                await service.deployment_action(
                    "container:external-stack", "start",
                )
                await spawn_started.wait()
                stop_request = asyncio.create_task(service.deployment_action(
                    "container:external-stack", "stop",
                ))
                await asyncio.sleep(0)
                self.assertFalse(stop_request.done())

                release_spawn.set()
                result = await stop_request
                stop_task = service._external_lifecycle_tasks["external-stack"]["task"]
                await stop_task

            self.assertTrue(start_process.terminated)
            self.assertEqual(result["status"], "stopped")
            await manager.http.aclose()
            await service.close()

    async def test_stop_kills_group_when_start_leader_exits_but_task_hangs(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "load_settings": {},
            "start_command": "/opt/stack/start.sh",
            "stop_command": "/opt/stack/stop.sh",
        }
        child_pipe_open = asyncio.Event()
        start_process = self.FakeProcess(blocker=child_pipe_open)
        # Model a shell leader that exited while a descendant inherited its
        # pipes, leaving the lifecycle task waiting for final cleanup.
        start_process.returncode = 0
        stop_process = self.FakeProcess()
        spawn = AsyncMock(side_effect=[start_process, stop_process])
        signals = []

        def signal_process(process, sig, group_id=None):
            signals.append((sig, group_id))
            if sig != signal.SIGTERM:
                process.kill()

        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            with (
                patch("asyncio.create_subprocess_shell", spawn),
                patch.object(
                    service, "_signal_process_group", side_effect=signal_process,
                ),
                patch("sparkdeck.service._EXTERNAL_HOOK_TERM_TIMEOUT", 0.01),
                patch("sparkdeck.service._EXTERNAL_HOOK_KILL_TIMEOUT", 0.1),
            ):
                await service.deployment_action(
                    "container:external-stack", "start",
                )
                await asyncio.sleep(0)
                result = await service.deployment_action(
                    "container:external-stack", "stop",
                )
                stop_task = service._external_lifecycle_tasks["external-stack"]["task"]
                await stop_task

            self.assertEqual(
                [item[0] for item in signals],
                [signal.SIGTERM, _EXTERNAL_HOOK_KILL_SIGNAL],
            )
            if os.name in {"posix", "nt"}:
                self.assertEqual(signals[0][1], start_process.pid)
            self.assertTrue(start_process.killed)
            self.assertEqual(result["status"], "stopped")
            await manager.http.aclose()
            await service.close()

    async def test_cancelling_hook_task_kills_spawned_process_group(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "load_settings": {},
            "start_command": "/opt/stack/start.sh",
        }
        blocker = asyncio.Event()
        process = self.FakeProcess(blocker=blocker)
        signals = []

        def signal_process(target, sig, group_id=None):
            signals.append((sig, group_id))
            target.kill()

        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            with (
                patch(
                    "asyncio.create_subprocess_shell",
                    AsyncMock(return_value=process),
                ),
                patch.object(
                    service, "_signal_process_group", side_effect=signal_process,
                ),
            ):
                await service.deployment_action(
                    "container:external-stack", "start",
                )
                await asyncio.sleep(0)
                task = service._external_lifecycle_tasks["external-stack"]["task"]
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

            self.assertEqual(signals[0][0], _EXTERNAL_HOOK_KILL_SIGNAL)
            if os.name in {"posix", "nt"}:
                self.assertEqual(signals[0][1], process.pid)
            self.assertTrue(process.killed)
            await manager.http.aclose()
            await service.close()

    async def test_stop_with_hook_spawns_script_instead_of_docker_stop(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "running", "load_settings": {},
            "start_command": "/opt/stack/start.sh",
            "stop_command": "/opt/stack/stop.sh",
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            spawn = AsyncMock(return_value=self.FakeProcess())
            with patch("asyncio.create_subprocess_shell", spawn):
                result = await service.deployment_action(
                    "container:external-stack", "stop",
                )
                task = service._external_lifecycle_tasks["external-stack"]["task"]
                await task

            spawn.assert_awaited_once_with(
                "/opt/stack/stop.sh",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **(
                    {"start_new_session": True}
                    if os.name == "posix"
                    else {
                        "creationflags": getattr(
                            subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200,
                        )
                    }
                    if os.name == "nt"
                    else {}
                ),
            )
            manager.stop_container.assert_not_awaited()
            self.assertEqual(result["status"], "stopped")
            self.assertNotIn(
                "external-stack", service._external_lifecycle_tasks,
            )
            await manager.http.aclose()
            await service.close()

    async def test_stop_without_hook_falls_back_to_docker_stop(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "running", "load_settings": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            spawn = AsyncMock(return_value=self.FakeProcess())
            with patch("asyncio.create_subprocess_shell", spawn):
                result = await service.deployment_action(
                    "container:external-stack", "stop",
                )

            spawn.assert_not_awaited()
            manager.stop_container.assert_awaited_once_with(
                "external-stack", explicit=True, managed=False,
            )
            self.assertEqual(result["status"], "stopped")
            await manager.http.aclose()
            await service.close()

    async def test_close_signals_hook_process_group_then_escalates(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "load_settings": {},
            "start_command": "/opt/stack/start.sh",
        }
        blocker = asyncio.Event()
        process = self.FakeProcess(blocker=blocker)
        signals_sent = []

        def fake_killpg(pgid, sig):
            signals_sent.append(sig)
            if len(signals_sent) > 1:
                # The escalation finally brings the group down.
                process.returncode = -9
                blocker.set()

        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            spawn = AsyncMock(return_value=process)
            with (
                patch.object(os, "name", "posix"),
                patch("asyncio.create_subprocess_shell", spawn),
                patch.object(os, "killpg", create=True, side_effect=fake_killpg),
                patch.object(os, "getpgid", create=True, return_value=999),
                patch("sparkdeck.service._EXTERNAL_HOOK_TERM_TIMEOUT", 0.01),
                patch("sparkdeck.service._EXTERNAL_HOOK_KILL_TIMEOUT", 0.1),
            ):
                await service.deployment_action(
                    "container:external-stack", "start",
                )
                # Yield so the hook task publishes its subprocess handle.
                await asyncio.sleep(0)

                await service.close()

            self.assertEqual(signals_sent[0], signal.SIGTERM)
            self.assertEqual(signals_sent[1], _EXTERNAL_HOOK_KILL_SIGNAL)
            self.assertFalse(process.terminated)
            self.assertNotIn(
                "external-stack", service._external_lifecycle_tasks,
            )
            await manager.http.aclose()

    async def test_close_falls_back_to_terminate_without_process_groups(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "load_settings": {},
            "start_command": "/opt/stack/start.sh",
        }
        blocker = asyncio.Event()
        process = self.FakeProcess(blocker=blocker)
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            spawn = AsyncMock(return_value=process)
            with (
                patch("asyncio.create_subprocess_shell", spawn),
                # Simulate a platform without process groups.
                patch.object(os, "killpg", None, create=True),
            ):
                await service.deployment_action(
                    "container:external-stack", "start",
                )
                # Yield so the hook task publishes its subprocess handle.
                await asyncio.sleep(0)

                # No killpg in this environment: terminate the bare shell.
                await service.close()

            self.assertTrue(process.terminated)
            self.assertNotIn(
                "external-stack", service._external_lifecycle_tasks,
            )
            await manager.http.aclose()

    def test_windows_group_signal_falls_back_to_taskkill_tree(self):
        process = self.FakeProcess()
        run = Mock()
        run.return_value.returncode = 0
        with (
            patch.object(os, "name", "nt"),
            patch.object(os, "kill", side_effect=OSError("no console")),
            patch("sparkdeck.service.subprocess.run", run),
        ):
            SparkDeckService._signal_process_group(
                process, signal.SIGTERM, process.pid,
            )

        run.assert_called_once()
        self.assertEqual(
            run.call_args.args[0],
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        )

    async def test_stop_interrupts_running_start_hook_before_docker_fallback(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "load_settings": {},
            # Only the start hook exists; stop would normally fall back to
            # docker stop, but not while the start script is running.
            "start_command": "/opt/stack/start.sh",
        }
        blocker = asyncio.Event()
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            start_process = self.FakeProcess(blocker=blocker)
            spawn = AsyncMock(return_value=start_process)
            with (
                patch("asyncio.create_subprocess_shell", spawn),
                patch.object(
                    service, "_signal_process_group",
                    side_effect=lambda process, _sig, _group_id=None: process.terminate(),
                ),
            ):
                await service.deployment_action(
                    "container:external-stack", "start",
                )
                await asyncio.sleep(0)
                result = await service.deployment_action(
                    "container:external-stack", "stop",
                )

                self.assertTrue(start_process.terminated)
                manager.stop_container.assert_awaited_once_with(
                    "external-stack", explicit=True, managed=False,
                )
                self.assertEqual(result["status"], "stopped")

            await manager.http.aclose()
            await service.close()

    async def test_hook_output_goes_to_data_dir_file_not_server_log(self):
        container = {
            "name": "external-stack", "model": "org/model", "engine": "vllm",
            "managed": False, "status": "exited", "load_settings": {},
            "start_command": "/opt/stack/start.sh",
        }
        with tempfile.TemporaryDirectory() as directory:
            service, manager = self._service(directory, container)
            spawn = AsyncMock(return_value=self.FakeProcess(
                stdout=(b"leaked-credential\n",),
                stderr=(b"secret-path\n",),
            ))
            with patch("asyncio.create_subprocess_shell", spawn):
                with self.assertLogs("sparkdeck.service", level="INFO") as logs:
                    await service.deployment_action(
                        "container:external-stack", "start",
                    )
                    task = (
                        service._external_lifecycle_tasks["external-stack"]["task"]
                    )
                    await task

            log_text = "\n".join(logs.output)
            self.assertIn("exited with code 0", log_text)
            self.assertNotIn("leaked-credential", log_text)
            self.assertNotIn("secret-path", log_text)
            output_file = (
                Path(directory) / "external-hooks" / "external-stack-start.log"
            )
            persisted = output_file.read_text()
            self.assertIn("leaked-credential", persisted)
            self.assertIn("secret-path", persisted)
            await manager.http.aclose()
            await service.close()

    async def test_bounded_output_keeps_only_the_stream_tail(self):
        service = SparkDeckService.__new__(SparkDeckService)
        process = self.FakeProcess(
            stdout=(b"x" * 131072, b"tail-end"),
        )

        stdout_tail, stderr_tail = await service._read_bounded_output(process)

        self.assertLessEqual(len(stdout_tail), 65536)
        self.assertTrue(stdout_tail.endswith(b"tail-end"))
        self.assertEqual(stderr_tail, b"")


class PublicContainerPayloadTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_containers_route_strips_raw_lifecycle_hooks(self):
        summary = {
            "name": "hooked", "status": "running",
            "start_command": "/opt/stack/start.sh --token secret",
            "stop_command": "/opt/stack/stop.sh",
        }
        with patch.object(
            server.manager, "list_containers", AsyncMock(return_value=[summary]),
        ):
            response = await self.client.get("/api/containers")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body[0]["name"], "hooked")
        payload = json.dumps(body)
        self.assertNotIn("start_command", payload)
        self.assertNotIn("stop_command", payload)
        self.assertNotIn("/opt/stack", payload)

    async def test_state_route_strips_raw_lifecycle_hooks(self):
        state = {
            "containers": [
                {"name": "hooked", "start_command": "/opt/stack/start.sh"},
            ],
            "nodes": [{
                "id": "local", "containers": [
                    {"name": "hooked", "stop_command": "/opt/stack/stop.sh"},
                ],
            }],
        }
        with patch.object(
            server.manager, "get_state", AsyncMock(return_value=state),
        ):
            response = await self.client.get("/api/state")

        self.assertEqual(response.status_code, 200)
        payload = json.dumps(response.json())
        self.assertNotIn("start_command", payload)
        self.assertNotIn("stop_command", payload)
        self.assertNotIn("/opt/stack", payload)


if __name__ == "__main__":
    unittest.main()
