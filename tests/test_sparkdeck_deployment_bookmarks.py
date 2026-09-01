"""Saved-deployment bookmarks: save, prepare weights via Virtual NAS, launch."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx

from manager import Manager
from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
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

    # Reuse Manager's real parser so llama.cpp controls surface correctly.
    _deployment_launch_controls = Manager._deployment_launch_controls
    _normalize_runtime_environment = staticmethod(
        Manager._normalize_runtime_environment
    )

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

    async def test_clone_copies_settings_as_a_stopped_bookmark_with_numbered_names(self):
        self.service.store.add_deployment(Deployment(
            id="running-1", alias="Chat model", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity(
                "org/model", revision="revision-1", quantization="fp8",
            ),
            container_name="sparkdeck-chat-model",
            settings={
                "context_length": 32768,
                "environment": {"NCCL_DEBUG": "WARN"},
                "node_ids": ["remote-1"],
                "deployment_mode": "single",
                "manager_deployment_id": "cluster-1",
                "managed_by": "sparkdeck-mcp",
                "automation_run_id": "run-1",
            },
        ), "http://127.0.0.1:8000")

        first = await self.service.clone_deployment("running-1")
        second = await self.service.clone_deployment(first["id"])

        self.assertEqual(first["alias"], "(Copy) Chat model")
        self.assertEqual(second["alias"], "(Copy 2) Chat model")
        self.assertEqual(first["status"], "saved")
        self.assertEqual(first["desired_state"], "stopped")
        self.assertEqual(first["node_ids"], ["remote-1"])
        self.assertEqual(first["model"]["revision"], "revision-1")
        self.assertEqual(first["model"]["quantization"], "fp8")
        self.assertEqual(first["settings"]["context_length"], 32768)
        self.assertEqual(first["settings"]["environment"], {"NCCL_DEBUG": "WARN"})
        self.assertFalse(first["base_url_set"])
        self.assertNotIn("manager_deployment_id", first["settings"])
        self.assertNotIn("managed_by", first["settings"])
        self.assertNotIn("automation_run_id", first["settings"])
        private = self.service.store.deployment(first["id"], include_private=True)
        self.assertIsNone(private["container_name"])
        self.assertIsNone(private.get("_base_url"))

    async def test_clone_uses_current_manager_argv_instead_of_stale_controls(self):
        current_args = [
            "--max-model-len", "400000",
            "--max-num-seqs", "10",
            "-tp", "4",
            "--kv-cache-dtype", "fp8",
            "--speculative-config",
            (
                '{"method":"dspark","draft_sample_method":"probabilistic",'
                '"num_speculative_tokens":3}'
            ),
        ]
        self.service.store.add_deployment(Deployment(
            id="deepseek", alias="DeepSeek", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/deepseek", revision="revision-1"),
            container_name="rank-0",
            settings={
                "manager_deployment_id": "deepseek-cluster",
                "node_ids": ["local", "remote-1"],
                "deployment_mode": "sharded",
                "launch_controls": {
                    "context_window": 32768,
                    "max_concurrency": 8,
                    "tensor_parallel_size": 2,
                    "draft_sample_method": "greedy",
                    "dspark_num_speculative_tokens": 5,
                    "max_cudagraph_capture_size": 48,
                },
            },
        ), "http://127.0.0.1:8000")
        self.manager.deployments = [{
            "id": "deepseek-cluster",
            "launch_settings": {
                "engine": "vllm",
                "image": "example/deepseek:current",
                "environment": {"NCCL_DEBUG": "WARN"},
                "extra_args": current_args,
                "node_ids": ["local", "remote-1"],
                "deployment_mode": "sharded",
            },
        }]

        cloned = await self.service.clone_deployment("deepseek")

        self.assertEqual(cloned["settings"]["extra_args"], current_args)
        self.assertEqual(
            cloned["settings"]["environment"], {"NCCL_DEBUG": "WARN"},
        )
        self.assertEqual(cloned["settings"]["max_concurrency"], 10)
        controls = cloned["settings"]["launch_controls"]
        self.assertEqual(controls["context_window"], 400000)
        self.assertEqual(controls["max_concurrency"], 10)
        self.assertEqual(controls["tensor_parallel_size"], 4)
        self.assertEqual(controls["draft_sample_method"], "probabilistic")
        self.assertEqual(controls["dspark_num_speculative_tokens"], 3)
        self.assertIsNone(controls["max_cudagraph_capture_size"])

        launch = self.service._cluster_launch_body(
            RuntimeKind.VLLM, "org/deepseek", cloned["alias"], cloned["id"],
            ModelIdentity("org/deepseek", revision="revision-1"),
            cloned["settings"], ["local", "remote-1"], "sharded",
            llama_artifact=None,
        )
        manager = object.__new__(Manager)
        final_args = manager._apply_deployment_launch_controls(
            launch["extra_args"], "vllm", launch["launch_controls"],
            launch["environment"],
        )
        reparsed = manager._deployment_launch_controls({
            "engine": "vllm", "extra_args": final_args,
            "environment": launch["environment"],
        })
        self.assertEqual(reparsed["context_window"], 400000)
        self.assertEqual(reparsed["max_concurrency"], 10)
        self.assertEqual(reparsed["tensor_parallel_size"], 4)
        self.assertEqual(reparsed["draft_sample_method"], "probabilistic")
        self.assertEqual(reparsed["dspark_num_speculative_tokens"], 3)
        self.assertIsNone(reparsed["max_cudagraph_capture_size"])

    async def test_clone_clears_scalars_absent_from_current_manager_argv(self):
        self.service.store.add_deployment(Deployment(
            id="stale-scalars", alias="Stale scalars", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
            settings={
                "manager_deployment_id": "current-cluster",
                "context_length": 32768,
                "max_concurrency": 8,
                "tensor_parallel_size": 4,
                "pipeline_parallel_size": 2,
            },
        ), "http://127.0.0.1:8000")
        self.manager.deployments = [{
            "id": "current-cluster",
            "launch_settings": {"engine": "vllm", "extra_args": []},
        }]

        cloned = await self.service.clone_deployment("stale-scalars")

        for key in (
            "context_length", "max_concurrency", "tensor_parallel_size",
            "pipeline_parallel_size",
        ):
            self.assertNotIn(key, cloned["settings"])

    async def test_clone_preserves_external_endpoint_and_credential(self):
        self.service.store.add_deployment(Deployment(
            id="external-1", alias="Hosted model", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.EXTERNAL, model=ModelIdentity("org/model"),
            settings={"context_length": 8192}, base_url_set=True,
        ), "https://models.example/v1", "keyring:external-1")

        with (
            patch.object(self.service, "_get_credential", return_value="secret") as get_key,
            patch.object(self.service, "_store_credential", return_value="keyring:copy") as store_key,
        ):
            cloned = await self.service.clone_deployment("external-1")

        private = self.service.store.deployment(cloned["id"], include_private=True)
        self.assertEqual(cloned["alias"], "(Copy) Hosted model")
        self.assertEqual(private["_base_url"], "https://models.example/v1")
        self.assertEqual(private["_credential_ref"], "keyring:copy")
        get_key.assert_called_once_with("external-1", "keyring:external-1")
        store_key.assert_called_once_with(cloned["id"], "secret")

    async def test_clone_normalizes_adopted_sglang_controls_for_relaunch(self):
        self.service.store.add_deployment(Deployment(
            id="sg-record", alias="SGLang model", runtime=RuntimeKind.SGLANG,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
            settings={
                "manager_deployment_id": "sg-cluster",
                "node_ids": ["local"],
                "deployment_mode": "single",
                "sg_tp_size": 2,
                "sg_context_length": 262144,
                "sg_max_running_requests": 10,
                "sg_mem_fraction": 0.85,
            },
        ))
        self.manager.deployments = [{
            "id": "sg-cluster",
            "launch_settings": {
                "engine": "sglang",
                "extra_args": ["--enable-metrics"],
                "port": 8123,
                "input_cost_per_1m": 1.25,
                "cache_cost_per_1m": 0.0,
                "output_cost_per_1m": 2.5,
                "sg_tp_size": 2,
                "sg_context_length": 262144,
                "sg_max_running_requests": 10,
                "sg_mem_fraction": 0.85,
            },
        }]

        cloned = await self.service.clone_deployment("sg-record")

        self.assertEqual(cloned["settings"]["tensor_parallel_size"], 2)
        self.assertEqual(cloned["settings"]["context_length"], 262144)
        self.assertEqual(cloned["settings"]["max_running_requests"], 10)
        self.assertEqual(cloned["settings"]["mem_fraction_static"], 0.85)
        self.assertNotIn("port", cloned["settings"])
        self.assertEqual(cloned["settings"]["input_cost_per_1m"], 1.25)
        self.assertEqual(cloned["settings"]["cache_cost_per_1m"], 0.0)
        self.assertEqual(cloned["settings"]["output_cost_per_1m"], 2.5)
        launch = self.service._cluster_launch_body(
            RuntimeKind.SGLANG, "org/model", cloned["alias"], cloned["id"],
            ModelIdentity("org/model"), cloned["settings"], ["local"],
            "single", llama_artifact=None,
        )
        self.assertEqual(launch["sg_tp_size"], 2)
        self.assertEqual(launch["sg_context_length"], 262144)
        self.assertEqual(launch["sg_max_running_requests"], 10)
        self.assertEqual(launch["sg_mem_fraction"], 0.85)
        self.assertNotIn("port", launch)
        self.assertEqual(launch["input_cost_per_1m"], 1.25)
        self.assertEqual(launch["cache_cost_per_1m"], 0.0)
        self.assertEqual(launch["output_cost_per_1m"], 2.5)

    async def test_clone_restores_adopted_llama_artifact_and_controls(self):
        revision = "a" * 40
        cached_artifact = (
            f"models--org--model/snapshots/{revision}/GGUF/model-Q4_K_M.gguf"
        )
        self.service.store.add_deployment(Deployment(
            id="llama-record", alias="Llama model", runtime=RuntimeKind.LLAMA_CPP,
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity(
                "org/model", artifact=cached_artifact,
                quantization="Q4_K_M",
            ),
            settings={
                "manager_deployment_id": "llama-cluster",
                "node_ids": ["remote-1"],
                "deployment_mode": "single",
            },
        ))
        self.manager.deployments = [{
            "id": "llama-cluster",
            "launch_settings": {
                "engine": "llama.cpp",
                "extra_args": [],
                "llama_artifact": cached_artifact,
                "llama_context_length": 16384,
                "llama_parallel_slots": 4,
                "llama_gpu_layers": 77,
            },
        }]

        cloned = await self.service.clone_deployment("llama-record")

        self.assertEqual(
            cloned["model"]["artifact"], "GGUF/model-Q4_K_M.gguf",
        )
        self.assertEqual(cloned["model"]["revision"], revision)
        self.assertEqual(cloned["settings"]["context_length"], 16384)
        self.assertEqual(cloned["settings"]["parallel_slots"], 4)
        self.assertEqual(cloned["settings"]["gpu_layers"], 77)
        prepared_artifact = self.service._hub_relative_llama_artifact(
            "org/model", cloned["model"]["artifact"], revision,
        )
        self.assertEqual(prepared_artifact, cached_artifact)
        self.assertEqual(
            self.service._clone_llama_artifact_identity(
                "org/model", r"C:\models\local.gguf",
            ),
            (r"C:\models\local.gguf", None),
        )
        launch = self.service._cluster_launch_body(
            RuntimeKind.LLAMA_CPP, "org/model", cloned["alias"], cloned["id"],
            ModelIdentity(
                "org/model", revision=revision,
                artifact=cloned["model"]["artifact"], quantization="Q4_K_M",
            ),
            cloned["settings"], ["remote-1"], "single",
            llama_artifact=prepared_artifact,
        )
        self.assertEqual(launch["llama_artifact"], cached_artifact)
        self.assertEqual(launch["llama_context_length"], 16384)
        self.assertEqual(launch["llama_parallel_slots"], 4)
        self.assertEqual(launch["llama_gpu_layers"], 77)

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

    async def test_vllm_environment_round_trips_and_launches_on_every_rank(self):
        environment = {
            "HF_HUB_OFFLINE": "1",
            "VLLM_CACHE_ROOT": "/cache/clusterops-runtime/vllm",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        created = await self.service.create_deployment({
            "model": "org/model", "alias": "env-bookmark", "runtime": "vllm",
            "node_ids": ["local"], "deployment_mode": "single",
            "settings": {"environment": environment},
        })

        self.assertEqual(created["settings"]["environment"], environment)
        detail = await self.service.deployment_detail(created["id"])
        self.assertEqual(detail["environment"], environment)

        updated = {**environment, "NCCL_DEBUG": "WARN"}
        detail = await self.service.update_deployment_settings(created["id"], {
            "environment": updated,
        })
        self.assertEqual(detail["environment"], updated)

        await self.service.deployment_action(created["id"], "start", ["remote-1"])
        launch = self.manager.create_deployment.await_args.args[0]
        self.assertEqual(launch["environment"], updated)

    async def test_runtime_environment_rejects_credentials_and_non_vllm_runtimes(self):
        with self.assertRaisesRegex(ValueError, "managed by SparkDeck"):
            await self.service.create_deployment({
                "model": "org/model", "alias": "secret-env", "runtime": "vllm",
                "node_ids": ["local"], "deployment_mode": "single",
                "settings": {"environment": {"HF_TOKEN": "secret"}},
            })

        with self.assertRaisesRegex(ValueError, "cannot contain a newline"):
            await self.service.create_deployment({
                "model": "org/model", "alias": "multiline-env", "runtime": "vllm",
                "node_ids": ["local"], "deployment_mode": "single",
                "settings": {"environment": {"VLLM_CONFIG": "first\nsecond"}},
            })

        with self.assertRaisesRegex(ValueError, "only supported for vLLM"):
            await self.service.create_deployment({
                "model": "org/model", "alias": "sg-env", "runtime": "sglang",
                "node_ids": ["local"], "deployment_mode": "single",
                "settings": {"environment": {"NCCL_DEBUG": "WARN"}},
            })

    async def test_controller_local_gguf_bookmark_is_saved_for_the_controller(self):
        artifact = Path(self.temp.name) / "local.gguf"
        artifact.write_bytes(b"gguf")

        created = await self.service.create_deployment({
            "model": "org/model", "alias": "local-bookmark",
            "runtime": "llama.cpp",
            "node_ids": ["local"], "deployment_mode": "single",
            "settings": {"artifact": str(artifact)},
        })

        self.assertEqual(created["status"], "saved")
        self.assertEqual(created["node_ids"], ["local"])

        with self.assertRaisesRegex(ValueError, "cannot be distributed"):
            await self.service.create_deployment({
                "model": "org/model", "alias": "remote-local",
                "runtime": "llama.cpp",
                "node_ids": ["local", "remote-1"], "deployment_mode": "replicated",
                "settings": {"artifact": str(artifact)},
            })

    async def test_saved_bookmark_response_exposes_required_node_count(self):
        created = await self.service.create_deployment({
            "model": "org/model", "alias": "replica-bookmark", "runtime": "vllm",
            "node_ids": ["local", "remote-1"], "deployment_mode": "replicated",
        })

        self.assertEqual(created["required_node_count"], 2)
        self.assertEqual(created["deployment_mode"], "replicated")
        listed = (await self.service.deployments())[0]
        self.assertEqual(listed["required_node_count"], 2)

    async def test_running_cluster_deployment_exposes_last_used_at(self):
        self.service.store.add_deployment(Deployment(
            id="running-1", alias="running-model",
            runtime=RuntimeKind.VLLM, kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/model"),
            settings={"manager_deployment_id": "cluster-9",
                      "deployment_mode": "single", "node_ids": ["remote-1"]},
        ), "http://127.0.0.1:8000", None)
        self.manager.deployments = [{
            "id": "cluster-9", "status": "running", "desired_state": "running",
            "node_ids": ["remote-1"], "api_port": 8000,
            "last_used_at": 1_700_000_000.0,
            "members": [{
                "rank": 0, "phase": {"phase": "ready", "message": "Ready"},
            }],
            "launch_settings": {"engine": "vllm", "extra_args": []},
        }]

        listed = next(
            item for item in await self.service.deployments()
            if item["id"] == "running-1"
        )

        self.assertEqual(listed["status"], "running")
        self.assertEqual(listed["last_used_at"], 1_700_000_000.0)

    async def test_saved_bookmark_keeps_launch_inputs_for_relaunch(self):
        await self.service.create_deployment({
            "model": "org/model", "alias": "with-image", "runtime": "vllm",
            "node_ids": ["remote-1"], "deployment_mode": "single",
            "settings": {"image": "private/image", "port": 8021,
                         "gpu_memory_gb": 40},
        })

        started = await self.service.deployment_action("with-image", "start")
        launch = self.manager.create_deployment.await_args.args[0]
        self.assertEqual(launch["image"], "private/image")
        self.assertEqual(launch["port"], 8021)
        self.assertEqual(launch["gpu_memory_gb"], 40)
        self.assertEqual(started["node_ids"], ["remote-1"])

    async def test_saved_vllm_bookmark_image_can_be_changed_before_launch(self):
        await self.service.create_deployment({
            "model": "org/model", "alias": "editable-image", "runtime": "vllm",
            "node_ids": ["remote-1"], "deployment_mode": "single",
            "settings": {"image": "registry.example/vllm:old"},
        })

        detail = await self.service.update_deployment_settings("editable-image", {
            "image": "registry.example/vllm:pinned",
        })
        self.assertEqual(detail["image"], "registry.example/vllm:pinned")

        await self.service.deployment_action("editable-image", "start")
        launch = self.manager.create_deployment.await_args.args[0]
        self.assertEqual(launch["image"], "registry.example/vllm:pinned")

    async def test_sensitive_launch_arguments_are_rejected_before_saving(self):
        self.manager._reject_sensitive_cli_credentials = (
            Manager._reject_sensitive_cli_credentials
        )
        with self.assertRaisesRegex(ValueError, "configure credentials"):
            await self.service.create_deployment({
                "model": "org/model", "alias": "credentialed", "runtime": "vllm",
                "node_ids": ["remote-1"], "deployment_mode": "single",
                "settings": {"extra_args": ["--hf-token", "super-secret"]},
            })
        self.assertIsNone(self.service.store.deployment("credentialed"))

    async def test_saved_deployment_mode_and_nodes_are_validated_together(self):
        await self.service.create_deployment({
            "model": "org/model", "alias": "bookmark", "runtime": "vllm",
            "node_ids": ["local"], "deployment_mode": "single",
        })

        detail = await self.service.update_deployment_settings("bookmark", {
            "deployment_mode": "replicated",
            "node_ids": ["local", "remote-1"],
        })
        self.assertEqual(detail["deployment_mode"], "replicated")
        self.assertEqual(detail["node_ids"], ["local", "remote-1"])

        with self.assertRaisesRegex(ValueError, "exactly one node"):
            await self.service.update_deployment_settings("bookmark", {
                "deployment_mode": "single",
            })

    async def test_controller_local_gguf_bookmark_stays_saved_after_creation(self):
        artifact = Path(self.temp.name) / "local.gguf"
        artifact.write_bytes(b"gguf")

        await self.service.create_deployment({
            "model": "org/model", "alias": "local-bookmark",
            "runtime": "llama.cpp",
            "node_ids": ["local"], "deployment_mode": "single",
            "settings": {"artifact": str(artifact)},
        })

        listed = (await self.service.deployments())[0]
        self.assertEqual(listed["status"], "saved")
        bookmark_id = self.service.store.deployment("local-bookmark")["id"]
        detail = await self.service.deployment_detail(bookmark_id)
        self.assertTrue(detail["editable"])

    async def test_edit_accepts_unchanged_controller_local_artifact(self):
        artifact = Path(self.temp.name) / "local.gguf"
        artifact.write_bytes(b"gguf")
        await self.service.create_deployment({
            "model": "org/model", "alias": "local-bookmark",
            "runtime": "llama.cpp",
            "node_ids": ["local"], "deployment_mode": "single",
            "settings": {"artifact": str(artifact), "context_length": 4096},
        })

        detail = await self.service.update_deployment_settings("local-bookmark", {
            "artifact": str(artifact),
            "context_length": 8192,
        })

        self.assertEqual(detail["status"], "saved")
        # The artifact lives on the model identity, not in filtered settings.
        stored = self.service.store.deployment("local-bookmark", include_private=True)
        self.assertEqual(stored["model"]["artifact"], str(artifact))

    async def test_llama_prepare_downloads_only_selected_files_per_node(self):
        revision = "c" * 40
        virtual_nas = Mock()
        virtual_nas.resolve_download_revision = AsyncMock(return_value={
            "resolved_revision": revision,
        })
        virtual_nas.download_model_files_checked = AsyncMock(return_value={"ok": True})
        virtual_nas.enabled = True
        virtual_nas.estimate_selected_files_size = AsyncMock(return_value=5_000_000_000)
        self.manager.virtual_nas = virtual_nas
        self.manager.node_registry = Mock()
        self.manager.node_registry.request = AsyncMock(return_value={"ok": True})
        self.manager.node_supports_selective_downloads = AsyncMock(return_value=True)
        self.manager.node_has_model_files = AsyncMock(return_value=False)
        self.manager.node_download_model_files = AsyncMock(return_value={"ok": True})
        self.manager.node_transfer_model_files = AsyncMock(return_value={"ok": True})
        self.manager.model_cache_inventory = AsyncMock(return_value=[
            {"id": "local", "models": [], "cache_free_size": 20_000_000_000},
            {"id": "remote-1", "models": [], "cache_free_size": 20_000_000_000},
        ])
        await self.service.create_deployment({
            "model": "org/model", "alias": "llama-bookmark",
            "runtime": "llama.cpp",
            "node_ids": ["local", "remote-1"], "deployment_mode": "replicated",
            "settings": {"artifact": "FP16/model-F16.gguf"},
        })

        plan = await self.service.deployment_preparation_preflight(
            "llama-bookmark", ["local", "remote-1"],
        )
        self.assertEqual(plan["action"], "download")
        self.assertTrue(plan["eligible"])
        for target in plan["targets"]:
            self.assertTrue(target["download_eligible"])
            self.assertIsNotNone(target["required_free_bytes"])

        # A node whose cache cannot hold the selected files is disabled with
        # a per-node capacity reason before the user confirms anything.
        small = await self.service.deployment_preparation_preflight(
            "llama-bookmark", ["remote-1"],
        )
        self.manager.model_cache_inventory = AsyncMock(return_value=[
            {"id": "remote-1", "models": [], "cache_free_size": 100},
        ])
        small = await self.service.deployment_preparation_preflight(
            "llama-bookmark", ["remote-1"],
        )
        self.assertFalse(small["eligible"])
        self.assertFalse(small["targets"][0]["download_eligible"])
        self.assertIn("cache space", small["targets"][0]["download_reason"])

        # Selectively prepared files satisfy readiness even though the
        # snapshot is deliberately marked partial in the inventory.
        self.manager.node_has_model_files = AsyncMock(return_value=True)
        ready = await self.service.deployment_preparation_preflight(
            "llama-bookmark", ["remote-1"],
        )
        self.assertEqual(ready["action"], "ready")

        # Unknown node IDs are never downloadable.
        unknown = await self.service.deployment_preparation_preflight(
            "llama-bookmark", ["ghost"],
        )
        self.assertFalse(unknown["eligible"])
        self.assertEqual(unknown["targets"][0]["download_reason"], "Unknown cluster node")

        # Restore the shared fixtures for the prepare + launch assertions.
        self.manager.node_has_model_files = AsyncMock(return_value=False)
        self.manager.model_cache_inventory = AsyncMock(return_value=[
            {"id": "local", "models": [], "cache_free_size": 20_000_000_000},
            {"id": "remote-1", "models": [], "cache_free_size": 20_000_000_000},
        ])
        await self.service.deployment_prepare("llama-bookmark", ["local", "remote-1"])

        # One selected node seeds the artifact from Hugging Face and the rest
        # receive file-scoped Virtual NAS streams.
        self.manager.node_download_model_files.assert_awaited_once_with(
            "local", "org/model", revision, ["FP16/model-F16.gguf"], "main",
        )
        self.manager.node_transfer_model_files.assert_awaited_once_with(
            "local", "remote-1", "org/model", revision,
            ["FP16/model-F16.gguf"], "main",
        )
        stored = self.service.store.deployment("llama-bookmark", include_private=True)
        self.assertEqual(stored["settings"]["prepared_revision"], revision)

        # The launch reuses the prepared revision instead of re-resolving.
        resolves_after_prepare = virtual_nas.resolve_download_revision.await_count
        await self.service.deployment_action("llama-bookmark", "start")
        self.assertEqual(
            virtual_nas.resolve_download_revision.await_count,
            resolves_after_prepare,
        )
        launch = self.manager.create_deployment.await_args.args[0]
        self.assertEqual(
            launch["llama_artifact"],
            f"models--org--model/snapshots/{revision}/FP16/model-F16.gguf",
        )

    async def test_stopped_llama_cluster_settings_remain_editable(self):
        launch_settings = {
            "engine": "llama.cpp", "model": "org/model",
            "extra_args": ["--ctx-size", "4096"],
            "deployment_mode": "single", "node_ids": ["remote-1"],
        }
        # A llama.cpp cluster that already launched: the SQLite row routes to
        # a stopped Manager record.
        self.service.store.add_deployment(Deployment(
            id="llama-cluster", alias="llama-cluster",
            runtime=RuntimeKind.LLAMA_CPP, kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/model"),
            settings={"manager_deployment_id": "cluster-1",
                      "deployment_mode": "single", "node_ids": ["remote-1"]},
        ), "http://127.0.0.1:8080", None)
        self.manager.deployments = [{
            "id": "cluster-1", "status": "stopped", "desired_state": "stopped",
            "launch_settings": dict(launch_settings), "node_ids": ["remote-1"],
        }]
        self.manager._deployment = lambda deployment_id: next(
            (item for item in self.manager.deployments if item["id"] == deployment_id),
            None,
        )
        self.manager.update_deployment_settings = Mock(return_value={
            "launch_settings": {
                **launch_settings,
                "extra_args": ["--ctx-size", "8192", "--parallel", "4"],
            },
            "mode": "single", "node_ids": ["remote-1"],
        })
        self.manager.recipe_deployment_contract = Mock(return_value={
            "deployment_mode": "single", "required_node_count": 1,
            "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
            "model_revision": None, "supported": True, "error": None,
        })

        bookmark_id = self.service.store.deployment("llama-cluster")["id"]
        detail = await self.service.deployment_detail(bookmark_id)
        self.assertTrue(detail["editable"])
        self.assertEqual(detail["launch_controls"]["context_window"], 4096)

        await self.service.update_deployment_settings("llama-cluster", {
            "launch_controls": {"context_window": 8192, "max_concurrency": 4},
        })
        self.assertEqual(
            self.manager.update_deployment_settings.call_args.args[1]["launch_controls"]["max_concurrency"],
            4,
        )

    async def test_local_model_directory_rejects_remote_nodes_for_any_runtime(self):
        model_dir = Path(self.temp.name) / "local-model"
        model_dir.mkdir()
        self.manager._resolve_local_path = Mock(return_value=str(model_dir))
        with self.assertRaisesRegex(ValueError, "local model paths"):
            await self.service.create_deployment({
                "model": str(model_dir), "alias": "local-vllm",
                "runtime": "vllm",
                "node_ids": ["local", "remote-1"], "deployment_mode": "replicated",
            })

        # The controller-only selection stays saveable.
        created = await self.service.create_deployment({
            "model": str(model_dir), "alias": "local-vllm",
            "runtime": "vllm",
            "node_ids": ["local"], "deployment_mode": "single",
        })
        self.assertEqual(created["status"], "saved")

    async def test_saved_bookmark_accepts_the_detail_editor_contract(self):
        await self.service.create_deployment({
            "model": "org/model", "alias": "bookmark", "runtime": "vllm",
            "node_ids": ["remote-1"], "deployment_mode": "single",
            "settings": {"context_length": 8192},
        })

        # The DeploymentPage editor submits the manager-backed contract; a
        # saved bookmark must translate it instead of rejecting the save, and
        # the complete structured controls persist for Manager to merge.
        detail = await self.service.update_deployment_settings("bookmark", {
            "launch_controls": {
                "context_window": 16384,
                "max_concurrency": 8,
                "kv_cache_dtype": "fp8",
                "thinking_mode": "default",
                "dspark_num_speculative_tokens": None,
                "max_cudagraph_capture_size": None,
                "max_num_batched_tokens": None,
            },
            "gpu_memory_gb": 24,
            "sg_tp_size": None,
            "sg_mem_fraction": None,
            "gpu_memory_utilization": None,
            "extra_args": ["--enable-prefix-caching"],
        })

        stored = self.service.store.deployment("bookmark", include_private=True)
        self.assertEqual(stored["settings"]["context_length"], 16384)
        self.assertEqual(stored["settings"]["max_running_requests"], 8)
        self.assertEqual(stored["settings"]["gpu_memory_gb"], 24)
        self.assertEqual(stored["settings"]["extra_args"], ["--enable-prefix-caching"])
        self.assertEqual(stored["settings"]["launch_controls"]["kv_cache_dtype"], "fp8")
        # The detail response round-trips the persisted structured controls
        # so a subsequent unchanged save cannot blank them out.
        detail = await self.service.deployment_detail(
            self.service.store.deployment("bookmark")["id"],
        )
        self.assertEqual(detail["launch_controls"]["kv_cache_dtype"], "fp8")
        self.assertEqual(detail["launch_controls"]["context_window"], 16384)
        self.assertEqual(detail["status"], "saved")

        # Clearing a control in the editor also drops the saved scalar.
        await self.service.update_deployment_settings("bookmark", {
            "launch_controls": {
                "context_window": None,
                "max_concurrency": None,
                "kv_cache_dtype": None,
                "thinking_mode": None,
                "dspark_num_speculative_tokens": None,
                "max_cudagraph_capture_size": None,
                "max_num_batched_tokens": None,
            },
        })
        cleared = await self.service.deployment_detail(
            self.service.store.deployment("bookmark")["id"],
        )
        self.assertIsNone(cleared["launch_controls"]["context_window"])
        stored = self.service.store.deployment("bookmark", include_private=True)
        self.assertIsNone(stored["settings"]["context_length"])

        # Alias changes save atomically with the settings in one request.
        detail = await self.service.update_deployment_settings("bookmark", {
            "alias": "renamed-bookmark",
            "context_length": 32768,
        })
        self.assertEqual(detail["alias"], "renamed-bookmark")
        self.assertIsNone(self.service.store.deployment("bookmark"))
        self.assertIsNotNone(self.service.store.deployment("renamed-bookmark"))
        # A second deployment's alias cannot be taken.
        await self.service.create_deployment({
            "model": "org/model", "alias": "other", "runtime": "vllm",
            "node_ids": ["remote-1"], "deployment_mode": "single",
        })
        with self.assertRaisesRegex(ValueError, "already in use"):
            await self.service.update_deployment_settings("renamed-bookmark", {
                "alias": "other",
            })
        self.assertEqual(
            self.service.store.deployment(
                "renamed-bookmark"
            )["settings"]["context_length"],
            32768,
        )

    async def test_saved_bookmark_launch_controls_update_layout_contract(self):
        await self.service.create_deployment({
            "model": "org/model", "alias": "sharded-bookmark", "runtime": "vllm",
            "node_ids": ["local", "remote-1"], "deployment_mode": "sharded",
            "settings": {"tensor_parallel_size": 2},
        })
        listed = (await self.service.deployments())[0]
        self.assertEqual(listed["required_node_count"], 2)

        # The editor is seeded with the saved scalar so an unchanged Save
        # cannot submit a null that strips the TP flag at first launch.
        detail = await self.service.deployment_detail(listed["id"])
        self.assertEqual(detail["launch_controls"]["tensor_parallel_size"], 2)

        # Editing the structured controls syncs the saved topology scalar and
        # the picker contract with it: TP4/PP1 fits the two saved nodes at
        # two ranks per node.
        await self.service.update_deployment_settings("sharded-bookmark", {
            "launch_controls": {
                "tensor_parallel_size": 4,
                "pipeline_parallel_size": 1,
            },
        })

        stored = self.service.store.deployment(
            "sharded-bookmark", include_private=True,
        )
        self.assertEqual(stored["settings"]["tensor_parallel_size"], 4)
        listed = (await self.service.deployments())[0]
        self.assertEqual(listed["required_node_count"], 2)

        # A layout the saved nodes cannot divide requires one node per rank.
        await self.service.update_deployment_settings("sharded-bookmark", {
            "launch_controls": {
                "tensor_parallel_size": 3,
                "pipeline_parallel_size": 1,
            },
        })
        listed = (await self.service.deployments())[0]
        self.assertEqual(listed["required_node_count"], 3)

        # Fractional and boolean controls are rejected before they persist;
        # bookmarks never see Manager's validators until their first Run.
        for bad in (1.5, True):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    await self.service.update_deployment_settings(
                        "sharded-bookmark",
                        {"launch_controls": {"tensor_parallel_size": bad}},
                    )

        # An explicit clear sticks instead of being re-seeded from the scalar.
        await self.service.update_deployment_settings("sharded-bookmark", {
            "launch_controls": {"tensor_parallel_size": None},
        })
        detail = await self.service.deployment_detail(
            self.service.store.deployment("sharded-bookmark")["id"],
        )
        self.assertIsNone(detail["launch_controls"]["tensor_parallel_size"])
        stored = self.service.store.deployment(
            "sharded-bookmark", include_private=True,
        )
        self.assertIsNone(stored["settings"]["tensor_parallel_size"])

    async def test_saved_bookmark_detail_seeds_controls_from_scalars(self):
        await self.service.create_deployment({
            "model": "org/model", "alias": "fresh", "runtime": "vllm",
            "node_ids": ["remote-1"], "deployment_mode": "single",
            "settings": {"context_length": 24576},
        })

        detail = await self.service.deployment_detail(
            self.service.store.deployment("fresh")["id"],
        )

        # First editor use shows the bookmark's own scalars instead of
        # blanks, so an unchanged Save/Run cannot strip the launch argument.
        self.assertEqual(detail["launch_controls"]["context_window"], 24576)
        self.assertEqual(detail["gpu_memory_gb"], None)

    async def test_saved_sglang_bookmark_detail_exposes_saved_scalars(self):
        await self.service.create_deployment({
            "model": "org/model", "alias": "sg-bookmark", "runtime": "sglang",
            "node_ids": ["remote-1"], "deployment_mode": "single",
            "settings": {"tensor_parallel_size": 2, "mem_fraction_static": 0.8},
        })

        detail = await self.service.deployment_detail(
            self.service.store.deployment("sg-bookmark")["id"],
        )

        self.assertEqual(detail["sg_tp_size"], 2)
        self.assertEqual(detail["sg_mem_fraction"], 0.8)

    async def test_quantization_only_change_is_revalidated_against_the_artifact(self):
        await self.service.create_deployment({
            "model": "org/model", "alias": "llama-bookmark",
            "runtime": "llama.cpp",
            "node_ids": ["remote-1"], "deployment_mode": "single",
            "settings": {"artifact": "model-Q4_K_M.gguf", "quantization": "Q4_K_M"},
        })

        with self.assertRaisesRegex(ValueError, "does not match the selected quantization"):
            await self.service.update_deployment_settings("llama-bookmark", {
                "quantization": "Q8_0",
            })

        stored = self.service.store.deployment("llama-bookmark", include_private=True)
        self.assertEqual(stored["model"]["quantization"], "Q4_K_M")

    async def test_detail_seeds_parsed_args_without_clobbering_them(self):
        # Controls that live only in extra_args parse into the editor and an
        # unchanged save must not strip them.
        await self.service.create_deployment({
            "model": "org/model", "alias": "argv-controls", "runtime": "vllm",
            "node_ids": ["remote-1"], "deployment_mode": "single",
            "settings": {
                "context_length": None,
                "extra_args": ["--max-model-len", "8192", "--max-num-seqs", "4"],
            },
        })

        detail = await self.service.deployment_detail(
            self.service.store.deployment("argv-controls")["id"],
        )

        self.assertEqual(detail["launch_controls"]["context_window"], 8192)
        self.assertEqual(detail["launch_controls"]["max_concurrency"], 4)

    async def test_bookmark_integer_fields_are_validated(self):
        await self.service.create_deployment({
            "model": "org/model", "alias": "bookmark", "runtime": "vllm",
            "node_ids": ["remote-1"], "deployment_mode": "single",
        })

        for bad in (0, -1, 1.5):
            with self.assertRaisesRegex(ValueError, "context_length"):
                await self.service.update_deployment_settings("bookmark", {
                    "context_length": bad,
                })
        # Non-finite and oversized numeric junk is a 400-shaped ValueError,
        # never an escaping OverflowError/TypeError. A numeric string such as
        # "8192" is accepted and normalized instead.
        for bad in (1e309, float("nan"), "not-a-number"):
            with self.assertRaisesRegex(ValueError, "context_length"):
                await self.service.update_deployment_settings("bookmark", {
                    "context_length": bad,
                })
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            await self.service.update_deployment_settings("bookmark", {
                "gpu_memory_utilization": 1.5,
            })
        stored = self.service.store.deployment("bookmark", include_private=True)
        self.assertNotIn("context_length", stored["settings"])

    async def test_saved_bookmark_accepts_fractional_gpu_memory_gb(self):
        await self.service.create_deployment({
            "model": "org/model", "alias": "bookmark", "runtime": "vllm",
            "node_ids": ["remote-1"], "deployment_mode": "single",
        })

        detail = await self.service.update_deployment_settings("bookmark", {
            "gpu_memory_gb": 24.5,
        })

        self.assertEqual(detail["gpu_memory_gb"], 24.5)
        stored = self.service.store.deployment("bookmark", include_private=True)
        self.assertEqual(stored["settings"]["gpu_memory_gb"], 24.5)

    async def test_gpu_layers_fractional_value_is_rejected(self):
        await self.service.create_deployment({
            "model": "org/model", "alias": "bookmark", "runtime": "llama.cpp",
            "node_ids": ["local"], "deployment_mode": "single",
            "settings": {"artifact": "model.gguf"},
        })

        with self.assertRaisesRegex(ValueError, "whole number"):
            await self.service.update_deployment_settings("bookmark", {
                "gpu_layers": 1.5,
            })

    async def test_alias_only_edit_survives_string_scalars_from_raw_api(self):
        # A raw-API create can persist numeric scalars as strings; a later
        # alias-only edit must still succeed (or cleanly reject), never 500.
        await self.service.create_deployment({
            "model": "org/model", "alias": "stringy", "runtime": "vllm",
            "node_ids": ["remote-1"], "deployment_mode": "single",
        })
        self.service.store.update_managed_routing(
            self.service.store.deployment("stringy")["id"],
            {"context_length": "8192", "node_ids": ["remote-1"],
             "deployment_mode": "single"},
            None, None,
        )

        detail = await self.service.update_deployment_settings("stringy", {
            "alias": "stringy-renamed",
        })

        self.assertEqual(detail["alias"], "stringy-renamed")

    async def test_creator_fields_are_rejected_for_launched_standalone(self):
        # A launched controller-only llama deployment owns a live container;
        # its creator-form identity fields must not change behind it.
        artifact = Path(self.temp.name) / "local.gguf"
        artifact.write_bytes(b"gguf")
        await self.service.create_deployment({
            "model": "org/model", "alias": "local-bookmark",
            "runtime": "llama.cpp",
            "node_ids": ["local"], "deployment_mode": "single",
            "settings": {"artifact": str(artifact)},
        })
        self.manager.evict_other_backends = AsyncMock(
            return_value={"ok": True}
        )
        with patch(
            "sparkdeck.service.launch_managed_container",
            AsyncMock(return_value={
                "name": "sparkdeck-local-bookmark", "port": 8080,
                "status": "running", "model_source": "local",
            }),
        ):
            await self.service.deployment_action("local-bookmark", "start")
        self.assertIsNotNone(
            self.service.store.deployment("local-bookmark")["container_name"]
        )

        with self.assertRaisesRegex(ValueError, "editable saved launch settings"):
            await self.service.update_deployment_settings("local-bookmark", {
                "context_length": 4096,
            })
        with self.assertRaisesRegex(ValueError, "editable saved launch settings"):
            await self.service.update_deployment_settings("local-bookmark", {
                "node_ids": ["remote-1"],
            })

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
