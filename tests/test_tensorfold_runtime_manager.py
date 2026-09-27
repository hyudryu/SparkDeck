"""Manager and service-side tests for the TensorFold runtime.

TensorFold is an OpenAI-compatible exact-decoding server: one complete engine
per node with speculative drafting that never changes output bytes. SparkDeck
treats it like the other single-engine runtimes — no tensor or pipeline
parallelism, weights resolved through the node's shared Hugging Face cache.
These tests pin the container contract the node agent must honour and the
launch-settings round-trip the controller relies on.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

from manager import (
    CONTROLLER_LABEL, DEFAULT_TENSORFOLD_IMAGE, DEPLOYMENT_LABEL, ENGINE_LABEL,
    Manager, MODE_LABEL, NNODES_LABEL, NODE_LABEL, RANK_LABEL,
    _SUPPORTED_ENGINES, _TENSORFOLD_SERVE_PORT,
)
from sparkdeck.runtime_environment import normalize_runtime_environment
from sparkdeck.service import SparkDeckService


def _manager() -> Manager:
    """A Manager with only the Docker-facing surface stubbed."""
    manager = Manager.__new__(Manager)
    container = Mock()
    container.reload = Mock()
    manager.client = Mock()
    manager.client.images.get = Mock()
    manager._allocate_port = AsyncMock(return_value=8123)
    manager._build_volumes = Mock(return_value={"/host/cache": {"bind": "/root/.cache/huggingface", "mode": "rw"}})
    manager._run_managed_container = Mock(return_value=container)
    manager._container_summary = Mock(return_value={
        "name": "tensorfold-org-model-8123", "port": 8123, "status": "running",
    })
    manager.settings = {"hf_cache": "/host/cache", "shm_size": "16g"}
    manager.cluster_member_launches = {}
    manager.evict_other_backends = AsyncMock()
    return manager


async def _launch(manager, **overrides):
    """Call the TensorFold container launcher with the full node-agent signature."""
    kwargs = {
        "model": "org/model", "port": None, "image": None,
        "environment": None, "extra_args": None, "name": None,
        "cluster_member": None, "hf_token": None,
        "sparkdeck_deployment_id": None, "shm_size": None,
    }
    kwargs.update(overrides)
    return await manager._create_tensorfold_container(**kwargs)


class TensorfoldContainerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # The launcher refuses to run without an NVIDIA driver, so the default
        # fixture pretends one is present; individual tests override it.
        driver = patch("manager._node_has_nvidia_driver", return_value=True)
        driver.start()
        self.addCleanup(driver.stop)

    async def test_tensorfold_container_runs_the_serve_command(self):
        manager = _manager()

        result = await _launch(
            manager, model="org/model",
            sparkdeck_deployment_id="dep-1",
        )

        options = manager._run_managed_container.call_args.args[0]
        self.assertEqual(options["image"], DEFAULT_TENSORFOLD_IMAGE)
        self.assertEqual(
            options["command"],
            [
                "serve", "org/model",
                "--host", "0.0.0.0",
                "--port", str(_TENSORFOLD_SERVE_PORT),
            ],
        )
        self.assertEqual(
            options["ports"], {f"{_TENSORFOLD_SERVE_PORT}/tcp": 8123},
        )
        self.assertEqual(options["labels"]["io.sparkdeck.runtime"], "tensorfold")
        self.assertEqual(options["labels"]["io.sparkdeck.deployment"], "dep-1")
        # The model reference is forwarded so a controller-local checkpoint
        # directory is mounted, not only the shared Hugging Face cache.
        manager._build_volumes.assert_called_once_with(
            "org/model", "/host/cache", DEFAULT_TENSORFOLD_IMAGE,
        )
        self.assertEqual(result["model_source"], "public_repository")

    async def test_a_chat_engine_launch_evicts_other_backends(self):
        manager = _manager()

        await _launch(manager)

        manager.evict_other_backends.assert_awaited_once_with(protect="tensorfold")

    async def test_hugging_face_credential_reaches_the_container(self):
        manager = _manager()
        manager._container_hf_environment = Mock(
            return_value={"HF_TOKEN": "secret", "HUGGING_FACE_HUB_TOKEN": "secret"}
        )

        await _launch(manager, hf_token="secret")

        options = manager._run_managed_container.call_args.args[0]
        self.assertEqual(
            options["environment"],
            {"HF_TOKEN": "secret", "HUGGING_FACE_HUB_TOKEN": "secret"},
        )
        manager._container_hf_environment.assert_called_once_with("secret")

    async def test_no_environment_block_without_variables(self):
        manager = _manager()
        manager._container_hf_environment = Mock(return_value={})

        await _launch(manager)

        options = manager._run_managed_container.call_args.args[0]
        self.assertNotIn("environment", options)

    async def test_gpu_request_is_always_attached(self):
        manager = _manager()
        with patch("manager._node_has_nvidia_driver", return_value=True):
            await _launch(manager)
        self.assertIn("device_requests", manager._run_managed_container.call_args.args[0])

    async def test_nodes_without_an_nvidia_driver_are_rejected_before_eviction(self):
        manager = _manager()
        with patch("manager._node_has_nvidia_driver", return_value=False):
            with self.assertRaisesRegex(ValueError, "requires an NVIDIA GPU"):
                await _launch(manager)
        # Healthy chat backends must not be evicted for a launch that cannot
        # start, and no container may be created.
        manager.evict_other_backends.assert_not_awaited()
        manager._run_managed_container.assert_not_called()

    async def test_extra_args_extend_the_served_configuration(self):
        manager = _manager()

        await _launch(
            manager, port=8200, image="registry.test/tensorfold:1",
            extra_args=["--context", "32768", "--no-thinking", "--mtp-drafts", "4"],
            name="tensorfold-test", shm_size="4g",
        )

        options = manager._run_managed_container.call_args.args[0]
        self.assertEqual(options["image"], "registry.test/tensorfold:1")
        self.assertEqual(
            options["command"],
            [
                "serve", "org/model", "--host", "0.0.0.0", "--port",
                str(_TENSORFOLD_SERVE_PORT),
                "--context", "32768", "--no-thinking", "--mtp-drafts", "4",
            ],
        )
        self.assertEqual(options["name"], "tensorfold-test")
        self.assertEqual(options["shm_size"], "4g")

    async def test_cluster_member_labels_are_recorded(self):
        manager = _manager()
        member = {
            "deployment_id": "dep-9", "node_id": "spark-2", "rank": 1,
            "mode": "replicated", "nnodes": 2,
        }

        await _launch(
            manager, name="tensorfold-replica", cluster_member=member,
            sparkdeck_deployment_id="dep-9",
        )

        labels = manager._run_managed_container.call_args.args[0]["labels"]
        self.assertEqual(labels[ENGINE_LABEL], "tensorfold")
        self.assertEqual(labels[DEPLOYMENT_LABEL], "dep-9")
        self.assertEqual(labels[NODE_LABEL], "spark-2")
        self.assertEqual(labels[RANK_LABEL], "1")
        self.assertEqual(labels[MODE_LABEL], "replicated")
        self.assertEqual(labels[NNODES_LABEL], "2")

    async def test_sharded_tensorfold_is_rejected(self):
        manager = _manager()

        with self.assertRaisesRegex(ValueError, "cannot run sharded"):
            await _launch(
                manager, name="tensorfold-shard",
                cluster_member={
                    "mode": "sharded", "deployment_id": "d",
                    "node_id": "n", "rank": 0,
                },
            )
        manager._run_managed_container.assert_not_called()

    async def test_missing_image_is_pulled_once(self):
        import docker

        manager = _manager()
        manager.client.images.get = Mock(
            side_effect=[docker.errors.ImageNotFound("missing"), Mock()]
        )

        await _launch(manager, image="registry.test/tensorfold:1", name="tensorfold-pull")

        manager.client.images.pull.assert_called_once_with("registry.test/tensorfold:1")

    async def test_launch_failure_is_reported_without_leaking_the_token(self):
        manager = _manager()
        manager.client.images.get = Mock(side_effect=RuntimeError("boom hf_secret_value"))
        manager._redact_hf_secret = Mock(return_value="boom [REDACTED]")

        with self.assertRaisesRegex(RuntimeError, r"boom \[REDACTED\]"):
            await _launch(manager, name="tensorfold-fail")

    def test_tensorfold_is_a_supported_engine(self):
        self.assertIn("tensorfold", _SUPPORTED_ENGINES)

    def test_tensorfold_accepts_operator_environment_variables(self):
        """TensorFold reads HF_HUB_OFFLINE-style variables from the container
        environment, so a validator that rejects every non-vLLM map would fail
        the launch before the container branch runs."""
        normalized = normalize_runtime_environment(
            {"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            "tensorfold",
        )
        self.assertEqual(
            normalized,
            {"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )
        # The credential guard still applies.
        with self.assertRaisesRegex(ValueError, "managed by SparkDeck"):
            normalize_runtime_environment({"HF_TOKEN": "x"}, "tensorfold")


class TensorfoldEvictionTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_engine_eviction_preserves_laya_but_stops_other_chat_engines(self):
        for engine in ("vllm", "sglang", "llama.cpp", "tensorfold"):
            with self.subTest(engine=engine):
                manager = _manager()
                # Use the real eviction path, not the launcher's stub.
                manager.evict_other_backends = Manager.evict_other_backends.__get__(manager)
                containers = [
                    {"name": runtime, "engine": runtime, "managed": True,
                     "status": "running"}
                    for runtime in ("laya", "vllm", "sglang", "llama.cpp", "tensorfold")
                ]
                manager.list_containers = AsyncMock(return_value=containers)
                manager.stop_container = AsyncMock()
                manager._activity = {item["name"]: 1 for item in containers}

                result = await manager.evict_other_backends(protect=engine)

                expected = {"vllm", "sglang", "llama.cpp", "tensorfold"} - {engine}
                self.assertEqual(set(result["stopped"]), expected)
                self.assertEqual(
                    {call.args[0] for call in manager.stop_container.await_args_list},
                    expected,
                )
                self.assertIn("laya", manager._activity)


class TensorfoldPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_preflight_rejects_sharded_tensorfold_layouts(self):
        manager = Manager.__new__(Manager)
        manager._reject_hf_cli_credentials = Mock()
        manager.cluster_nodes = Mock()
        manager._validate_runtime_file_mount_nodes = Mock()

        with self.assertRaisesRegex(ValueError, "single and replicated layouts"):
            await manager._preflight_deployment_launch({
                "model": "org/model",
                "engine": "tensorfold",
                "deployment_mode": "sharded",
                "node_ids": ["local", "spark-2"],
            })

    async def test_preflight_accepts_a_single_node_tensorfold_deployment(self):
        manager = Manager.__new__(Manager)
        manager._reject_hf_cli_credentials = Mock()
        manager._validate_runtime_file_mount_nodes = Mock()

        async def cluster_nodes():
            return [{
                "id": "spark-2", "name": "Spark 2", "online": True,
                "docker_ready": True,
            }]

        manager.cluster_nodes = cluster_nodes

        plan = await manager._preflight_deployment_launch({
            "model": "org/model",
            "engine": "tensorfold",
            "deployment_mode": "single",
            "node_ids": ["spark-2"],
        })

        self.assertEqual(plan["engine"], "tensorfold")
        self.assertEqual(plan["mode"], "single")
        self.assertEqual(plan["node_ids"], ["spark-2"])


class TensorfoldLaunchControlsTests(unittest.TestCase):
    """The structured editor's controls must round-trip through TensorFold argv."""

    def test_launch_controls_parse_context_and_thinking(self):
        controls = Manager._deployment_launch_controls({
            "engine": "tensorfold",
            "extra_args": ["--context", "16384", "--no-thinking"],
        })
        self.assertEqual(controls["context_window"], 16384)
        self.assertEqual(controls["thinking_mode"], "disabled")
        self.assertIsNone(controls["max_concurrency"])
        self.assertIsNone(controls["kv_cache_dtype"])

    def test_apply_maps_context_and_thinking_back_to_flags(self):
        manager = Manager.__new__(Manager)
        args = manager._apply_deployment_launch_controls(
            ["--context", "8192"], "tensorfold",
            {"context_window": 32768, "thinking_mode": "enabled"},
        )
        self.assertEqual(
            args, ["--context", "32768", "--thinking"],
        )

    def test_default_thinking_clears_the_override(self):
        manager = Manager.__new__(Manager)
        args = manager._apply_deployment_launch_controls(
            ["--context", "8192", "--no-thinking"], "tensorfold",
            {"context_window": 8192, "thinking_mode": "default"},
        )
        self.assertEqual(args, ["--context", "8192"])


class FakeClusterManager:
    """The slice of Manager the cluster-record launch path needs."""

    def __init__(self):
        import httpx

        self.http = httpx.AsyncClient()
        self.deployments = []
        self.selected_cluster_nodes = AsyncMock(
            return_value=[{"id": "spark-2", "name": "Spark 2"}],
        )
        self.create_deployment = AsyncMock(return_value={
            "id": "cluster-tf", "status": "starting", "api_port": 8123,
            "members": [], "model_source": "public_repository",
        })
        self.public_target_node = Mock(side_effect=lambda node: node)


class TensorfoldClusterLaunchTests(unittest.IsolatedAsyncioTestCase):
    async def test_cached_revision_is_not_injected_for_tensorfold(self):
        """A shared cached snapshot must not add the unsupported --revision
        flag to a TensorFold launch: TensorFold resolves checkpoints itself
        and fails on unknown flags."""
        import tempfile
        from pathlib import Path

        from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
        from sparkdeck.service import SparkDeckService

        # The store keeps a SQLite handle open, so the directory must be
        # cleaned up only after the service is closed (notably on Windows).
        temp = tempfile.TemporaryDirectory()
        manager = FakeClusterManager()
        service = SparkDeckService(manager, Path(temp.name))
        service._validate_start_selection = AsyncMock(return_value="b" * 40)
        service._link_cluster_record = Mock()
        record = Deployment(
            id="record-tf", alias="tf-model",
            runtime=RuntimeKind.TENSORFOLD, kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/model"), settings={},
        )

        try:
            await service._launch_cluster_record(
                record, {}, "org/model", "", ["spark-2"],
            )

            service._validate_start_selection.assert_not_awaited()
            body = manager.create_deployment.await_args.args[0]
            self.assertEqual(body["engine"], "tensorfold")
            self.assertNotIn("--revision", body["extra_args"])
        finally:
            await manager.http.aclose()
            await service.close()
            temp.cleanup()


class TensorfoldProxyTests(unittest.IsolatedAsyncioTestCase):
    """Managed TensorFold traffic must route through Manager's member-aware path."""

    async def test_managed_tensorfold_uses_manager_member_routing(self):
        import tempfile
        from pathlib import Path

        import httpx

        from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
        from sparkdeck.service import SparkDeckService

        class FakeProxyManager:
            def __init__(self, transport):
                self.http = httpx.AsyncClient(transport=transport)
                self.deployments = []
                self.list_containers = AsyncMock(return_value=[])
                self._prompt_waiting_requests = {}
                self.proxy_cluster_inference = AsyncMock()

            def source_ip_routing_rule(self, caller_ip, requested_model):
                return None

            @staticmethod
            async def _await_or_cancel(coro, cancel):
                return await coro

        with tempfile.TemporaryDirectory() as directory:
            manager = FakeProxyManager(httpx.MockTransport(
                lambda request: httpx.Response(200, json={})
            ))
            manager.proxy_cluster_inference.return_value = {
                "id": "chatcmpl-1", "object": "chat.completion", "created": 0,
                "model": "org/model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
            service = SparkDeckService(manager, Path(directory))
            service.store.add_deployment(Deployment(
                id="record-tf", alias="tf-model",
                runtime=RuntimeKind.TENSORFOLD, kind=DeploymentKind.MANAGED,
                model=ModelIdentity("org/model"),
                container_name="tensorfold-org-model-8123", status="running",
                settings={"manager_deployment_id": "cluster-tf"},
            ))
            service.store.update_managed_routing(
                "record-tf", {"manager_deployment_id": "cluster-tf"},
                "tensorfold-org-model-8123", "http://127.0.0.1:8123",
            )

            try:
                await service.proxy(
                    {"model": "tf-model", "messages": [{"role": "user", "content": "hi"}]},
                    "chat/completions",
                )

                manager.proxy_cluster_inference.assert_awaited_once()
                args = manager.proxy_cluster_inference.await_args.args
                self.assertEqual(args[0], "cluster-tf")
                self.assertEqual(args[1], "org/model")
            finally:
                await manager.http.aclose()
                await service.close()


class TensorfoldLaunchSettingsTests(unittest.TestCase):
    """Typed TensorFold settings must survive both launch paths."""

    def _body(self, settings):
        from sparkdeck.models import ModelIdentity, RuntimeKind

        return SparkDeckService.__new__(SparkDeckService)._cluster_launch_body(
            RuntimeKind.TENSORFOLD, "org/model", "tf-model", "record-1",
            ModelIdentity("org/model"), settings,
            ["spark-2"], "single", None,
        )

    def test_cluster_launch_translates_typed_settings_to_flags(self):
        body = self._body({"context_length": 16384, "thinking": False})

        extra = body["extra_args"]
        self.assertEqual(extra[extra.index("--context") + 1], "16384")
        self.assertIn("--no-thinking", extra)
        self.assertEqual(body["engine"], "tensorfold")

    def test_unset_settings_do_not_emit_flags(self):
        body = self._body({"context_length": None, "thinking": None})

        self.assertNotIn("--context", body["extra_args"])
        self.assertNotIn("--no-thinking", body["extra_args"])

    def test_revision_is_not_forwarded_for_tensorfold(self):
        from sparkdeck.models import ModelIdentity, RuntimeKind

        body = SparkDeckService.__new__(SparkDeckService)._cluster_launch_body(
            RuntimeKind.TENSORFOLD, "org/model", "tf-model", "record-1",
            ModelIdentity("org/model", revision="a" * 40),
            {}, ["spark-2"], "single", None,
        )

        self.assertNotIn("--revision", body["extra_args"])

    def test_service_configuration_keeps_tensorfold_inputs(self):
        """_safe_configuration runs on every persisted record."""
        configuration = SparkDeckService._safe_configuration({
            "context_length": 16384, "secrets": "must not survive",
        })

        self.assertEqual(configuration.get("context_length"), 16384)
        self.assertNotIn("secrets", configuration)


if __name__ == "__main__":
    unittest.main()
