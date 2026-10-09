"""Manager and service-side tests for the Strata runtime.

Strata (https://github.com/Niko1221/Strata) is an OpenAI-compatible
expert-offload engine: one server holds the whole model, and the container
configures itself through the upstream entrypoint's environment variables
instead of launch flags. These tests pin the container contract the node
agent must honour, the environment round-trip the controller relies on, and
the single-engine preflight rules shared with the other one-copy runtimes.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call, patch

from manager import (
    DEFAULT_STRATA_IMAGE, DEPLOYMENT_LABEL, ENGINE_LABEL,
    Manager, MODE_LABEL, NNODES_LABEL, NODE_LABEL, RANK_LABEL,
    _STRATA_SERVE_PORT, _SUPPORTED_ENGINES,
)
from cluster import STRATA_CAPABILITY
from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.runtime_environment import (
    discovered_runtime_environment,
    normalize_runtime_environment,
)
from sparkdeck.runtimes import (
    RuntimeRegistry,
    apply_strata_launch_controls,
    strata_launch_environment,
    validate_strata_model,
)
from sparkdeck.service import SparkDeckService


def _manager() -> Manager:
    """A Manager with only the Docker-facing surface stubbed."""
    manager = Manager.__new__(Manager)
    container = Mock()
    container.reload = Mock()
    manager.client = Mock()
    image = Mock()
    image.attrs = {"Config": {"Env": []}}
    manager.client.images.get = Mock(return_value=image)
    manager._allocate_port = AsyncMock(return_value=8123)
    manager._build_volumes = Mock(return_value={
        "/host/cache": {"bind": "/root/.cache/huggingface", "mode": "rw"},
    })
    manager._run_managed_container = Mock(return_value=container)
    manager._container_summary = Mock(return_value={
        "name": "strata-org-model-8123", "port": 8123, "status": "running",
    })
    manager.settings = {"hf_cache": "/host/cache", "shm_size": "16g"}
    manager.cluster_member_launches = {}
    manager.evict_other_backends = AsyncMock()
    return manager


async def _launch(manager, **overrides):
    """Call the Strata container launcher with the full node-agent signature."""
    kwargs = {
        "model": "org/model", "port": None, "image": None,
        "environment": None, "extra_args": None, "name": None,
        "cluster_member": None, "hf_token": None,
        "sparkdeck_deployment_id": None, "shm_size": None,
    }
    kwargs.update(overrides)
    return await manager._create_strata_container(**kwargs)


class StrataContainerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # The launcher refuses to run without an NVIDIA driver, so the default
        # fixture pretends one is present; individual tests override it.
        driver = patch("manager._node_has_nvidia_driver", return_value=True)
        driver.start()
        self.addCleanup(driver.stop)

    async def test_strata_container_lets_the_entrypoint_serve(self):
        manager = _manager()

        result = await _launch(
            manager, model="org/model", sparkdeck_deployment_id="dep-1",
        )

        options = manager._run_managed_container.call_args.args[0]
        self.assertEqual(options["image"], DEFAULT_STRATA_IMAGE)
        # The upstream entrypoint builds and starts the server, so the launch
        # must not override the image's command.
        self.assertNotIn("command", options)
        self.assertEqual(
            options["ports"], {f"{_STRATA_SERVE_PORT}/tcp": 8123},
        )
        self.assertEqual(options["labels"]["io.sparkdeck.runtime"], "strata")
        self.assertEqual(options["labels"]["io.sparkdeck.deployment"], "dep-1")
        self.assertEqual(result["model_source"], "public_repository")

    async def test_strata_environment_carries_the_upstream_defaults(self):
        manager = _manager()

        await _launch(manager)

        options = manager._run_managed_container.call_args.args[0]
        self.assertEqual(options["environment"]["MODEL"], "IQ2_XS")
        self.assertEqual(options["environment"]["FAMILY"], "qwen")
        self.assertNotIn("CONTEXT", options["environment"])

    async def test_the_typed_context_seeds_the_context_variable(self):
        manager = _manager()

        await _launch(manager, environment={"CONTEXT": "16384"})

        options = manager._run_managed_container.call_args.args[0]
        self.assertEqual(options["environment"]["CONTEXT"], "16384")

    async def test_operator_environment_reaches_the_container(self):
        manager = _manager()

        await _launch(manager, environment={"MODEL": "IQ3_S", "LOW_RAM": "on"})

        options = manager._run_managed_container.call_args.args[0]
        self.assertEqual(options["environment"]["MODEL"], "IQ3_S")
        self.assertEqual(options["environment"]["LOW_RAM"], "on")

    async def test_hugging_face_credential_reaches_the_container(self):
        manager = _manager()
        manager._container_hf_environment = Mock(
            return_value={"HF_TOKEN": "secret", "HUGGING_FACE_HUB_TOKEN": "secret"}
        )

        await _launch(manager, hf_token="secret")

        options = manager._run_managed_container.call_args.args[0]
        self.assertEqual(
            options["environment"],
            {
                "MODEL": "IQ2_XS",
                "FAMILY": "qwen",
                "HF_TOKEN": "secret",
                "HUGGING_FACE_HUB_TOKEN": "secret",
            },
        )
        manager._container_hf_environment.assert_called_once_with("secret")

    async def test_strata_data_volume_is_mounted_read_write(self):
        """The entrypoint prepares its pack under /data, and a restart skips
        that setup when the files survive, so the volume must outlive the
        container."""
        manager = _manager()

        await _launch(manager, name="strata-replica")

        volumes = manager._run_managed_container.call_args.args[0]["volumes"]
        self.assertEqual(
            volumes["sparkdeck-strata-strata-replica"],
            {"bind": "/data", "mode": "rw"},
        )

    async def test_memlock_ulimit_is_lifted(self):
        """The engine locks part of host RAM for the expert residency; the
        upstream launch instructions lift Docker's memlock cap."""
        import docker

        manager = _manager()

        await _launch(manager)

        (ulimit,) = manager._run_managed_container.call_args.args[0]["ulimits"]
        self.assertIsInstance(ulimit, docker.types.Ulimit)
        self.assertEqual(ulimit.name, "memlock")
        self.assertEqual(ulimit.soft, -1)
        self.assertEqual(ulimit.hard, -1)

    async def test_gpu_request_is_always_attached(self):
        import docker

        manager = _manager()

        await _launch(manager)

        (request,) = manager._run_managed_container.call_args.args[0][
            "device_requests"
        ]
        self.assertIsInstance(request, docker.types.DeviceRequest)

    async def test_extra_flags_are_rejected(self):
        """The upstream entrypoint has no flag surface, so extra argv would
        silently never reach the engine."""
        manager = _manager()

        with self.assertRaisesRegex(ValueError, "do not accept extra flags"):
            await _launch(manager, extra_args=["--some-flag"])
        manager._run_managed_container.assert_not_called()

    async def test_an_unknown_model_size_is_rejected_before_eviction(self):
        manager = _manager()

        with self.assertRaisesRegex(ValueError, "strata model size"):
            await _launch(manager, environment={"MODEL": "Q9_ZZZ"})
        manager.evict_other_backends.assert_not_awaited()
        manager._run_managed_container.assert_not_called()

    async def test_a_model_size_from_another_family_is_rejected(self):
        manager = _manager()

        with self.assertRaisesRegex(ValueError, "requires family coder"):
            await _launch(manager, environment={"MODEL": "IQ1_M"})

    async def test_nodes_without_an_nvidia_driver_are_rejected_before_eviction(self):
        manager = _manager()

        with patch("manager._node_has_nvidia_driver", return_value=False):
            with self.assertRaisesRegex(ValueError, "NVIDIA driver"):
                await _launch(manager)
        manager.evict_other_backends.assert_not_awaited()
        manager._run_managed_container.assert_not_called()

    async def test_a_chat_engine_launch_evicts_other_backends(self):
        manager = _manager()

        await _launch(manager)

        manager.evict_other_backends.assert_awaited_once_with(protect="strata")

    async def test_sharded_strata_is_rejected(self):
        manager = _manager()

        with self.assertRaisesRegex(ValueError, "cannot run sharded"):
            await _launch(
                manager, name="strata-shard",
                cluster_member={
                    "mode": "sharded", "deployment_id": "d",
                    "node_id": "n", "rank": 0,
                },
            )
        manager._run_managed_container.assert_not_called()

    async def test_cluster_member_labels_are_recorded(self):
        manager = _manager()
        member = {
            "deployment_id": "dep-9", "node_id": "spark-2", "rank": 1,
            "mode": "replicated", "nnodes": 2,
        }

        await _launch(
            manager, name="strata-replica", cluster_member=member,
            sparkdeck_deployment_id="dep-9",
        )

        labels = manager._run_managed_container.call_args.args[0]["labels"]
        self.assertEqual(labels[ENGINE_LABEL], "strata")
        self.assertEqual(labels[DEPLOYMENT_LABEL], "dep-9")
        self.assertEqual(labels[NODE_LABEL], "spark-2")
        self.assertEqual(labels[RANK_LABEL], "1")
        self.assertEqual(labels[MODE_LABEL], "replicated")
        self.assertEqual(labels[NNODES_LABEL], "2")

    async def test_missing_image_is_pulled_once(self):
        import docker

        manager = _manager()
        manager.client.images.get = Mock(
            side_effect=docker.errors.ImageNotFound("missing"),
        )

        await _launch(manager)

        manager.client.images.pull.assert_called_once_with(DEFAULT_STRATA_IMAGE)
        manager._run_managed_container.assert_called_once()

    async def test_launch_failure_is_reported_without_leaking_the_token(self):
        manager = _manager()
        manager._container_hf_environment = Mock(return_value={"HF_TOKEN": "secret"})
        manager._cluster_launch_update = Mock()
        manager._run_managed_container = Mock(side_effect=RuntimeError("boom"))

        with self.assertRaisesRegex(RuntimeError, "boom"):
            await _launch(manager, hf_token="secret")
        update = manager._cluster_launch_update.call_args
        self.assertNotIn("secret", str(update))


class StrataEvictionTests(unittest.IsolatedAsyncioTestCase):
    def _manager_with_containers(self, runtimes):
        manager = Manager.__new__(Manager)
        manager._activity = {}
        containers = [
            {"name": f"c-{index}", "managed": True, "status": "running",
             "engine": runtime}
            for index, runtime in enumerate(runtimes)
        ]
        manager.list_containers = AsyncMock(return_value=containers)
        manager.stop_container = AsyncMock()
        return manager

    async def test_chat_engine_eviction_preserves_laya_but_stops_other_chat_engines(self):
        manager = self._manager_with_containers(["laya", "vllm", "tensorfold"])

        await manager.evict_other_backends(protect="strata")

        self.assertEqual(
            [invocation.args[0] for invocation in manager.stop_container.await_args_list],
            ["c-1", "c-2"],
        )


class StrataPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_preflight_rejects_sharded_strata_layouts(self):
        manager = Manager.__new__(Manager)
        manager._reject_hf_cli_credentials = Mock()
        manager.cluster_nodes = Mock()
        manager._validate_runtime_file_mount_nodes = Mock()

        with self.assertRaisesRegex(ValueError, "single and replicated layouts"):
            await manager._preflight_deployment_launch({
                "model": "org/model",
                "engine": "strata",
                "deployment_mode": "sharded",
                "node_ids": ["local", "spark-2"],
            })

    async def test_preflight_rejects_agents_without_strata_support(self):
        """Older agents reject the unknown engine at container creation, so a
        mixed-version selection must fail before any replica launches."""
        manager = Manager.__new__(Manager)
        manager._reject_hf_cli_credentials = Mock()
        manager._validate_runtime_file_mount_nodes = Mock()

        async def cluster_nodes():
            return [
                {"id": "local", "name": "Coordinator", "online": True,
                 "docker_ready": True,
                 "capabilities": [STRATA_CAPABILITY],
                 "stats": {"gpus": [{"name": "RTX 5090", "memory_total_gb": 32}]}},
                {"id": "old-node", "name": "Old Node", "online": True,
                 "docker_ready": True,
                 "capabilities": ["runtime-file-mounts-v1"],
                 "stats": {"gpus": [{"name": "RTX 5090", "memory_total_gb": 32}]}},
            ]

        manager.cluster_nodes = cluster_nodes

        with self.assertRaisesRegex(ValueError, "Strata requires updated"):
            await manager._preflight_deployment_launch({
                "model": "org/model",
                "engine": "strata",
                "deployment_mode": "replicated",
                "node_ids": ["local", "old-node"],
            })


class StrataContractTests(unittest.TestCase):
    def test_strata_is_a_supported_engine(self):
        self.assertIn("strata", _SUPPORTED_ENGINES)

    def test_registry_serves_the_strata_adapter(self):
        adapter = RuntimeRegistry().get("strata")
        self.assertIs(adapter.kind, RuntimeKind.STRATA)
        self.assertEqual(adapter.default_image, DEFAULT_STRATA_IMAGE)

    def test_launch_spec_configures_through_environment(self):
        spec = RuntimeRegistry().get("strata").launch_spec(
            "org/model", {"context_length": 16384, "kv_cache_dtype": "q4_0"},
        )
        self.assertEqual(spec.command, [])
        self.assertEqual(spec.environment["MODEL"], "IQ2_XS")
        self.assertEqual(spec.environment["FAMILY"], "qwen")
        self.assertEqual(spec.environment["CONTEXT"], "16384")
        self.assertEqual(spec.environment["KV"], "q4_0")

    def test_an_explicit_context_variable_wins_over_the_typed_control(self):
        environment = strata_launch_environment(
            {"CONTEXT": "8192"}, 32768,
        )
        self.assertEqual(environment["CONTEXT"], "8192")

    def test_a_malformed_context_variable_fails_the_launch(self):
        with self.assertRaisesRegex(ValueError, "CONTEXT"):
            strata_launch_environment({"CONTEXT": "soon"})

    def test_an_unknown_kv_variable_fails_the_launch(self):
        with self.assertRaisesRegex(ValueError, "KV"):
            strata_launch_environment({"KV": "fp8"})

    def test_validate_strata_model_defaults_and_matrix(self):
        self.assertEqual(validate_strata_model("", ""), ("IQ2_XS", "qwen"))
        self.assertEqual(
            validate_strata_model("IQ1_M", "coder"), ("IQ1_M", "coder"),
        )
        self.assertEqual(
            validate_strata_model("IQ2_XS", "swift"), ("IQ2_XS", "swift"),
        )

    def test_validate_strata_model_rejects_unknown_inputs(self):
        with self.assertRaisesRegex(ValueError, "strata model size"):
            validate_strata_model("Q9_ZZZ", "qwen")
        with self.assertRaisesRegex(ValueError, "strata family"):
            validate_strata_model("IQ2_XS", "gemma")
        with self.assertRaisesRegex(ValueError, "requires family"):
            validate_strata_model("UD-IQ4_XS", "qwen")

    def test_strata_accepts_operator_environment_variables(self):
        self.assertEqual(
            normalize_runtime_environment({"MODEL": "IQ2_XS"}, "strata"),
            {"MODEL": "IQ2_XS"},
        )
        with self.assertRaisesRegex(ValueError, "only supported for"):
            normalize_runtime_environment({"MODEL": "IQ2_XS"}, "llama.cpp")

    def test_discovered_environment_recovers_only_strata_setup_variables(self):
        self.assertEqual(
            discovered_runtime_environment(
                {"MODEL": "IQ2_XS", "CONTEXT": "8192", "NCCL_DEBUG": "WARN"},
                "strata",
            ),
            {"MODEL": "IQ2_XS", "CONTEXT": "8192"},
        )

    def test_launch_controls_parse_the_environment_variables(self):
        controls = Manager._deployment_launch_controls({
            "engine": "strata",
            "extra_args": [],
            "environment": {"CONTEXT": "8192", "KV": "k8v4"},
        })
        self.assertEqual(controls["context_window"], 8192)
        self.assertEqual(controls["kv_cache_dtype"], "k8v4")
        self.assertIsNone(controls["max_concurrency"])
        self.assertIsNone(controls["thinking_mode"])

    def test_launch_controls_without_variables_stay_unset(self):
        controls = Manager._deployment_launch_controls({
            "engine": "strata", "extra_args": [], "environment": {},
        })
        self.assertIsNone(controls["context_window"])
        self.assertIsNone(controls["kv_cache_dtype"])

    def test_apply_leaves_the_flags_untouched(self):
        args = Manager._apply_deployment_launch_controls(
            Mock(), [], "strata",
            {"context_window": 8192, "kv_cache_dtype": "int8"},
        )
        self.assertEqual(args, [])

    def test_apply_strata_launch_controls_rewrites_submitted_keys_only(self):
        environment = apply_strata_launch_controls(
            {"CONTEXT": "8192", "MODEL": "IQ3_S"},
            {"kv_cache_dtype": "q4_0"},
        )
        self.assertEqual(environment["CONTEXT"], "8192")
        self.assertEqual(environment["KV"], "q4_0")
        self.assertEqual(environment["MODEL"], "IQ3_S")

    def test_apply_strata_launch_controls_clears_on_explicit_null(self):
        environment = apply_strata_launch_controls(
            {"CONTEXT": "8192", "KV": "int8"},
            {"context_window": None, "kv_cache_dtype": None},
        )
        self.assertNotIn("CONTEXT", environment)
        self.assertNotIn("KV", environment)

    def test_apply_strata_launch_controls_rejects_bad_values(self):
        with self.assertRaisesRegex(ValueError, "context_window"):
            apply_strata_launch_controls({}, {"context_window": "soon"})
        with self.assertRaisesRegex(ValueError, "kv_cache_dtype"):
            apply_strata_launch_controls({}, {"kv_cache_dtype": "fp8"})


class _FakeClusterManager:
    """The slice of Manager the cluster-record launch path needs."""

    def __init__(self):
        import httpx

        self.http = httpx.AsyncClient()
        self.deployments = []
        self.selected_cluster_nodes = AsyncMock(
            return_value=[{"id": "spark-2", "name": "Spark 2"}],
        )
        self.create_deployment = AsyncMock(return_value={
            "id": "cluster-strata", "status": "starting", "api_port": 8123,
            "members": [{"container_name": "strata-org-model-8123"}],
            "model_source": "public_repository",
        })
        self.public_target_node = Mock(side_effect=lambda node: node)


class StrataServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        driver = patch("manager._node_has_nvidia_driver", return_value=True)
        driver.start()
        self.addCleanup(driver.stop)

    def _service(self):
        temp = tempfile.TemporaryDirectory()
        manager = _FakeClusterManager()
        service = SparkDeckService(manager, Path(temp.name))
        return manager, service, temp

    async def test_cluster_launch_body_seeds_the_environment(self):
        manager, service, temp = self._service()
        try:
            body = service._cluster_launch_body(
                RuntimeKind.STRATA, "org/model", "alias", "dep-1",
                ModelIdentity("org/model"),
                {
                    "context_length": 16384,
                    "kv_cache_dtype": "int8",
                    "environment": {"MODEL": "IQ3_S"},
                },
                ["spark-2"], "single", None,
            )
            self.assertEqual(body["engine"], "strata")
            self.assertEqual(body["environment"]["MODEL"], "IQ3_S")
            self.assertEqual(body["environment"]["CONTEXT"], "16384")
            self.assertEqual(body["environment"]["KV"], "int8")
        finally:
            await manager.http.aclose()
            await service.close()
            temp.cleanup()

    async def test_a_revision_pin_is_rejected_at_creation(self):
        manager, service, temp = self._service()
        try:
            with self.assertRaisesRegex(ValueError, "cannot pin a model revision"):
                await service.create_deployment({
                    "model": "org/model",
                    "alias": "strata-pinned",
                    "runtime": "strata",
                    "revision": "a" * 40,
                    "node_ids": ["spark-2"],
                    "deployment_mode": "single",
                })
            self.assertIsNone(
                service.store.deployment("strata-pinned", include_private=True),
            )
        finally:
            await manager.http.aclose()
            await service.close()
            temp.cleanup()

    async def test_a_sharded_bookmark_is_rejected(self):
        manager, service, temp = self._service()
        try:
            with self.assertRaisesRegex(ValueError, "single and replicated layouts"):
                await service.create_deployment({
                    "model": "org/model",
                    "alias": "strata-sharded",
                    "runtime": "strata",
                    "node_ids": ["spark-2", "spark-3"],
                    "deployment_mode": "sharded",
                })
        finally:
            await manager.http.aclose()
            await service.close()
            temp.cleanup()

    async def test_a_saved_strata_record_needs_no_artifact(self):
        manager, service, temp = self._service()
        try:
            result = await service.create_deployment({
                "model": "org/model",
                "alias": "strata-saved",
                "runtime": "strata",
                "node_ids": ["spark-2"],
                "deployment_mode": "single",
            })

            record = service.store.deployment(result["id"], include_private=True)
            self.assertIsNotNone(record)
            self.assertEqual(record["runtime"], "strata")
        finally:
            await manager.http.aclose()
            await service.close()
            temp.cleanup()


if __name__ == "__main__":
    unittest.main()
