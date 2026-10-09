"""Manager and service-side tests for the NInfer runtime.

NInfer is a single-GPU OpenAI-compatible engine that loads one compiled v3
``.ninfer`` artifact per server. SparkDeck treats it like the other
single-engine runtimes — no tensor or pipeline parallelism, the artifact
resolved through the node's shared Hugging Face cache exactly like a llama.cpp
GGUF artifact. These tests pin the container contract the node agent must
honour and the launch-settings round-trip the controller relies on.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from manager import (
    DEFAULT_NINFER_IMAGE, DEPLOYMENT_LABEL, ENGINE_LABEL,
    Manager, MODE_LABEL, NNODES_LABEL, NODE_LABEL, RANK_LABEL,
    _NINFER_SERVE_PORT, _SUPPORTED_ENGINES,
)
from cluster import NINFER_CAPABILITY
from sparkdeck.runtime_environment import normalize_runtime_environment
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
    manager._build_volumes = Mock(return_value={"/host/cache": {"bind": "/root/.cache/huggingface", "mode": "rw"}})
    manager._run_managed_container = Mock(return_value=container)
    manager._container_summary = Mock(return_value={
        "name": "ninfer-org-model-8123", "port": 8123, "status": "running",
    })
    manager.settings = {"hf_cache": "/host/cache", "shm_size": "16g"}
    manager.cluster_member_launches = {}
    manager.evict_other_backends = AsyncMock()
    return manager


class _ArtifactCache:
    """A temporary hub cache holding one compiled .ninfer artifact."""

    def __init__(self):
        self._temp = tempfile.TemporaryDirectory()
        hub = Path(self._temp.name) / "hub"
        snapshot = hub / "models--org--model" / "snapshots" / ("a" * 40)
        snapshot.mkdir(parents=True)
        self.artifact = "models--org--model/snapshots/{}/model.ninfer".format("a" * 40)
        (snapshot / "model.ninfer").write_bytes(b"ninfer")

    @property
    def root(self) -> str:
        return self._temp.name

    def cleanup(self):
        self._temp.cleanup()


async def _launch(manager, **overrides):
    """Call the NInfer container launcher with the full node-agent signature."""
    kwargs = {
        "model": "org/model", "port": None, "image": None,
        "environment": None, "extra_args": None, "name": None,
        "ninfer_artifact": None, "cluster_member": None, "hf_token": None,
        "sparkdeck_deployment_id": None, "shm_size": None,
    }
    kwargs.update(overrides)
    return await manager._create_ninfer_container(**kwargs)


class NinferContainerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # The launcher refuses to run without an NVIDIA driver, so the default
        # fixture pretends one is present; individual tests override it.
        driver = patch("manager._node_has_nvidia_driver", return_value=True)
        driver.start()
        self.addCleanup(driver.stop)
        cache = _ArtifactCache()
        self.addCleanup(cache.cleanup)
        self.cache = cache

    def _resolved_command(self, options):
        """The launch argv with the host-specific artifact path replaced."""
        command = list(options["command"])
        index = next(
            i for i, part in enumerate(command)
            if Path(part).name == "model.ninfer"
        )
        command[index] = "<artifact>"
        return command

    async def test_ninfer_container_runs_the_serve_command(self):
        manager = _manager()
        manager.settings["hf_cache"] = self.cache.root

        result = await _launch(
            manager, model="org/model",
            ninfer_artifact=self.cache.artifact,
            sparkdeck_deployment_id="dep-1",
        )

        options = manager._run_managed_container.call_args.args[0]
        self.assertEqual(options["image"], DEFAULT_NINFER_IMAGE)
        self.assertEqual(
            self._resolved_command(options),
            [
                "ninfer-serve", "<artifact>",
                "--host", "0.0.0.0",
                "--port", str(_NINFER_SERVE_PORT),
                # The proxy routes by the repository id, so the served alias
                # must match it rather than the artifact's embedded name.
                "--model-id", "org/model",
            ],
        )
        self.assertIn("model.ninfer", options["command"][1])
        self.assertEqual(
            options["ports"], {f"{_NINFER_SERVE_PORT}/tcp": 8123},
        )
        self.assertEqual(options["labels"]["io.sparkdeck.runtime"], "ninfer")
        self.assertEqual(options["labels"]["io.sparkdeck.deployment"], "dep-1")
        # The artifact reference is resolved against the shared cache mount.
        manager._build_volumes.assert_called_once_with(
            "org/model", self.cache.root, DEFAULT_NINFER_IMAGE,
        )
        self.assertEqual(result["model_source"], "public_repository")

    async def test_a_chat_engine_launch_evicts_other_backends(self):
        manager = _manager()
        manager.settings["hf_cache"] = self.cache.root

        await _launch(manager, ninfer_artifact=self.cache.artifact)

        manager.evict_other_backends.assert_awaited_once_with(protect="ninfer")

    async def test_hugging_face_credential_reaches_the_container(self):
        manager = _manager()
        manager.settings["hf_cache"] = self.cache.root
        manager._container_hf_environment = Mock(
            return_value={"HF_TOKEN": "secret", "HUGGING_FACE_HUB_TOKEN": "secret"}
        )

        await _launch(manager, hf_token="secret", ninfer_artifact=self.cache.artifact)

        options = manager._run_managed_container.call_args.args[0]
        self.assertEqual(
            options["environment"],
            {"HF_TOKEN": "secret", "HUGGING_FACE_HUB_TOKEN": "secret"},
        )
        manager._container_hf_environment.assert_called_once_with("secret")

    async def test_no_environment_block_without_variables(self):
        manager = _manager()
        manager.settings["hf_cache"] = self.cache.root
        manager._container_hf_environment = Mock(return_value={})

        await _launch(manager, ninfer_artifact=self.cache.artifact)

        options = manager._run_managed_container.call_args.args[0]
        self.assertNotIn("environment", options)

    async def test_gpu_request_is_always_attached(self):
        manager = _manager()
        manager.settings["hf_cache"] = self.cache.root
        with patch("manager._node_has_nvidia_driver", return_value=True):
            await _launch(manager, ninfer_artifact=self.cache.artifact)
        self.assertIn("device_requests", manager._run_managed_container.call_args.args[0])

    async def test_nodes_without_an_nvidia_driver_are_rejected_before_eviction(self):
        manager = _manager()
        with patch("manager._node_has_nvidia_driver", return_value=False):
            with self.assertRaisesRegex(ValueError, "requires an NVIDIA GPU"):
                await _launch(manager, ninfer_artifact=self.cache.artifact)
        # Healthy chat backends must not be evicted for a launch that cannot
        # start, and no container may be created.
        manager.evict_other_backends.assert_not_awaited()
        manager._run_managed_container.assert_not_called()

    async def test_missing_artifact_reference_is_rejected(self):
        manager = _manager()

        with self.assertRaisesRegex(ValueError, "require a .ninfer artifact"):
            await _launch(manager)

        manager._run_managed_container.assert_not_called()

    async def test_uncached_artifact_is_rejected(self):
        manager = _manager()
        manager.settings["hf_cache"] = self.cache.root

        # The artifact existence check runs inside the container-creation
        # thread, so the launch wrapper reports it as a RuntimeError — the
        # same contract llama.cpp artifact launches use.
        with self.assertRaisesRegex(RuntimeError, "not cached on this node"):
            await _launch(
                manager,
                ninfer_artifact="models--org--model/snapshots/missing/model.ninfer",
            )

        manager._run_managed_container.assert_not_called()

    async def test_artifact_from_another_repository_is_rejected(self):
        """A retargeted model must never silently serve the old repository's
        compiled artifact."""
        manager = _manager()
        manager.settings["hf_cache"] = self.cache.root

        # The artifact/containment check runs inside the container-creation
        # thread, so the launch wrapper reports it as a RuntimeError — the
        # same contract llama.cpp artifact launches use.
        with self.assertRaisesRegex(
            RuntimeError, "does not belong to org/model",
        ):
            await _launch(
                manager,
                ninfer_artifact=(
                    "models--other--repo/snapshots/"
                    + ("a" * 40) + "/model.ninfer"
                ),
            )

        manager.evict_other_backends.assert_not_awaited()
        manager._run_managed_container.assert_not_called()

    async def test_extra_args_extend_the_served_configuration(self):
        manager = _manager()
        manager.settings["hf_cache"] = self.cache.root

        await _launch(
            manager, port=8200, image="registry.test/ninfer:1",
            ninfer_artifact=self.cache.artifact,
            extra_args=[
                "--max-context", "240000", "--kv-dtype", "fp8",
                "--spec", "mtp", "--draft-tokens", "3",
            ],
            name="ninfer-test", shm_size="4g",
        )

        options = manager._run_managed_container.call_args.args[0]
        self.assertEqual(options["image"], "registry.test/ninfer:1")
        command = self._resolved_command(options)
        self.assertEqual(
            command,
            [
                "ninfer-serve", "<artifact>", "--host", "0.0.0.0", "--port",
                str(_NINFER_SERVE_PORT),
                "--model-id", "org/model",
                "--max-context", "240000", "--kv-dtype", "fp8",
                "--spec", "mtp", "--draft-tokens", "3",
            ],
        )
        self.assertEqual(options["name"], "ninfer-test")
        self.assertEqual(options["shm_size"], "4g")

    async def test_a_foreign_model_id_override_is_rejected(self):
        """The proxy routes requests by the repository id and NInfer rejects
        any other request model, so a custom alias would break a healthy
        deployment — reject it before any backend is evicted."""
        manager = _manager()

        with self.assertRaisesRegex(ValueError, "custom --model-id alias"):
            await _launch(
                manager, ninfer_artifact=self.cache.artifact,
                extra_args=["--model-id", "custom-alias"],
            )

        manager.evict_other_backends.assert_not_awaited()
        manager._run_managed_container.assert_not_called()

    async def test_the_repository_id_as_model_id_is_honoured(self):
        manager = _manager()
        manager.settings["hf_cache"] = self.cache.root

        await _launch(
            manager, ninfer_artifact=self.cache.artifact,
            extra_args=["--model-id", "org/model"],
        )

        command = manager._run_managed_container.call_args.args[0]["command"]
        self.assertEqual(command.count("--model-id"), 1)
        self.assertEqual(
            command[command.index("--model-id") + 1], "org/model",
        )

    async def test_cluster_member_labels_are_recorded(self):
        manager = _manager()
        manager.settings["hf_cache"] = self.cache.root
        member = {
            "deployment_id": "dep-9", "node_id": "spark-2", "rank": 1,
            "mode": "replicated", "nnodes": 2,
        }

        await _launch(
            manager, name="ninfer-replica", cluster_member=member,
            ninfer_artifact=self.cache.artifact,
            sparkdeck_deployment_id="dep-9",
        )

        labels = manager._run_managed_container.call_args.args[0]["labels"]
        self.assertEqual(labels[ENGINE_LABEL], "ninfer")
        self.assertEqual(labels[DEPLOYMENT_LABEL], "dep-9")
        self.assertEqual(labels[NODE_LABEL], "spark-2")
        self.assertEqual(labels[RANK_LABEL], "1")
        self.assertEqual(labels[MODE_LABEL], "replicated")
        self.assertEqual(labels[NNODES_LABEL], "2")

    async def test_sharded_ninfer_is_rejected(self):
        manager = _manager()

        with self.assertRaisesRegex(ValueError, "cannot run sharded"):
            await _launch(
                manager, name="ninfer-shard",
                ninfer_artifact=self.cache.artifact,
                cluster_member={
                    "mode": "sharded", "deployment_id": "d",
                    "node_id": "n", "rank": 0,
                },
            )
        manager._run_managed_container.assert_not_called()

    async def test_missing_image_is_pulled_once(self):
        import docker

        manager = _manager()
        manager.settings["hf_cache"] = self.cache.root
        image = Mock()
        image.attrs = {"Config": {"Env": []}}
        manager.client.images.get = Mock(
            side_effect=[docker.errors.ImageNotFound("missing"), image]
        )

        await _launch(
            manager, image="registry.test/ninfer:1", name="ninfer-pull",
            ninfer_artifact=self.cache.artifact,
        )

        manager.client.images.pull.assert_called_once_with("registry.test/ninfer:1")

    async def test_launch_failure_is_reported_without_leaking_the_token(self):
        manager = _manager()
        manager.client.images.get = Mock(side_effect=RuntimeError("boom hf_secret_value"))
        manager._redact_hf_secret = Mock(return_value="boom [REDACTED]")

        with self.assertRaisesRegex(RuntimeError, r"boom \[REDACTED\]"):
            await _launch(manager, name="ninfer-fail", ninfer_artifact=self.cache.artifact)

        # A launch that fails before the container is created must not have
        # stopped healthy chat backends.
        manager.evict_other_backends.assert_not_awaited()

    async def test_missing_image_is_rejected_before_eviction(self):
        import docker

        manager = _manager()
        manager.client.images.get = Mock(
            side_effect=docker.errors.ImageNotFound("missing")
        )
        manager.client.images.pull = Mock(side_effect=RuntimeError("pull failed"))

        with self.assertRaisesRegex(RuntimeError, "pull failed"):
            await _launch(manager, ninfer_artifact=self.cache.artifact)

        manager.evict_other_backends.assert_not_awaited()
        manager._run_managed_container.assert_not_called()

    def test_ninfer_is_a_supported_engine(self):
        self.assertIn("ninfer", _SUPPORTED_ENGINES)

    def test_ninfer_accepts_operator_environment_variables(self):
        """NInfer reads HF_HUB_OFFLINE-style variables from the container
        environment, so a validator that rejects every non-vLLM map would fail
        the launch before the container branch runs."""
        normalized = normalize_runtime_environment(
            {"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            "ninfer",
        )
        self.assertEqual(
            normalized,
            {"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )
        # The credential guard still applies.
        with self.assertRaisesRegex(ValueError, "managed by SparkDeck"):
            normalize_runtime_environment({"HF_TOKEN": "x"}, "ninfer")


class NinferContainerInspectionTests(unittest.TestCase):
    """Discovered NInfer containers must parse through their own command shape."""

    def test_container_load_settings_parses_the_ninfer_argv(self):
        settings = Manager._container_load_settings(
            Manager.__new__(Manager),
            [
                "ninfer-serve",
                "/root/.cache/huggingface/hub/models--org--model/snapshots/"
                + ("a" * 40) + "/model.ninfer",
                "--host", "0.0.0.0", "--port", "8080",
                "--max-context", "240000", "--max-concurrency", "2",
                "--kv-dtype", "fp8", "--spec", "mtp", "--draft-tokens", "3",
                "--no-thinking", "--model-id", "org/model",
            ],
            "ninfer",
            "org/model",
        )

        self.assertTrue(settings["editable"])
        self.assertEqual(settings["engine"], "ninfer")
        self.assertEqual(settings["context_window"], 240000)
        self.assertEqual(settings["max_concurrency"], 2)
        self.assertEqual(settings["kv_cache_dtype"], "fp8")
        self.assertEqual(settings["thinking_mode"], "disabled")
        self.assertIsNone(settings["tensor_parallel_size"])
        # Engine-owned flags and the positional artifact are stripped; the
        # speculative pair has no dedicated settings keys here, so it stays
        # in extra_args for the launch-controls parser.
        self.assertEqual(
            settings["extra_args"],
            ["--spec", "mtp", "--draft-tokens", "3", "--model-id", "org/model"],
        )
        self.assertEqual(
            settings["artifact_path"],
            "/root/.cache/huggingface/hub/models--org--model/snapshots/"
            + ("a" * 40) + "/model.ninfer",
        )

    def test_container_load_settings_keeps_preserve_thinking(self):
        """--preserve-thinking is an independent extra flag, so inspection
        must retain it instead of folding it into thinking_mode."""
        settings = Manager._container_load_settings(
            Manager.__new__(Manager),
            [
                "ninfer-serve", "/cache/hub/models--org--model/snapshots/"
                + ("a" * 40) + "/model.ninfer",
                "--host", "0.0.0.0", "--port", "8080",
                "--preserve-thinking",
            ],
            "ninfer",
            "org/model",
        )

        self.assertIn("--preserve-thinking", settings["extra_args"])
        self.assertIsNone(settings["thinking_mode"])


class NinferSelectiveArtifactTests(unittest.TestCase):
    """Preparation must download only the selected .ninfer artifact."""

    def test_selective_artifact_recognizes_ninfer(self):
        service = SparkDeckService.__new__(SparkDeckService)

        files = service._llama_selective_artifact(
            {"runtime": "ninfer", "model": {"artifact": "model.ninfer"}},
            "org/model",
        )
        self.assertEqual(files, ["model.ninfer"])

    def test_selective_artifact_ignores_other_runtimes(self):
        service = SparkDeckService.__new__(SparkDeckService)

        self.assertIsNone(service._llama_selective_artifact(
            {"runtime": "vllm", "model": {"artifact": "model.ninfer"}},
            "org/model",
        ))


class NinferEvictionTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_engine_eviction_preserves_laya_but_stops_other_chat_engines(self):
        for engine in ("vllm", "sglang", "llama.cpp", "tensorfold", "ninfer"):
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


class NinferPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_preflight_rejects_sharded_ninfer_layouts(self):
        manager = Manager.__new__(Manager)
        manager._reject_hf_cli_credentials = Mock()
        manager.cluster_nodes = Mock()
        manager._validate_runtime_file_mount_nodes = Mock()

        with self.assertRaisesRegex(ValueError, "single and replicated layouts"):
            await manager._preflight_deployment_launch({
                "model": "org/model",
                "engine": "ninfer",
                "deployment_mode": "sharded",
                "node_ids": ["local", "spark-2"],
            })

    async def test_preflight_rejects_nodes_without_reported_gpus(self):
        """A GPU-less node must fail before any replica evicts healthy
        chat backends, using the same telemetry the launch would rely on."""
        manager = Manager.__new__(Manager)
        manager._reject_hf_cli_credentials = Mock()
        manager._validate_runtime_file_mount_nodes = Mock()

        async def cluster_nodes():
            return [
                {"id": "spark-2", "name": "Spark 2", "online": True,
                 "docker_ready": True,
                 "capabilities": [NINFER_CAPABILITY],
                 "stats": {"gpus": [{"name": "RTX 5090", "memory_total_gb": 32}]}},
                {"id": "cpu-node", "name": "CPU Node", "online": True,
                 "docker_ready": True,
                 "capabilities": [NINFER_CAPABILITY],
                 "stats": {"gpus": []}},
            ]

        manager.cluster_nodes = cluster_nodes

        with self.assertRaisesRegex(ValueError, "none is reported on: CPU Node"):
            await manager._preflight_deployment_launch({
                "model": "org/model",
                "engine": "ninfer",
                "deployment_mode": "replicated",
                "node_ids": ["spark-2", "cpu-node"],
            })

    async def test_preflight_rejects_agents_without_ninfer_support(self):
        """Older agents reject the unknown engine at container creation, so a
        mixed-version selection must fail before any replica launches."""
        manager = Manager.__new__(Manager)
        manager._reject_hf_cli_credentials = Mock()
        manager._validate_runtime_file_mount_nodes = Mock()

        async def cluster_nodes():
            return [
                {"id": "local", "name": "Coordinator", "online": True,
                 "docker_ready": True,
                 "capabilities": [NINFER_CAPABILITY],
                 "stats": {"gpus": [{"name": "RTX 5090", "memory_total_gb": 32}]}},
                {"id": "old-node", "name": "Old Node", "online": True,
                 "docker_ready": True,
                 "capabilities": ["runtime-file-mounts-v1"],
                 "stats": {"gpus": [{"name": "RTX 5090", "memory_total_gb": 32}]}},
            ]

        manager.cluster_nodes = cluster_nodes

        with self.assertRaisesRegex(
            ValueError, "requires updated SparkDeck agents on: Old Node",
        ):
            await manager._preflight_deployment_launch({
                "model": "org/model",
                "engine": "ninfer",
                "deployment_mode": "replicated",
                "node_ids": ["local", "old-node"],
            })

    async def test_preflight_accepts_a_single_node_ninfer_deployment(self):
        manager = Manager.__new__(Manager)
        manager._reject_hf_cli_credentials = Mock()
        manager._validate_runtime_file_mount_nodes = Mock()

        async def cluster_nodes():
            return [{
                "id": "spark-2", "name": "Spark 2", "online": True,
                "docker_ready": True,
                "capabilities": [NINFER_CAPABILITY],
            }]

        manager.cluster_nodes = cluster_nodes

        plan = await manager._preflight_deployment_launch({
            "model": "org/model",
            "engine": "ninfer",
            "deployment_mode": "single",
            "node_ids": ["spark-2"],
        })

        self.assertEqual(plan["engine"], "ninfer")
        self.assertEqual(plan["mode"], "single")
        self.assertEqual(plan["node_ids"], ["spark-2"])


class NinferLaunchControlsTests(unittest.IsolatedAsyncioTestCase):
    """The structured editor's controls must round-trip through NInfer argv."""

    def test_launch_controls_parse_scalars_spec_and_thinking(self):
        controls = Manager._deployment_launch_controls({
            "engine": "ninfer",
            "extra_args": [
                "--max-context", "240000", "--max-concurrency", "2",
                "--kv-dtype", "fp8", "--spec", "mtp", "--draft-tokens", "3",
                "--no-thinking",
            ],
        })
        self.assertEqual(controls["context_window"], 240000)
        self.assertEqual(controls["max_concurrency"], 2)
        self.assertEqual(controls["kv_cache_dtype"], "fp8")
        self.assertEqual(controls["speculative_method"], "mtp")
        self.assertEqual(controls["dspark_num_speculative_tokens"], 3)
        self.assertEqual(controls["thinking_mode"], "disabled")
        self.assertIsNone(controls["tensor_parallel_size"])

    def test_preserve_thinking_is_not_the_thinking_switch(self):
        """NInfer thinks by default; --preserve-thinking is an independent
        assistant-reasoning switch, so it must not read as 'enabled'."""
        controls = Manager._deployment_launch_controls({
            "engine": "ninfer",
            "extra_args": ["--preserve-thinking"],
        })
        self.assertIsNone(controls["thinking_mode"])

    def test_apply_maps_scalars_spec_and_thinking_back_to_flags(self):
        manager = Manager.__new__(Manager)
        args = manager._apply_deployment_launch_controls(
            ["--max-context", "8192", "--kv-dtype", "bf16"], "ninfer",
            {
                "context_window": 240000, "max_concurrency": 4,
                "kv_cache_dtype": "fp8", "speculative_method": "dflash2",
                "dspark_num_speculative_tokens": 7,
                "thinking_mode": "disabled",
            },
        )
        self.assertEqual(
            args,
            [
                "--max-context", "240000", "--max-concurrency", "4",
                "--kv-dtype", "fp8", "--no-thinking",
                "--spec", "dflash2", "--draft-tokens", "7",
            ],
        )

    def test_enabled_thinking_emits_no_flag(self):
        """NInfer thinks by default: 'enabled' only clears --no-thinking and
        must not emit the independent --preserve-thinking switch."""
        manager = Manager.__new__(Manager)
        args = manager._apply_deployment_launch_controls(
            ["--no-thinking", "--preserve-thinking"], "ninfer",
            {"thinking_mode": "enabled"},
        )
        self.assertNotIn("--no-thinking", args)
        self.assertIn("--preserve-thinking", args)

    def test_default_thinking_clears_the_override(self):
        manager = Manager.__new__(Manager)
        args = manager._apply_deployment_launch_controls(
            ["--max-context", "8192", "--no-thinking"], "ninfer",
            {"context_window": 8192, "thinking_mode": "default"},
        )
        self.assertEqual(args, ["--max-context", "8192"])

    def test_clearing_the_spec_method_clears_the_draft_pair(self):
        manager = Manager.__new__(Manager)
        args = manager._apply_deployment_launch_controls(
            ["--spec", "mtp", "--draft-tokens", "3"], "ninfer",
            {"speculative_method": None, "dspark_num_speculative_tokens": None},
        )
        self.assertNotIn("--spec", args)
        self.assertNotIn("--draft-tokens", args)

    def test_mtp_rejects_more_than_five_draft_tokens(self):
        manager = Manager.__new__(Manager)
        with self.assertRaisesRegex(ValueError, "between 1 and 5"):
            manager._apply_deployment_launch_controls(
                ["--spec", "mtp", "--draft-tokens", "3"], "ninfer",
                {
                    "speculative_method": "mtp",
                    "dspark_num_speculative_tokens": 7,
                },
            )

    def test_dflash2_accepts_up_to_fifteen_draft_tokens(self):
        manager = Manager.__new__(Manager)
        args = manager._apply_deployment_launch_controls(
            ["--spec", "dflash2"], "ninfer",
            {
                "speculative_method": "dflash2",
                "dspark_num_speculative_tokens": 15,
            },
        )
        self.assertEqual(args[args.index("--draft-tokens") + 1], "15")

    def test_dflash_rejects_more_than_fifteen_draft_tokens(self):
        manager = Manager.__new__(Manager)
        with self.assertRaisesRegex(ValueError, "between 1 and 15"):
            manager._apply_deployment_launch_controls(
                ["--spec", "dflash"], "ninfer",
                {
                    "speculative_method": "dflash",
                    "dspark_num_speculative_tokens": 16,
                },
            )

    def test_unsupported_speculative_method_is_rejected(self):
        manager = Manager.__new__(Manager)
        with self.assertRaisesRegex(ValueError, "mtp, dflash, or dflash2"):
            manager._apply_deployment_launch_controls(
                [], "ninfer",
                {"speculative_method": "eagle3"},
            )

    def test_rebuild_applies_submitted_speculative_controls(self):
        """A discovered-container edit that changes the speculative pair must
        rewrite the flags instead of preserving the old backend."""
        manager = Manager.__new__(Manager)
        original = [
            "ninfer-serve", "/cache/hub/models--org--model/snapshots/"
            + ("a" * 40) + "/model.ninfer",
            "--host", "0.0.0.0", "--port", "8080",
            "--spec", "mtp", "--draft-tokens", "3",
        ]

        argv = manager._updated_container_command(
            original, "ninfer", "org/model",
            {
                "speculative_method": "dflash2",
                "dspark_num_speculative_tokens": 7,
            },
        )

        self.assertEqual(argv[argv.index("--spec") + 1], "dflash2")
        self.assertEqual(argv[argv.index("--draft-tokens") + 1], "7")

    def test_rebuild_clears_speculation_when_the_method_is_removed(self):
        manager = Manager.__new__(Manager)
        original = [
            "ninfer-serve", "/cache/hub/models--org--model/snapshots/"
            + ("a" * 40) + "/model.ninfer",
            "--host", "0.0.0.0", "--port", "8080",
            "--spec", "mtp", "--draft-tokens", "3",
        ]

        argv = manager._updated_container_command(
            original, "ninfer", "org/model",
            {"speculative_method": None, "dspark_num_speculative_tokens": None},
        )

        self.assertNotIn("--spec", argv)
        self.assertNotIn("--draft-tokens", argv)

    def test_validator_rejects_foreign_methods(self):
        with self.assertRaisesRegex(ValueError, "mtp, dflash, or dflash2"):
            Manager._validated_ninfer_speculation("eagle3", None)

    def test_unsupported_kv_dtype_is_rejected(self):
        manager = Manager.__new__(Manager)
        with self.assertRaisesRegex(ValueError, "bf16, int8, fp8, nvfp4, or k8v4"):
            manager._apply_deployment_launch_controls(
                [], "ninfer", {"kv_cache_dtype": "fp8_e4m3"},
            )

    def test_max_concurrency_above_eight_is_rejected(self):
        """NInfer's documented admission range is 1..8, enforced on both the
        structured apply path and the discovered-container rebuild."""
        manager = Manager.__new__(Manager)
        command = [
            "ninfer-serve", "/cache/hub/models--org--model/snapshots/"
            + ("a" * 40) + "/model.ninfer",
            "--host", "0.0.0.0", "--port", "8080",
            "--max-concurrency", "4",
        ]
        with self.assertRaisesRegex(ValueError, "between 1 and 8"):
            manager._apply_deployment_launch_controls(
                ["--max-context", "8192"], "ninfer",
                {"context_window": 8192, "max_concurrency": 9},
            )
        with self.assertRaisesRegex(ValueError, "between 1 and 8"):
            manager._updated_container_command(
                command, "ninfer", "org/model", {"max_concurrency": 12},
            )

    def test_discovered_environment_includes_ninfer(self):
        from sparkdeck.runtime_environment import discovered_runtime_environment

        discovered = discovered_runtime_environment(
            {"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            "ninfer",
        )
        self.assertEqual(discovered, {
            "HF_HUB_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })

    async def test_immediate_cluster_launch_carries_the_artifact(self):
        """Creating with launch=true and explicit nodes must derive the same
        cache-relative reference the saved-launch path computes."""
        from sparkdeck.service import SparkDeckService

        temp = tempfile.TemporaryDirectory()
        manager = FakeClusterManager()
        service = SparkDeckService(manager, Path(temp.name))
        service._prepare_public_gguf_artifact = AsyncMock(
            return_value="/cache/hub/models--org--model/snapshots/x/model.ninfer"
        )
        service._resolved_model_revision = AsyncMock(return_value="c" * 40)
        service._link_cluster_record = Mock()

        try:
            await service.create_deployment({
                "model": "org/model",
                "alias": "ni-immediate",
                "runtime": "ninfer",
                "artifact": "model.ninfer",
                "node_ids": ["spark-2"],
                "deployment_mode": "single",
            }, launch=True)

            body = manager.create_deployment.await_args.args[0]
            self.assertEqual(
                body["ninfer_artifact"],
                "models--org--model/snapshots/{}/model.ninfer".format("c" * 40),
            )
        finally:
            await manager.http.aclose()
            await service.close()
            temp.cleanup()

    def test_revision_pin_is_not_injected_for_ninfer(self):
        args = Manager._with_saved_launch_identity(
            ["--max-context", "8192"], "ninfer", model_revision="a" * 40,
        )
        self.assertNotIn("--revision", args)


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
            "id": "cluster-ni", "status": "starting", "api_port": 8123,
            "members": [{"container_name": "ninfer-org-model-8123"}],
            "model_source": "public_repository",
        })
        self.public_target_node = Mock(side_effect=lambda node: node)


class NinferClusterLaunchTests(unittest.IsolatedAsyncioTestCase):
    async def test_revision_pin_is_rejected_at_creation(self):
        """NInfer pins its snapshot inside the prepared artifact path, so a
        persisted revision pin would claim provenance the server never loads."""
        from sparkdeck.service import SparkDeckService

        # The store keeps a SQLite handle open, so the directory must be
        # cleaned up only after the service is closed (notably on Windows).
        temp = tempfile.TemporaryDirectory()
        manager = FakeClusterManager()
        service = SparkDeckService(manager, Path(temp.name))

        try:
            with self.assertRaisesRegex(
                ValueError, "cannot pin a model revision",
            ):
                await service.create_deployment({
                    "model": "org/model",
                    "alias": "ni-pinned",
                    "runtime": "ninfer",
                    "artifact": "model.ninfer",
                    "revision": "a" * 40,
                    "node_ids": ["spark-2"],
                    "deployment_mode": "single",
                })
            # The rejected record must not be persisted under its alias.
            self.assertIsNone(
                service.store.deployment("ni-pinned", include_private=True),
            )
        finally:
            await manager.http.aclose()
            await service.close()
            temp.cleanup()

    async def test_relaunch_revalidates_the_selective_artifact_before_removing_ranks(self):
        """The per-file .ninfer presence check is the only pre-removal
        readiness gate for a selective snapshot: without it a bad node
        selection would remove the serving ranks before the container
        create discovers the missing artifact."""
        from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
        from sparkdeck.service import SparkDeckService

        temp = tempfile.TemporaryDirectory()
        manager = FakeClusterManager()
        manager.deployment_action = AsyncMock(return_value={"ok": True, "errors": []})
        manager.node_has_model_files = AsyncMock(return_value=False)
        service = SparkDeckService(manager, Path(temp.name))
        service._resolved_model_revision = AsyncMock(return_value="a" * 40)

        service.store.add_deployment(Deployment(
            id="record-ni", alias="ni-live", runtime=RuntimeKind.NINFER,
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/model", artifact="model.ninfer"),
            container_name="cluster-old-r0-model",
            settings={
                "manager_deployment_id": "cluster-old",
                "node_ids": ["spark-2"], "deployment_mode": "single",
            },
        ), "http://127.0.0.1:8123")
        manager.deployments = [{
            "id": "cluster-old", "status": "stopped", "engine": "ninfer",
            "mode": "single", "node_ids": ["spark-2"],
            "sparkdeck_record_id": "record-ni",
            "launch_settings": {
                "engine": "ninfer", "deployment_mode": "single",
                "node_ids": ["spark-2"], "extra_args": [],
                "ninfer_artifact": (
                    f"models--org--model/snapshots/{'a' * 40}/model.ninfer"
                ),
            },
            "members": [{
                "node_id": "spark-2", "rank": 0,
                "container_name": "cluster-old-r0-model",
            }],
        }]

        try:
            with self.assertRaisesRegex(
                ValueError, "model weights are not available",
            ):
                await service.deployment_action("record-ni", "start", ["spark-2"])
            manager.deployment_action.assert_not_awaited()
        finally:
            await manager.http.aclose()
            await service.close()
            temp.cleanup()

    async def test_relaunch_validates_the_persisted_ninfer_snapshot(self):
        """The relaunch resolves the snapshot pinned in the persisted
        artifact reference, not the bookmark's current mutable revision, so
        validation must check that exact snapshot."""
        from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
        from sparkdeck.service import SparkDeckService

        temp = tempfile.TemporaryDirectory()
        manager = FakeClusterManager()
        manager.deployment_action = AsyncMock(return_value={"ok": True, "errors": []})
        manager.node_has_model_files = AsyncMock(return_value=True)
        service = SparkDeckService(manager, Path(temp.name))
        # The bookmark revision would resolve to a different head; the
        # persisted artifact reference must win instead.
        service._resolved_model_revision = AsyncMock(return_value="c" * 40)

        service.store.add_deployment(Deployment(
            id="record-ni", alias="ni-live", runtime=RuntimeKind.NINFER,
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/model", artifact="model.ninfer"),
            container_name="cluster-old-r0-model",
            settings={
                "manager_deployment_id": "cluster-old",
                "node_ids": ["spark-2"], "deployment_mode": "single",
            },
        ), "http://127.0.0.1:8123")
        manager.deployments = [{
            "id": "cluster-old", "status": "stopped", "engine": "ninfer",
            "mode": "single", "node_ids": ["spark-2"],
            "sparkdeck_record_id": "record-ni",
            "launch_settings": {
                "engine": "ninfer", "deployment_mode": "single",
                "node_ids": ["spark-2"], "extra_args": [],
                "ninfer_artifact": (
                    f"models--org--model/snapshots/{'a' * 40}/model.ninfer"
                ),
            },
            "members": [{
                "node_id": "spark-2", "rank": 0,
                "container_name": "cluster-old-r0-model",
            }],
        }]

        try:
            await service.deployment_action("record-ni", "start", ["spark-2"])

            args = manager.node_has_model_files.await_args.args
            self.assertEqual(args[0], "spark-2")
            self.assertEqual(args[1], "org/model")
            self.assertEqual(args[2], "a" * 40)
            self.assertEqual(args[3], ["model.ninfer"])
            service._resolved_model_revision.assert_not_awaited()
            # The validated snapshot feeds the relaunch (Manager suppresses
            # the flag itself for NInfer; the artifact path pins it).
            manager.deployment_action.assert_awaited_once_with(
                "cluster-old", "start", ["spark-2"],
                model_revision="a" * 40,
            )
        finally:
            await manager.http.aclose()
            await service.close()
            temp.cleanup()

    async def test_cached_revision_is_not_injected_for_ninfer(self):
        """A shared cached snapshot must not add the unsupported --revision
        flag to a NInfer launch: the artifact reference pins the snapshot."""
        from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
        from sparkdeck.service import SparkDeckService

        temp = tempfile.TemporaryDirectory()
        manager = FakeClusterManager()
        service = SparkDeckService(manager, Path(temp.name))
        service._validate_start_selection = AsyncMock(return_value="b" * 40)
        service._resolved_model_revision = AsyncMock(return_value="c" * 40)
        service._link_cluster_record = Mock()
        record = Deployment(
            id="record-ni", alias="ni-model",
            runtime=RuntimeKind.NINFER, kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/model"), settings={},
        )

        try:
            await service._launch_cluster_record(
                record, {}, "org/model", "model.ninfer", ["spark-2"],
            )

            service._validate_start_selection.assert_not_awaited()
            body = manager.create_deployment.await_args.args[0]
            self.assertEqual(body["engine"], "ninfer")
            self.assertEqual(
                body["ninfer_artifact"],
                "models--org--model/snapshots/{}/model.ninfer".format("c" * 40),
            )
            self.assertNotIn("--revision", body["extra_args"])
        finally:
            await manager.http.aclose()
            await service.close()
            temp.cleanup()

    async def test_ninfer_launch_requires_an_artifact(self):
        from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
        from sparkdeck.service import SparkDeckService

        temp = tempfile.TemporaryDirectory()
        manager = FakeClusterManager()
        service = SparkDeckService(manager, Path(temp.name))
        service._link_cluster_record = Mock()
        record = Deployment(
            id="record-ni", alias="ni-model",
            runtime=RuntimeKind.NINFER, kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/model"), settings={},
        )

        try:
            with self.assertRaisesRegex(ValueError, "require a .ninfer artifact"):
                await service._launch_cluster_record(
                    record, {}, "org/model", "", ["spark-2"],
                )
        finally:
            await manager.http.aclose()
            await service.close()
            temp.cleanup()

    async def test_local_artifact_paths_are_rejected_at_creation(self):
        """There is no controller-local NInfer launch path, so a local .ninfer
        file must be rejected when the deployment is saved instead of
        producing a bookmark that can never start."""
        from sparkdeck.service import SparkDeckService

        temp = tempfile.TemporaryDirectory()
        manager = FakeClusterManager()
        service = SparkDeckService(manager, Path(temp.name))

        try:
            with self.assertRaisesRegex(
                ValueError, "require a repo-relative Hub .ninfer artifact",
            ):
                await service.create_deployment({
                    "model": "org/model",
                    "alias": "ni-local",
                    "runtime": "ninfer",
                    "artifact": str(Path(temp.name) / "model.ninfer"),
                    "node_ids": ["spark-2"],
                    "deployment_mode": "single",
                })
            self.assertIsNone(
                service.store.deployment("ni-local", include_private=True),
            )
        finally:
            await manager.http.aclose()
            await service.close()
            temp.cleanup()

    async def test_creation_without_an_artifact_is_rejected(self):
        """A REST bookmark without an artifact can never launch, so creation
        must reject it instead of persisting the record."""
        from sparkdeck.service import SparkDeckService

        temp = tempfile.TemporaryDirectory()
        manager = FakeClusterManager()
        service = SparkDeckService(manager, Path(temp.name))

        try:
            with self.assertRaisesRegex(
                ValueError, "require a .ninfer artifact",
            ):
                await service.create_deployment({
                    "model": "org/model",
                    "alias": "ni-bare",
                    "runtime": "ninfer",
                    "node_ids": ["spark-2"],
                    "deployment_mode": "single",
                })
            self.assertIsNone(
                service.store.deployment("ni-bare", include_private=True),
            )
        finally:
            await manager.http.aclose()
            await service.close()
            temp.cleanup()


class NinferProxyTests(unittest.IsolatedAsyncioTestCase):
    """Managed NInfer traffic must route through Manager's member-aware path."""

    async def test_managed_ninfer_uses_manager_member_routing(self):
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
                id="record-ni", alias="ni-model",
                runtime=RuntimeKind.NINFER, kind=DeploymentKind.MANAGED,
                model=ModelIdentity("org/model"),
                container_name="ninfer-org-model-8123", status="running",
                settings={"manager_deployment_id": "cluster-ni"},
            ))
            service.store.update_managed_routing(
                "record-ni", {"manager_deployment_id": "cluster-ni"},
                "ninfer-org-model-8123", "http://127.0.0.1:8123",
            )

            try:
                await service.proxy(
                    {"model": "ni-model", "messages": [{"role": "user", "content": "hi"}]},
                    "chat/completions",
                )

                manager.proxy_cluster_inference.assert_awaited_once()
                args = manager.proxy_cluster_inference.await_args.args
                self.assertEqual(args[0], "cluster-ni")
                self.assertEqual(args[1], "org/model")
            finally:
                await manager.http.aclose()
                await service.close()


class NinferLaunchSettingsTests(unittest.TestCase):
    """Typed NInfer settings must survive both launch paths."""

    def _body(self, settings):
        from sparkdeck.models import ModelIdentity, RuntimeKind

        return SparkDeckService.__new__(SparkDeckService)._cluster_launch_body(
            RuntimeKind.NINFER, "org/model", "ni-model", "record-1",
            ModelIdentity("org/model"), settings,
            ["spark-2"], "single", None,
        )

    def test_cluster_launch_translates_typed_settings_to_flags(self):
        body = self._body({"context_length": 16384, "thinking": False})

        extra = body["extra_args"]
        self.assertEqual(extra[extra.index("--max-context") + 1], "16384")
        self.assertIn("--no-thinking", extra)
        self.assertEqual(body["engine"], "ninfer")

    def test_cluster_launch_translates_the_typed_concurrency_limit(self):
        body = self._body({"max_concurrency": 4})

        extra = body["extra_args"]
        self.assertEqual(extra[extra.index("--max-concurrency") + 1], "4")

    def test_unset_settings_do_not_emit_flags(self):
        body = self._body({"context_length": None, "thinking": None})

        self.assertNotIn("--max-context", body["extra_args"])
        self.assertNotIn("--no-thinking", body["extra_args"])

    def test_cluster_launch_carries_the_artifact_reference(self):
        from sparkdeck.models import ModelIdentity, RuntimeKind

        body = SparkDeckService.__new__(SparkDeckService)._cluster_launch_body(
            RuntimeKind.NINFER, "org/model", "ni-model", "record-1",
            ModelIdentity("org/model"), {},
            ["spark-2"], "single", None,
            ninfer_artifact="models--org--model/snapshots/x/model.ninfer",
        )

        self.assertEqual(
            body["ninfer_artifact"],
            "models--org--model/snapshots/x/model.ninfer",
        )

    def test_revision_is_not_forwarded_for_ninfer(self):
        from sparkdeck.models import ModelIdentity, RuntimeKind

        body = SparkDeckService.__new__(SparkDeckService)._cluster_launch_body(
            RuntimeKind.NINFER, "org/model", "ni-model", "record-1",
            ModelIdentity("org/model", revision="a" * 40),
            {}, ["spark-2"], "single", None,
        )

        self.assertNotIn("--revision", body["extra_args"])

    def test_service_configuration_keeps_ninfer_inputs(self):
        """_safe_configuration runs on every persisted record."""
        configuration = SparkDeckService._safe_configuration({
            "context_length": 16384, "secrets": "must not survive",
        })

        self.assertEqual(configuration.get("context_length"), 16384)
        self.assertNotIn("secrets", configuration)


class NinferPromotionTests(unittest.IsolatedAsyncioTestCase):
    """Promoting a discovered NInfer container must recover the artifact."""

    def test_saved_bookmark_editor_seeds_the_typed_concurrency(self):
        """An unchanged editor save must not null out a concurrency limit
        that was saved in settings.max_concurrency."""
        from sparkdeck.service import SparkDeckService

        service = SparkDeckService.__new__(SparkDeckService)
        service.manager = Manager
        controls = service._saved_bookmark_launch_controls(
            "ninfer", ["--spec", "mtp"], {"max_concurrency": 4},
        )
        self.assertEqual(controls["max_concurrency"], 4)

    def test_recovery_rebuilds_the_artifact_reference(self):
        manager = Manager.__new__(Manager)
        cache_root = "/host/cache"
        image = "sparkdeck/ninfer:latest"
        image_mock = Mock()
        image_mock.attrs = {"Config": {"Env": [f"HF_HOME={cache_root}"]}}
        manager.client = Mock()
        manager.client.images.get = Mock(return_value=image_mock)
        manager._image_hf_cache_target = (
            Manager._image_hf_cache_target.__get__(manager)
        )

        recovered = manager._recovered_deployment_launch_settings(
            {
                "name": "promoted", "model": "org/model", "engine": "ninfer",
                "mode": "single", "node_ids": ["local"],
            },
            {
                "image": image,
                "load_settings": {
                    "command_flags": "--spec mtp --draft-tokens 3",
                    "extra_args": ["--spec", "mtp", "--draft-tokens", "3"],
                    "artifact_path": (
                        f"{cache_root}/hub/models--org--model/snapshots/"
                        + ("a" * 40) + "/compiled/model.ninfer"
                    ),
                    # Structured controls the parser stripped from the flags.
                    "context_window": 240000,
                    "max_concurrency": 2,
                    "kv_cache_dtype": "fp8",
                    "thinking_mode": "disabled",
                },
            },
        )

        self.assertEqual(
            recovered.get("ninfer_artifact"),
            "models--org--model/snapshots/{}/compiled/model.ninfer".format("a" * 40),
        )
        # The stripped structured controls must return as flags so promotion
        # does not silently start with NInfer defaults.
        extra = recovered.get("extra_args") or []
        self.assertEqual(extra[extra.index("--max-context") + 1], "240000")
        self.assertEqual(extra[extra.index("--max-concurrency") + 1], "2")
        self.assertEqual(extra[extra.index("--kv-dtype") + 1], "fp8")
        self.assertIn("--no-thinking", extra)
        self.assertEqual(extra[extra.index("--spec") + 1], "mtp")
        self.assertEqual(extra[extra.index("--draft-tokens") + 1], "3")

    def test_updated_command_rebuilds_the_ninfer_argv(self):
        manager = Manager.__new__(Manager)
        original = [
            "ninfer-serve",
            "/root/.cache/huggingface/hub/models--org--model/snapshots/"
            + ("a" * 40) + "/model.ninfer",
            "--host", "0.0.0.0", "--port", "8080",
            "--max-context", "8192", "--spec", "mtp", "--draft-tokens", "3",
        ]

        argv = manager._updated_container_command(
            original, "ninfer", "org/model",
            {"context_window": 240000, "max_concurrency": 4,
             "kv_cache_dtype": "fp8", "thinking_mode": "disabled"},
        )

        self.assertEqual(argv[0], "ninfer-serve")
        self.assertTrue(argv[1].endswith("model.ninfer"))
        # Host and port are preserved so Docker's port mapping stays valid.
        self.assertEqual(
            argv[argv.index("--host") + 1:argv.index("--host") + 2], ["0.0.0.0"],
        )
        self.assertEqual(argv[argv.index("--port") + 1], "8080")
        self.assertEqual(argv[argv.index("--max-context") + 1], "240000")
        self.assertEqual(argv[argv.index("--max-concurrency") + 1], "4")
        self.assertEqual(argv[argv.index("--kv-dtype") + 1], "fp8")
        self.assertIn("--no-thinking", argv)
        # The vLLM spellings must not leak into the rebuilt NInfer argv.
        self.assertNotIn("--kv-cache-dtype", argv)
        self.assertNotIn("--max-num-seqs", argv)
        self.assertNotIn("--max-model-len", argv)
        # Speculation rides in the flags and survives the rebuild.
        self.assertEqual(argv[argv.index("--spec") + 1], "mtp")
        self.assertEqual(argv[argv.index("--draft-tokens") + 1], "3")


if __name__ == "__main__":
    unittest.main()
