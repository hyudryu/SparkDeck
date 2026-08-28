import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock

from sparkdeck.runtimes import (
    LlamaCppAdapter, RuntimeRegistry, SglangAdapter, VllmAdapter,
    launch_managed_container, normalize_openai_base_url,
)


class RuntimeAdapterTests(unittest.TestCase):
    def test_registry_supports_exactly_three_runtimes(self):
        self.assertEqual(set(RuntimeRegistry().kinds), {"vllm", "llama.cpp", "sglang"})

    def test_vllm_launch_settings(self):
        command = VllmAdapter().launch_spec("org/model", {
            "tensor_parallel_size": 2, "pipeline_parallel_size": 3,
            "context_length": 8192,
            "quantization": "awq", "revision": "release-1",
        }).command
        self.assertEqual(command[:3], ["vllm", "serve", "org/model"])
        self.assertIn("--tensor-parallel-size", command)
        self.assertEqual(command[command.index("--pipeline-parallel-size") + 1], "3")
        self.assertIn("--max-model-len", command)
        self.assertEqual(command[command.index("--revision") + 1], "release-1")

    def test_llama_server_uses_real_gguf_flags(self):
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.gguf"
            artifact.touch()
            spec = LlamaCppAdapter().launch_spec("org/model", {
                "artifact": str(artifact), "context_length": 4096,
                "parallel_slots": 4, "gpu_layers": 99,
                "split_mode": "layer", "tensor_split": "1,1",
            })
        self.assertIn("--model", spec.command)
        self.assertIn("--ctx-size", spec.command)
        self.assertIn("--parallel", spec.command)
        self.assertIn("--n-gpu-layers", spec.command)
        self.assertNotIn("--tensor-parallel-size", spec.command)
        self.assertEqual(spec.command[spec.command.index("--model") + 1], "/models/model.gguf")
        self.assertIn(str(artifact.resolve()), spec.volumes)

    def test_llama_server_rejects_repository_without_gguf_artifact(self):
        with self.assertRaisesRegex(ValueError, "existing local GGUF artifact"):
            LlamaCppAdapter().launch_spec("org/model", {})

    def test_llama_server_mounts_complete_shard_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "model-Q4_K_M-00001-of-00002.gguf"
            second = root / "model-Q4_K_M-00002-of-00002.gguf"
            first.touch()
            second.touch()

            spec = LlamaCppAdapter().launch_spec(
                "org/model", {"artifact": str(second)},
            )

        self.assertEqual(
            spec.command[spec.command.index("--model") + 1],
            "/models/model-Q4_K_M-00001-of-00002.gguf",
        )
        self.assertEqual(
            spec.volumes,
            {str(root.resolve()): {"bind": "/models", "mode": "ro"}},
        )

    def test_llama_server_rejects_incomplete_shard_set(self):
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "model-Q4_K_M-00001-of-00002.gguf"
            artifact.touch()

            with self.assertRaisesRegex(ValueError, "every GGUF shard"):
                LlamaCppAdapter().launch_spec(
                    "org/model", {"artifact": str(artifact)},
                )

    @unittest.skipIf(os.name == "nt", "creating cache symlinks requires privileges")
    def test_llama_server_mounts_hub_symlink_shards_under_logical_names(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            blobs = root / "blobs"
            snapshot = root / "snapshots" / "revision"
            blobs.mkdir()
            snapshot.mkdir(parents=True)
            first_blob = blobs / "first-content-hash"
            second_blob = blobs / "second-content-hash"
            first_blob.write_bytes(b"first")
            second_blob.write_bytes(b"second")
            first = snapshot / "model-00001-of-00002.gguf"
            second = snapshot / "model-00002-of-00002.gguf"
            first.symlink_to(Path("../../blobs/first-content-hash"))
            second.symlink_to(Path("../../blobs/second-content-hash"))

            spec = LlamaCppAdapter().launch_spec(
                "org/model", {"artifact": str(second)},
            )

        self.assertEqual(
            spec.command[spec.command.index("--model") + 1],
            "/models/model-00001-of-00002.gguf",
        )
        self.assertEqual(spec.volumes, {
            str(first_blob.resolve()): {
                "bind": "/models/model-00001-of-00002.gguf", "mode": "ro",
            },
            str(second_blob.resolve()): {
                "bind": "/models/model-00002-of-00002.gguf", "mode": "ro",
            },
        })

    def test_sglang_uses_runtime_native_parallel_flags(self):
        command = SglangAdapter().launch_spec("org/model", {
            "tensor_parallel_size": 2, "data_parallel_size": 3,
            "context_length": 32768,
        }).command
        self.assertIn("--tp-size", command)
        self.assertIn("--dp-size", command)
        self.assertIn("--context-length", command)

    def test_openai_base_url_normalizes_one_v1_prefix(self):
        self.assertEqual(
            normalize_openai_base_url("https://example.test/openai/v1/"),
            "https://example.test/openai",
        )


class ManagedLaunchBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_does_not_duplicate_v1_prefix(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {"data": []}
        http = Mock()
        http.get = AsyncMock(return_value=response)

        await VllmAdapter().health(http, "https://example.test/openai/v1/")

        self.assertEqual(
            http.get.await_args.args[0], "https://example.test/openai/v1/models"
        )

    async def test_sglang_bridge_keeps_data_parallelism_and_quantization(self):
        manager = Mock()
        manager.create_container = AsyncMock(return_value={"name": "sparkdeck-model", "port": 8000})

        await launch_managed_container(
            manager, SglangAdapter(), "dep-1", "model", "org/model",
            {"data_parallel_size": 2, "quantization": "fp8"},
        )

        extra = manager.create_container.await_args.kwargs["extra_args"]
        self.assertEqual(extra, ["--dp-size", "2", "--quantization", "fp8"])

    async def test_vllm_bridge_keeps_pipeline_parallelism(self):
        manager = Mock()
        manager.create_container = AsyncMock(return_value={"name": "sparkdeck-model", "port": 8000})

        await launch_managed_container(
            manager, VllmAdapter(), "dep-1", "model", "org/model",
            {"tensor_parallel_size": 2, "pipeline_parallel_size": 3},
        )

        extra = manager.create_container.await_args.kwargs["extra_args"]
        self.assertEqual(extra, [
            "--tensor-parallel-size", "2",
            "--pipeline-parallel-size", "3",
        ])

    async def test_managed_launches_keep_requested_revision(self):
        for adapter in (VllmAdapter(), SglangAdapter()):
            with self.subTest(runtime=adapter.kind.value):
                manager = Mock()
                manager.create_container = AsyncMock(
                    return_value={"name": "sparkdeck-model", "port": 8000}
                )
                await launch_managed_container(
                    manager, adapter, "dep-1", "model", "org/model",
                    {"revision": "revision-abc"},
                )
                extra = manager.create_container.await_args.kwargs["extra_args"]
                self.assertEqual(extra[extra.index("--revision") + 1], "revision-abc")

    async def test_llama_bridge_uses_durable_managed_create(self):
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.gguf"
            artifact.touch()
            container = Mock()
            container.reload = Mock()
            manager = Mock()
            manager.settings = {"shm_size": "16g"}
            manager.evict_other_backends = AsyncMock()
            manager._allocate_port = AsyncMock(return_value=8100)
            manager._run_managed_container = Mock(return_value=container)
            manager._container_summary = Mock(return_value={
                "name": "sparkdeck-llama-dep-1", "port": 8100,
            })
            manager.client.images.get = Mock()

            await launch_managed_container(
                manager, LlamaCppAdapter(), "dep-1", "llama", "org/model",
                {"artifact": str(artifact)},
            )

        options = manager._run_managed_container.call_args.args[0]
        self.assertEqual(options["labels"]["io.sparkdeck.managed"], "1")
        self.assertEqual(options["labels"]["io.sparkdeck.deployment"], "dep-1")
