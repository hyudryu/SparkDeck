import unittest
from unittest.mock import AsyncMock, Mock

from sparkdeck.runtimes import (
    LlamaCppAdapter, RuntimeRegistry, SglangAdapter, VllmAdapter,
    launch_managed_container,
)


class RuntimeAdapterTests(unittest.TestCase):
    def test_registry_supports_exactly_three_runtimes(self):
        self.assertEqual(set(RuntimeRegistry().kinds), {"vllm", "llama.cpp", "sglang"})

    def test_vllm_launch_settings(self):
        command = VllmAdapter().launch_spec("org/model", {
            "tensor_parallel_size": 2, "context_length": 8192, "quantization": "awq",
        }).command
        self.assertEqual(command[:3], ["vllm", "serve", "org/model"])
        self.assertIn("--tensor-parallel-size", command)
        self.assertIn("--max-model-len", command)

    def test_llama_server_uses_real_gguf_flags(self):
        spec = LlamaCppAdapter().launch_spec("model.gguf", {
            "context_length": 4096, "parallel_slots": 4, "gpu_layers": 99,
            "split_mode": "layer", "tensor_split": "1,1",
        })
        self.assertIn("--model", spec.command)
        self.assertIn("--ctx-size", spec.command)
        self.assertIn("--parallel", spec.command)
        self.assertIn("--n-gpu-layers", spec.command)
        self.assertNotIn("--tensor-parallel-size", spec.command)

    def test_sglang_uses_runtime_native_parallel_flags(self):
        command = SglangAdapter().launch_spec("org/model", {
            "tensor_parallel_size": 2, "data_parallel_size": 3,
            "context_length": 32768,
        }).command
        self.assertIn("--tp-size", command)
        self.assertIn("--dp-size", command)
        self.assertIn("--context-length", command)


class ManagedLaunchBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_sglang_bridge_keeps_data_parallelism_and_quantization(self):
        manager = Mock()
        manager.create_container = AsyncMock(return_value={"name": "sparkdeck-model", "port": 8000})

        await launch_managed_container(
            manager, SglangAdapter(), "dep-1", "model", "org/model",
            {"data_parallel_size": 2, "quantization": "fp8"},
        )

        extra = manager.create_container.await_args.kwargs["extra_args"]
        self.assertEqual(extra, ["--dp-size", "2", "--quantization", "fp8"])
