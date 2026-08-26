import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from sparkdeck.service import SparkDeckService


class FakeManager:
    def __init__(self):
        self.http = httpx.AsyncClient()
        self.list_containers = AsyncMock(return_value=[])


class BenchmarkCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = FakeManager()
        self.service = SparkDeckService(self.manager, Path(self.temp.name))

    async def asyncTearDown(self):
        await self.manager.http.aclose()
        await self.service.close()
        self.temp.cleanup()

    async def test_response_metrics_persist_without_request_or_output_content(self):
        self.service.store.set_setting("community_consent", True)
        response = {
            "choices": [{"message": {"content": "generated secret"}}],
            "usage": {"prompt_tokens": 32, "completion_tokens": 24},
            "timings": {"predicted_per_second": 96.0, "prompt_per_second": 128.0},
        }
        self.service._record_response(
            None, "org/model", "vllm", {"context_size": 4096, "api_key": "secret"},
            time.monotonic() - 0.25, response,
        )
        items, total = self.service.store.benchmarks()
        self.assertEqual(total, 1)
        payload = str(items[0])
        self.assertNotIn("generated secret", payload)
        self.assertNotIn("api_key", payload)
        self.assertEqual(items[0]["output_tokens"], 24)
        self.assertTrue(items[0]["eligible_for_community"])
        self.assertEqual(
            self.service.store.sync_status()["outbox"]["waiting_for_account"], 1
        )

    async def test_cluster_routing_identifiers_never_enter_benchmark_configuration(self):
        self.service._record_response(
            "dep-1", "org/model", "vllm",
            {
                "context_length": 4096,
                "node_ids": ["private-worker-id"],
                "manager_deployment_id": "private-cluster-uuid",
                "deployment_mode": "single",
            },
            time.monotonic() - 0.2,
            {"usage": {"prompt_tokens": 32, "completion_tokens": 24}},
        )

        items, _ = self.service.store.benchmarks()
        self.assertEqual(items[0]["configuration"], {"context_length": 4096})

    async def test_gpu_memory_utilization_is_safely_persisted_and_benchmarked(self):
        settings = {
            "gpu_memory_utilization": 0.82,
            "api_key": "secret",
        }

        persisted = self.service._local_configuration(settings)
        self.service._record_response(
            "dep-1", "org/model", "vllm", settings,
            time.monotonic() - 0.2,
            {"usage": {"prompt_tokens": 32, "completion_tokens": 24}},
        )
        items, _ = self.service.store.benchmarks()

        self.assertEqual(persisted, {"gpu_memory_utilization": 0.82})
        self.assertEqual(
            items[0]["configuration"], {"gpu_memory_utilization": 0.82},
        )
        for unsafe in ("0.82", True, float("nan"), 0, 1.01):
            self.assertNotIn(
                "gpu_memory_utilization",
                self.service._local_configuration({
                    "gpu_memory_utilization": unsafe,
                }),
            )

    async def test_hardware_snapshot_uses_public_hardware_class_contract(self):
        self.manager._stats_cache = {
            "gpus": [{"name": "NVIDIA GB10", "mem_total_mib": 128000}]
        }

        snapshot = self.service._hardware_snapshot()

        self.assertEqual(snapshot["hardware_class"], "dgx-spark")
        self.assertNotIn("device_class", snapshot)

    async def test_remote_primary_benchmark_uses_serving_node_hardware(self):
        self.manager._cluster_primary_member = lambda _: (
            {"id": "manager-deployment"},
            {"node_id": "worker-1", "container_name": "rank-0"},
        )
        self.manager.node_registry = SimpleNamespace(request=AsyncMock(return_value={
            "gpus": [{"name": "NVIDIA GB10", "mem_total_mib": 128000}],
        }))
        self.manager.proxy_cluster_inference = AsyncMock(return_value={
            "usage": {"prompt_tokens": 24, "completion_tokens": 20},
            "timings": {"predicted_per_second": 80.0},
        })
        deployment = {
            "id": "dep-remote", "alias": "remote-model", "kind": "managed",
            "runtime": "vllm", "model": {"repository": "org/model"},
            "settings": {"manager_deployment_id": "manager-deployment"},
        }

        await self.service._proxy_managed(
            deployment, {"model": "remote-model", "stream": False},
            "chat/completions", None,
        )

        items, _ = self.service.store.benchmarks()
        self.assertEqual(items[0]["hardware"]["hardware_class"], "dgx-spark")
        self.assertEqual(items[0]["hardware"]["gpus"][0]["model"], "NVIDIA GB10")
        self.assertTrue(items[0]["eligible_for_community"])
        self.manager.node_registry.request.assert_awaited_once_with(
            "worker-1", "GET", "/api/agent/stats", timeout=5,
        )

    async def test_unknown_endpoint_hardware_stays_local_only(self):
        self.service.store.set_setting("community_consent", True)

        self.service._record_response(
            "external", "org/model", "vllm", {}, time.monotonic() - 0.2,
            {"usage": {"prompt_tokens": 24, "completion_tokens": 20}},
            hardware=self.service._unknown_hardware_snapshot(),
            hardware_verified=False,
        )

        items, _ = self.service.store.benchmarks()
        self.assertEqual(items[0]["hardware"]["hardware_class"], "unknown")
        self.assertFalse(items[0]["eligible_for_community"])
        self.assertEqual(self.service.store.outbox_batch(), [])

    async def test_short_sample_remains_local_and_is_not_queued(self):
        self.service.store.set_setting("community_consent", True)
        self.service._record_response(
            None, "org/model", "llama.cpp", {}, time.monotonic() - 0.1,
            {"usage": {"prompt_tokens": 4, "completion_tokens": 2}},
        )
        items, _ = self.service.store.benchmarks()
        self.assertFalse(items[0]["eligible_for_community"])
        self.assertEqual(
            self.service.store.sync_status()["outbox"]["waiting_for_account"], 0
        )

    async def test_non_stream_without_native_timing_is_local_only_and_keeps_revision(self):
        self.service.store.set_setting("community_consent", True)
        self.service._record_response(
            "dep-1", "org/model", "sglang", {}, time.monotonic() - 0.2,
            {"usage": {"prompt_tokens": 24, "completion_tokens": 20}},
            revision="revision-abc",
        )
        items, _ = self.service.store.benchmarks()
        self.assertFalse(items[0]["eligible_for_community"])
        self.assertIsNone(items[0]["generation_tokens_per_second"])
        self.assertEqual(items[0]["model"]["revision"], "revision-abc")

    async def test_local_artifact_and_private_image_never_enter_outbox(self):
        self.service.store.set_setting("community_consent", True)
        self.service._record_response(
            None, "models/customer.gguf", "llama.cpp",
            {"image": "registry.private/team/runtime:latest", "context_length": 4096},
            time.monotonic() - 0.25,
            {
                "usage": {"prompt_tokens": 32, "completion_tokens": 24},
                "timings": {"predicted_per_second": 96.0},
            },
        )
        items, _ = self.service.store.benchmarks()
        self.assertEqual(items[0]["model"]["repository"], "local-model")
        self.assertNotIn("image", items[0]["configuration"])
        self.assertFalse(items[0]["eligible_for_community"])
        self.assertEqual(self.service.store.outbox_batch(), [])

    async def test_repository_shaped_existing_relative_path_is_redacted(self):
        self.service.store.set_setting("community_consent", True)
        with tempfile.TemporaryDirectory(prefix="private-model-", dir="tests") as directory:
            relative_model = Path(directory).as_posix()
            self.service._record_response(
                None, relative_model, "vllm", {}, time.monotonic() - 0.2,
                {"usage": {"prompt_tokens": 24, "completion_tokens": 20}},
            )
        items, _ = self.service.store.benchmarks()
        self.assertEqual(items[0]["model"]["repository"], "local-model")
        self.assertFalse(items[0]["eligible_for_community"])
        self.assertEqual(self.service.store.outbox_batch(), [])
