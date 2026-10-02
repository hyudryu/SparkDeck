import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sparkdeck.benchmark_hardware import benchmark_hardware
from sparkdeck.models import BenchmarkSample, ModelIdentity, RuntimeKind
from sparkdeck.service import SparkDeckService, _public_community_aggregates
from sparkdeck.storage import SparkDeckStore


def hardware(model, memory=98304, architecture="x86_64"):
    return {"architecture": architecture, "gpus": [{"model": model, "memory_mib": memory}]}


def sample(sample_id, model, speed):
    return BenchmarkSample(
        id=sample_id, created_at="2026-10-01T00:00:00+00:00", deployment_id="dep",
        model=ModelIdentity("org/model", quantization="NVFP4"), runtime=RuntimeKind.VLLM,
        runtime_version=None, hardware=model, configuration={"context_length": 4096},
        input_tokens=400, output_tokens=200, latency_ms=1000, ttft_ms=10,
        generation_tokens_per_second=speed, prompt_tokens_per_second=1000,
        cold_start=False, eligible_for_community=True,
    )


class HardwareProfileTests(unittest.TestCase):
    def test_gpu_names_override_class_and_driver_memory_does_not_split_cohort(self):
        ws = hardware("NVIDIA RTX PRO 6000 Blackwell Workstation Edition")
        ws["hardware_class"] = "dgx-spark"
        profile = benchmark_hardware(ws)
        self.assertEqual(profile["hardware"]["hardware_class"], "workstation")
        self.assertEqual(profile["hardware_key"], benchmark_hardware(
            hardware(ws["gpus"][0]["model"], 97887, "unknown"))["hardware_key"])
        self.assertNotEqual(profile["hardware_key"], benchmark_hardware(hardware("NVIDIA GB10"))["hardware_key"])
        self.assertEqual(benchmark_hardware({"hardware_class": "dgx-spark"})["hardware_key"], "unknown")

    def test_spark_aliases_and_multi_gpu_composition(self):
        spark = benchmark_hardware(hardware("NVIDIA GB10"))
        self.assertEqual(spark["hardware_key"], benchmark_hardware(hardware("DGX Spark"))["hardware_key"])
        dual = benchmark_hardware({"gpus": hardware("NVIDIA GB10")["gpus"] * 2})
        self.assertNotEqual(spark["hardware_key"], dual["hardware_key"])
        self.assertEqual(dual["hardware"]["gpu_count"], 2)

    def test_legacy_hosted_rows_stay_unknown_and_distinct_hardware_is_not_duplicate(self):
        row = {"model_id": "org/model", "quantization": "NVFP4", "prompt_tokens_bucket": 400,
               "tensor_parallel_size": 1, "inference_tokens_per_second": 40, "sample_count": 10}
        rows = _public_community_aggregates({"items": [row, {**row, "hardware": hardware("NVIDIA GB10")},
                                                     {**row, "hardware": hardware("NVIDIA RTX PRO 6000 Blackwell")}]})
        self.assertEqual([item["hardware"]["hardware_class"] for item in rows],
                         ["unknown", "dgx-spark", "workstation"])


class HardwareStorageTests(unittest.TestCase):
    def test_local_history_keeps_older_models_cohorts_and_maxima_beyond_community_limit(self):
        with tempfile.TemporaryDirectory() as directory, closing(SparkDeckStore(Path(directory) / "db.sqlite")) as store:
            for index, model_id, gpu, speed in [
                (0, "org/older-model", "NVIDIA GB10", 500),
                (1, "org/model", "NVIDIA RTX PRO 6000 Blackwell", 200),
                (2, "org/model", "NVIDIA GB10", 120),
                (3, "org/model", "NVIDIA GB10", 20),
                (4, "org/model", "NVIDIA GB10", 40),
            ]:
                created_at = f"2026-10-01T00:00:0{index}+00:00"
                point = {"id": f"history-{index}", "created_at": created_at,
                         "model_id": model_id, "context_window_size": 8192 if index == 1 else 4096,
                         "concurrency": 1, "tensor_parallel_size": 2 if index == 1 else 1,
                         "prompt_tokens_per_second": speed * 10,
                         "generation_tokens_per_second": speed, "request_count": 1}
                store.add_coordinated_benchmark(point, replace(
                    sample(str(index), hardware(gpu), speed), created_at=created_at,
                    model=ModelIdentity(model_id, quantization="NVFP4")), queue=False)
            with patch("sparkdeck.storage._COMMUNITY_AGGREGATE_ROW_LIMIT", 2):
                summaries = store.benchmark_model_summaries()
                detail = store.benchmark_model_detail("org/model")
            with self.subTest(view="summaries"):
                self.assertEqual(len(summaries), 3)
                self.assertIn("org/older-model", {row["model_id"] for row in summaries})
                spark = next(row for row in summaries if row["model_id"] == "org/model"
                             and row["hardware"]["hardware_class"] == "dgx-spark")
                self.assertEqual(spark["run_count"], 3)
                self.assertEqual(spark["best_generation_tokens_per_second"], 120)
                self.assertEqual(spark["best_prompt_tokens_per_second"], 1200)
                workstation = next(row for row in summaries
                                   if row["hardware"]["hardware_class"] == "workstation")
                self.assertEqual(workstation["context_windows"], [8192])
                self.assertEqual(workstation["tensor_parallel_sizes"], [2])
            with self.subTest(view="detail"):
                self.assertEqual(len(detail["points"]), 2)
                spark = next(row for row in detail["points"]
                             if row["hardware"]["hardware_class"] == "dgx-spark")
                self.assertEqual(spark["sample_count"], 3)
                self.assertEqual(spark["generation_tokens_per_second"], 60)
                self.assertEqual(spark["prompt_tokens_per_second"], 600)
                workstation = next(row for row in detail["points"]
                                   if row["hardware"]["hardware_class"] == "workstation")
                self.assertEqual(workstation["context_window_size"], 8192)
                self.assertEqual(workstation["tensor_parallel_size"], 2)

    def test_upload_means_and_local_evidence_never_mix_spark_and_workstation(self):
        with tempfile.TemporaryDirectory() as directory, closing(SparkDeckStore(Path(directory) / "db.sqlite")) as store:
            store.set_setting("device_pairing", {"status": "paired"})
            store.set_community_consent(True)
            for index, gpu, speed in [(1, "NVIDIA GB10", 40), (2, "NVIDIA GB10", 60),
                                      (3, "NVIDIA RTX PRO 6000 Blackwell", 200)]:
                store.add_benchmark(sample(str(index), hardware(gpu), speed), queue=True)
            payloads = store.outbox_batch()
            self.assertEqual([item["inference_tokens_per_second"] for item in payloads], [50, 50, 200])
            self.assertEqual(len(store.community_aggregates()), 2)
            history = store.benchmark_history_models()
            self.assertEqual(len(history), 2)
            self.assertEqual(sorted(item["sample_count"] for item in history), [1, 2])
            for index, gpu, speed in [(4, "NVIDIA GB10", 50), (5, "NVIDIA RTX PRO 6000 Blackwell", 200)]:
                point = {"id": f"point-{index}", "created_at": "2026-10-01T00:00:00+00:00",
                         "model_id": "org/model", "context_window_size": 4096, "concurrency": 1,
                         "tensor_parallel_size": 1, "prompt_tokens_per_second": 1000,
                         "generation_tokens_per_second": speed, "request_count": 1}
                store.add_coordinated_benchmark(point, sample(str(index), hardware(gpu), speed), queue=True)
            self.assertEqual(len(store.benchmark_model_summaries()), 2)
            points = store.benchmark_model_detail("org/model")["points"]
            self.assertEqual(sorted(item["generation_tokens_per_second"] for item in points), [50, 200])


class ServingHardwareTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_gpu_entry_invalidates_whole_serving_inventory(self):
        service = SparkDeckService.__new__(SparkDeckService)
        stats = {"architecture": "aarch64", "gpus": [
            {"name": "NVIDIA GB10", "mem_total_mib": 128000}, {},
        ]}
        service.manager = SimpleNamespace(_stats_cache=stats)
        snapshot, verified = await service._managed_hardware_snapshot({"kind": "managed", "settings": {}})
        self.assertFalse(verified)
        self.assertEqual(snapshot["hardware_class"], "unknown")
        self.assertIsNone(snapshot["gpu_count"])

    async def test_remote_architecture_and_all_serving_shards_not_other_replicas(self):
        members = [{"node_id": "spark", "instance_id": 0}, {"node_id": "ws1", "instance_id": 0},
                   {"node_id": "other", "instance_id": 1}]
        cluster = {"mode": "grouped_sharded", "members": members}
        async def stats(node_id, *args, **kwargs):
            return {"architecture": "aarch64" if node_id == "spark" else "x86_64",
                    "gpus": [{"name": "NVIDIA GB10" if node_id == "spark" else "NVIDIA RTX PRO 6000 Blackwell",
                              "mem_total_mib": 98304}]}
        service = SparkDeckService.__new__(SparkDeckService)
        request = AsyncMock(side_effect=stats)
        service.manager = SimpleNamespace(_deployment=lambda _: cluster, node_registry=SimpleNamespace(request=request))
        snapshot, verified = await service._managed_hardware_snapshot(
            {"kind": "managed", "settings": {"manager_deployment_id": "dep"}}, members[0])
        self.assertTrue(verified)
        self.assertEqual(snapshot["hardware_class"], "mixed")
        self.assertEqual(snapshot["architecture"], "mixed")
        self.assertEqual(snapshot["gpu_count"], 2)
        self.assertEqual({call.args[0] for call in request.await_args_list}, {"spark", "ws1"})
