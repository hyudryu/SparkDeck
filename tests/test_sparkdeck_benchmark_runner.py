import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock, patch

import httpx

from sparkdeck.benchmark_runner import (
    CSV_COLUMNS,
    BenchmarkRunnerError,
    BenchmarkRunnerService,
    _flatten_report,
    _write_csv,
)


def _report_payload(**overrides):
    def metric(mean, std=0.5):
        return {"mean": mean, "std": std, "values": [mean - std, mean + std]}

    payload = {
        "version": "0.1.2",
        "timestamp": "2026-08-28T10:00:00Z",
        "latency_mode": "api",
        "latency_ms": 12.5,
        "model": "unsloth/Qwen3-4B-GGUF",
        "prefix_caching_enabled": False,
        "max_concurrency": 2,
        "benchmarks": [
            {
                "concurrency": 1,
                "context_size": 0,
                "prompt_size": 2048,
                "response_size": 128,
                "is_context_prefill_phase": False,
                "pp_throughput": metric(1800.0),
                "pp_req_throughput": metric(1800.0),
                "tg_throughput": metric(42.5),
                "tg_req_throughput": metric(42.5),
                "peak_throughput": metric(45.0),
                "peak_req_throughput": metric(45.0),
                "ttfr": metric(480.0),
                "est_ppt": metric(1100.0),
                "e2e_ttft": metric(505.0),
            },
            {
                "concurrency": 2,
                "context_size": 0,
                "prompt_size": 2048,
                "response_size": 128,
                "is_context_prefill_phase": False,
                "pp_throughput": metric(3200.0),
                "pp_req_throughput": metric(1600.0),
                "tg_throughput": metric(60.0),
                "tg_req_throughput": metric(30.0),
                "peak_throughput": metric(63.0),
                "peak_req_throughput": metric(31.5),
                "ttfr": metric(700.0),
                "est_ppt": metric(640.0),
                "e2e_ttft": metric(720.0),
            },
        ],
    }
    payload.update(overrides)
    return payload


def _target():
    return {
        "id": "unsloth/Qwen3-4B-GGUF", "label": "Qwen3-4B-GGUF",
        "runtime": "llama.cpp", "deployment_id": None,
        "model": "unsloth/Qwen3-4B-GGUF", "quantization": "Q4_K_M",
        "base_url": "http://127.0.0.1:8080",
    }


def _installed():
    return {
        "installed": True, "version": "0.1.2",
        "launch_mode": "python_module", "path_on_host": False,
    }


class FakeManager:
    def __init__(self, llama_running=False, containers=None, primary_node_id=None):
        self.settings = {"llama_server_host": "127.0.0.1"}
        self._llama_model = "unsloth/Qwen3-4B-GGUF" if llama_running else None
        self._llama_port = 8080 if llama_running else 0
        self._running = llama_running
        self.list_containers = AsyncMock(return_value=containers or [])
        self._primary_node_id = primary_node_id

    def _llama_running(self):
        return self._running

    def _unsloth_variant(self, model):
        return "Q4_K_M" if model.upper().endswith("GGUF") else ""

    @staticmethod
    def _upstream_model_id(container, requested_model):
        served = container.get("served_models") or []
        if requested_model in served:
            return requested_model
        return served[0] if served else requested_model

    def _cluster_primary_member(self, deployment_id):
        if self._primary_node_id is None:
            raise LookupError("cluster deployment not found")
        return {"id": deployment_id}, {"node_id": self._primary_node_id}


class FakeSparkdeck:
    def __init__(self, stored=None):
        self.store = Mock()
        self.store.deployment = Mock(return_value=stored)
        self.models = AsyncMock(return_value={"data": []})

    def _get_credential(self, deployment_id, credential_ref):
        return "secret-key" if credential_ref == "cred-1" else None


class FakeFinishedProcess:
    def __init__(self, returncode=0):
        self.returncode = returncode

    def kill(self):
        raise AssertionError("a finished process should never be killed")

    async def wait(self):
        return self.returncode


class FakeLiveProcess:
    def __init__(self):
        self.returncode = None
        self._done = asyncio.Event()

    def terminate(self):
        self.returncode = -15
        self._done.set()

    def kill(self):
        self.returncode = -9
        self._done.set()

    async def wait(self):
        await self._done.wait()
        return self.returncode


async def _wait_terminal(service, run_id):
    for _ in range(200):
        if service.get_run(run_id)["status"] not in ("pending", "running"):
            return
        await asyncio.sleep(0.01)


class ValidateConfigTests(unittest.TestCase):
    def setUp(self):
        self.service = BenchmarkRunnerService.__new__(BenchmarkRunnerService)

    def test_defaults_fill_missing_fields(self):
        config = self.service._validate_config({"model_id": "m"})
        self.assertEqual(config["model_id"], "m")
        self.assertEqual(config["prompt_sizes"], [2048])
        self.assertEqual(config["response_sizes"], [128])
        self.assertEqual(config["concurrency_levels"], [1, 2, 5, 10])
        self.assertEqual(
            config["context_depths"],
            [0, 4096, 8192, 16384, 32768, 65535, 100000],
        )
        self.assertEqual(config["runs"], 3)
        self.assertEqual(config["warmup_runs"], 1)
        self.assertTrue(config["enable_prefix_caching"])
        self.assertFalse(config["exact_tg"])

    def test_requires_model_id(self):
        with self.assertRaises(BenchmarkRunnerError):
            self.service._validate_config({"prompt_sizes": [512]})

    def test_rejects_empty_and_non_integer_lists(self):
        for body in (
            {"model_id": "m", "prompt_sizes": []},
            {"model_id": "m", "prompt_sizes": ["512"]},
            {"model_id": "m", "concurrency_levels": [True]},
            {"model_id": "m", "runs": "3"},
            {"model_id": "m", "runs": 99},
        ):
            with self.assertRaises(BenchmarkRunnerError, msg=str(body)):
                self.service._validate_config(body)

    def test_requires_at_least_one_measured_run(self):
        with self.assertRaises(BenchmarkRunnerError):
            self.service._validate_config({"model_id": "m", "runs": 0})
        config = self.service._validate_config(
            {"model_id": "m", "runs": 1, "warmup_runs": 0}
        )
        self.assertEqual(config["runs"], 1)
        self.assertEqual(config["warmup_runs"], 0)

    def test_rejects_more_than_one_warmup_run(self):
        with self.assertRaises(BenchmarkRunnerError):
            self.service._validate_config({"model_id": "m", "warmup_runs": 2})

    def test_requires_boolean_exact_tg(self):
        for bad in ("false", "true", 1):
            with self.assertRaises(BenchmarkRunnerError, msg=str(bad)):
                self.service._validate_config({"model_id": "m", "exact_tg": bad})
        self.assertFalse(
            self.service._validate_config({"model_id": "m"})["exact_tg"]
        )
        self.assertTrue(
            self.service._validate_config({"model_id": "m", "exact_tg": True})["exact_tg"]
        )

    def test_requires_boolean_enable_prefix_caching(self):
        for bad in ("false", "true", 1):
            with self.assertRaises(BenchmarkRunnerError, msg=str(bad)):
                self.service._validate_config({
                    "model_id": "m", "enable_prefix_caching": bad,
                })
        self.assertTrue(
            self.service._validate_config({"model_id": "m"})["enable_prefix_caching"]
        )
        self.assertFalse(self.service._validate_config({
            "model_id": "m", "enable_prefix_caching": False,
        })["enable_prefix_caching"])

    def test_default_argv_matches_benchmark_sweep(self):
        config = self.service._validate_config({"model_id": "m"})
        with patch.object(self.service, "_argv_prefix", return_value=["llama-benchy"]):
            argv = self.service._build_argv(
                {"config": config},
                {"base_url": "http://localhost:8000/v1", "model": "model-name"},
                Path("run"),
                {},
            )

        self.assertEqual(argv[:5], [
            "llama-benchy", "--base-url", "http://localhost:8000/v1",
            "--model", "model-name",
        ])
        self.assertEqual(argv[argv.index("--depth") + 1:argv.index("--runs")], [
            "0", "4096", "8192", "16384", "32768", "65535", "100000",
        ])
        self.assertEqual(
            argv[argv.index("--concurrency") + 1:argv.index("--depth")],
            ["1", "2", "5", "10"],
        )
        self.assertIn("--enable-prefix-caching", argv)
        self.assertNotIn("--warmup-runs", argv)

    def test_zero_warmups_uses_supported_no_warmup_flag(self):
        config = self.service._validate_config({"model_id": "m", "warmup_runs": 0})
        with patch.object(self.service, "_argv_prefix", return_value=["llama-benchy"]):
            argv = self.service._build_argv(
                {"config": config},
                {"base_url": "http://localhost:8000/v1", "model": "model-name"},
                Path("run"),
                {},
            )
        self.assertIn("--no-warmup", argv)

    def test_rejects_explosive_shape_combinations(self):
        body = {
            "model_id": "m",
            "prompt_sizes": [128, 256, 512, 1024, 2048, 4096, 8192],
            "response_sizes": [64, 128, 256, 512, 1024, 2048, 4096, 8192],
            "concurrency_levels": [1, 2, 4, 8, 16, 32, 64, 128],
        }
        with self.assertRaises(BenchmarkRunnerError):
            self.service._validate_config(body)

    def test_rejects_oversized_lists(self):
        body = {
            "model_id": "m",
            "prompt_sizes": [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        }
        with self.assertRaises(BenchmarkRunnerError):
            self.service._validate_config(body)


class ReportFlatteningTests(unittest.TestCase):
    def _run(self, results):
        return {
            "id": "run-1", "started_at": "2026-08-28T10:00:00+00:00",
            "model": "unsloth/Qwen3-4B-GGUF", "model_id": "unsloth/Qwen3-4B-GGUF",
            "quantization": "Q4_K_M", "runtime": "llama.cpp",
            "base_url": "http://127.0.0.1:8080", "benchy_version": "0.1.2",
            "config": {
                "runs": 3, "warmup_runs": 1, "exact_tg": False,
                "enable_prefix_caching": True,
            },
            "report": {
                "latency_mode": "api", "latency_ms": 12.5,
                "prefix_caching_enabled": False,
            },
            "results": results,
        }

    def test_rows_keep_run_metrics_and_metadata(self):
        results = _flatten_report("run-1", _report_payload())
        self.assertEqual(len(results), 2)
        with TemporaryDirectory() as temp:
            csv_path = Path(temp) / "results.csv"
            _write_csv(csv_path, self._run(results))
            lines = csv_path.read_text(encoding="utf-8").strip().splitlines()

        header = lines[0].split(",")
        self.assertEqual(header, CSV_COLUMNS)
        self.assertEqual(len(lines), 3)
        first = dict(zip(header, lines[1].split(",")))
        self.assertEqual(first["model"], "unsloth/Qwen3-4B-GGUF")
        self.assertEqual(first["quantization"], "Q4_K_M")
        self.assertEqual(first["prompt_size"], "2048")
        self.assertEqual(first["concurrency"], "1")
        self.assertEqual(first["prefix_caching_enabled"], "False")
        self.assertEqual(first["tg_tokens_per_second"], "42.5")
        self.assertEqual(first["pp_tokens_per_second"], "1800.0")
        second = dict(zip(header, lines[2].split(",")))
        self.assertEqual(second["concurrency"], "2")
        self.assertEqual(second["tg_tokens_per_second"], "60.0")
        self.assertEqual(second["tg_tokens_per_second_request"], "30.0")

    def test_csv_falls_back_to_requested_prefix_caching_mode(self):
        results = _flatten_report("run-1", _report_payload())
        run = self._run(results)
        run["report"].pop("prefix_caching_enabled")
        with TemporaryDirectory() as temp:
            csv_path = Path(temp) / "results.csv"
            _write_csv(csv_path, run)
            lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        first = dict(zip(lines[0].split(","), lines[1].split(",")))
        self.assertEqual(first["prefix_caching_enabled"], "True")

    def test_missing_metrics_render_as_empty_cells(self):
        report = _report_payload()
        report["benchmarks"][0]["tg_throughput"] = None
        results = _flatten_report("run-1", report)
        self.assertIsNone(results[0]["tg_tokens_per_second"])
        with TemporaryDirectory() as temp:
            csv_path = Path(temp) / "results.csv"
            _write_csv(csv_path, self._run(results))
            row = csv_path.read_text(encoding="utf-8").strip().splitlines()[1]
            cells = row.split(",")
            self.assertEqual(cells[CSV_COLUMNS.index("tg_tokens_per_second")], "")


class ServedModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_llama_model_is_listed_with_quantization(self):
        service = BenchmarkRunnerService(FakeManager(llama_running=True), FakeSparkdeck(), Path("data-unused"))
        models = await service.served_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["id"], "unsloth/Qwen3-4B-GGUF")
        self.assertEqual(models[0]["quantization"], "Q4_K_M")
        self.assertEqual(models[0]["base_url"], "http://127.0.0.1:8080")

    async def test_external_deployment_uses_private_base_url(self):
        sparkdeck = FakeSparkdeck(stored={
            "id": "dep-1", "kind": "external", "_base_url": "http://10.0.0.9:8000/",
        })
        sparkdeck.models = AsyncMock(return_value={"data": [{
            "id": "org/model", "runtime": "vllm", "deployment_id": "dep-1",
            "model": {"repository": "org/model", "quantization": "FP8"},
        }]})
        service = BenchmarkRunnerService(FakeManager(), sparkdeck, Path("data-unused"))
        models = await service.served_models()
        self.assertEqual(models[0]["base_url"], "http://10.0.0.9:8000")
        self.assertEqual(models[0]["model"], "org/model")
        self.assertEqual(models[0]["quantization"], "FP8")

    async def test_keyed_deployment_benchmarks_through_controller_proxy(self):
        sparkdeck = FakeSparkdeck(stored={
            "id": "dep-1", "kind": "external",
            "_base_url": "http://10.0.0.9:8000", "_credential_ref": "cred-1",
        })
        sparkdeck.models = AsyncMock(return_value={"data": [{
            "id": "org/model", "runtime": "vllm", "deployment_id": "dep-1",
            "model": {"repository": "org/model", "quantization": None},
        }]})
        service = BenchmarkRunnerService(FakeManager(), sparkdeck, Path("data-unused"))
        models = await service.served_models()
        self.assertEqual(models[0]["base_url"], "http://127.0.0.1:7878")
        # The proxy resolves the alias to the upstream model server-side.
        self.assertEqual(models[0]["model"], "org/model")

    async def test_unkeyed_deployment_keeps_direct_endpoint(self):
        sparkdeck = FakeSparkdeck(stored={
            "id": "dep-1", "kind": "external", "_base_url": "http://10.0.0.9:8000",
        })
        sparkdeck.models = AsyncMock(return_value={"data": [{
            "id": "org/model", "runtime": "vllm", "deployment_id": "dep-1",
            "model": {"repository": "org/model", "quantization": None},
        }]})
        service = BenchmarkRunnerService(FakeManager(), sparkdeck, Path("data-unused"))
        models = await service.served_models()
        self.assertEqual(models[0]["base_url"], "http://10.0.0.9:8000")

    async def test_managed_deployment_uses_served_model_id(self):
        sparkdeck = FakeSparkdeck(stored={
            "id": "dep-2", "kind": "managed", "_base_url": None,
        })
        sparkdeck.models = AsyncMock(return_value={"data": [{
            "id": "alias", "runtime": "vllm", "deployment_id": "dep-2",
            "port": 8123, "container_name": "c1",
            "model": {"repository": "org/model", "quantization": None},
        }]})
        manager = FakeManager(containers=[
            {"name": "c1", "served_models": ["served-model-name"]},
        ])
        service = BenchmarkRunnerService(manager, sparkdeck, Path("data-unused"))
        models = await service.served_models()
        self.assertEqual(models[0]["base_url"], "http://127.0.0.1:8123")
        self.assertEqual(models[0]["model"], "served-model-name")

    async def test_remote_managed_deployment_uses_controller_proxy(self):
        sparkdeck = FakeSparkdeck(stored={
            "id": "dep-2", "kind": "managed", "_base_url": None,
            "settings": {"manager_deployment_id": "md-1"},
        })
        sparkdeck.models = AsyncMock(return_value={"data": [{
            "id": "alias", "runtime": "vllm", "deployment_id": "dep-2",
            "port": 8123, "container_name": "c1",
            "model": {"repository": "org/model", "quantization": None},
        }]})
        remote = BenchmarkRunnerService(FakeManager(primary_node_id="node-2"), sparkdeck, Path("data-unused"))
        remote_models = await remote.served_models()
        self.assertEqual(remote_models[0]["base_url"], "http://127.0.0.1:7878")
        self.assertEqual(remote_models[0]["model"], "alias")
        local = BenchmarkRunnerService(FakeManager(primary_node_id="local"), sparkdeck, Path("data-unused"))
        self.assertEqual(len(await local.served_models()), 1)

    async def test_discovered_container_is_benchmarkable(self):
        sparkdeck = FakeSparkdeck(stored=None)
        sparkdeck.models = AsyncMock(return_value={"data": [{
            "id": "container:legacy-vllm", "runtime": "vllm",
            "deployment_id": "container:legacy-vllm", "port": 8222,
            "container_name": "legacy-vllm",
            "model": {"repository": "org/legacy", "quantization": None},
        }]})
        manager = FakeManager(containers=[
            {"name": "legacy-vllm", "served_models": ["legacy-served"]},
        ])
        service = BenchmarkRunnerService(manager, sparkdeck, Path("data-unused"))
        models = await service.served_models()
        self.assertEqual(models[0]["base_url"], "http://127.0.0.1:8222")
        self.assertEqual(models[0]["model"], "legacy-served")

    async def test_registered_deployment_wins_over_native_llama_alias(self):
        sparkdeck = FakeSparkdeck(stored={
            "id": "dep-1", "kind": "external", "_base_url": "http://10.0.0.9:8000",
        })
        sparkdeck.models = AsyncMock(return_value={"data": [{
            "id": "unsloth/Qwen3-4B-GGUF", "runtime": "vllm", "deployment_id": "dep-1",
            "model": {"repository": "unsloth/Qwen3-4B-GGUF", "quantization": None},
        }]})
        service = BenchmarkRunnerService(FakeManager(llama_running=True), sparkdeck, Path("data-unused"))
        models = await service.served_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["base_url"], "http://10.0.0.9:8000")

    async def test_managed_deployment_uses_local_port(self):
        sparkdeck = FakeSparkdeck(stored={"id": "dep-2", "kind": "managed", "_base_url": None})
        sparkdeck.models = AsyncMock(return_value={"data": [{
            "id": "alias", "runtime": "vllm", "deployment_id": "dep-2", "port": 8123,
            "model": {"repository": "org/model", "quantization": None},
        }]})
        service = BenchmarkRunnerService(FakeManager(), sparkdeck, Path("data-unused"))
        models = await service.served_models()
        self.assertEqual(models[0]["base_url"], "http://127.0.0.1:8123")

    async def test_unreachable_deployment_is_skipped(self):
        sparkdeck = FakeSparkdeck(stored={"id": "dep-3", "kind": "external", "_base_url": None})
        sparkdeck.models = AsyncMock(return_value={"data": [{
            "id": "alias", "runtime": "vllm", "deployment_id": "dep-3",
            "model": {"repository": "org/model", "quantization": None},
        }]})
        service = BenchmarkRunnerService(FakeManager(), sparkdeck, Path("data-unused"))
        self.assertEqual(await service.served_models(), [])


class RunLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def _service(self):
        return BenchmarkRunnerService(FakeManager(llama_running=True), FakeSparkdeck(), Path(self.temp.name))

    def test_argv_measures_latency_through_targeted_generation(self):
        service = self._service()
        run = {
            "config": self._validated_config(),
        }
        argv = service._build_argv(
            run, _target(), Path(self.temp.name), _installed(),
        )

        latency_index = argv.index("--latency-mode")
        self.assertEqual(argv[latency_index + 1], "generation")

    def _validated_config(self):
        return self._service()._validate_config({
            "model_id": "unsloth/Qwen3-4B-GGUF",
        })

    async def test_completed_run_parses_report_and_writes_csv(self):
        service = self._service()

        async def fake_spawn(run):
            run.pop("_argv")
            run_dir = Path(run["_run_dir"])
            (run_dir / "report.json").write_text(
                json.dumps(_report_payload()), encoding="utf-8")
            (run_dir / "progress.jsonl").write_text(
                json.dumps({"type": "request_start", "prompt_size": 2048,
                            "response_size": 128, "context_size": 0,
                            "concurrency": 1}) + "\n" +
                json.dumps({"type": "request_end", "total_tokens": 128,
                            "prompt_tokens": 2048, "decode_seconds": 3.0}) + "\n",
                encoding="utf-8")
            process = FakeFinishedProcess(returncode=0)
            service._processes[run["id"]] = process
            return process

        with patch.object(service, "detect", AsyncMock(return_value=_installed())), \
                patch.object(service, "served_models", AsyncMock(return_value=[_target()])), \
                patch.object(service, "_spawn", AsyncMock(side_effect=fake_spawn)):
            started = await service.start_run({
                "model_id": "unsloth/Qwen3-4B-GGUF",
                "prompt_sizes": [2048],
                "response_sizes": [128],
                "concurrency_levels": [1, 2],
            })
            await _wait_terminal(service, started["id"])

        run = service.get_run(started["id"])
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["quantization"], "Q4_K_M")
        self.assertEqual(run["result_count"], 2)
        self.assertEqual(run["progress"]["requests_done"], 1)
        self.assertEqual(run["progress"]["current"]["prompt_size"], 2048)
        self.assertIsNone(run["error"])
        csv_path = service.runs_dir / run["id"] / "results.csv"
        self.assertTrue(csv_path.is_file())
        self.assertIn(
            "quantization", csv_path.read_text(encoding="utf-8").splitlines()[0],
        )
        # The completed state survives a service reload.
        reloaded = BenchmarkRunnerService(FakeManager(), FakeSparkdeck(), Path(self.temp.name))
        self.assertEqual(reloaded.get_run(run["id"])["status"], "completed")

    async def test_failed_run_records_output_tail(self):
        service = self._service()

        async def fake_spawn(run):
            run.pop("_argv")
            run_dir = Path(run["_run_dir"])
            (run_dir / "output.log").write_text(
                "Traceback: endpoint unreachable", encoding="utf-8")
            process = FakeFinishedProcess(returncode=2)
            service._processes[run["id"]] = process
            return process

        with patch.object(service, "detect", AsyncMock(return_value=_installed())), \
                patch.object(service, "served_models", AsyncMock(return_value=[_target()])), \
                patch.object(service, "_spawn", AsyncMock(side_effect=fake_spawn)):
            started = await service.start_run({"model_id": "unsloth/Qwen3-4B-GGUF"})
            await _wait_terminal(service, started["id"])

        run = service.get_run(started["id"])
        self.assertEqual(run["status"], "failed")
        self.assertIn("exited with code 2", run["error"])
        self.assertIn("endpoint unreachable", run["error"])

    async def test_second_run_is_refused_while_one_is_active(self):
        service = self._service()

        async def fake_spawn(run):
            run.pop("_argv")
            process = FakeLiveProcess()
            service._processes[run["id"]] = process
            return process

        with patch.object(service, "detect", AsyncMock(return_value=_installed())), \
                patch.object(service, "served_models", AsyncMock(return_value=[_target()])), \
                patch.object(service, "_spawn", AsyncMock(side_effect=fake_spawn)):
            first = await service.start_run({"model_id": "unsloth/Qwen3-4B-GGUF"})
            with self.assertRaises(BenchmarkRunnerError):
                await service.start_run({"model_id": "unsloth/Qwen3-4B-GGUF"})
            self.assertEqual(service.active_run()["id"], first["id"])

            await service.cancel_run(first["id"])
            await _wait_terminal(service, first["id"])
            cancelled = service.get_run(first["id"])

        self.assertEqual(cancelled["status"], "cancelled")
        service.delete_run(first["id"])
        self.assertEqual(service.list_runs(), [])

    async def test_concurrent_starts_start_only_one_run(self):
        service = self._service()

        async def fake_spawn(run):
            run.pop("_argv")
            process = FakeLiveProcess()
            service._processes[run["id"]] = process
            return process

        with patch.object(service, "detect", AsyncMock(return_value=_installed())), \
                patch.object(service, "served_models", AsyncMock(return_value=[_target()])), \
                patch.object(service, "_spawn", AsyncMock(side_effect=fake_spawn)):
            results = await asyncio.gather(
                service.start_run({"model_id": "unsloth/Qwen3-4B-GGUF"}),
                service.start_run({"model_id": "unsloth/Qwen3-4B-GGUF"}),
                return_exceptions=True,
            )
            started = [item for item in results if not isinstance(item, Exception)]
            errors = [item for item in results if isinstance(item, BenchmarkRunnerError)]
            self.assertEqual(len(started), 1)
            self.assertEqual(len(errors), 1)
            self.assertIn("already in progress", str(errors[0]))

            await service.cancel_run(started[0]["id"])
            await _wait_terminal(service, started[0]["id"])
            service.delete_run(started[0]["id"])

    async def test_monitor_cancellation_terminates_child(self):
        service = self._service()
        run_id = "20260101-000000-cancel"
        run_dir = Path(self.temp.name) / "benchmark-runner" / "runs" / run_id
        run_dir.mkdir(parents=True)
        run = {
            "id": run_id, "status": "running",
            "created_at": "2026-01-01T00:00:00+00:00",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": None, "duration_seconds": None,
            "config": {"runs": 3, "warmup_runs": 1, "exact_tg": False},
            "error": None, "progress": {"requests_done": 0, "requests_failed": 0},
            "results": [], "result_count": 0, "csv_filename": None, "report": None,
        }
        process = FakeLiveProcess()
        service._processes[run_id] = process
        task = asyncio.create_task(service._monitor_run(run, run_dir))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertIsNotNone(process.returncode, "child must not survive cancellation")
        self.assertNotIn(run_id, service._processes)
        self.assertEqual(run["status"], "failed")
        self.assertIn("interrupted", run["error"])
        persisted = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "failed")

    async def test_install_cancellation_terminates_pip(self):
        class HangingPipProcess:
            def __init__(self):
                self.returncode = None
                self._done = asyncio.Event()
                self.terminated = False

            def terminate(self):
                self.terminated = True
                self.returncode = -15
                self._done.set()

            def kill(self):
                self.returncode = -9
                self._done.set()

            async def wait(self):
                await self._done.wait()
                return self.returncode

            async def communicate(self):
                await self._done.wait()
                return b"", None

        pip_process = HangingPipProcess()
        with patch("sparkdeck.benchmark_runner.asyncio.create_subprocess_exec",
                   AsyncMock(return_value=pip_process)):
            service = BenchmarkRunnerService(FakeManager(), FakeSparkdeck(), Path("data-unused"))
            task = asyncio.create_task(service.install())
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(pip_process.terminated, "pip must not survive cancellation")
        self.assertIsNotNone(pip_process.returncode)

    async def test_start_run_rolls_back_when_state_persist_fails(self):
        service = self._service()
        with patch.object(service, "detect", AsyncMock(return_value=_installed())), \
                patch.object(service, "served_models", AsyncMock(return_value=[_target()])), \
                patch.object(service, "_save_state", Mock(side_effect=OSError("disk full"))):
            with self.assertRaises(BenchmarkRunnerError) as caught:
                await service.start_run({"model_id": "unsloth/Qwen3-4B-GGUF"})
        self.assertIn("could not record benchmark state", str(caught.exception))
        self.assertEqual(service.runs, {})
        self.assertIsNone(service.active_run())
        # No half-created run directory survives the rollback.
        self.assertEqual(list(service.runs_dir.iterdir()), [])

    async def test_delete_failure_keeps_run_recorded(self):
        service = self._service()
        run_id = "20260101-000000-stuck"
        run_dir = Path(self.temp.name) / "benchmark-runner" / "runs" / run_id
        run_dir.mkdir(parents=True)
        service.runs[run_id] = {
            "id": run_id, "status": "completed",
            "created_at": "2026-01-01T00:00:00+00:00",
            "config": {"runs": 1, "warmup_runs": 0, "exact_tg": False},
        }
        with patch("sparkdeck.benchmark_runner.shutil.rmtree", side_effect=OSError("file busy")):
            with self.assertRaises(OSError):
                service.delete_run(run_id)
        self.assertIn(run_id, service.runs)
        self.assertEqual(service.list_runs()[0]["id"], run_id)

    async def test_unknown_model_is_rejected(self):
        service = self._service()
        with patch.object(service, "detect", AsyncMock(return_value=_installed())), \
                patch.object(service, "served_models", AsyncMock(return_value=[])):
            with self.assertRaises(BenchmarkRunnerError):
                await service.start_run({"model_id": "ghost/model"})

    async def test_uninstalled_tool_is_rejected(self):
        service = self._service()
        with patch.object(service, "detect", AsyncMock(return_value={
                "installed": False, "version": None,
                "launch_mode": None, "path_on_host": False})):
            with self.assertRaises(BenchmarkRunnerError):
                await service.start_run({"model_id": "unsloth/Qwen3-4B-GGUF"})

    async def test_orphaned_active_runs_fail_on_restart(self):
        BenchmarkRunnerService(FakeManager(), FakeSparkdeck(), Path(self.temp.name))
        run_id = "20260101-000000-abcdef"
        run_dir = Path(self.temp.name) / "benchmark-runner" / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "state.json").write_text(json.dumps({
            "id": run_id, "status": "running", "created_at": "2026-01-01T00:00:00+00:00",
        }), encoding="utf-8")
        reloaded = BenchmarkRunnerService(FakeManager(), FakeSparkdeck(), Path(self.temp.name))
        run = reloaded.get_run(run_id)
        self.assertEqual(run["status"], "failed")
        self.assertIn("restart", run["error"])


class InstallTests(unittest.IsolatedAsyncioTestCase):
    async def test_install_failure_raises_with_output_tail(self):
        class FailingProcess:
            returncode = 1

            def kill(self):
                pass

            async def wait(self):
                return self.returncode

            async def communicate(self):
                return b"ERROR: no matching distribution", None

        with patch("sparkdeck.benchmark_runner.asyncio.create_subprocess_exec",
                   AsyncMock(return_value=FailingProcess())):
            service = BenchmarkRunnerService(FakeManager(), FakeSparkdeck(), Path("data-unused"))
            with self.assertRaises(BenchmarkRunnerError) as caught:
                await service.install()
        self.assertIn("no matching distribution", str(caught.exception))

    async def test_install_success_refreshes_detection(self):
        class OkProcess:
            returncode = 0

            def kill(self):
                pass

            async def wait(self):
                return self.returncode

            async def communicate(self):
                return b"Successfully installed llama-benchy-0.1.2", None

        with patch("sparkdeck.benchmark_runner.asyncio.create_subprocess_exec",
                   AsyncMock(return_value=OkProcess())):
            service = BenchmarkRunnerService(FakeManager(), FakeSparkdeck(), Path("data-unused"))
            with patch.object(service, "_probe_version", AsyncMock(return_value="0.1.2")):
                status = await service.install()
        self.assertTrue(status["installed"])
        self.assertEqual(status["version"], "0.1.2")


class BenchmarkRunnerApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        with patch("docker.from_env", return_value=Mock()):
            import server
        self.server = server
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_status_reports_detection(self):
        detect = AsyncMock(return_value={
            "installed": True, "version": "0.1.2",
            "launch_mode": "path", "path_on_host": True,
        })
        with patch.object(self.server.benchmark_runner, "detect", detect), \
                patch.object(self.server.benchmark_runner, "active_run", Mock(return_value=None)):
            response = await self.client.get("/api/v1/benchmark-runner/status")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["installed"])
        self.assertEqual(body["version"], "0.1.2")
        self.assertIsNone(body["active_run_id"])

    async def test_models_list_delegates_to_service(self):
        served = AsyncMock(return_value=[_target()])
        with patch.object(self.server.benchmark_runner, "served_models", served):
            response = await self.client.get("/api/v1/benchmark-runner/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"][0]["id"], "unsloth/Qwen3-4B-GGUF")

    async def test_models_list_strips_internal_credential_fields(self):
        served = AsyncMock(return_value=[{**_target(), "_api_key": "secret-key"}])
        with patch.object(self.server.benchmark_runner, "served_models", served):
            response = await self.client.get("/api/v1/benchmark-runner/models")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("secret-key", response.text)
        self.assertNotIn("_api_key", response.text)

    async def test_local_history_models_returns_consolidated_rows(self):
        history = AsyncMock(return_value=[{
            "id": "sample-new", "model": {"repository": "org/model"},
            "sample_count": 12,
        }])
        with patch.object(
            self.server.sparkdeck, "benchmark_history_models", history,
        ):
            response = await self.client.get("/api/v1/benchmark-history/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"][0]["sample_count"], 12)

    async def test_delete_local_history_model_deletes_all_samples(self):
        deleted = AsyncMock(return_value=12)
        with patch.object(self.server.sparkdeck, "delete_benchmark_model", deleted):
            response = await self.client.delete(
                "/api/v1/benchmark-history/models/org%2Fmodel"
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["deleted"], 12)
        deleted.assert_awaited_once_with("org/model")

    async def test_start_run_passes_body_through(self):
        started = AsyncMock(return_value={"id": "run-1", "status": "running"})
        with patch.object(self.server.benchmark_runner, "start_run", started):
            response = await self.client.post("/api/v1/benchmark-runner/runs", json={
                "model_id": "m", "prompt_sizes": [512], "concurrency_levels": [1, 2],
            })
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["id"], "run-1")

    async def test_start_run_maps_validation_errors_to_400(self):
        with patch.object(
            self.server.benchmark_runner, "start_run",
            AsyncMock(side_effect=BenchmarkRunnerError("model_id is required")),
        ):
            response = await self.client.post("/api/v1/benchmark-runner/runs", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("model_id", response.json()["detail"])

    async def test_missing_run_returns_404(self):
        missing = LookupError("benchmark run not found")
        with patch.object(self.server.benchmark_runner, "get_run", Mock(side_effect=missing)), \
                patch.object(self.server.benchmark_runner, "csv_path", Mock(side_effect=missing)), \
                patch.object(self.server.benchmark_runner, "delete_run", Mock(side_effect=missing)):
            detail = await self.client.get("/api/v1/benchmark-runner/runs/missing")
            csv = await self.client.get("/api/v1/benchmark-runner/runs/missing/csv")
            deleted = await self.client.delete("/api/v1/benchmark-runner/runs/missing")
        for response in (detail, csv, deleted):
            self.assertEqual(response.status_code, 404)

    async def test_runs_list_shape(self):
        with patch.object(self.server.benchmark_runner, "list_runs", Mock(return_value=[])):
            response = await self.client.get("/api/v1/benchmark-runner/runs")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"items": []})

    async def test_csv_download_serves_file(self):
        with TemporaryDirectory() as temp:
            csv_file = Path(temp) / "results.csv"
            csv_file.write_text("run_id,model\n1,m\n", encoding="utf-8")
            with patch.object(self.server.benchmark_runner, "csv_path", Mock(return_value=csv_file)):
                response = await self.client.get("/api/v1/benchmark-runner/runs/run-1/csv")
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/csv", response.headers["content-type"])
            self.assertIn("benchmark-run-run-1.csv", response.headers["content-disposition"])


if __name__ == "__main__":
    unittest.main()
