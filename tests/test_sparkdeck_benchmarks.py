import asyncio
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

from manager import Manager
from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import (
    SparkDeckService,
    _COMMUNITY_MAX_RESPONSE_BYTES,
    _chunk_has_output,
    _read_bounded_community_response,
)


class FakeManager:
    _deployment_launch_controls = Manager._deployment_launch_controls
    recipe_deployment_contract = Manager.recipe_deployment_contract
    _cli_option = staticmethod(Manager._cli_option)

    def __init__(self):
        self.http = httpx.AsyncClient()
        self.list_containers = AsyncMock(return_value=[])
        self.deployments = []
        self.community_http_transport = None
        self.community_resolver = lambda host, port, **kwargs: [(
            socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            ("8.8.8.8", port),
        )]

    def _deployment(self, deployment_id):
        return next(
            (item for item in self.deployments if item.get("id") == deployment_id),
            None,
        )


class StreamChunkTests(unittest.TestCase):
    def test_reasoning_alias_marks_the_first_output_token(self):
        self.assertTrue(_chunk_has_output({
            "choices": [{"delta": {"reasoning": "Working through it"}}],
        }))


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

    async def test_coordinated_run_records_real_concurrency_and_strict_upload_dimensions(self):
        self.service.store.add_deployment(Deployment(
            id="dep-series", alias="series", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
            settings={
                "context_length": 16384, "tensor_parallel_size": 2,
                "model_source": "public_repository",
            },
            base_url_set=True,
        ), "http://127.0.0.1:8000")
        self.service.store.set_setting("device_pairing", {"status": "paired"})
        self.service.store.set_community_consent(True)

        point = await self.service.record_benchmark_series_point({
            "deployment_id": "dep-series", "concurrency": 5,
            "request_count": 10, "prompt_tokens": 1000,
            "generation_tokens": 500, "prompt_seconds": 0.5,
            "wall_seconds": 2.0,
        })

        self.assertEqual(point["context_window_size"], 16384)
        self.assertEqual(point["tensor_parallel_size"], 2)
        self.assertEqual(point["prompt_tokens_per_second"], 2000.0)
        self.assertEqual(point["generation_tokens_per_second"], 250.0)
        self.assertEqual(self.service.store.benchmark_model_detail("org/model")["points"][0]["concurrency"], 5)
        self.assertEqual(self.service.store.outbox_batch(), [{
            "model_id": "org/model", "context_window_size": 16384,
            "inference_tokens_per_second": 250.0,
            "concurrency": 5, "tensor_parallel_size": 2,
        }])

    async def test_coordinated_run_groups_private_models_by_distinct_opaque_local_ids(self):
        self.service.store.set_setting("device_pairing", {"status": "paired"})
        self.service.store.set_community_consent(True)
        point_ids = []
        for deployment_id, model in (
            ("private-path", r"C:\private\customer-model.gguf"),
            ("private-url", "https://models.private.example/customer-model"),
        ):
            with self.subTest(model=model):
                self.service.store.add_deployment(Deployment(
                    id=deployment_id, alias=deployment_id,
                    runtime=RuntimeKind.VLLM, kind=DeploymentKind.MANAGED,
                    model=ModelIdentity(model), settings={"context_length": 4096},
                ))

                point = await self.service.record_benchmark_series_point({
                    "deployment_id": deployment_id, "concurrency": 1,
                    "request_count": 2, "prompt_tokens": 100,
                    "generation_tokens": 50, "prompt_seconds": 0.25,
                    "wall_seconds": 1,
                })

                point_ids.append(point["model_id"])
                self.assertRegex(point["model_id"], r"^local-model-[0-9a-f]{16}$")
                self.assertNotIn(model, str(point))

        samples, total = self.service.store.benchmarks()
        self.assertEqual(total, 2)
        self.assertEqual(len(set(point_ids)), 2)
        self.assertTrue(all(
            sample["model"]["repository"] == "local-model"
            and not sample["eligible_for_community"]
            for sample in samples
        ))
        self.assertEqual(self.service.store.outbox_batch(), [])
        summaries = self.service.store.benchmark_model_summaries()
        self.assertEqual({item["model_id"] for item in summaries}, set(point_ids))
        self.assertTrue(all(item["run_count"] == 1 for item in summaries))
        self.assertTrue(all(
            self.service.store.benchmark_model_detail(model_id)["points"][0][
                "sample_count"
            ] == 1
            for model_id in point_ids
        ))
        persisted = str(summaries) + str(samples)
        self.assertNotIn(r"C:\private\customer-model.gguf", persisted)
        self.assertNotIn("models.private.example", persisted)

    async def test_coordinated_external_run_keeps_hardware_unverified(self):
        self.manager._stats_cache = {
            "gpus": [{"name": "NVIDIA GB10", "mem_total_mib": 128000}],
        }
        self.service.store.set_setting("device_pairing", {"status": "paired"})
        self.service.store.set_community_consent(True)
        self.service.store.add_deployment(Deployment(
            id="external-series", alias="external-series",
            runtime=RuntimeKind.VLLM, kind=DeploymentKind.EXTERNAL,
            model=ModelIdentity("org/external-model"),
            settings={"context_length": 4096}, base_url_set=True,
        ), "http://127.0.0.1:8000")

        await self.service.record_benchmark_series_point({
            "deployment_id": "external-series", "concurrency": 1,
            "request_count": 2, "prompt_tokens": 100,
            "generation_tokens": 50, "prompt_seconds": 0.25,
            "wall_seconds": 1,
        })

        samples, total = self.service.store.benchmarks()
        self.assertEqual(total, 1)
        self.assertEqual(samples[0]["hardware"], {
            "hardware_class": "unknown", "gpu_count": None, "gpus": [],
        })
        self.assertFalse(samples[0]["eligible_for_community"])
        self.assertEqual(self.service.store.outbox_batch(), [])

    async def test_coordinated_run_resolves_manager_only_launch_metadata(self):
        manager_deployment = {
            "id": "manager-only", "model": "org/manager-model", "status": "ready",
            "model_source": "public_repository",
            "node_ids": ["local"],
            "launch_settings": {
                "model": "org/manager-model", "engine": "vllm",
                "extra_args": ["--max-model-len", "32768", "--tensor-parallel-size", "2"],
            },
            "launch_controls": {"context_window": 32768},
        }
        self.manager.deployments = [manager_deployment]
        self.manager._cluster_primary_member = lambda _deployment_id: (
            manager_deployment, {"node_id": "local"},
        )

        point = await self.service.record_benchmark_series_point({
            "deployment_id": "manager-only", "concurrency": 2,
            "request_count": 4, "prompt_tokens": 400,
            "generation_tokens": 80, "prompt_seconds": 0.5,
            "wall_seconds": 2,
        })

        self.assertEqual(point["deployment_id"], "manager-only")
        self.assertEqual(point["model_id"], "org/manager-model")
        self.assertEqual(point["context_window_size"], 32768)
        self.assertEqual(point["tensor_parallel_size"], 2)
        samples, total = self.service.store.benchmarks()
        self.assertEqual(total, 1)
        self.assertEqual(samples[0]["configuration"], {
            "context_length": 32768,
            "benchmark_concurrency": 2,
            "tensor_parallel_size": 2,
        })

    async def test_coordinated_manager_run_uses_runtime_provenance_for_relative_local_model(self):
        manager_deployment = {
            "id": "manager-relative", "model": "models/customer",
            "status": "ready", "node_ids": ["worker-1"],
            "model_source": "local",
            "launch_settings": {
                "model": "models/customer", "engine": "vllm",
                "extra_args": ["--max-model-len", "8192"],
                "model_source": "local",
            },
        }
        self.manager.deployments = [manager_deployment]
        self.manager._cluster_primary_member = lambda _deployment_id: (
            manager_deployment, {"node_id": "worker-1"},
        )
        self.manager.node_registry = SimpleNamespace(request=AsyncMock(return_value={
            "gpus": [{"name": "NVIDIA GB10", "mem_total_mib": 128000}],
        }))
        self.service.store.set_setting("device_pairing", {"status": "paired"})
        self.service.store.set_community_consent(True)

        point = await self.service.record_benchmark_series_point({
            "deployment_id": "manager-relative", "concurrency": 1,
            "request_count": 2, "prompt_tokens": 100,
            "generation_tokens": 50, "prompt_seconds": 0.25,
            "wall_seconds": 1,
        })

        self.assertRegex(point["model_id"], r"^local-model-[0-9a-f]{16}$")
        samples, total = self.service.store.benchmarks()
        self.assertEqual(total, 1)
        self.assertEqual(samples[0]["model"]["repository"], "local-model")
        self.assertNotIn("model_source", samples[0]["configuration"])
        self.assertFalse(samples[0]["eligible_for_community"])
        self.assertEqual(self.service.store.outbox_batch(), [])
        self.assertEqual(
            self.service.store.benchmark_model_summaries()[0]["model_id"],
            point["model_id"],
        )
        self.assertNotIn("models/customer", str(point) + str(samples))

    async def test_coordinated_run_enriches_normalized_manager_record(self):
        self.service.store.add_deployment(Deployment(
            id="sparkdeck-record", alias="stored", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/stored-model"),
            settings={
                "manager_deployment_id": "manager-stored",
                "max_model_len": 4096,
                "context_length": 8192,
                "context_size": 2048,
            },
            base_url_set=True,
        ), "http://127.0.0.1:8000")
        manager_deployment = {
            "id": "manager-stored", "sparkdeck_record_id": "sparkdeck-record",
            "model": "org/stored-model", "status": "ready", "node_ids": ["local"],
            "model_source": "public_repository",
            "members": [],
            "launch_settings": {
                "model": "org/stored-model", "engine": "vllm",
                "extra_args": ["--max-model-len", "16384", "--tensor-parallel-size", "4"],
            },
            "launch_controls": {"context_window": 16384},
        }
        self.manager.deployments = [manager_deployment]
        self.manager.cluster_nodes = AsyncMock(return_value=[])
        self.manager._cluster_primary_member = lambda _deployment_id: (
            manager_deployment, {"node_id": "local"},
        )

        point = await self.service.record_benchmark_series_point({
            "deployment_id": "manager-stored", "concurrency": 1,
            "request_count": 2, "prompt_tokens": 200,
            "generation_tokens": 40, "prompt_seconds": 0.25,
            "wall_seconds": 1,
        })

        self.assertEqual(point["deployment_id"], "sparkdeck-record")
        self.assertEqual(point["context_window_size"], 16384)
        self.assertEqual(point["tensor_parallel_size"], 4)
        samples, total = self.service.store.benchmarks()
        self.assertEqual(total, 1)
        self.assertEqual(samples[0]["configuration"], {
            "context_length": 16384,
            "benchmark_concurrency": 1,
            "tensor_parallel_size": 4,
        })

    async def test_coordinated_run_rejects_unmeasured_concurrency(self):
        self.service.store.add_deployment(Deployment(
            id="dep-series", alias="series", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.EXTERNAL, model=ModelIdentity("org/model"),
            settings={"context_length": 4096}, base_url_set=True,
        ), "http://127.0.0.1:8000")

        with self.assertRaisesRegex(ValueError, "one of 1, 2, 5, or 10"):
            await self.service.record_benchmark_series_point({
                "deployment_id": "dep-series", "concurrency": 3,
                "request_count": 2, "prompt_tokens": 100,
                "generation_tokens": 50, "prompt_seconds": 0.25,
                "wall_seconds": 1,
            })

        with self.assertRaisesRegex(ValueError, "at least the measured concurrency"):
            await self.service.record_benchmark_series_point({
                "deployment_id": "dep-series", "concurrency": 10,
                "request_count": 5, "prompt_tokens": 100,
                "generation_tokens": 50, "prompt_seconds": 0.25,
                "wall_seconds": 1,
            })

        with self.assertRaisesRegex(ValueError, "divisible by the measured concurrency"):
            await self.service.record_benchmark_series_point({
                "deployment_id": "dep-series", "concurrency": 5,
                "request_count": 6, "prompt_tokens": 100,
                "generation_tokens": 50, "prompt_seconds": 0.25,
                "wall_seconds": 1,
            })

        base = {
            "deployment_id": "dep-series", "concurrency": 1,
            "request_count": 2, "prompt_tokens": 100,
            "generation_tokens": 50, "prompt_seconds": 0.25,
            "wall_seconds": 1,
        }
        for field in ("concurrency", "request_count", "prompt_tokens", "generation_tokens"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, f"{field} must be an integer"):
                    await self.service.record_benchmark_series_point({
                        **base, field: 1.9,
                    })

        without_prompt_timing = dict(base)
        without_prompt_timing.pop("prompt_seconds")
        with self.assertRaisesRegex(ValueError, "prompt_seconds must be a positive number"):
            await self.service.record_benchmark_series_point(without_prompt_timing)

        with self.assertRaisesRegex(ValueError, "without measured prompt tokens"):
            await self.service.record_benchmark_series_point({
                **base, "prompt_tokens": 0,
            })

        with self.assertRaisesRegex(ValueError, "derived benchmark throughput"):
            await self.service.record_benchmark_series_point({
                **base, "wall_seconds": 5e-324,
            })
        self.assertIsNone(self.service.store.benchmark_model_detail("org/model"))

    async def test_coordinated_run_applies_community_evidence_quality_gates(self):
        self.service.store.add_deployment(Deployment(
            id="dep-quality", alias="quality", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.EXTERNAL, model=ModelIdentity("org/model"),
            settings={"context_length": 4096}, base_url_set=True,
        ), "http://127.0.0.1:8000")
        self.service.store.set_setting("device_pairing", {"status": "paired"})
        self.service.store.set_community_consent(True)

        await self.service.record_benchmark_series_point({
            "deployment_id": "dep-quality", "concurrency": 1,
            "request_count": 2, "prompt_tokens": 100,
            "generation_tokens": 1, "prompt_seconds": 0.25,
            "wall_seconds": 1,
        })
        self.service._managed_hardware_snapshot = AsyncMock(return_value=(
            {"hardware_class": "unknown", "gpu_count": None, "gpus": []}, False,
        ))
        await self.service.record_benchmark_series_point({
            "deployment_id": "dep-quality", "concurrency": 1,
            "request_count": 2, "prompt_tokens": 100,
            "generation_tokens": 50, "prompt_seconds": 0.25,
            "wall_seconds": 1,
        })

        samples, total = self.service.store.benchmarks()
        self.assertEqual(total, 2)
        self.assertTrue(all(not sample["eligible_for_community"] for sample in samples))
        self.assertEqual(self.service.store.outbox_batch(), [])
        self.assertEqual(
            self.service.store.benchmark_model_detail("org/model")["points"][0]["sample_count"],
            2,
        )

    async def test_coordinated_queueing_serializes_with_consent_withdrawal(self):
        self.service.store.add_deployment(Deployment(
            id="dep-consent-race", alias="consent-race",
            runtime=RuntimeKind.VLLM, kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/model"),
            settings={"context_length": 4096},
        ))
        self.service.store.set_setting("device_pairing", {"status": "paired"})
        self.service.store.set_community_consent(True)
        self.service._managed_hardware_snapshot = AsyncMock(return_value=(
            {"hardware_class": "dgx-spark", "gpu_count": 1, "gpus": []}, True,
        ))
        insertion_started = threading.Event()
        release_insertion = threading.Event()
        original_add = self.service.store.add_coordinated_benchmark

        def blocking_add(*args, **kwargs):
            insertion_started.set()
            if not release_insertion.wait(timeout=5):
                raise TimeoutError("test did not release benchmark insertion")
            return original_add(*args, **kwargs)

        with patch.object(
            self.service.store, "add_coordinated_benchmark", side_effect=blocking_add,
        ):
            record_task = asyncio.create_task(
                self.service.record_benchmark_series_point({
                    "deployment_id": "dep-consent-race", "concurrency": 1,
                    "request_count": 2, "prompt_tokens": 100,
                    "generation_tokens": 50, "prompt_seconds": 0.25,
                    "wall_seconds": 1,
                })
            )
            self.assertTrue(await asyncio.to_thread(insertion_started.wait, 2))
            withdraw_task = asyncio.create_task(
                self.service.set_community_consent(False)
            )
            await asyncio.sleep(0.05)
            self.assertFalse(withdraw_task.done())
            release_insertion.set()
            await asyncio.gather(record_task, withdraw_task)

        self.assertFalse(self.service.store.sync_status()["consent"])
        self.assertEqual(self.service.store.outbox_batch(), [])

    async def test_configured_community_aggregates_are_fetched_and_sanitized(self):
        requested = []

        def respond(request: httpx.Request) -> httpx.Response:
            requested.append(request)
            return httpx.Response(200, json={"items": [{
                "model_id": "org/community-model",
                "context_window_size": 8192,
                "inference_tokens_per_second": 42.5,
                "sample_count": 12,
                "private_remote_field": "discard-me",
            }]}, request=request)

        self.manager.community_http_transport = httpx.MockTransport(respond)
        url_patch = patch(
            "sparkdeck.service.COMMUNITY_API_URL", "https://community.example/api/v1/community",
        )
        url_patch.start()
        self.addCleanup(url_patch.stop)

        result = await self.service.community_aggregates()

        self.assertEqual(
            [str(request.url) for request in requested],
            ["https://8.8.8.8/api/v1/community/aggregates"],
        )
        self.assertEqual(requested[0].headers["host"], "community.example")
        self.assertEqual(
            requested[0].extensions["sni_hostname"], "community.example",
        )
        self.assertEqual(result["availability"], "available")
        self.assertEqual(result["items"], [{
            "model_id": "org/community-model",
            "context_window_size": 8192,
            "inference_tokens_per_second": 42.5,
            "sample_count": 12,
        }])
        self.assertEqual(
            result["evidence_policy"]["exact_match_dimensions"],
            ["model_id", "context_window_size"],
        )

    async def test_local_community_database_work_runs_off_event_loop(self):
        # An empty constant keeps the installation fully local.
        url_patch = patch(
            "sparkdeck.service.COMMUNITY_API_URL", "",
        )
        url_patch.start()
        self.addCleanup(url_patch.stop)
        event_loop_thread = threading.get_ident()
        aggregate_threads = []
        community_aggregates = self.service.store.community_aggregates

        def tracked_aggregates():
            aggregate_threads.append(threading.get_ident())
            return community_aggregates()

        with patch.object(
            self.service.store, "community_aggregates", tracked_aggregates,
        ):
            result = await self.service.community_aggregates()

        self.assertEqual(result["availability"], "not_configured")
        self.assertTrue(aggregate_threads)
        self.assertNotIn(event_loop_thread, aggregate_threads)

    async def test_invalid_configured_community_response_fails_visibly(self):
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"items": [{
                "model_id": "org/model",
                "context_window_size": 4096,
                "inference_tokens_per_second": float("nan"),
                "sample_count": 10,
            }]}, request=request)

        self.manager.community_http_transport = httpx.MockTransport(respond)
        url_patch = patch(
            "sparkdeck.service.COMMUNITY_API_URL", "https://community.example/aggregates",
        )
        url_patch.start()
        self.addCleanup(url_patch.stop)

        with self.assertRaisesRegex(RuntimeError, "service is unavailable"):
            await self.service.community_aggregates()

    async def test_community_aggregates_reject_private_literal_without_connecting(self):
        requests = []
        self.manager.community_http_transport = httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(200),
        )
        url_patch = patch(
            "sparkdeck.service.COMMUNITY_API_URL", "http://169.254.169.254/latest/meta-data",
        )
        url_patch.start()
        self.addCleanup(url_patch.stop)

        with self.assertRaisesRegex(RuntimeError, "service is unavailable"):
            await self.service.community_aggregates()

        self.assertEqual(requests, [])

    async def test_community_aggregates_reject_mixed_public_private_dns(self):
        requests = []
        self.manager.community_http_transport = httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(200),
        )
        self.manager.community_resolver = lambda host, port, **kwargs: [
            (
                socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                ("8.8.8.8", port),
            ),
            (
                socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                ("127.0.0.1", port),
            ),
        ]
        url_patch = patch(
            "sparkdeck.service.COMMUNITY_API_URL", "https://community.example/api",
        )
        url_patch.start()
        self.addCleanup(url_patch.stop)

        with self.assertRaisesRegex(RuntimeError, "service is unavailable"):
            await self.service.community_aggregates()

        self.assertEqual(requests, [])

    async def test_community_redirect_to_private_address_is_rejected(self):
        requests = []

        def redirect(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                302,
                headers={"location": "http://127.0.0.1/admin"},
                request=request,
            )

        self.manager.community_http_transport = httpx.MockTransport(redirect)
        url_patch = patch(
            "sparkdeck.service.COMMUNITY_API_URL", "https://community.example/api",
        )
        url_patch.start()
        self.addCleanup(url_patch.stop)

        with self.assertRaisesRegex(RuntimeError, "service is unavailable"):
            await self.service.community_aggregates()

        self.assertEqual(len(requests), 1)
        self.assertEqual(str(requests[0].url), "https://8.8.8.8/api/aggregates")

    async def test_community_dns_result_is_pinned_before_connection(self):
        resolver_calls = []
        requests = []

        def rebind(host, port, **kwargs):
            resolver_calls.append((host, port))
            address = "8.8.4.4" if len(resolver_calls) == 1 else "127.0.0.1"
            return [(
                socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                (address, port),
            )]

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"items": []}, request=request)

        self.manager.community_resolver = rebind
        self.manager.community_http_transport = httpx.MockTransport(respond)
        url_patch = patch(
            "sparkdeck.service.COMMUNITY_API_URL", "https://community.example/api?next=/aggregates",
        )
        url_patch.start()
        self.addCleanup(url_patch.stop)

        result = await self.service.community_aggregates()

        self.assertEqual(result["availability"], "available")
        self.assertEqual(resolver_calls, [("community.example", 443)])
        self.assertEqual(
            str(requests[0].url),
            "https://8.8.4.4/api/aggregates?next=/aggregates",
        )
        self.assertEqual(requests[0].headers["host"], "community.example")

    async def test_community_fetch_tries_each_validated_public_address(self):
        resolver_calls = []
        requests = []

        def resolve(host, port, **kwargs):
            resolver_calls.append((host, port))
            return [
                (
                    socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                    ("8.8.4.4", port),
                ),
                (
                    socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                    ("1.1.1.1", port),
                ),
            ]

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if len(requests) == 1:
                raise httpx.ConnectError("first address unavailable", request=request)
            return httpx.Response(200, json={"items": []}, request=request)

        self.manager.community_resolver = resolve
        self.manager.community_http_transport = httpx.MockTransport(respond)
        url_patch = patch(
            "sparkdeck.service.COMMUNITY_API_URL", "https://community.example/api",
        )
        url_patch.start()
        self.addCleanup(url_patch.stop)

        result = await self.service.community_aggregates()

        self.assertEqual(result["availability"], "available")
        self.assertEqual(resolver_calls, [("community.example", 443)])
        self.assertEqual([str(request.url) for request in requests], [
            "https://8.8.4.4/api/aggregates",
            "https://1.1.1.1/api/aggregates",
        ])
        self.assertTrue(all(
            request.headers["host"] == "community.example"
            and request.extensions["sni_hostname"] == "community.example"
            for request in requests
        ))

    async def test_community_fetch_rejects_huge_ignored_response_field(self):
        body = (
            b'{"items":[],"ignored":"'
            + b"x" * _COMMUNITY_MAX_RESPONSE_BYTES
            + b'"}'
        )
        self.manager.community_http_transport = httpx.MockTransport(
            lambda request: httpx.Response(200, content=body, request=request),
        )
        url_patch = patch(
            "sparkdeck.service.COMMUNITY_API_URL", "https://community.example/api",
        )
        url_patch.start()
        self.addCleanup(url_patch.stop)

        with self.assertRaisesRegex(RuntimeError, "service is unavailable") as raised:
            await self.service.community_aggregates()
        self.assertRegex(str(raised.exception.__cause__), "response is too large")

    async def test_community_fetch_rejects_invalid_declared_lengths(self):
        for value in ("-1", "not-a-number"):
            with self.subTest(content_length=value):
                response = httpx.Response(
                    200,
                    headers={"content-length": value},
                    content=b"{}",
                )
                with self.assertRaisesRegex(ValueError, "invalid length"):
                    await _read_bounded_community_response(response)

    async def test_community_fetch_stops_continuous_chunks_at_byte_cap(self):
        class EndlessBody(httpx.AsyncByteStream):
            def __init__(self):
                self.chunks_sent = 0
                self.closed = False

            async def __aiter__(self):
                while self.chunks_sent < 10_000:
                    self.chunks_sent += 1
                    yield b"x" * 65_536

            async def aclose(self):
                self.closed = True

        body = EndlessBody()
        self.manager.community_http_transport = httpx.MockTransport(
            lambda request: httpx.Response(200, stream=body, request=request),
        )
        url_patch = patch(
            "sparkdeck.service.COMMUNITY_API_URL", "https://community.example/api",
        )
        url_patch.start()
        self.addCleanup(url_patch.stop)

        with self.assertRaisesRegex(RuntimeError, "service is unavailable"):
            await self.service.community_aggregates()

        self.assertLess(body.chunks_sent, 10_000)
        self.assertTrue(body.closed)

    async def test_cluster_routing_identifiers_never_enter_benchmark_configuration(self):
        self.service._record_response(
            "dep-1", "org/model", "vllm",
            {
                "context_length": 4096,
                "node_ids": ["private-worker-id"],
                "manager_deployment_id": "private-cluster-uuid",
                "deployment_mode": "single",
                "benchmark_concurrency": 1,
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

    async def test_replicated_failover_benchmark_uses_serving_node_hardware(self):
        self.manager._cluster_primary_member = lambda _: (
            {"id": "manager-deployment"},
            {"node_id": "worker-1", "container_name": "rank-0"},
        )
        async def stats(node_id, *args, **kwargs):
            return {"gpus": [{
                "name": "NVIDIA RTX 5090" if node_id == "worker-2" else "NVIDIA GB10",
                "mem_total_mib": 32768 if node_id == "worker-2" else 128000,
            }]}

        self.manager.node_registry = SimpleNamespace(request=AsyncMock(side_effect=stats))

        async def proxy(*args, route_observation=None, **kwargs):
            # Rank 0 failed before committing output; rank 1 served the request.
            route_observation["member"] = {
                "node_id": "worker-2", "container_name": "rank-1",
            }
            return {
                "usage": {"prompt_tokens": 24, "completion_tokens": 20},
                "timings": {"predicted_per_second": 80.0},
            }

        self.manager.proxy_cluster_inference = AsyncMock(side_effect=proxy)
        deployment = {
            "id": "dep-remote", "alias": "remote-model", "kind": "managed",
            "runtime": "vllm", "model": {"repository": "org/model"},
            "settings": {
                "manager_deployment_id": "manager-deployment",
                "context_length": 4096,
            },
        }

        await self.service._proxy_managed(
            deployment, {"model": "remote-model", "stream": False},
            "chat/completions", None,
        )

        items, _ = self.service.store.benchmarks()
        self.assertEqual(items[0]["hardware"]["hardware_class"], "local")
        self.assertEqual(
            items[0]["hardware"]["gpus"][0]["model"], "NVIDIA RTX 5090",
        )
        self.assertTrue(items[0]["eligible_for_community"])
        self.manager.node_registry.request.assert_awaited_once_with(
            "worker-2", "GET", "/api/agent/stats", timeout=5,
        )

    async def test_stream_hardware_lookup_waits_for_serving_member(self):
        self.manager.node_registry = SimpleNamespace(request=AsyncMock(return_value={
            "gpus": [{"name": "NVIDIA RTX 5090", "mem_total_mib": 32768}],
        }))

        async def proxy(*args, route_observation=None, **kwargs):
            async def stream():
                route_observation["member"] = {
                    "node_id": "worker-2", "container_name": "rank-1",
                }
                yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                yield 'data: {"choices":[],"usage":{"prompt_tokens":24,"completion_tokens":20}}\n\n'
                yield "data: [DONE]\n\n"

            return stream()

        self.manager.proxy_cluster_inference = AsyncMock(side_effect=proxy)
        deployment = {
            "id": "dep-stream", "alias": "stream-model", "kind": "managed",
            "runtime": "vllm", "model": {"repository": "org/model"},
            "settings": {
                "manager_deployment_id": "manager-deployment",
                "context_length": 4096,
            },
        }

        stream = await self.service._proxy_managed(
            deployment, {"model": "stream-model", "stream": True},
            "chat/completions", None,
        )
        self.manager.node_registry.request.assert_not_awaited()
        _ = [chunk async for chunk in stream]

        items, _ = self.service.store.benchmarks()
        self.assertEqual(
            items[0]["hardware"]["gpus"][0]["model"], "NVIDIA RTX 5090",
        )
        self.assertTrue(items[0]["eligible_for_community"])
        self.manager.node_registry.request.assert_awaited_once_with(
            "worker-2", "GET", "/api/agent/stats", timeout=5,
        )

    async def test_managed_llama_uses_local_hardware_but_external_stays_unknown(self):
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "model": "org/model", "choices": [],
                "usage": {"prompt_tokens": 32, "completion_tokens": 24},
                "timings": {"predicted_per_second": 80.0},
            }, request=request)

        await self.manager.http.aclose()
        self.manager.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.manager._stats_cache = {
            "gpus": [{"name": "NVIDIA GB10", "mem_total_mib": 128000}],
        }
        for deployment_id, alias, kind in (
            ("managed-llama", "managed", DeploymentKind.MANAGED),
            ("external-llama", "external", DeploymentKind.EXTERNAL),
        ):
            self.service.store.add_deployment(Deployment(
                id=deployment_id, alias=alias, runtime=RuntimeKind.LLAMA_CPP,
                kind=kind, model=ModelIdentity("org/model"), base_url_set=True,
                settings={"context_length": 4096},
            ), "http://127.0.0.1:8080")
            await self.service.proxy(
                {"model": alias, "messages": [], "stream": False},
                "chat/completions",
            )

        items, _ = self.service.store.benchmarks()
        by_deployment = {item["deployment_id"]: item for item in items}
        managed = by_deployment["managed-llama"]
        external = by_deployment["external-llama"]

        self.assertEqual(managed["hardware"]["hardware_class"], "dgx-spark")
        self.assertEqual(managed["hardware"]["gpus"][0]["model"], "NVIDIA GB10")
        self.assertTrue(managed["eligible_for_community"])
        self.assertEqual(external["hardware"]["hardware_class"], "unknown")
        self.assertFalse(external["eligible_for_community"])

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
