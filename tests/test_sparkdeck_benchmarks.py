import socket
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import (
    SparkDeckService,
    _COMMUNITY_MAX_RESPONSE_BYTES,
    _read_bounded_community_response,
)


class FakeManager:
    def __init__(self):
        self.http = httpx.AsyncClient()
        self.list_containers = AsyncMock(return_value=[])
        self.community_http_transport = None
        self.community_resolver = lambda host, port, **kwargs: [(
            socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            ("8.8.8.8", port),
        )]


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

    async def test_upload_worker_drains_exact_privacy_payload_with_idempotency(self):
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(202, request=request)

        self.service._community_upload_url = (
            "https://community.example/api/v1/community"
        )
        self.service._community_upload_token = "node-scoped-token"
        self.manager.community_http_transport = httpx.MockTransport(respond)
        self.service.store.set_setting("device_pairing", {
            "status": "paired", "sub": "user-sub-123",
        })
        self.service.store.set_community_consent(True)
        self.service._record_response(
            None, "org/model", "vllm", {"context_size": 4096},
            time.monotonic() - 0.25,
            {
                "usage": {"prompt_tokens": 32, "completion_tokens": 24},
                "timings": {"predicted_per_second": 96.0},
            },
        )

        synced = await self.service.upload_community_once()

        self.assertEqual(synced, 1)
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.headers["host"], "community.example")
        self.assertEqual(
            request.headers["authorization"], "Bearer node-scoped-token",
        )
        self.assertTrue(request.headers["idempotency-key"])
        self.assertEqual(request.url.path, "/api/v1/community/benchmarks")
        self.assertEqual(set(json.loads(request.content)), {
            "model_id", "context_window_size",
            "inference_tokens_per_second",
        })
        self.assertEqual(
            self.service.store.sync_status()["outbox"]["synced"], 1,
        )

    async def test_upload_worker_requires_node_scoped_configuration(self):
        requested = []
        self.manager.community_http_transport = httpx.MockTransport(
            lambda request: requested.append(request) or httpx.Response(200),
        )
        self.service.store.set_setting("device_pairing", {
            "status": "paired", "sub": "user-sub-123",
        })
        self.service.store.set_community_consent(True)

        self.assertEqual(await self.service.upload_community_once(), 0)
        self.assertEqual(requested, [])

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
        self.service.store.set_setting(
            "community_api_url", "https://community.example/api/v1/community",
        )

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
        event_loop_thread = threading.get_ident()
        setting_threads = []
        aggregate_threads = []
        get_setting = self.service.store.get_setting
        community_aggregates = self.service.store.community_aggregates

        def tracked_get_setting(*args):
            setting_threads.append(threading.get_ident())
            return get_setting(*args)

        def tracked_aggregates():
            aggregate_threads.append(threading.get_ident())
            return community_aggregates()

        with (
            patch.object(self.service.store, "get_setting", tracked_get_setting),
            patch.object(
                self.service.store, "community_aggregates", tracked_aggregates,
            ),
        ):
            result = await self.service.community_aggregates()

        self.assertEqual(result["availability"], "not_configured")
        self.assertTrue(setting_threads)
        self.assertTrue(aggregate_threads)
        self.assertNotIn(event_loop_thread, setting_threads)
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
        self.service.store.set_setting(
            "community_api_url", "https://community.example/aggregates",
        )

        with self.assertRaisesRegex(RuntimeError, "service is unavailable"):
            await self.service.community_aggregates()

    async def test_community_aggregates_reject_private_literal_without_connecting(self):
        requests = []
        self.manager.community_http_transport = httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(200),
        )
        self.service.store.set_setting(
            "community_api_url", "http://169.254.169.254/latest/meta-data",
        )

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
        self.service.store.set_setting(
            "community_api_url", "https://community.example/api",
        )

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
        self.service.store.set_setting(
            "community_api_url", "https://community.example/api",
        )

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
        self.service.store.set_setting(
            "community_api_url",
            "https://community.example/api?next=/aggregates",
        )

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
        self.service.store.set_setting(
            "community_api_url", "https://community.example/api",
        )

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
        self.service.store.set_setting(
            "community_api_url", "https://community.example/api",
        )

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
        self.service.store.set_setting(
            "community_api_url", "https://community.example/api",
        )

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
        self.assertEqual(items[0]["hardware"]["hardware_class"], "dgx-spark")
        self.assertEqual(items[0]["hardware"]["gpus"][0]["model"], "NVIDIA GB10")
        self.assertTrue(items[0]["eligible_for_community"])
        self.manager.node_registry.request.assert_awaited_once_with(
            "worker-1", "GET", "/api/agent/stats", timeout=5,
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
