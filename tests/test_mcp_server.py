import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from mcp.server.mcpserver.exceptions import UnexpectedToolError

from mcp_server import ControllerClient, ControllerError, build_server


class CountingErrorStream(httpx.AsyncByteStream):
    def __init__(self, chunk: bytes = b"private-error-detail-", count: int = 100):
        self.chunk = chunk
        self.count = count
        self.chunks_yielded = 0

    async def __aiter__(self):
        for _ in range(self.count):
            self.chunks_yielded += 1
            yield self.chunk


class ControllerClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_clone_gets_distinct_recipe_id_and_preserves_controls(self) -> None:
        requests = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.method == "GET":
                return httpx.Response(200, json={
                    "recipes": [{
                        "id": "base-1", "name": "Base", "model": "model/a",
                        "engine": "vllm", "extra_args": [],
                        "environment": {"NCCL_DEBUG": "WARN"},
                        "deployment_mode": "single", "node_ids": ["local"],
                    }]
                })
            body = json.loads(request.content)
            return httpx.Response(200, json={"id": "clone-1", **body})

        client = ControllerClient(transport=httpx.MockTransport(handler))
        result = await client.clone_recipe(
            "base-1", "Variant", {"launch_controls": {"max_concurrency": 8}}
        )

        payload = json.loads(requests[-1].content)
        self.assertEqual(result["id"], "clone-1")
        self.assertTrue(payload["force_new"])
        self.assertEqual(payload["launch_controls"]["max_concurrency"], 8)
        self.assertEqual(payload["environment"], {"NCCL_DEBUG": "WARN"})
        self.assertNotIn("id", payload)

    async def test_delete_refuses_deployment_not_owned_by_mcp(self) -> None:
        methods = []

        async def handler(request: httpx.Request) -> httpx.Response:
            methods.append(request.method)
            if request.url.path == "/api/state":
                return httpx.Response(200, json={
                    "deployments": [{"id": "user-1", "managed_by": None}]
                })
            return httpx.Response(200, json={
                "items": [{"id": "user-record", "managed_by": None,
                           "settings": {"manager_deployment_id": "user-1"}}],
            })

        client = ControllerClient(transport=httpx.MockTransport(handler))
        with self.assertRaisesRegex(ControllerError, "not created by this MCP"):
            await client.action("user-1", "remove")
        self.assertEqual(methods, ["GET", "GET"])

    async def test_deployment_configuration_and_lifecycle_use_stable_v1_id(self) -> None:
        requests: list[tuple[str, str, dict | None]] = []
        timeouts: list[tuple[str, float]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else None
            requests.append((request.method, request.url.path, body))
            if request.method == "POST":
                timeouts.append((
                    request.url.path,
                    request.extensions["timeout"]["read"],
                ))
            if request.url.path == "/api/state":
                return httpx.Response(200, json={
                    "deployments": [{
                        "id": "manager-1",
                        "sparkdeck_record_id": "record-1",
                        # Manager can lose this during a settings relaunch; the
                        # durable catalog record remains authoritative.
                        "managed_by": None,
                    }],
                })
            if request.url.path == "/api/v1/deployments":
                return httpx.Response(200, json={"items": [{
                    "id": "record-1", "managed_by": "sparkdeck-mcp",
                    "settings": {"manager_deployment_id": "manager-1"},
                }]})
            if request.method == "GET":
                return httpx.Response(200, json={
                    "id": "record-1", "editable": True,
                    "desired_state": "stopped", "extra_args": [],
                    "environment": {}, "launch_controls": {},
                })
            return httpx.Response(200, json={
                "ok": True, "id": "record-1", "changes": body,
            })

        client = ControllerClient(transport=httpx.MockTransport(handler))
        configuration = await client.deployment_configuration("manager-1")
        updated = await client.update_deployment_configuration(
            "record-1",
            {
                "environment": {"NCCL_DEBUG": "WARN"},
                "extra_args": ["--enable-prefix-caching"],
                "launch_controls": {"max_concurrency": 8},
            },
        )
        started = await client.action("manager-1", "start")
        stopped = await client.action("record-1", "stop")
        removed = await client.action("record-1", "remove")

        self.assertTrue(configuration["editable"])
        self.assertEqual(updated["changes"]["environment"], {"NCCL_DEBUG": "WARN"})
        self.assertTrue(started["ok"])
        self.assertTrue(stopped["ok"])
        self.assertTrue(removed["ok"])
        mutations = [request for request in requests if request[0] != "GET"]
        self.assertEqual(mutations, [
            ("PUT", "/api/v1/deployments/record-1/settings", {
                "environment": {"NCCL_DEBUG": "WARN"},
                "extra_args": ["--enable-prefix-caching"],
                "launch_controls": {"max_concurrency": 8},
            }),
            ("POST", "/api/v1/deployments/record-1/start", None),
            ("POST", "/api/v1/deployments/record-1/stop", None),
            ("POST", "/api/deployments/manager-1/remove", None),
        ])
        self.assertEqual(
            sum(path == "/api/v1/deployments" for _, path, _ in requests), 5,
        )
        self.assertIn(
            ("GET", "/api/v1/deployments/record-1", None), requests,
        )
        self.assertEqual(timeouts, [
            ("/api/v1/deployments/record-1/start", 1800),
            ("/api/v1/deployments/record-1/stop", 300),
            ("/api/deployments/manager-1/remove", 300),
        ])

    async def test_legacy_manager_id_is_reconciled_before_v1_action(self) -> None:
        paths = []

        async def handler(request: httpx.Request) -> httpx.Response:
            paths.append((request.method, request.url.path))
            if request.url.path == "/api/state":
                return httpx.Response(200, json={
                    "deployments": [{
                        "id": "legacy-manager", "managed_by": "sparkdeck-mcp",
                    }],
                })
            if request.url.path == "/api/v1/deployments":
                return httpx.Response(200, json={"items": [{
                    "id": "adopted-record", "managed_by": "sparkdeck-mcp",
                    "settings": {"manager_deployment_id": "legacy-manager"},
                }]})
            return httpx.Response(200, json={"ok": True})

        client = ControllerClient(transport=httpx.MockTransport(handler))

        result = await client.action("legacy-manager", "start")

        self.assertTrue(result["ok"])
        self.assertEqual(paths, [
            ("GET", "/api/state"),
            ("GET", "/api/v1/deployments"),
            ("POST", "/api/v1/deployments/adopted-record/start"),
        ])

    async def test_start_forwards_explicit_node_selection(self) -> None:
        requests: list = []

        async def handler(request: httpx.Request) -> httpx.Response:
            body = None
            if request.content:
                try:
                    body = json.loads(request.content)
                except ValueError:
                    body = None
            requests.append((request.method, request.url.path, body))
            if request.url.path == "/api/state":
                return httpx.Response(200, json={
                    "deployments": [{
                        "id": "manager-1",
                        "sparkdeck_record_id": "record-1",
                        "managed_by": "sparkdeck-mcp",
                    }],
                })
            if request.url.path == "/api/v1/deployments":
                return httpx.Response(200, json={"items": [{
                    "id": "record-1",
                    "managed_by": "sparkdeck-mcp",
                    "settings": {"manager_deployment_id": "manager-1"},
                }]})
            return httpx.Response(200, json={"ok": True})

        client = ControllerClient(transport=httpx.MockTransport(handler))

        result = await client.action(
            "record-1", "start", node_ids=["local", "worker-1"],
        )

        self.assertTrue(result["ok"])
        self.assertIn(
            ("POST", "/api/v1/deployments/record-1/start",
             {"node_ids": ["local", "worker-1"]}),
            requests,
        )

        with self.assertRaisesRegex(ControllerError, "only supported for start"):
            await client.action("record-1", "stop", node_ids=["local"])
        with self.assertRaisesRegex(ControllerError, "non-empty node IDs"):
            await client.action("record-1", "start", node_ids=[])

    async def test_configuration_update_requires_explicit_unowned_override(self) -> None:
        requests: list[tuple[str, str]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.method, request.url.path))
            if request.url.path == "/api/state":
                return httpx.Response(200, json={
                    "deployments": [{
                        "id": "user-manager",
                        "sparkdeck_record_id": "user-record",
                        "managed_by": None,
                    }],
                })
            if request.url.path == "/api/v1/deployments":
                return httpx.Response(200, json={"items": [{
                    "id": "user-record", "managed_by": None,
                    "settings": {"manager_deployment_id": "user-manager"},
                }]})
            return httpx.Response(200, json={"id": "user-record", "editable": True})

        client = ControllerClient(transport=httpx.MockTransport(handler))
        with self.assertRaisesRegex(ControllerError, "not created by this MCP"):
            await client.update_deployment_configuration(
                "user-manager", {"environment": {}},
            )
        updated = await client.update_deployment_configuration(
            "user-record", {"environment": {}}, require_owned=False,
        )

        self.assertTrue(updated["editable"])
        self.assertEqual(requests, [
            ("GET", "/api/state"),
            ("GET", "/api/v1/deployments"),
            ("GET", "/api/state"),
            ("GET", "/api/v1/deployments"),
            ("PUT", "/api/v1/deployments/user-record/settings"),
        ])

    async def test_configuration_update_surfaces_stopped_state_requirement(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/state":
                return httpx.Response(200, json={
                    "deployments": [{
                        "id": "manager-1",
                        "sparkdeck_record_id": "record-1",
                        "managed_by": "sparkdeck-mcp",
                    }],
                })
            if request.url.path == "/api/v1/deployments":
                return httpx.Response(200, json={"items": [{
                    "id": "record-1", "managed_by": "sparkdeck-mcp",
                    "settings": {"manager_deployment_id": "manager-1"},
                }]})
            return httpx.Response(
                409,
                json={"detail": "Stop the cluster before changing launch settings"},
            )

        client = ControllerClient(transport=httpx.MockTransport(handler))

        with self.assertRaisesRegex(
            ControllerError, "409.*Stop the cluster",
        ):
            await client.update_deployment_configuration(
                "manager-1", {"extra_args": []},
            )

    async def test_storage_operations_reuse_controller_storage_contracts(self) -> None:
        requests: list[tuple[str, str, bytes, dict | None]] = []
        storage = {
            "enabled": True,
            "nodes": [
                {"id": "local", "name": "Controller", "models": [{
                    "model_id": "org/model", "size_bytes": 10,
                }]},
                {"id": "worker-1", "name": "Worker", "models": []},
            ],
            "jobs": [
                {"id": "job-1", "status": "running", "progress": 0.5},
                {"id": "job-2", "status": "completed", "progress": 1.0},
            ],
        }

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else None
            requests.append((
                request.method, request.url.path, request.url.raw_path, body,
            ))
            if request.method == "GET":
                return httpx.Response(200, json=storage)
            return httpx.Response(202, json={"ok": True, "request": body})

        client = ControllerClient(transport=httpx.MockTransport(handler))

        inventory = await client.storage_weights("local")
        running = await client.storage_transfers("RUNNING")
        job = await client.storage_transfer("job-2")
        pulled = await client.pull_storage_weights(
            "org/model", ["local", "worker-1"],
            revision="release-1", download_node_id="local",
        )
        transferred = await client.transfer_storage_weights(
            "org/model", "local", ["worker-1"], revision="release-1",
        )
        deleted = await client.delete_storage_weights("worker-1", "org/model")

        self.assertEqual([node["id"] for node in inventory["nodes"]], ["local"])
        self.assertEqual([item["id"] for item in running["jobs"]], ["job-1"])
        self.assertEqual(job["id"], "job-2")
        self.assertEqual(pulled["request"], {
            "model_id": "org/model",
            "node_ids": ["local", "worker-1"],
            "revision": "release-1",
            "download_node_id": "local",
        })
        self.assertEqual(transferred["request"], {
            "model_id": "org/model",
            "source_node_id": "local",
            "target_node_ids": ["worker-1"],
            "revision": "release-1",
        })
        self.assertTrue(deleted["ok"])
        self.assertIn(
            ("DELETE", "/api/v1/storage/nodes/worker-1/models/org/model",
             b"/api/v1/storage/nodes/worker-1/models/org%2Fmodel", None),
            requests,
        )

    async def test_storage_lookups_reject_unknown_node_and_job(self) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "enabled": True,
                "nodes": [{"id": "local", "models": []}],
                "jobs": [],
            })

        client = ControllerClient(transport=httpx.MockTransport(handler))

        with self.assertRaisesRegex(ControllerError, "storage node not found"):
            await client.storage_weights("missing")
        with self.assertRaisesRegex(ControllerError, "storage transfer not found"):
            await client.storage_transfer("missing")

    async def test_wait_ready_observes_state_transition(self) -> None:
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            status = "starting" if calls == 1 else "ready"
            return httpx.Response(200, json={
                "deployments": [{"id": "mcp-1", "status": status}]
            })

        client = ControllerClient(transport=httpx.MockTransport(handler))
        result = await client.wait_ready("mcp-1", poll_seconds=0.01)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(calls, 2)

    async def test_benchmark_uses_union_for_overlapping_c10_prompt_timing(self) -> None:
        recorded_bodies = []

        async def controller_handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                recorded_bodies.append(json.loads(request.content))
                return httpx.Response(201, json={"id": "run-1"})
            return httpx.Response(200, json={
                "deployments": [{
                    "id": "mcp-1", "status": "ready", "api_port": 8000,
                }]
            })

        async def inference_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "served-model"}]})
            request_body = json.loads(request.content)
            self.assertTrue(request_body["stream"])
            self.assertEqual(request_body["stream_options"], {"include_usage": True})
            await asyncio.sleep(0.005)
            return httpx.Response(
                200,
                content=(
                    'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
                    'data: {"choices":[{"delta":{"content":"done"}}]}\n\n'
                    'data: {"choices":[],"usage":{"prompt_tokens":5,'
                    '"completion_tokens":10}}\n\n'
                    'data: [DONE]\n\n'
                ),
                headers={"content-type": "text/event-stream"},
            )

        client = ControllerClient(
            transport=httpx.MockTransport(controller_handler),
            inference_transport=httpx.MockTransport(inference_handler),
        )
        result = await client.benchmark(
            "mcp-1", prompts=["one", "two"], repetitions=1,
            concurrency=10, max_tokens=10, warmup_requests=0,
        )

        self.assertEqual(result["model"], "served-model")
        self.assertEqual(result["configuration"]["requests"], 10)
        self.assertEqual(result["metrics"]["completion_tokens"], 100)
        self.assertGreater(result["metrics"]["prompt_tokens_per_second"], 0)
        self.assertGreater(result["metrics"]["prompt_seconds"], 0)
        self.assertGreater(result["metrics"]["mean_time_to_first_token_seconds"], 0)
        summed_ttft = sum(
            sample["time_to_first_token_seconds"] for sample in result["samples"]
        )
        self.assertLess(result["metrics"]["prompt_seconds"], summed_ttft / 2)
        self.assertGreater(result["metrics"]["output_tokens_per_second"], 0)
        self.assertEqual(result["recording"], {"status": "recorded", "id": "run-1"})
        self.assertEqual(len(recorded_bodies), 1)
        self.assertEqual(
            recorded_bodies[0]["prompt_seconds"],
            result["metrics"]["prompt_seconds"],
        )
        self.assertNotIn("prompts", recorded_bodies[0])
        self.assertNotIn("samples", recorded_bodies[0])

    async def test_benchmark_pads_default_c5_run_to_two_full_waves(self) -> None:
        recorded_bodies = []
        arrivals = [0, 0]
        wave_ready = [asyncio.Event(), asyncio.Event()]
        request_count = 0

        async def controller_handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                recorded_bodies.append(json.loads(request.content))
                return httpx.Response(201, json={"id": "run-c5"})
            return httpx.Response(200, json={
                "deployments": [{
                    "id": "mcp-1", "status": "ready", "api_port": 8000,
                }]
            })

        async def inference_handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "served-model"}]})
            request_body = json.loads(request.content)
            if request_body["max_tokens"] == 1:
                return httpx.Response(
                    200,
                    content=(
                        'data: {"choices":[{"delta":{"content":"probe"}}]}\n\n'
                        'data: {"choices":[],"usage":{"prompt_tokens":5,'
                        '"completion_tokens":1}}\n\n'
                        'data: [DONE]\n\n'
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            wave = request_count // 5
            request_count += 1
            arrivals[wave] += 1
            if arrivals[wave] == 5:
                wave_ready[wave].set()
            await asyncio.wait_for(wave_ready[wave].wait(), timeout=1)
            return httpx.Response(
                200,
                content=(
                    'data: {"choices":[{"delta":{"content":"done"}}]}\n\n'
                    'data: {"choices":[],"usage":{"prompt_tokens":5,'
                    '"completion_tokens":10}}\n\n'
                    'data: [DONE]\n\n'
                ),
                headers={"content-type": "text/event-stream"},
            )

        client = ControllerClient(
            transport=httpx.MockTransport(controller_handler),
            inference_transport=httpx.MockTransport(inference_handler),
        )
        result = await asyncio.wait_for(
            client.benchmark("mcp-1", concurrency=5, warmup_requests=0),
            timeout=3,
        )

        self.assertEqual(result["configuration"]["requests"], 10)
        self.assertEqual(arrivals, [5, 5])
        self.assertEqual(recorded_bodies[0]["request_count"], 10)

    async def test_benchmark_probes_and_caches_stream_options_before_timing(self) -> None:
        attempts = []
        recorded_bodies = []

        async def controller_handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                recorded_bodies.append(json.loads(request.content))
                return httpx.Response(201, json={"id": "run-fallback"})
            return httpx.Response(200, json={
                "deployments": [{
                    "id": "mcp-1", "status": "ready", "api_port": 8000,
                }]
            })

        async def inference_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "served-model"}]})
            request_body = json.loads(request.content)
            attempts.append((
                "stream_options" in request_body,
                request_body["max_tokens"],
                request_body["messages"][0]["content"],
            ))
            if "stream_options" in request_body:
                await asyncio.sleep(0.2)
                return httpx.Response(400, json={"detail": "unsupported stream_options"})
            await asyncio.sleep(0.005)
            return httpx.Response(
                200,
                content=(
                    'data: {"choices":[{"delta":{"content":"done"}}]}\n\n'
                    'data: {"choices":[],"usage":{"prompt_tokens":5,'
                    '"completion_tokens":10}}\n\n'
                    'data: [DONE]\n\n'
                ),
                headers={"content-type": "text/event-stream"},
            )

        client = ControllerClient(
            transport=httpx.MockTransport(controller_handler),
            inference_transport=httpx.MockTransport(inference_handler),
        )
        result = await client.benchmark(
            "mcp-1", prompts=["one"], repetitions=1,
            concurrency=2, max_tokens=10, warmup_requests=0,
        )

        self.assertEqual(
            [(with_options, tokens) for with_options, tokens, _ in attempts],
            [(True, 1), (False, 1), (False, 10), (False, 10)],
        )
        probe_prompt = attempts[0][2]
        self.assertEqual(attempts[1][2], probe_prompt)
        self.assertNotEqual(probe_prompt, "one")
        self.assertEqual([attempt[2] for attempt in attempts[2:]], ["one", "one"])
        self.assertLess(result["metrics"]["wall_seconds"], 0.1)
        self.assertEqual(result["recording"], {
            "status": "recorded", "id": "run-fallback",
        })
        self.assertEqual(recorded_bodies[0]["request_count"], 2)

    async def test_benchmark_bounds_streaming_capability_error_body(self) -> None:
        error_stream = CountingErrorStream()

        async def controller_handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "deployments": [{
                    "id": "mcp-1", "status": "ready", "api_port": 8000,
                }]
            })

        async def inference_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "served-model"}]})
            return httpx.Response(503, stream=error_stream)

        client = ControllerClient(
            transport=httpx.MockTransport(controller_handler),
            inference_transport=httpx.MockTransport(inference_handler),
        )
        with self.assertRaises(ControllerError) as raised:
            await client.benchmark(
                "mcp-1", prompts=["one"], repetitions=1,
                concurrency=1, warmup_requests=0,
            )

        message = str(raised.exception)
        self.assertIn("benchmark capability probe failed (503)", message)
        self.assertIn("private-error-detail", message)
        self.assertLess(len(message), 650)
        self.assertLess(error_stream.chunks_yielded, error_stream.count)

    async def test_benchmark_bounds_streaming_measured_request_error_body(self) -> None:
        error_stream = CountingErrorStream(chunk=b"measured-error-detail-")
        inference_calls = 0

        async def controller_handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "deployments": [{
                    "id": "mcp-1", "status": "ready", "api_port": 8000,
                }]
            })

        async def inference_handler(request: httpx.Request) -> httpx.Response:
            nonlocal inference_calls
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "served-model"}]})
            inference_calls += 1
            if inference_calls == 1:
                return httpx.Response(
                    200,
                    content=(
                        'data: {"choices":[{"delta":{"content":"probe"}}]}\n\n'
                        'data: {"choices":[],"usage":{"prompt_tokens":5,'
                        '"completion_tokens":1}}\n\n'
                        'data: [DONE]\n\n'
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(502, stream=error_stream)

        client = ControllerClient(
            transport=httpx.MockTransport(controller_handler),
            inference_transport=httpx.MockTransport(inference_handler),
        )
        with self.assertRaises(ControllerError) as raised:
            await client.benchmark(
                "mcp-1", prompts=["one"], repetitions=1,
                concurrency=1, warmup_requests=0,
            )

        message = str(raised.exception)
        self.assertIn("inference request failed (502)", message)
        self.assertIn("measured-error-detail", message)
        self.assertLess(len(message), 650)
        self.assertLess(error_stream.chunks_yielded, error_stream.count)

    async def test_benchmark_does_not_record_without_stream_usage(self) -> None:
        recorded_bodies = []

        async def controller_handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                recorded_bodies.append(json.loads(request.content))
                return httpx.Response(201, json={"id": "unexpected"})
            return httpx.Response(200, json={
                "deployments": [{
                    "id": "mcp-1", "status": "ready", "api_port": 8000,
                }]
            })

        async def inference_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "served-model"}]})
            return httpx.Response(
                200,
                content=(
                    'data: {"choices":[{"delta":{"content":"done"}}]}\n\n'
                    'data: [DONE]\n\n'
                ),
                headers={"content-type": "text/event-stream"},
            )

        client = ControllerClient(
            transport=httpx.MockTransport(controller_handler),
            inference_transport=httpx.MockTransport(inference_handler),
        )
        result = await client.benchmark(
            "mcp-1", prompts=["one"], repetitions=1,
            concurrency=1, max_tokens=10, warmup_requests=0,
        )

        self.assertEqual(result["recording"]["status"], "not_recorded")
        self.assertIn("prompt throughput unavailable", result["recording"]["reason"])
        self.assertIsNone(result["metrics"]["prompt_tokens_per_second"])
        self.assertIsNone(result["metrics"]["prompt_seconds"])
        self.assertEqual(recorded_bodies, [])

    async def test_benchmark_rejects_unbounded_concurrency_before_controller_request(self) -> None:
        async def unexpected_request(_request: httpx.Request) -> httpx.Response:
            self.fail("invalid concurrency must fail before making a request")

        client = ControllerClient(transport=httpx.MockTransport(unexpected_request))

        with self.assertRaisesRegex(
            ControllerError, "concurrency must be one of 1, 2, 5, or 10"
        ):
            await client.benchmark("mcp-1", concurrency=1_000_000)


class MCPToolSchemaTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_server_publishes_ab_and_lifecycle_tools(self) -> None:
        server = build_server(ControllerClient())
        tools = {tool.name: tool for tool in await server.list_tools()}

        self.assertIn("run_cluster_ab_test", tools)
        self.assertIn("benchmark_cluster_deployment", tools)
        self.assertIn("delete_cluster_deployment", tools)
        for name in (
            "get_cluster_deployment_configuration",
            "update_cluster_deployment_configuration",
            "start_cluster_deployment", "stop_cluster_deployment",
        ):
            self.assertIn(name, tools)
        update = tools["update_cluster_deployment_configuration"]
        self.assertIn("allow_unowned", update.input_schema["properties"])
        self.assertIn("environment", update.description)
        self.assertIn("extra_args", update.description)
        self.assertIn(
            "launch_controls",
            tools["get_cluster_deployment_configuration"].description,
        )
        for name in (
            "list_storage_weights", "pull_storage_weights",
            "transfer_storage_weights", "list_storage_transfers",
            "get_storage_transfer", "delete_storage_weights",
        ):
            self.assertIn(name, tools)
        self.assertIn("confirm", tools["delete_storage_weights"].input_schema["properties"])
        self.assertIn("Virtual NAS", tools["transfer_storage_weights"].description)
        for name in (
            "create_cluster_recipe", "update_cluster_recipe",
            "clone_cluster_recipe", "deploy_cluster_recipe", "run_cluster_ab_test",
        ):
            self.assertIn("environment", tools[name].description)
        self.assertNotIn(
            "ctx", tools["run_cluster_ab_test"].input_schema.get("properties", {})
        )

    async def test_deployment_configuration_and_lifecycle_tools_delegate(self) -> None:
        calls = []

        class FakeClient:
            async def deployment_configuration(self, deployment_id):
                calls.append(("get", deployment_id))
                return {"id": deployment_id, "editable": True}

            async def update_deployment_configuration(
                self, deployment_id, changes, *, require_owned,
            ):
                calls.append(("update", deployment_id, changes, require_owned))
                return {"id": deployment_id, "environment": changes["environment"]}

            async def action(self, deployment_id, action, *, require_owned, node_ids=None):
                calls.append((action, deployment_id, require_owned))
                return {"ok": True, "id": deployment_id}

        server = build_server(FakeClient())
        viewed = await server.call_tool(
            "get_cluster_deployment_configuration", {"deployment_id": "dep-1"},
        )
        updated = await server.call_tool(
            "update_cluster_deployment_configuration", {
                "deployment_id": "dep-1",
                "changes": {"environment": {"NCCL_DEBUG": "WARN"}},
            },
        )
        started = await server.call_tool("start_cluster_deployment", {
            "deployment_id": "dep-1", "allow_unowned": True,
        })
        stopped = await server.call_tool("stop_cluster_deployment", {
            "deployment_id": "dep-1",
        })

        self.assertTrue(viewed.structured_content["editable"])
        self.assertEqual(
            updated.structured_content["environment"], {"NCCL_DEBUG": "WARN"},
        )
        self.assertTrue(started.structured_content["ok"])
        self.assertTrue(stopped.structured_content["ok"])
        self.assertEqual(calls, [
            ("get", "dep-1"),
            ("update", "dep-1", {"environment": {"NCCL_DEBUG": "WARN"}}, True),
            ("start", "dep-1", False),
            ("stop", "dep-1", True),
        ])

    async def test_storage_tools_delegate_and_require_delete_confirmation(self) -> None:
        calls = []

        class FakeClient:
            async def storage_weights(self, node_id):
                calls.append(("inventory", node_id))
                return {"nodes": [{"id": node_id or "local"}]}

            async def pull_storage_weights(self, model_id, node_ids, **kwargs):
                calls.append(("pull", model_id, node_ids, kwargs))
                return {"job_ids": ["download-1"]}

            async def transfer_storage_weights(
                self, model_id, source_node_id, target_node_ids, **kwargs,
            ):
                calls.append((
                    "transfer", model_id, source_node_id, target_node_ids, kwargs,
                ))
                return {"job_ids": ["transfer-1"]}

            async def storage_transfers(self, status):
                calls.append(("transfers", status))
                return {"jobs": [{"id": "transfer-1"}]}

            async def storage_transfer(self, job_id):
                calls.append(("status", job_id))
                return {"id": job_id, "status": "running"}

            async def delete_storage_weights(self, node_id, model_id):
                calls.append(("delete", node_id, model_id))
                return {"ok": True}

        server = build_server(FakeClient())

        inventory = await server.call_tool(
            "list_storage_weights", {"node_id": "local"},
        )
        pulled = await server.call_tool("pull_storage_weights", {
            "model_id": "org/model", "node_ids": ["local", "worker-1"],
            "revision": "main", "download_node_id": "local",
        })
        transferred = await server.call_tool("transfer_storage_weights", {
            "model_id": "org/model", "source_node_id": "local",
            "target_node_ids": ["worker-1"],
        })
        await server.call_tool("list_storage_transfers", {"status": "running"})
        status = await server.call_tool(
            "get_storage_transfer", {"job_id": "transfer-1"},
        )
        with self.assertRaisesRegex(UnexpectedToolError, "Error executing tool"):
            await server.call_tool("delete_storage_weights", {
                "node_id": "worker-1", "model_id": "org/model",
            })
        deleted = await server.call_tool("delete_storage_weights", {
            "node_id": "worker-1", "model_id": "org/model", "confirm": True,
        })

        self.assertEqual(inventory.structured_content["nodes"][0]["id"], "local")
        self.assertEqual(pulled.structured_content["job_ids"], ["download-1"])
        self.assertEqual(transferred.structured_content["job_ids"], ["transfer-1"])
        self.assertEqual(status.structured_content["status"], "running")
        self.assertTrue(deleted.structured_content["ok"])
        self.assertEqual(calls, [
            ("inventory", "local"),
            ("pull", "org/model", ["local", "worker-1"], {
                "revision": "main", "download_node_id": "local",
            }),
            ("transfer", "org/model", "local", ["worker-1"], {
                "revision": None,
            }),
            ("transfers", "running"),
            ("status", "transfer-1"),
            ("delete", "worker-1", "org/model"),
        ])

    async def test_ab_tool_rejects_cluster_with_stopping_deployment(self) -> None:
        class FakeClient:
            async def state(self):
                return {"deployments": [{"id": "dep-1", "status": "stopping"}]}

        server = build_server(FakeClient())
        with self.assertRaises(UnexpectedToolError) as raised:
            await server.call_tool("run_cluster_ab_test", {
                "recipe_id": "recipe-1",
                "variant_a_overrides": {},
                "variant_b_overrides": {},
            })
        self.assertIsInstance(raised.exception.__cause__, ControllerError)
        self.assertIn("cluster is not idle", str(raised.exception.__cause__))

    async def test_ab_tool_runs_variants_sequentially_and_cleans_up(self) -> None:
        events = []

        class FakeClient:
            async def state(self):
                return {"deployments": []}

            async def recipe(self, recipe_id):
                return {"id": recipe_id, "model": "model/a", "extra_args": []}

            def variant_payload(self, base, overrides):
                return {**base, **overrides}

            async def deploy(self, payload, *, run_id, deployment_name):
                label = deployment_name[-1]
                events.append(("deploy", label))
                return {"id": f"deployment-{label.lower()}"}

            async def wait_ready(self, deployment_id, **kwargs):
                events.append(("ready", deployment_id))
                return {"id": deployment_id, "status": "ready"}

            async def benchmark(self, deployment_id, **kwargs):
                events.append(("benchmark", deployment_id))
                rate = 10.0 if deployment_id.endswith("a") else 12.0
                return {"metrics": {"output_tokens_per_second": rate}}

            async def action(self, deployment_id, action, *, require_owned, node_ids=None):
                events.append((action, deployment_id))
                return {"ok": True, "errors": []}

        server = build_server(FakeClient())
        with patch("mcp_server._save_ab_result"):
            result = await server.call_tool("run_cluster_ab_test", {
                "recipe_id": "recipe-1",
                "variant_a_overrides": {"launch_controls": {"max_concurrency": 8}},
                "variant_b_overrides": {"launch_controls": {"max_concurrency": 12}},
            })

        self.assertEqual(result.structured_content["winner"], "B")
        self.assertEqual(events, [
            ("deploy", "A"),
            ("ready", "deployment-a"),
            ("benchmark", "deployment-a"),
            ("remove", "deployment-a"),
            ("deploy", "B"),
            ("ready", "deployment-b"),
            ("benchmark", "deployment-b"),
            ("remove", "deployment-b"),
        ])


if __name__ == "__main__":
    unittest.main()
