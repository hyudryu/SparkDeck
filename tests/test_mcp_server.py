import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from mcp_server import ControllerClient, ControllerError, build_server


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
        self.assertNotIn("id", payload)

    async def test_delete_refuses_deployment_not_owned_by_mcp(self) -> None:
        methods = []

        async def handler(request: httpx.Request) -> httpx.Response:
            methods.append(request.method)
            return httpx.Response(200, json={
                "deployments": [{"id": "user-1", "managed_by": None}]
            })

        client = ControllerClient(transport=httpx.MockTransport(handler))
        with self.assertRaisesRegex(ControllerError, "not created by this MCP"):
            await client.action("user-1", "remove")
        self.assertEqual(methods, ["GET"])

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

    async def test_benchmark_reports_concurrent_output_throughput(self) -> None:
        async def controller_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "deployments": [{
                    "id": "mcp-1", "status": "ready", "api_port": 8000,
                }]
            })

        async def inference_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "served-model"}]})
            await asyncio.sleep(0.005)
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "done"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 10},
            })

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
        self.assertGreater(result["metrics"]["output_tokens_per_second"], 0)
        self.assertEqual(result["recording"]["status"], "recorded")

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
        self.assertNotIn(
            "ctx", tools["run_cluster_ab_test"].input_schema.get("properties", {})
        )

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

            async def action(self, deployment_id, action, *, require_owned):
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
