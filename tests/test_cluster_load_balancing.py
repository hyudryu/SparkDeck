import asyncio
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi import HTTPException

from manager import ClientAbort, Manager


def status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(
        f"upstream returned HTTP {status}", request=request, response=response,
    )


def upstream_error_stream(status: int):
    """Emulate a ``_vllm_stream`` that reports an upstream HTTP error."""

    async def gen():
        yield (
            "data: " + json.dumps({
                "error": {
                    "message": f"HTTP {status}: boom",
                    "type": "upstream_error",
                    "code": status,
                },
            }) + "\n\n"
        )
        yield "data: [DONE]\n\n"

    return gen()


class StreamResponse:
    status_code = 200

    def __init__(self):
        self.closed = False

    async def aiter_lines(self):
        yield 'data: {"choices": []}'
        yield "data: [DONE]"

    async def aclose(self):
        self.closed = True


class ErrorStreamResponse(StreamResponse):
    def __init__(self, status=None):
        super().__init__()
        self.status = status

    async def aiter_lines(self):
        error = {
            "message": "runtime disappeared",
            "type": "upstream_error",
        }
        if self.status is not None:
            error["code"] = self.status
        yield "data: " + json.dumps({"error": error})
        yield "data: [DONE]"


def member(rank: int, node_id: str, container_name: str) -> dict:
    return {
        "rank": rank,
        "node_id": node_id,
        "node_name": f"Node {rank}",
        "container_name": container_name,
    }


def replicated_deployment() -> dict:
    return {
        "id": "repl-1",
        "model": "org/model",
        "mode": "replicated",
        "launch_settings": {},
        "members": [
            member(0, "remote-1", "repl-1-r0"),
            member(1, "remote-2", "repl-1-r1"),
        ],
    }


def build_manager(deployment: dict) -> Manager:
    manager = Manager.__new__(Manager)
    manager.deployments = [deployment]
    manager.node_registry = Mock()
    manager.node_registry.request = AsyncMock(return_value={"choices": []})
    manager.node_registry.open_stream = AsyncMock()
    manager._acquire_inference_slot = AsyncMock(return_value=None)
    manager._release_inference_slot = Mock()
    return manager


def proxied_containers(manager: Manager) -> list[str]:
    return [
        call.kwargs["json_body"]["_sparkdeck_container_name"]
        for call in manager.node_registry.request.await_args_list
    ]


def member_loads(manager: Manager, deployment: dict) -> list[int]:
    return [
        manager._cluster_member_active("repl-1", m)
        for m in deployment["members"]
    ]


class ReplicaBalancingTests(unittest.IsolatedAsyncioTestCase):
    async def proxy(self, manager: Manager, **overrides):
        body = {"model": "org/model", "messages": [], "stream": False, **overrides}
        return await manager.proxy_cluster_inference(
            "repl-1", "org/model", body, "chat/completions",
        )

    async def test_idle_replicas_alternate_requests(self):
        manager = build_manager(replicated_deployment())

        for _ in range(4):
            await self.proxy(manager)

        self.assertEqual(
            proxied_containers(manager),
            ["repl-1-r0", "repl-1-r1", "repl-1-r0", "repl-1-r1"],
        )
        nodes = [
            call.args[0] for call in manager.node_registry.request.await_args_list
        ]
        self.assertEqual(nodes, ["remote-1", "remote-2", "remote-1", "remote-2"])

    async def test_remote_request_tracks_and_forwards_original_caller_ip(self):
        manager = build_manager(replicated_deployment())
        manager._req_seq = 0
        manager._active_reqs = {}
        manager._trailing_window = 5.0

        async def respond(*args, **kwargs):
            self.assertEqual(
                manager.active_requests()["org/model"]["caller_ips"],
                {"192.0.2.45": 1},
            )
            self.assertEqual(
                kwargs["json_body"]["_sparkdeck_caller_ip"], "192.0.2.45",
            )
            return {"choices": []}

        manager.node_registry.request = AsyncMock(side_effect=respond)

        await manager.proxy_cluster_inference(
            "repl-1", "org/model",
            {"model": "org/model", "messages": [], "stream": False},
            "chat/completions", caller_ip="192.0.2.45",
        )

        self.assertNotIn("org/model", manager.active_requests())

    async def test_least_loaded_replica_receives_concurrent_request(self):
        manager = build_manager(replicated_deployment())
        first_started = asyncio.Event()

        async def slow_rank_zero(node_id, method, path, *, json_body=None, timeout=30):
            if json_body["_sparkdeck_container_name"] == "repl-1-r0":
                first_started.set()
                await asyncio.sleep(0.05)
            return {"choices": []}

        manager.node_registry.request = AsyncMock(side_effect=slow_rank_zero)

        first = asyncio.create_task(self.proxy(manager))
        await first_started.wait()
        second = await self.proxy(manager)
        await first

        self.assertEqual(second, {"choices": []})
        # Rank 0 is still busy with the first request, so the second one
        # must route to the idle replica.
        self.assertEqual(proxied_containers(manager)[1], "repl-1-r1")
        self.assertEqual(
            member_loads(manager, manager.deployments[0]), [0, 0]
        )

    def test_balanced_selection_prefers_least_loaded_member(self):
        manager = build_manager(replicated_deployment())
        deployment = manager.deployments[0]
        members = Manager._cluster_members_sorted(deployment)

        manager._acquire_cluster_member("repl-1", members[0])
        manager._acquire_cluster_member("repl-1", members[0])

        order = manager._cluster_route_order(deployment)
        self.assertEqual(order[0]["container_name"], "repl-1-r1")

    def test_single_mode_deployment_always_routes_to_primary(self):
        deployment = replicated_deployment()
        deployment["mode"] = "single"
        deployment["members"] = deployment["members"][:1]
        manager = build_manager(deployment)

        order = manager._cluster_route_order(deployment)
        self.assertEqual([m["container_name"] for m in order], ["repl-1-r0"])

    def test_sharded_deployment_stays_on_rank_zero_under_load(self):
        deployment = replicated_deployment()
        deployment["mode"] = "sharded"
        manager = build_manager(deployment)
        members = Manager._cluster_members_sorted(deployment)
        manager._acquire_cluster_member("repl-1", members[0])

        order = manager._cluster_route_order(deployment)
        self.assertEqual([m["container_name"] for m in order], ["repl-1-r0"])

    def test_failover_candidates_follow_least_loaded_order(self):
        deployment = replicated_deployment()
        deployment["members"].append(member(2, "remote-3", "repl-1-r2"))
        manager = build_manager(deployment)
        members = Manager._cluster_members_sorted(deployment)
        for _ in range(5):
            manager._acquire_cluster_member("repl-1", members[0])
        manager._acquire_cluster_member("repl-1", members[2])
        manager._acquire_cluster_member("repl-1", members[2])

        order = manager._cluster_route_order(deployment)
        self.assertEqual(
            [m["container_name"] for m in order],
            ["repl-1-r1", "repl-1-r2", "repl-1-r0"],
        )


class ReplicaFailoverTests(unittest.IsolatedAsyncioTestCase):
    async def proxy(self, manager: Manager, **overrides):
        body = {"model": "org/model", "messages": [], "stream": False, **overrides}
        return await manager.proxy_cluster_inference(
            "repl-1", "org/model", body, "chat/completions",
        )

    async def test_unreachable_replica_fails_over_to_next(self):
        manager = build_manager(replicated_deployment())
        manager.node_registry.request = AsyncMock(side_effect=[
            RuntimeError("could not contact Node 0: connect failed"),
            {"choices": [], "ok": True},
        ])

        result = await self.proxy(manager)

        self.assertEqual(result, {"choices": [], "ok": True})
        self.assertEqual(manager.node_registry.request.await_count, 2)
        second = manager.node_registry.request.await_args_list[1]
        self.assertEqual(second.args[0], "remote-2")
        self.assertEqual(
            second.kwargs["json_body"]["_sparkdeck_container_name"], "repl-1-r1"
        )
        # The failed attempt must not leak a load slot.
        self.assertEqual(
            member_loads(manager, manager.deployments[0]), [0, 0]
        )

    async def test_nonstream_failover_observes_actual_serving_member(self):
        manager = build_manager(replicated_deployment())
        manager.node_registry.request = AsyncMock(side_effect=[
            RuntimeError("could not contact Node 0: connect failed"),
            {"choices": [], "ok": True},
        ])
        route_observation = {}

        await manager.proxy_cluster_inference(
            "repl-1", "org/model",
            {"model": "org/model", "messages": [], "stream": False},
            "chat/completions", route_observation=route_observation,
        )

        self.assertEqual(
            route_observation["member"]["node_id"], "remote-2",
        )

    async def test_exhausted_replicas_raise_last_error(self):
        manager = build_manager(replicated_deployment())
        manager.node_registry.request = AsyncMock(side_effect=[
            RuntimeError("could not contact Node 0"),
            RuntimeError("could not contact Node 1"),
        ])

        with self.assertRaisesRegex(RuntimeError, "Node 1"):
            await self.proxy(manager)
        self.assertEqual(manager.node_registry.request.await_count, 2)
        self.assertEqual(
            member_loads(manager, manager.deployments[0]), [0, 0]
        )

    async def test_client_abort_does_not_fail_over(self):
        manager = build_manager(replicated_deployment())
        manager.node_registry.request = AsyncMock(
            side_effect=ClientAbort("client disconnected")
        )

        with self.assertRaises(ClientAbort):
            await self.proxy(manager)
        manager.node_registry.request.assert_awaited_once()

    async def test_admission_limits_are_scoped_per_replica(self):
        manager = build_manager(replicated_deployment())
        manager.node_registry.request = AsyncMock(side_effect=[
            RuntimeError("could not contact Node 0: connect failed"),
            {"choices": [], "ok": True},
        ])
        captured = []

        async def capture_slot(container, model, cancel):
            captured.append(dict(container))
            return None

        manager._acquire_inference_slot = AsyncMock(side_effect=capture_slot)

        await self.proxy(manager)

        self.assertEqual(len(captured), 2)
        self.assertEqual(
            {c["name"] for c in captured}, {"repl-1-r0", "repl-1-r1"},
        )
        for container in captured:
            self.assertNotIn("deployment_id", container)
        targets = {
            Manager._inference_admission_config(container, "org/model")[0]
            for container in captured
        }
        self.assertEqual(len(targets), 2)

    async def test_admission_cancel_releases_replica_load(self):
        manager = build_manager(replicated_deployment())
        manager._acquire_inference_slot = AsyncMock(
            side_effect=ClientAbort("client disconnected while queued")
        )

        with self.assertRaises(ClientAbort):
            await self.proxy(manager)

        manager.node_registry.request.assert_not_awaited()
        self.assertEqual(member_loads(manager, manager.deployments[0]), [0, 0])

    async def test_remote_upstream_client_error_does_not_fail_over(self):
        manager = build_manager(replicated_deployment())
        manager.node_registry.request = AsyncMock(side_effect=[
            RuntimeError("Node 0 agent error: HTTP 400: invalid parameters"),
        ])

        with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
            await self.proxy(manager)

        manager.node_registry.request.assert_awaited_once()
        self.assertEqual(member_loads(manager, manager.deployments[0]), [0, 0])

    async def test_remote_upstream_server_error_fails_over(self):
        manager = build_manager(replicated_deployment())
        manager.node_registry.request = AsyncMock(side_effect=[
            RuntimeError("Node 0 agent error: HTTP 503: model overloaded"),
            {"choices": [], "ok": True},
        ])

        result = await self.proxy(manager)

        self.assertEqual(result["ok"], True)
        self.assertEqual(manager.node_registry.request.await_count, 2)
        self.assertEqual(member_loads(manager, manager.deployments[0]), [0, 0])

    async def test_remote_missing_container_fails_over_without_stream(self):
        # The agent types a missing container as a 404 with a
        # ``replica_unavailable`` detail; ``NodeRegistry.request`` wraps that
        # as a RuntimeError, which must still be treated as availability.
        detail = json.dumps({
            "detail": {
                "type": "replica_unavailable",
                "message": "No managed container found for repl-1-r0",
            },
        })
        manager = build_manager(replicated_deployment())
        manager.node_registry.request = AsyncMock(side_effect=[
            RuntimeError(f"Node 0 agent error: HTTP 404: {detail}"),
            {"choices": [], "ok": True},
        ])

        result = await self.proxy(manager)

        self.assertEqual(result["ok"], True)
        self.assertEqual(manager.node_registry.request.await_count, 2)
        second = manager.node_registry.request.await_args_list[1]
        self.assertEqual(second.args[0], "remote-2")
        self.assertEqual(member_loads(manager, manager.deployments[0]), [0, 0])


class ReplicaStreamTests(unittest.IsolatedAsyncioTestCase):
    class UnavailableResponse:
        status_code = 503

        async def aread(self):
            return b"unavailable"

        async def aclose(self):
            pass

    async def test_stream_releases_replica_load_after_consumption(self):
        manager = build_manager(replicated_deployment())
        response = StreamResponse()
        manager.node_registry.open_stream = AsyncMock(return_value=response)
        deployment = manager.deployments[0]
        body = {"model": "org/model", "messages": [], "stream": True}

        stream = await manager.proxy_cluster_inference(
            "repl-1", "org/model", body, "chat/completions",
        )
        self.assertEqual(sum(member_loads(manager, deployment)), 1)

        chunks = [chunk async for chunk in stream]

        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertTrue(response.closed)
        self.assertEqual(member_loads(manager, deployment), [0, 0])

    async def test_failed_stream_open_fails_over_and_releases(self):
        manager = build_manager(replicated_deployment())
        manager.node_registry.open_stream = AsyncMock(side_effect=[
            self.UnavailableResponse(),
            StreamResponse(),
        ])
        deployment = manager.deployments[0]
        body = {"model": "org/model", "messages": [], "stream": True}

        stream = await manager.proxy_cluster_inference(
            "repl-1", "org/model", body, "chat/completions",
        )
        chunks = [chunk async for chunk in stream]

        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertEqual(manager.node_registry.open_stream.await_count, 2)
        self.assertEqual(member_loads(manager, deployment), [0, 0])

    async def test_remote_stream_error_before_output_fails_over(self):
        manager = build_manager(replicated_deployment())
        manager.node_registry.open_stream = AsyncMock(side_effect=[
            ErrorStreamResponse(503), StreamResponse(),
        ])
        route_observation = {}

        stream = await manager.proxy_cluster_inference(
            "repl-1", "org/model",
            {"model": "org/model", "messages": [], "stream": True},
            "chat/completions",
            route_observation=route_observation,
        )
        self.assertEqual(route_observation, {})
        chunks = [chunk async for chunk in stream]

        self.assertEqual(chunks[0], 'data: {"choices": []}\n\n')
        self.assertNotIn("runtime disappeared", "".join(chunks))
        self.assertEqual(manager.node_registry.open_stream.await_count, 2)
        self.assertEqual(
            route_observation["member"]["node_id"], "remote-2",
        )

    async def test_remote_codeless_stream_error_before_output_fails_over(self):
        manager = build_manager(replicated_deployment())
        manager.node_registry.open_stream = AsyncMock(side_effect=[
            ErrorStreamResponse(), StreamResponse(),
        ])

        stream = await manager.proxy_cluster_inference(
            "repl-1", "org/model",
            {"model": "org/model", "messages": [], "stream": True},
            "chat/completions",
        )
        chunks = [chunk async for chunk in stream]

        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertEqual(manager.node_registry.open_stream.await_count, 2)

    async def test_missing_remote_container_404_fails_over(self):
        manager = build_manager(replicated_deployment())
        request = httpx.Request("POST", "http://remote/agent/inference")
        missing = httpx.Response(
            404,
            json={"detail": {
                "type": "replica_unavailable",
                "message": "No managed container found",
            }},
            request=request,
        )
        manager.node_registry.open_stream = AsyncMock(side_effect=[
            missing, StreamResponse(),
        ])

        stream = await manager.proxy_cluster_inference(
            "repl-1", "org/model",
            {"model": "org/model", "messages": [], "stream": True},
            "chat/completions",
        )
        chunks = [chunk async for chunk in stream]

        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertEqual(manager.node_registry.open_stream.await_count, 2)

    async def test_remote_upstream_404_does_not_fail_over(self):
        manager = build_manager(replicated_deployment())
        request = httpx.Request("POST", "http://remote/agent/inference")
        rejected = httpx.Response(
            404,
            json={"detail": {
                "type": "upstream_error", "message": "model not found",
            }},
            request=request,
        )
        manager.node_registry.open_stream = AsyncMock(return_value=rejected)

        with self.assertRaises(httpx.HTTPStatusError) as raised:
            await manager.proxy_cluster_inference(
                "repl-1", "org/model",
                {"model": "org/model", "messages": [], "stream": True},
                "chat/completions",
            )

        self.assertEqual(raised.exception.response.status_code, 404)
        manager.node_registry.open_stream.assert_awaited_once()


class LocalReplicaTests(unittest.IsolatedAsyncioTestCase):
    def manager(self):
        deployment = replicated_deployment()
        deployment["members"] = [
            member(0, "local", "repl-1-r0"),
            member(1, "remote-2", "repl-1-r1"),
        ]
        manager = build_manager(deployment)
        manager._vllm_chat = AsyncMock(return_value={"choices": []})
        manager._vllm_completions = AsyncMock(return_value={"choices": []})
        return manager, deployment

    async def test_local_and_remote_replicas_share_work(self):
        manager, deployment = self.manager()

        for _ in range(2):
            await self.manager_proxy(manager)

        manager._vllm_chat.assert_awaited_once()
        self.assertEqual(
            manager._vllm_chat.await_args.kwargs["container_name"], "repl-1-r0"
        )
        remote_calls = manager.node_registry.request.await_args_list
        self.assertEqual(len(remote_calls), 1)
        self.assertEqual(remote_calls[0].args[0], "remote-2")
        self.assertEqual(member_loads(manager, deployment), [0, 0])

    async def test_local_replica_failure_fails_over_to_remote(self):
        manager, deployment = self.manager()
        manager._vllm_chat = AsyncMock(side_effect=LookupError(
            "No managed container found for model 'org/model'"
        ))
        manager.node_registry.request = AsyncMock(
            return_value={"choices": [], "ok": True}
        )

        result = await self.manager_proxy(manager)

        self.assertEqual(result["ok"], True)
        manager.node_registry.request.assert_awaited_once()
        self.assertEqual(member_loads(manager, deployment), [0, 0])

    async def test_local_stream_releases_load_after_consumption(self):
        manager, deployment = self.manager()

        async def fake_stream(*args, **kwargs):
            async def gen():
                yield "chunk-1"
                yield "chunk-2"

            return gen()

        manager._vllm_chat = AsyncMock(side_effect=fake_stream)

        stream = await manager.proxy_cluster_inference(
            "repl-1", "org/model",
            {"model": "org/model", "messages": [], "stream": True},
            "chat/completions",
        )
        chunks = [chunk async for chunk in stream]

        self.assertEqual(chunks, ["chunk-1", "chunk-2"])
        self.assertEqual(member_loads(manager, deployment), [0, 0])

    async def test_local_stream_server_error_fails_over_to_remote(self):
        manager, deployment = self.manager()
        manager._vllm_chat = AsyncMock(
            side_effect=lambda *a, **k: upstream_error_stream(503)
        )
        manager.node_registry.open_stream = AsyncMock(return_value=StreamResponse())

        stream = await self.manager_proxy(manager, stream=True)
        chunks = [chunk async for chunk in stream]

        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        manager._vllm_chat.assert_awaited_once()
        manager.node_registry.open_stream.assert_awaited_once()
        self.assertEqual(member_loads(manager, deployment), [0, 0])

    async def test_local_stream_client_error_does_not_fail_over(self):
        manager, deployment = self.manager()
        manager._vllm_chat = AsyncMock(
            side_effect=lambda *a, **k: upstream_error_stream(400)
        )

        stream = await self.manager_proxy(manager, stream=True)
        chunks = [chunk async for chunk in stream]

        self.assertIn('"code": 400', chunks[0])
        manager.node_registry.open_stream.assert_not_awaited()
        manager.node_registry.request.assert_not_awaited()
        self.assertEqual(member_loads(manager, deployment), [0, 0])

    async def test_local_codeless_stream_error_fails_over_to_remote(self):
        manager, deployment = self.manager()

        async def unavailable(*args, **kwargs):
            async def gen():
                yield "data: " + json.dumps({"error": {
                    "message": "connect failed", "type": "upstream_error",
                }}) + "\n\n"
            return gen()

        manager._vllm_chat = AsyncMock(side_effect=unavailable)
        manager.node_registry.open_stream = AsyncMock(return_value=StreamResponse())

        stream = await self.manager_proxy(manager, stream=True)
        chunks = [chunk async for chunk in stream]

        self.assertEqual(chunks[0], 'data: {"choices": []}\n\n')
        manager.node_registry.open_stream.assert_awaited_once()
        self.assertEqual(member_loads(manager, deployment), [0, 0])

    async def test_local_stream_http_error_preserves_status_before_headers(self):
        manager, deployment = self.manager()
        manager._vllm_chat = AsyncMock(side_effect=status_error(422))

        with self.assertRaises(httpx.HTTPStatusError) as raised:
            await self.manager_proxy(manager, stream=True)

        self.assertEqual(raised.exception.response.status_code, 422)
        manager.node_registry.open_stream.assert_not_awaited()
        self.assertEqual(member_loads(manager, deployment), [0, 0])

    async def test_local_upstream_client_error_does_not_fail_over(self):
        manager, deployment = self.manager()
        manager._vllm_chat = AsyncMock(side_effect=status_error(400))

        with self.assertRaises(httpx.HTTPStatusError):
            await self.manager_proxy(manager)

        manager.node_registry.request.assert_not_awaited()
        self.assertEqual(member_loads(manager, deployment), [0, 0])

    async def manager_proxy(self, manager: Manager, **overrides):
        body = {"model": "org/model", "messages": [], "stream": False, **overrides}
        return await manager.proxy_cluster_inference(
            "repl-1", "org/model", body, "chat/completions",
        )


class AgentInferenceErrorTests(unittest.IsolatedAsyncioTestCase):
    class Request:
        headers = {}

        async def json(self):
            return {"model": "org/model", "messages": [], "stream": True}

        async def stream(self):
            yield json.dumps(await self.json()).encode()

        async def is_disconnected(self):
            return False

    async def test_agent_preserves_upstream_http_status(self):
        import server

        with (
            patch.object(server, "_require_agent"),
            patch.object(
                server.manager, "_vllm_chat",
                AsyncMock(side_effect=status_error(422)),
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await server.agent_inference("chat/completions", self.Request())

        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.detail["type"], "upstream_error")

    async def test_agent_types_missing_container_as_replica_unavailable(self):
        import server

        with (
            patch.object(server, "_require_agent"),
            patch.object(
                server.manager, "_vllm_chat",
                AsyncMock(side_effect=LookupError("No managed container found")),
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await server.agent_inference("chat/completions", self.Request())

        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(
            raised.exception.detail["type"], "replica_unavailable",
        )

    async def test_agent_health_is_observational_and_never_resolves_or_starts(self):
        import server

        request = self.Request()
        request.json = AsyncMock(return_value={
            "model": "org/model",
            "_sparkdeck_container_name": "rank-0",
            "_sparkdeck_deployment_id": "deployment-1",
        })
        health = AsyncMock(return_value=False)
        with (
            patch.object(server, "_require_agent"),
            patch.object(server.manager, "inference_target_health", health),
            patch.object(server.manager, "_resolve_vllm_target", AsyncMock()) as resolve,
        ):
            result = await server.agent_inference_health(request)

        self.assertEqual(result, {"ready": False, "model": "org/model"})
        health.assert_awaited_once_with(
            "org/model", container_name="rank-0", deployment_id="deployment-1",
        )
        resolve.assert_not_awaited()


class ReplicaHealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_replicated_health_ready_when_any_replica_ready(self):
        manager = build_manager(replicated_deployment())
        manager.node_registry.request = AsyncMock(side_effect=[
            {"ready": False}, {"ready": True},
        ])

        self.assertTrue(
            await manager.cluster_deployment_health("repl-1", "org/model")
        )
        containers = [
            call.kwargs["json_body"]["_sparkdeck_container_name"]
            for call in manager.node_registry.request.await_args_list
        ]
        self.assertEqual(sorted(containers), ["repl-1-r0", "repl-1-r1"])

    async def test_replicated_health_down_when_all_replicas_unready(self):
        manager = build_manager(replicated_deployment())
        manager.node_registry.request = AsyncMock(return_value={"ready": False})

        self.assertFalse(
            await manager.cluster_deployment_health("repl-1", "org/model")
        )

    async def test_replica_agent_error_counts_as_unhealthy(self):
        manager = build_manager(replicated_deployment())
        manager.node_registry.request = AsyncMock(side_effect=[
            RuntimeError("could not contact Node 0"), {"ready": True},
        ])

        self.assertTrue(
            await manager.cluster_deployment_health("repl-1", "org/model")
        )

    async def test_legacy_deployment_without_mode_keeps_primary_health(self):
        deployment = replicated_deployment()
        deployment.pop("mode")
        manager = build_manager(deployment)
        manager.node_registry.request = AsyncMock(return_value={"ready": True})

        self.assertTrue(
            await manager.cluster_deployment_health("repl-1", "org/model")
        )
        manager.node_registry.request.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
