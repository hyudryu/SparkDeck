"""A momentary connectivity blip must not fail requests for a serving model.

When the controller briefly loses a node (Wi-Fi reassociation, a Tailscale
re-key), the routing snapshot publishes the node's deployment as degraded,
unknown, or starting. Requests by the deployment's served model name used to
lose their owner and fall through to the legacy local-container lookup,
which answered a terminal HTTP 404 ("No managed container found") even
though the deployment was still meant to serve. These tests pin that the
served name keeps its owner and that the cluster proxy rides out a short
outage instead of failing on the first unanswered readiness probe.
"""
import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import httpx

from cluster import NodeAgentResponseError
from manager import ClientAbort, ClusterReplicaUnavailable, Manager
from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import SparkDeckService
from test_cluster_load_balancing import HEALTH_PATH, build_manager, member
from test_sparkdeck_proxy_fixes import FakeManager

SERVED = "glm-5.3-flash"
LEGACY_404 = f"No managed container found for model '{SERVED}'"


def observed(record_id: str, manager_id: str, status: str, **extra) -> dict:
    return {
        "id": record_id, "alias": f"{record_id} alias", "runtime": "vllm",
        "kind": "managed", "status": status, "served_models": [SERVED],
        "model": {"repository": f"/models/{record_id}"},
        "settings": {"manager_deployment_id": manager_id},
        **extra,
    }


class ServedNameKeepsItsOwnerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.manager = FakeManager()
        self.manager.proxy_cluster_inference = AsyncMock(return_value={
            "model": SERVED, "choices": [], "usage": {},
        })
        self.manager.proxy_chat_completions = AsyncMock(
            side_effect=LookupError(LEGACY_404),
        )
        self.service = SparkDeckService(self.manager, Path(self.directory.name))

    async def asyncTearDown(self):
        await self.manager.http.aclose()
        await self.service.close()
        self.directory.cleanup()

    def store(self, record_id: str, manager_id: str, desired_state: str = "running"):
        # The alias differs from the served name, as it does for a card named
        # "GLM-5.3-Flash-FP4 TP2 (...)" serving "glm-5.3-flash".
        self.service.store.add_deployment(Deployment(
            id=record_id, alias=f"{record_id} alias", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity(f"/models/{record_id}"),
            settings={"manager_deployment_id": manager_id},
            desired_state=desired_state,
        ))

    async def proxy(self):
        return await self.service.proxy(
            {"model": SERVED, "messages": [], "stream": False},
            "chat/completions",
        )

    async def test_blip_statuses_keep_routing_to_the_cluster_deployment(self):
        self.store("glm", "cluster-glm")
        for status in ("degraded", "unknown", "starting"):
            with self.subTest(status=status):
                self.manager.proxy_cluster_inference.reset_mock()
                self.service.deployments = AsyncMock(return_value=[
                    observed("glm", "cluster-glm", status),
                ])

                response = await self.proxy()

                self.assertEqual(response["model"], SERVED)
                self.assertEqual(
                    self.manager.proxy_cluster_inference.await_args.args[0],
                    "cluster-glm",
                )
                self.manager.proxy_chat_completions.assert_not_awaited()

    async def test_live_owner_still_wins_over_a_blipped_one(self):
        self.store("blipped", "cluster-blipped")
        self.store("live", "cluster-live")
        self.service.deployments = AsyncMock(return_value=[
            observed("blipped", "cluster-blipped", "degraded",
                     last_deployed_at="2026-11-05T00:00:00+00:00"),
            observed("live", "cluster-live", "running",
                     last_deployed_at="2026-01-01T00:00:00+00:00"),
        ])

        await self.proxy()

        self.assertEqual(
            self.manager.proxy_cluster_inference.await_args.args[0],
            "cluster-live",
        )

    async def test_stop_intent_and_confirmed_states_keep_the_legacy_path(self):
        # Only a deployment that is meant to be serving and merely cannot be
        # observed right now keeps the name; an explicit stop, a confirmed
        # stop, or a launch error stays on the unchanged legacy behavior.
        cases = [
            ("stopped-intent", "degraded", "stopped"),
            ("confirmed-stop", "stopped", "running"),
            ("launch-error", "error", "running"),
        ]
        for record_id, status, desired_state in cases:
            with self.subTest(case=record_id):
                self.store(record_id, f"cluster-{record_id}", desired_state)
                self.service.deployments = AsyncMock(return_value=[
                    observed(record_id, f"cluster-{record_id}", status,
                             desired_state=desired_state),
                ])
                self.manager.proxy_chat_completions.reset_mock()

                with self.assertRaisesRegex(LookupError, "No managed container"):
                    await self.proxy()

                self.manager.proxy_cluster_inference.assert_not_awaited()
                self.manager.proxy_chat_completions.assert_awaited_once()
                self.service.store.delete_deployment(record_id)

    async def test_policy_priority_routes_transient_owners(self):
        # Both owners are transiently unobserved, so the live selectors
        # never saw them; the configured policy's priority must still
        # decide instead of the newest blipped owner winning.
        self.store("glm-a", "cluster-a")
        self.store("glm-b", "cluster-b")
        self.service.deployments = AsyncMock(return_value=[
            observed("glm-a", "cluster-a", "degraded",
                     last_deployed_at="2026-01-01T00:00:00+00:00"),
            observed("glm-b", "cluster-b", "degraded",
                     last_deployed_at="2026-11-05T00:00:00+00:00"),
        ])
        self.service._model_routing_policies = {
            SERVED: {
                "model": SERVED,
                "members": [
                    {"deployment_id": "glm-a", "max_concurrency": None},
                    {"deployment_id": "glm-b", "max_concurrency": None},
                ],
                "updated_at": None,
            },
        }

        await self.proxy()

        self.assertEqual(
            self.manager.proxy_cluster_inference.await_args.args[0],
            "cluster-a",
        )

    async def test_policy_cap_overflows_off_a_saturated_transient_owner(self):
        self.store("glm-a", "cluster-a")
        self.store("glm-b", "cluster-b")
        self.service.deployments = AsyncMock(return_value=[
            observed("glm-a", "cluster-a", "degraded"),
            observed("glm-b", "cluster-b", "degraded"),
        ])
        self.service._model_routing_policies = {
            SERVED: {
                "model": SERVED,
                "members": [
                    {"deployment_id": "glm-a", "max_concurrency": 1},
                    {"deployment_id": "glm-b", "max_concurrency": None},
                ],
                "updated_at": None,
            },
        }
        self.service._model_routing_inflight["glm-a"] = 1

        await self.proxy()

        self.assertEqual(
            self.manager.proxy_cluster_inference.await_args.args[0],
            "cluster-b",
        )

    async def test_transient_owners_balance_without_a_policy(self):
        # Without a configured policy the owners share the request by
        # least-in-flight balancing, not by stacking on the newest one.
        self.store("glm-a", "cluster-a")
        self.store("glm-b", "cluster-b")
        self.service.deployments = AsyncMock(return_value=[
            observed("glm-a", "cluster-a", "degraded"),
            observed("glm-b", "cluster-b", "degraded"),
        ])

        await self.proxy()
        await self.proxy()

        targets = {
            call.args[0]
            for call in self.manager.proxy_cluster_inference.await_args_list
        }
        self.assertEqual(targets, {"cluster-a", "cluster-b"})


def single_remote_deployment() -> dict:
    return {
        "id": "glm", "model": "/models/glm", "mode": "single",
        "launch_settings": {},
        "members": [member(0, "threadripper", "glm-r0")],
    }


def triple_remote_deployment() -> dict:
    return {
        "id": "glm", "model": "/models/glm", "mode": "replicated",
        "launch_settings": {},
        "members": [
            member(0, "node-a", "glm-r0"),
            member(1, "node-b", "glm-r1"),
            member(2, "node-c", "glm-r2"),
        ],
    }


def dual_remote_deployment() -> dict:
    return {
        "id": "glm", "model": "/models/glm", "mode": "replicated",
        "launch_settings": {},
        "members": [
            member(0, "node-a", "glm-r0"),
            member(1, "node-b", "glm-r1"),
        ],
    }


def connect_failure(message: str = "could not contact threadripper") -> RuntimeError:
    error = RuntimeError(message)
    error.__cause__ = httpx.ConnectTimeout("connect timed out")
    return error


class ClusterRideThroughTests(unittest.IsolatedAsyncioTestCase):
    def manager(self, *, window: float = 30.0, pause: bool = False) -> Manager:
        manager = build_manager(single_remote_deployment())
        manager._cluster_ride_through_seconds = window
        if not pause:
            manager._cluster_ride_through_pause = AsyncMock()
        return manager

    async def proxy(self, manager: Manager, cancel=None, **overrides):
        body = {"model": "/models/glm", "messages": [], "stream": False, **overrides}
        return await manager.proxy_cluster_inference(
            "glm", "/models/glm", body, "chat/completions", cancel,
        )

    def forwarded(self, manager: Manager) -> int:
        return sum(
            1 for call in manager.node_registry.request.await_args_list
            if call.args[2] != HEALTH_PATH
        )

    def assert_no_reservation_leaked(self, manager: Manager) -> None:
        # Each pass reserves the member for balancing and must release it.
        member = manager.deployments[0]["members"][0]
        self.assertEqual(manager._cluster_member_active("glm", member), 0)

    async def test_unreachable_node_is_ridden_through_until_it_answers(self):
        manager = self.manager()
        manager.node_registry.request = AsyncMock(side_effect=[
            connect_failure(),
            {"ready": False},
            {"ready": True},
            {"choices": [], "ok": True},
        ])

        result = await self.proxy(manager)

        self.assertEqual(result, {"choices": [], "ok": True})
        self.assertEqual(manager._cluster_ride_through_pause.await_count, 2)
        self.assertEqual(self.forwarded(manager), 1)
        self.assert_no_reservation_leaked(manager)

    async def test_streams_are_ridden_through_before_any_byte_is_sent(self):
        manager = self.manager()
        manager.node_registry.request = AsyncMock(side_effect=[
            {"ready": False}, {"ready": True},
        ])
        opened = AsyncMock()
        opened.status_code = 200

        async def lines():
            yield 'data: {"choices": []}'
            yield "data: [DONE]"

        opened.aiter_lines = lines
        manager.node_registry.open_stream = AsyncMock(return_value=opened)
        manager._close_inference_transport = AsyncMock()

        stream = await self.proxy(manager, stream=True)
        chunks = [chunk async for chunk in stream]

        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        manager._cluster_ride_through_pause.assert_awaited_once()
        manager.node_registry.open_stream.assert_awaited_once()

    async def test_a_connect_failure_on_dispatch_is_ridden_through(self):
        # The request never reached the engine, so replaying it is safe.
        manager = self.manager()
        manager.node_registry.request = AsyncMock(side_effect=[
            {"ready": True},
            connect_failure(),
            {"ready": True},
            {"choices": [], "ok": True},
        ])

        result = await self.proxy(manager)

        self.assertEqual(result, {"choices": [], "ok": True})
        manager._cluster_ride_through_pause.assert_awaited_once()

    async def test_a_failure_after_dispatch_is_not_replayed(self):
        # A read timeout means the engine may already be generating; the
        # ride-through must never send the same prompt twice.
        manager = self.manager()
        dispatched = RuntimeError("could not contact threadripper: read timed out")
        dispatched.__cause__ = httpx.ReadTimeout("read timed out")
        manager.node_registry.request = AsyncMock(side_effect=[
            {"ready": True}, dispatched,
        ])

        with self.assertRaisesRegex(RuntimeError, "read timed out"):
            await self.proxy(manager)

        manager._cluster_ride_through_pause.assert_not_awaited()

    async def test_deterministic_rejections_fail_immediately(self):
        manager = self.manager()
        manager.node_registry.request = AsyncMock(side_effect=[
            {"ready": True},
            RuntimeError("Node 0 agent error: HTTP 400: invalid parameters"),
        ])

        with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
            await self.proxy(manager)

        manager._cluster_ride_through_pause.assert_not_awaited()

    async def test_stopped_deployment_fails_immediately(self):
        manager = self.manager()
        manager.deployments[0]["desired_state"] = "stopped"

        with self.assertRaisesRegex(RuntimeError, "deployment is stopped"):
            await self.proxy(manager)

        manager._cluster_ride_through_pause.assert_not_awaited()

    async def test_gives_up_with_a_retryable_error_after_the_window(self):
        manager = self.manager(window=0.05, pause=True)
        manager.node_registry.request = AsyncMock(return_value={"ready": False})

        with self.assertRaisesRegex(ClusterReplicaUnavailable, "not ready"):
            await asyncio.wait_for(self.proxy(manager), 2)

        self.assertGreaterEqual(manager.node_registry.request.await_count, 2)
        self.assertEqual(self.forwarded(manager), 0)
        self.assert_no_reservation_leaked(manager)

    async def test_zero_window_keeps_single_pass_behavior(self):
        manager = self.manager(window=0)
        manager.node_registry.request = AsyncMock(return_value={"ready": False})

        with self.assertRaises(ClusterReplicaUnavailable):
            await self.proxy(manager)

        manager._cluster_ride_through_pause.assert_not_awaited()
        self.assertEqual(manager.node_registry.request.await_count, 1)

    async def test_client_disconnect_ends_the_wait(self):
        manager = self.manager(window=30.0, pause=True)
        cancel = asyncio.Event()

        async def unready(*args, **kwargs):
            asyncio.get_running_loop().call_later(0.01, cancel.set)
            return {"ready": False}

        manager.node_registry.request = AsyncMock(side_effect=unready)

        with self.assertRaises(ClientAbort):
            await asyncio.wait_for(self.proxy(manager, cancel=cancel), 2)
        self.assert_no_reservation_leaked(manager)

    async def test_a_deterministic_probe_rejection_fails_fast(self):
        # The agent rejecting the readiness probe with a client error (bad
        # credentials) answers the same way on every retry; the ride-through
        # must surface that cause instead of waiting the window out.
        manager = self.manager()
        manager.node_registry.request = AsyncMock(side_effect=[
            NodeAgentResponseError("Node 0", 401, "invalid credentials"),
        ])

        with self.assertRaisesRegex(ClusterReplicaUnavailable, "HTTP 401"):
            await self.proxy(manager)

        manager._cluster_ride_through_pause.assert_not_awaited()
        self.assertEqual(manager.node_registry.request.await_count, 1)
        self.assert_no_reservation_leaked(manager)

    async def test_a_missing_container_probe_is_ridden_through(self):
        # The agent's typed missing-container 404 is transient (the
        # container may be restarting), so it still rides the window out.
        manager = self.manager()
        manager.node_registry.request = AsyncMock(side_effect=[
            NodeAgentResponseError("Node 0", 404, json.dumps({
                "detail": {
                    "type": "replica_unavailable",
                    "message": "No managed container found for glm-r0",
                },
            })),
            {"ready": True},
            {"choices": [], "ok": True},
        ])

        result = await self.proxy(manager)

        self.assertEqual(result, {"choices": [], "ok": True})
        manager._cluster_ride_through_pause.assert_awaited_once()

    async def test_the_window_bounds_the_route_pass_itself(self):
        # Three members whose probes each hang past the window: the pass
        # must give up when the window closes instead of finishing the
        # whole route first, and each probe is bounded by the remaining
        # time rather than its full 10-second timeout.
        manager = build_manager(triple_remote_deployment())
        manager._cluster_ride_through_seconds = 0.5
        manager._cluster_ride_through_pause = AsyncMock()
        started = time.monotonic()

        async def blackholed(node_id, method, path, *, json_body=None, timeout=30):
            await asyncio.sleep(0.3)
            raise RuntimeError("could not contact node")

        manager.node_registry.request = AsyncMock(side_effect=blackholed)

        with self.assertRaisesRegex(
            ClusterReplicaUnavailable, "ride-through window",
        ):
            await self.proxy(manager)

        self.assertLess(time.monotonic() - started, 0.9)
        self.assertEqual(manager.node_registry.request.await_count, 2)
        self.assertLess(
            manager.node_registry.request.await_args_list[1].kwargs["timeout"],
            0.5,
        )
        self.assert_no_reservation_leaked(manager)

    async def test_a_zero_window_still_probes_every_member(self):
        # With ride-through off, the legacy single pass still probes every
        # candidate before reporting the outage.
        manager = build_manager(triple_remote_deployment())
        manager._cluster_ride_through_seconds = 0
        manager.node_registry.request = AsyncMock(return_value={"ready": False})

        with self.assertRaisesRegex(ClusterReplicaUnavailable, "not ready"):
            await self.proxy(manager)

        self.assertEqual(manager.node_registry.request.await_count, 3)
        self.assert_no_reservation_leaked(manager)

    async def test_a_disabled_node_fails_fast(self):
        # The registry refuses a disabled node's request before dialing;
        # waiting out the window cannot bring it back, so the cause must
        # surface immediately.
        manager = self.manager()
        manager.node_registry.request = AsyncMock(side_effect=[
            ValueError("node threadripper is disabled"),
        ])

        with self.assertRaisesRegex(ClusterReplicaUnavailable, "disabled"):
            await self.proxy(manager)

        manager._cluster_ride_through_pause.assert_not_awaited()
        self.assertEqual(manager.node_registry.request.await_count, 1)
        self.assert_no_reservation_leaked(manager)

    async def test_a_removed_node_fails_fast(self):
        manager = self.manager()
        manager.node_registry.request = AsyncMock(side_effect=[
            ValueError("remote node not found"),
        ])

        with self.assertRaisesRegex(
            ClusterReplicaUnavailable, "remote node not found",
        ):
            await self.proxy(manager)

        manager._cluster_ride_through_pause.assert_not_awaited()
        self.assertEqual(manager.node_registry.request.await_count, 1)

    async def test_a_post_dispatch_failure_is_not_replayed_on_a_later_probe(self):
        # Replica 0's dispatch fails after the prompt was sent (read
        # timeout), then replica 1's probe failure ends the pass: the
        # ride-through must not replay the route, because the first engine
        # may still be generating the prompt.
        manager = build_manager(dual_remote_deployment())
        manager._cluster_ride_through_pause = AsyncMock()
        dispatched = RuntimeError("could not contact node-a: read timed out")
        dispatched.__cause__ = httpx.ReadTimeout("read timed out")
        manager.node_registry.request = AsyncMock(side_effect=[
            {"ready": True},
            dispatched,
            {"ready": False},
        ])

        with self.assertRaisesRegex(ClusterReplicaUnavailable, "not ready"):
            await self.proxy(manager)

        manager._cluster_ride_through_pause.assert_not_awaited()
        self.assertEqual(self.forwarded(manager), 1)
        self.assert_no_reservation_leaked(manager)


if __name__ == "__main__":
    unittest.main()
