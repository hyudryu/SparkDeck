import asyncio
import contextvars
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

from manager import Manager
from sparkdeck.service import SparkDeckService

from sparkdeck.startup_benchmark import StartupBenchmarkMonitor, StartupTarget, startup_fingerprint


class StartupBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = {}
        self.snapshot = {"enabled": True, "generation": 1}
        self.store = SimpleNamespace(
            community_consent_snapshot=lambda: dict(self.snapshot),
            get_setting=lambda key, default=None: self.settings.get(key, default),
            set_setting=lambda key, value: self.settings.__setitem__(key, value),
            deployment=Mock(return_value=None),
        )
        self.manager = SimpleNamespace(
            deployments=[], cluster_nodes=AsyncMock(return_value=[]),
            list_containers=AsyncMock(return_value=[]),
        )
        self.service = SimpleNamespace(
            manager=self.manager, store=self.store,
            deployments=AsyncMock(return_value=[]),
            _community_observation_scopes=Mock(return_value=frozenset({"node:local"})),
            _community_active_observations={},
            _community_observation_start=Mock(return_value={"enabled": True}),
            _community_observation_end=Mock(),
            _community_observation=contextvars.ContextVar("test_observation", default=None),
        )
        self.monitor = StartupBenchmarkMonitor(self.service)
        self.deployment = {
            "id": "deployment", "kind": "managed", "runtime": "vllm",
            "model": {"repository": "org/model"}, "container_name": "engine", "settings": {},
        }
        self.target = StartupTarget(self.deployment, "boot1")

    async def test_opted_out_does_not_discover_or_probe(self):
        self.snapshot["enabled"] = False
        await self.monitor.tick()
        self.service.deployments.assert_not_awaited()

    async def test_unhealthy_engine_retried_then_benchmarked_only_once(self):
        self.monitor.targets = AsyncMock(return_value=[self.target])
        self.monitor._healthy = AsyncMock(return_value=False)
        self.monitor._benchmark = AsyncMock()
        await self.monitor._attempt(self.target, dict(self.snapshot))
        self.monitor._benchmark.assert_not_awaited()
        self.assertFalse(self.settings.get(
            self.monitor._seen_key(self.target.fingerprint), False,
        ))
        self.monitor._healthy.return_value = True
        await self.monitor.tick()
        await asyncio.gather(*list(self.monitor._tasks.values()))
        await asyncio.sleep(0)
        await self.monitor.tick()
        self.monitor._benchmark.assert_awaited_once()
        # A recorded boot persists its seen marker atomically with the sample;
        # simulate that storage commit so the boot identity survives restart.
        self.service.store.set_setting(self.monitor._seen_key(self.target.fingerprint), True)
        restarted = StartupBenchmarkMonitor(self.service)
        restarted.targets = AsyncMock(return_value=[self.target])
        await restarted.tick()
        self.assertFalse(restarted._tasks)

    async def test_queued_boot_replaced_or_consent_changed_is_not_measured(self):
        self.monitor._healthy = AsyncMock(return_value=True)
        self.monitor.targets = AsyncMock(return_value=[StartupTarget(self.deployment, "boot2")])
        self.monitor._benchmark = AsyncMock()
        await self.monitor._attempt(self.target, dict(self.snapshot))
        self.monitor.targets.return_value = [self.target]
        await self.monitor._attempt(self.target, {"enabled": True, "generation": 0})
        self.monitor._benchmark.assert_not_awaited()
        self.assertFalse(self.settings)

    async def test_busy_user_hardware_defers_benchmark(self):
        self.monitor.targets = AsyncMock(return_value=[self.target])
        self.monitor._healthy = AsyncMock(return_value=True)
        self.monitor._benchmark = AsyncMock()
        self.service._community_active_observations["user"] = {"scopes": {"node:local"}}
        await self.monitor._attempt(self.target, dict(self.snapshot))
        self.monitor._benchmark.assert_not_awaited()
        self.assertFalse(self.settings)

    async def test_manual_benchmark_busy_callback_defers_startup_request(self):
        self.monitor.targets = AsyncMock(return_value=[self.target])
        self.monitor._healthy = AsyncMock(return_value=True)
        self.monitor._benchmark = AsyncMock()
        self.service._startup_benchmark_busy = lambda: True
        await self.monitor._attempt(self.target, dict(self.snapshot))
        self.monitor._benchmark.assert_not_awaited()
        self.assertFalse(self.settings)

    async def test_request_is_200_tokens_no_context_and_stream_is_closed(self):
        observed = []
        closed = []
        async def stream():
            try:
                observed.append(dict(self.service._community_observation.get()))
                yield "data: [DONE]\n\n"
            finally:
                closed.append(True)
        self.service._proxy_registered = AsyncMock(return_value=stream())
        await self.monitor._benchmark(self.target, self.snapshot)
        body = self.service._proxy_registered.call_args.args[1]
        self.assertEqual(body["max_tokens"], 200)
        self.assertTrue(body["ignore_eos"])
        self.assertTrue(body["stream"])
        self.assertNotIn("messages", body)
        self.assertTrue(self.service._proxy_registered.call_args.kwargs["startup_benchmark"])
        self.assertTrue(observed[0]["startup_benchmark"])
        self.assertEqual(observed[0]["generation"], 1)
        self.assertEqual(closed, [True])
        self.assertIsNone(self.service._community_observation.get())
        self.service._community_observation_end.assert_called_once()
    async def test_real_manager_health_requires_200_without_models_fallback(self):
        manager = Manager.__new__(Manager)
        manager.deployments = []
        container = {"name": "engine", "model": "org/model", "status": "running", "port": 8000}
        manager.list_containers = AsyncMock(return_value=[container])
        manager.ensure_loaded = AsyncMock()
        paths = []
        health_status = 503
        def respond(request):
            paths.append(request.url.path)
            return httpx.Response(health_status if request.url.path == "/health" else 200)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            manager.http = client
            self.assertFalse(await manager.inference_target_health(
                "org/model", container_name="engine", strict_health=True,
            ))
            self.assertEqual(paths, ["/health"])
            paths.clear()
            self.assertTrue(await manager.inference_target_health("org/model", container_name="engine"))
            self.assertEqual(paths, ["/health", "/v1/models"])
            paths.clear()
            health_status = 200
            self.assertTrue(await manager.inference_target_health(
                "org/model", container_name="engine", strict_health=True,
            ))
            self.assertEqual(paths, ["/health"])
        manager.ensure_loaded.assert_not_awaited()

    async def test_real_local_llama_stream_adds_one_trusted_startup_sample(self):
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[{"text":"garden"}]}\n\n'
                await asyncio.sleep(0.01)
                yield b'data: {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":200}}\n\n'
                yield b'data: [DONE]\n\n'

        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(200, stream=Stream())
        manager = Manager.__new__(Manager)
        manager.deployments = []
        manager._stats_cache = {"gpus": [{"name": "NVIDIA GB10", "mem_total_mib": 128000}]}
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            manager.http = client
            with tempfile.TemporaryDirectory() as directory:
                service = SparkDeckService(manager, Path(directory))
                try:
                    service.store.set_community_consent(True)
                    deployment = {
                        **self.deployment, "runtime": "llama.cpp", "alias": "garden",
                        "_base_url": "http://localhost:8080", "settings": {"tensor_parallel_size": 1},
                    }
                    await StartupBenchmarkMonitor(service)._benchmark(
                        StartupTarget(deployment, "boot"), service.store.community_consent_snapshot(),
                    )
                    samples, total = service.store.benchmarks()
                    self.assertEqual(total, 1)
                    self.assertEqual(samples[0]["output_tokens"], 200)
                    self.assertTrue(samples[0]["eligible_for_community"])
                    self.assertGreater(samples[0]["generation_tokens_per_second"], 0)
                    self.assertEqual(requests[0].url.path, "/v1/completions")
                finally:
                    await service.close()

    async def test_agent_strict_health_envelope_proves_health_200(self):
        import server

        class Request:
            headers = {}
            async def stream(self):
                yield json.dumps({
                    "model": "org/model", "_sparkdeck_container_name": "rank-0",
                    "_sparkdeck_deployment_id": "cluster", "strict_health": True,
                }).encode()

        health = AsyncMock(return_value=False)
        with patch.object(server, "_require_agent"), patch.object(server.manager, "inference_target_health", health):
            result = await server.agent_inference_health(Request())
            self.assertEqual(result, {"ready": False, "model": "org/model", "health_status": None})
            health.assert_awaited_once_with(
                "org/model", container_name="rank-0", deployment_id="cluster", strict_health=True,
            )
            health.return_value = True
            result = await server.agent_inference_health(Request())
            self.assertEqual(result, {"ready": True, "model": "org/model", "health_status": 200})

    async def test_agent_inference_forwards_remote_startup_marker(self):
        import server

        class Request:
            headers = {}

        async def completions(model, body, stream, cancel, **kwargs):
            return {"choices": [], "usage": {}}

        with patch.object(server, "_require_agent"), \
             patch.object(server, "read_limited_json", return_value={
                 "model": "org/model", "stream": False,
                 "_sparkdeck_startup_benchmark": True,
                 "_sparkdeck_container_name": "rank-0",
             }), \
             patch.object(server.manager, "_vllm_completions",
                          AsyncMock(side_effect=completions)) as manager_call:
            result = await server.agent_inference("completions", Request())
        self.assertEqual(result["choices"], [])
        self.assertEqual(manager_call.call_args.kwargs["startup_benchmark"], True)
        # The internal marker is consumed by the agent, not forwarded upstream.
        forwarded_body = manager_call.call_args.args[1]
        self.assertNotIn("_sparkdeck_startup_benchmark", forwarded_body)
        self.assertEqual(forwarded_body["model"], "org/model")

    async def test_grouped_inventory_selects_coordinators_and_tracks_all_rank_boots(self):
        self.deployment["settings"]["manager_deployment_id"] = "cluster"
        members = [
            {"node_id": f"n{instance}{rank}", "container_name": f"engine{instance}{rank}",
             "rank": rank, "instance_id": instance}
            for instance in (0, 1) for rank in (0, 1)
        ]
        cluster = {"id": "cluster", "mode": "grouped_sharded", "members": members}
        self.manager.deployments = [cluster]
        self.manager._cluster_members_sorted = lambda cluster: cluster["members"]
        self.manager._grouped_coordinators = lambda cluster: [m for m in members if m["rank"] == 0]
        nodes = [{"id": m["node_id"], "online": True, "containers": [{
            "name": m["container_name"], "status": "running", "started_at": "boot1",
        }]} for m in members]
        self.manager.cluster_nodes.return_value = nodes
        self.service.deployments.return_value = [self.deployment]
        before = await self.monitor.targets()
        self.assertEqual([t.member["instance_id"] for t in before], [0, 1])
        nodes[1]["containers"][0]["started_at"] = "boot2"
        after = await self.monitor.targets()
        self.assertNotEqual(before[0].fingerprint, after[0].fingerprint)
        self.assertEqual(before[1].fingerprint, after[1].fingerprint)

    async def test_cluster_benchmark_routes_to_exact_new_member_with_admission(self):
        member = {"node_id": "remote", "container_name": "new-replica"}
        cluster = {"id": "cluster"}
        target = StartupTarget(self.deployment, "boot", cluster, member)
        async def upstream():
            yield "data: [DONE]\n\n"
        self.manager._proxy_cluster_member = AsyncMock(return_value=upstream())
        self.service._model_observation_settings = Mock(return_value={"tensor_parallel_size": 2})
        self.service._observe_stream = Mock(side_effect=lambda stream, *args, **kwargs: stream)
        await self.monitor._benchmark(target, self.snapshot)
        args = self.manager._proxy_cluster_member.call_args.args
        self.assertIs(args[0], cluster)
        self.assertIs(args[1], member)
        self.assertEqual(args[4], "completions")
        self.assertIn("hardware_resolver", self.service._observe_stream.call_args.kwargs)

    def test_fingerprint_requires_start_evidence_and_changes_on_restart(self):
        self.assertIsNone(startup_fingerprint([("local", {"name": "engine"})]))
        before = [("local", {"name": "engine", "started_at": "first", "restart_count": 0})]
        after = [("local", {"name": "engine", "started_at": "second", "restart_count": 1})]
        self.assertNotEqual(startup_fingerprint(before), startup_fingerprint(after))

    async def test_agent_does_not_benchmark_controller_owned_rank(self):
        self.deployment["id"] = "container:engine"
        self.service.deployments.return_value = [self.deployment]
        self.manager.list_containers.return_value = [{
            "name": "engine", "deployment_id": "controller-deployment",
            "status": "running", "started_at": "boot1",
        }]
        self.assertEqual(await self.monitor.targets(), [])

    async def test_local_health_probe_is_read_only_and_requires_health_200(self):
        self.manager.inference_target_health = AsyncMock(return_value=False)
        self.assertFalse(await self.monitor._healthy(self.target))
        self.manager.inference_target_health.assert_awaited_once_with(
            "org/model", container_name="engine", deployment_id="deployment", strict_health=True,
        )

    async def test_remote_health_fails_closed_for_old_agent_models_fallback(self):
        self.manager.node_registry = SimpleNamespace(request=AsyncMock(return_value={"ready": True}))
        target = StartupTarget(self.deployment, "boot", {"id": "cluster"},
                               {"node_id": "remote", "container_name": "new"})
        self.assertFalse(await self.monitor._healthy(target))
        self.manager.node_registry.request.return_value = {"ready": True, "health_status": 200}
        self.assertTrue(await self.monitor._healthy(target))

    async def test_cancelling_benchmark_closes_stream_and_observation(self):
        entered = asyncio.Event()
        closed = []
        async def stream():
            try:
                entered.set()
                await asyncio.Event().wait()
                yield "never"
            finally:
                closed.append(True)
        self.service._proxy_registered = AsyncMock(return_value=stream())
        task = asyncio.create_task(self.monitor._benchmark(self.target, self.snapshot))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(closed, [True])
        self.service._community_observation_end.assert_called_once()

    async def test_no_sample_recording_enters_retry_backoff(self):
        self.monitor.targets = AsyncMock(return_value=[self.target])
        self.monitor._healthy = AsyncMock(return_value=True)
        # A benchmark that records no eligible sample must not consume the boot.
        self.monitor._benchmark = AsyncMock(return_value=False)
        await self.monitor._attempt(self.target, dict(self.snapshot))
        self.assertFalse(self.settings.get(
            self.monitor._seen_key(self.target.fingerprint), False,
        ))
        self.assertIn(self.target.fingerprint, self.monitor._retry_after)
        persisted = self.service.store.get_setting(
            self.monitor._retry_key(self.target.fingerprint), {},
        )
        self.assertEqual(persisted["attempts"], 1)
        # While the fingerprint backs off, it is not re-attempted on the next tick.
        self.monitor._benchmark.reset_mock()
        self.monitor.targets = AsyncMock(return_value=[self.target])
        self.assertFalse(await self.monitor.tick())
        self.monitor._benchmark.assert_not_awaited()
        self.assertFalse(self.monitor._tasks)
        # A successful record clears the retry state instead of re-queuing.
        self.monitor._benchmark = AsyncMock(return_value=True)
        await self.monitor._attempt(self.target, dict(self.snapshot))
        self.assertNotIn(self.target.fingerprint, self.monitor._retry_after)
        self.assertIsNone(self.service.store.get_setting(
            self.monitor._retry_key(self.target.fingerprint),
        ))

    async def test_failed_probe_persists_backoff_across_monitor_restart(self):
        self.monitor._healthy = AsyncMock(return_value=True)
        self.monitor.targets = AsyncMock(return_value=[self.target])
        self.monitor._benchmark = AsyncMock(side_effect=TimeoutError("probe timed out"))
        await self.monitor._attempt(self.target, dict(self.snapshot))

        restarted = StartupBenchmarkMonitor(self.service)
        restarted.targets = AsyncMock(return_value=[self.target])
        self.assertFalse(await restarted.tick())
        self.assertFalse(restarted._tasks)

    async def test_benchmark_returns_whether_a_sample_was_recorded(self):
        observed = []
        closed = []
        async def stream():
            try:
                obs = self.service._community_observation.get()
                obs["startup_recorded"] = True
                observed.append(dict(obs))
                yield "data: [DONE]\n\n"
            finally:
                closed.append(True)
        self.service._proxy_registered = AsyncMock(return_value=stream())
        self.assertTrue(await self.monitor._benchmark(self.target, self.snapshot))
        self.assertTrue(observed[0].get("startup_recorded"))
        self.assertEqual(closed, [True])
        self.assertIsNone(self.service._community_observation.get())
        self.service._community_observation_end.assert_called_once()

    def test_startup_probe_suppresses_ordinary_usage_persistence(self):
        manager = Manager.__new__(Manager)
        manager._record_tokens = Mock()
        manager._mark_deployment_used = Mock()
        manager.deployments = []
        manager._stats_cache = {}
        # A synthetic startup probe must not pollute the token/throughput history.
        manager._record_usage(
            "model", {"prompt_tokens": 10, "completion_tokens": 200}, 1.0, 1.0,
            startup_benchmark=True,
        )
        manager._record_tokens.assert_not_called()
        # Ordinary requests still persist.
        manager._record_usage("model", {"prompt_tokens": 10, "completion_tokens": 5}, 1.0, 1.0)
        manager._record_tokens.assert_called_once()
        # A startup probe must not refresh the deployment last-used timestamp.
        startup_rid = manager._track_start("model", deployment_id="dep", startup_benchmark=True)
        manager._track_end(startup_rid)
        manager._mark_deployment_used.assert_not_called()
        # Ordinary requests still refresh at completion.
        manager._track_start("model", deployment_id="dep")
        manager._mark_deployment_used.assert_called()

    async def test_cancel_active_cancels_inflight_probes(self):
        started = asyncio.Event()
        async def idle():
            started.set()
            await asyncio.Event().wait()
        task = asyncio.create_task(idle())
        self.monitor._tasks["boot"] = task
        await started.wait()
        self.monitor.cancel_active()
        await asyncio.sleep(0)
        self.assertTrue(task.cancelled())
        self.assertIn("boot", self.monitor._tasks)

    async def test_tick_reports_progress_only_for_unseen_boots(self):
        self.monitor.targets = AsyncMock(return_value=[self.target])
        self.assertTrue(await self.monitor.tick())
        for task in list(self.monitor._tasks.values()):
            task.cancel()
        await asyncio.gather(*list(self.monitor._tasks.values()), return_exceptions=True)
        # A fresh monitor where the boot is already seen reports no progress.
        restarted = StartupBenchmarkMonitor(self.service)
        restarted.targets = AsyncMock(return_value=[self.target])
        self.service.store.set_setting(restarted._seen_key(self.target.fingerprint), True)
        self.assertFalse(await restarted.tick())
        self.assertFalse(restarted._tasks)
