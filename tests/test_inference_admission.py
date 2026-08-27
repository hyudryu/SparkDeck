import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx

from manager import ClientAbort, Manager


def container(limit: int = 1, deployment_id: str = "deployment-a") -> dict:
    return {
        "name": f"cluster-{deployment_id}-r0-model",
        "deployment_id": deployment_id,
        "model": "model",
        "served_model": "model",
        "served_models": ["model"],
        "stats_key": "model [test]",
        "port": 8000,
        "load_settings": {"max_concurrency": limit},
    }


class InferenceAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def manager(self) -> Manager:
        instance = Manager.__new__(Manager)
        instance._inference_admission = {}
        return instance

    async def test_excess_requests_wait_fifo(self) -> None:
        instance = self.manager()
        target = await instance._acquire_inference_slot(
            container(), "model [test]", None,
        )
        second = asyncio.create_task(instance._acquire_inference_slot(
            container(), "model [test]", None,
        ))
        third = asyncio.create_task(instance._acquire_inference_slot(
            container(), "model [test]", None,
        ))
        await asyncio.sleep(0)

        self.assertEqual(instance.inference_admission()["deployment-a"], {
            "model": "model [test]",
            "limit": 1,
            "running": 1,
            "queued": 2,
            "oldest_wait_seconds": 0.0,
        })

        instance._release_inference_slot(target)
        self.assertEqual(await second, "deployment-a")
        self.assertFalse(third.done())
        instance._release_inference_slot("deployment-a")
        self.assertEqual(await third, "deployment-a")
        instance._release_inference_slot("deployment-a")
        self.assertEqual(instance.inference_admission(), {})

    async def test_disconnected_queued_request_is_removed(self) -> None:
        instance = self.manager()
        target = await instance._acquire_inference_slot(container(), "model", None)
        cancel = asyncio.Event()
        queued = asyncio.create_task(instance._acquire_inference_slot(
            container(), "model", cancel,
        ))
        await asyncio.sleep(0)
        cancel.set()

        with self.assertRaises(ClientAbort):
            await queued
        self.assertEqual(
            instance.inference_admission()["deployment-a"]["queued"], 0,
        )
        instance._release_inference_slot(target)

    async def test_deployments_have_independent_limits(self) -> None:
        instance = self.manager()

        first = await instance._acquire_inference_slot(
            container(deployment_id="a"), "model-a", None,
        )
        second = await instance._acquire_inference_slot(
            container(deployment_id="b"), "model-b", None,
        )

        self.assertEqual(first, "a")
        self.assertEqual(second, "b")
        self.assertEqual(instance.inference_admission()["a"]["running"], 1)
        self.assertEqual(instance.inference_admission()["b"]["running"], 1)
        instance._release_inference_slot(first)
        instance._release_inference_slot(second)

    async def test_active_requests_includes_admitted_non_streaming_work(self) -> None:
        instance = self.manager()
        instance._req_seq = 0
        instance._active_reqs = {}
        instance._trailing_window = 5.0
        target = await instance._acquire_inference_slot(
            container(), "model [test]", None,
        )

        rates = instance.active_requests()

        self.assertEqual(rates["model [test]"]["connections"], 1)
        instance._release_inference_slot(target)

    async def test_target_is_refreshed_after_admission(self) -> None:
        instance = self.manager()
        old = container()
        fresh = {**old, "port": 8123}
        instance._resolve_vllm_target = mock.AsyncMock(side_effect=[old, fresh])
        instance._req_seq = 0
        instance._active_reqs = {}
        instance._trailing_window = 5.0
        instance._record_usage = mock.Mock()
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [], "usage": {}}
        instance.http = SimpleNamespace(post=mock.AsyncMock(return_value=response))

        await instance._vllm_chat(
            "model", {"model": "model", "messages": []}, stream=False,
        )

        self.assertIn(":8123/", instance.http.post.await_args.args[0])

    async def test_sparkrun_parallel_stream_limit_reaches_admission(self) -> None:
        instance = self.manager()
        instance.spark_runs = {
            "run": {
                "status": "running", "started_at": 10,
                "max_concurrency": 3,
            },
        }
        docker_container = SimpleNamespace(name="sparkrun_example_solo")
        instance.client = SimpleNamespace(
            containers=SimpleNamespace(list=lambda **kwargs: [docker_container])
        )
        instance._container_summary = mock.Mock(return_value={
            "name": docker_container.name, "load_settings": {},
        })
        response = mock.Mock(status_code=200)
        response.json.return_value = {"data": [{"id": "model"}]}
        instance.http = SimpleNamespace(
            get=mock.AsyncMock(return_value=response),
        )

        targets = await instance._sparkrun_targets()

        self.assertEqual(
            targets["model"]["load_settings"]["max_concurrency"], 3,
        )

    async def test_chat_and_completions_share_the_same_proxy_limit(self) -> None:
        instance = self.manager()
        target_container = container()
        instance._resolve_vllm_target = mock.AsyncMock(
            return_value=target_container,
        )
        instance._req_seq = 0
        instance._active_reqs = {}
        instance._trailing_window = 5.0
        instance._record_usage = mock.Mock()

        first_release = asyncio.Event()
        second_release = asyncio.Event()
        upstream_calls = []

        async def post(url, **kwargs):
            upstream_calls.append(url)
            if len(upstream_calls) == 1:
                await first_release.wait()
            else:
                await second_release.wait()
            response = mock.Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {"choices": [], "usage": {}}
            return response

        instance.http = SimpleNamespace(post=post)
        chat = asyncio.create_task(instance._vllm_chat(
            "model", {"model": "model", "messages": []}, stream=False,
        ))
        while len(upstream_calls) < 1:
            await asyncio.sleep(0)
        completion = asyncio.create_task(instance._vllm_completions(
            "model", {"model": "model", "prompt": "hello"}, stream=False,
        ))
        await asyncio.sleep(0)

        self.assertEqual(len(upstream_calls), 1)
        self.assertEqual(
            instance.inference_admission()["deployment-a"]["queued"], 1,
        )

        first_release.set()
        await chat
        while len(upstream_calls) < 2:
            await asyncio.sleep(0)
        self.assertIn("/v1/completions", upstream_calls[1])
        second_release.set()
        await completion
        self.assertEqual(instance.inference_admission(), {})

    async def test_stream_waits_before_opening_upstream_connection(self) -> None:
        instance = self.manager()
        target_container = container()
        instance._resolve_vllm_target = mock.AsyncMock(
            return_value=target_container,
        )
        instance._req_seq = 0
        instance._active_reqs = {}
        instance._trailing_window = 5.0
        instance._record_usage = mock.Mock()

        releases = [asyncio.Event(), asyncio.Event()]
        upstream_calls = []

        class Response:
            status_code = 200

            def __init__(self, release):
                self.release = release

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def aiter_lines(self):
                await self.release.wait()
                yield "data: [DONE]"

        class Http:
            def stream(self, method, url, **kwargs):
                upstream_calls.append(url)
                return Response(releases[len(upstream_calls) - 1])

        instance.http = Http()
        first_stream = await instance._vllm_chat(
            "model", {"model": "model", "messages": [], "stream": True},
            stream=True,
        )
        second_open = asyncio.create_task(
            instance._vllm_chat(
                "model", {"model": "model", "messages": [], "stream": True},
                stream=True,
            )
        )

        async def consume(stream):
            return [chunk async for chunk in stream]

        await asyncio.sleep(0)

        self.assertEqual(len(upstream_calls), 1)
        self.assertEqual(
            instance.inference_admission()["deployment-a"]["queued"], 1,
        )
        releases[0].set()
        await consume(first_stream)
        while len(upstream_calls) < 2:
            await asyncio.sleep(0)
        second_stream = await second_open
        releases[1].set()
        await consume(second_stream)
        self.assertEqual(instance.inference_admission(), {})

    async def test_stream_disconnect_cancels_stalled_response_headers(self) -> None:
        instance = self.manager()
        target_container = container()
        instance._resolve_vllm_target = mock.AsyncMock(
            return_value=target_container,
        )
        instance._req_seq = 0
        instance._active_reqs = {}
        instance._trailing_window = 5.0
        entered = asyncio.Event()

        class StalledContext:
            async def __aenter__(self):
                entered.set()
                await asyncio.Event().wait()

            async def __aexit__(self, *args):
                return False

        instance.http = SimpleNamespace(
            stream=lambda *args, **kwargs: StalledContext(),
        )
        cancel = asyncio.Event()
        opening = asyncio.create_task(
            instance._vllm_chat(
                "model", {"model": "model", "messages": [], "stream": True},
                stream=True, cancel=cancel,
            )
        )
        await entered.wait()
        cancel.set()

        with self.assertRaises(ClientAbort):
            await opening
        self.assertEqual(instance.inference_admission(), {})

    async def test_stream_preserves_rejection_status_during_preparation(self) -> None:
        instance = self.manager()
        instance._resolve_vllm_target = mock.AsyncMock(return_value=container())
        instance._req_seq = 0
        instance._active_reqs = {}
        instance._trailing_window = 5.0
        instance.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(
                422, text="invalid parameters", request=request,
            )
        ))

        try:
            with self.assertRaises(httpx.HTTPStatusError) as raised:
                await instance._vllm_chat(
                    "model",
                    {"model": "model", "messages": [], "stream": True},
                    stream=True,
                )
        finally:
            await instance.http.aclose()

        self.assertEqual(raised.exception.response.status_code, 422)
        self.assertEqual(instance.inference_admission(), {})

    async def test_nudger_replays_only_newer_zero_output_stream(self) -> None:
        instance = self.manager()
        instance._req_seq = 0
        instance._active_reqs = {}
        instance._trailing_window = 5.0
        instance._nudger_slow_since = {}
        instance.settings = {
            "vllm_nudger_enabled": True,
            "vllm_nudger_rate_threshold": 5.0,
            "vllm_nudger_stall_seconds": 3.0,
        }
        first_slot = await instance._acquire_inference_slot(
            container(limit=3), "model [test]", None,
        )
        second_slot = await instance._acquire_inference_slot(
            container(limit=3), "model [test]", None,
        )
        first_event, second_event = asyncio.Event(), asyncio.Event()
        first = instance._track_start(
            "model [test]", True, first_slot, first_event,
        )
        second = instance._track_start(
            "model [test]", True, second_slot, second_event,
        )
        instance._active_reqs[first]["started_at"] = 90.0
        instance._active_reqs[second]["started_at"] = 91.0

        instance._inference_nudger_tick(now=91.0)
        instance._inference_nudger_tick(now=94.1)

        state = instance.inference_admission()["deployment-a"]
        self.assertEqual(state["effective_limit"], 1)
        self.assertEqual(state["nudger"]["status"], "nudging")
        self.assertEqual(state["nudger"]["survivor_request"], first)
        self.assertFalse(first_event.is_set())
        self.assertTrue(second_event.is_set())

        instance._track_end(first)
        instance._track_end(second)
        instance._release_inference_slot(first_slot)
        instance._release_inference_slot(second_slot)

    async def test_nudger_never_interrupts_partial_output(self) -> None:
        instance = self.manager()
        instance._req_seq = 0
        instance._active_reqs = {}
        instance._trailing_window = 5.0
        instance._nudger_slow_since = {}
        instance.settings = {
            "vllm_nudger_enabled": True,
            "vllm_nudger_rate_threshold": 5.0,
            "vllm_nudger_stall_seconds": 3.0,
        }
        slots = [
            await instance._acquire_inference_slot(
                container(limit=3), "model [test]", None,
            )
            for _ in range(2)
        ]
        events = [asyncio.Event(), asyncio.Event()]
        requests = [
            instance._track_start("model [test]", True, slots[i], events[i])
            for i in range(2)
        ]
        for index, rid in enumerate(requests):
            instance._active_reqs[rid]["started_at"] = 90.0 + index
        instance._track_output(requests[1], 91.0, "thinking", 1)
        instance._active_reqs[requests[1]]["forwarded_chunks"] = 1

        instance._inference_nudger_tick(now=91.0)
        instance._inference_nudger_tick(now=94.1)

        state = instance.inference_admission()["deployment-a"]
        self.assertEqual(state["effective_limit"], 1)
        self.assertEqual(
            state["nudger"]["status"], "blocked_partial_output",
        )
        self.assertFalse(any(event.is_set() for event in events))

        for rid in requests:
            instance._track_end(rid)
        for slot in slots:
            instance._release_inference_slot(slot)

    async def test_effective_limit_stops_new_grants_until_running_is_one(self) -> None:
        instance = self.manager()
        first = await instance._acquire_inference_slot(
            container(limit=3), "model [test]", None,
        )
        second = await instance._acquire_inference_slot(
            container(limit=3), "model [test]", None,
        )
        instance._inference_admission["deployment-a"]["effective_limit"] = 1
        third = asyncio.create_task(instance._acquire_inference_slot(
            container(limit=3), "model [test]", None,
        ))
        await asyncio.sleep(0)

        instance._release_inference_slot(first)
        await asyncio.sleep(0)
        self.assertFalse(third.done())
        instance._release_inference_slot(second)
        self.assertEqual(await third, "deployment-a")
        instance._release_inference_slot("deployment-a")

    async def test_nudger_restores_one_slot_per_stall_interval(self) -> None:
        instance = self.manager()
        instance._nudger_slow_since = {}
        instance._active_reqs = {}
        instance.settings = {
            "vllm_nudger_enabled": True,
            "vllm_nudger_rate_threshold": 5.0,
            "vllm_nudger_stall_seconds": 3.0,
        }
        first = await instance._acquire_inference_slot(
            container(limit=3), "model [test]", None,
        )
        state = instance._inference_admission["deployment-a"]
        state["effective_limit"] = 1
        state["_nudger_next_step_at"] = 103.0
        second = asyncio.create_task(instance._acquire_inference_slot(
            container(limit=3), "model [test]", None,
        ))
        third = asyncio.create_task(instance._acquire_inference_slot(
            container(limit=3), "model [test]", None,
        ))
        await asyncio.sleep(0)

        instance._inference_nudger_tick(now=102.9)
        self.assertFalse(second.done())
        self.assertEqual(state["effective_limit"], 1)

        instance._inference_nudger_tick(now=103.0)
        self.assertEqual(await second, "deployment-a")
        self.assertFalse(third.done())
        self.assertEqual(state["effective_limit"], 2)
        self.assertEqual(state["_nudger_next_step_at"], 106.0)

        instance._inference_nudger_tick(now=105.9)
        self.assertFalse(third.done())
        instance._inference_nudger_tick(now=106.0)
        self.assertEqual(await third, "deployment-a")
        self.assertNotIn("effective_limit", state)
        self.assertNotIn("_nudger_next_step_at", state)

        for target in (first, "deployment-a", "deployment-a"):
            instance._release_inference_slot(target)

    async def test_low_throughput_during_recovery_returns_limit_to_one(self) -> None:
        instance = self.manager()
        instance._req_seq = 0
        instance._active_reqs = {}
        instance._trailing_window = 5.0
        instance._nudger_slow_since = {}
        instance.settings = {
            "vllm_nudger_enabled": True,
            "vllm_nudger_rate_threshold": 5.0,
            "vllm_nudger_stall_seconds": 3.0,
        }
        slots = [
            await instance._acquire_inference_slot(
                container(limit=3), "model [test]", None,
            )
            for _ in range(2)
        ]
        events = [asyncio.Event(), asyncio.Event()]
        requests = [
            instance._track_start("model [test]", True, slots[i], events[i])
            for i in range(2)
        ]
        for rid in requests:
            instance._active_reqs[rid]["started_at"] = 100.0
        state = instance._inference_admission["deployment-a"]
        state["effective_limit"] = 2
        state["_nudger_next_step_at"] = 103.0

        instance._inference_nudger_tick(now=100.0)
        instance._inference_nudger_tick(now=103.1)

        self.assertEqual(state["effective_limit"], 1)
        self.assertEqual(state["_nudger_next_step_at"], 106.1)
        self.assertEqual(state["nudger"]["status"], "nudging")
        self.assertTrue(events[1].is_set())

        for rid in requests:
            instance._track_end(rid)
        for slot in slots:
            instance._release_inference_slot(slot)

    async def test_zero_output_nudge_transparently_reopens_upstream(self) -> None:
        instance = self.manager()
        target_container = container(limit=3)
        instance._resolve_vllm_target = mock.AsyncMock(
            return_value=target_container,
        )
        instance._req_seq = 0
        instance._active_reqs = {}
        instance._trailing_window = 5.0
        instance._record_usage = mock.Mock()
        upstream_calls = []
        first_opened = asyncio.Event()

        class Response:
            status_code = 200

            def __init__(self, first):
                self.first = first

            async def __aenter__(self):
                if self.first:
                    first_opened.set()
                return self

            async def __aexit__(self, *args):
                return False

            async def aiter_lines(self):
                if self.first:
                    await asyncio.Event().wait()
                yield 'data: {"choices":[{"delta":{"content":"ok"},"token_ids":[1]}]}'
                yield "data: [DONE]"

        class Http:
            def stream(self, method, url, **kwargs):
                upstream_calls.append(url)
                return Response(len(upstream_calls) == 1)

        instance.http = Http()
        stream = await instance._vllm_chat(
            "model", {"model": "model", "messages": [], "stream": True},
            stream=True,
        )

        async def consume():
            return [chunk async for chunk in stream]

        task = asyncio.create_task(consume())
        await first_opened.wait()
        rec = next(iter(instance._active_reqs.values()))
        rec["nudge_event"].set()
        chunks = await asyncio.wait_for(task, timeout=1.0)

        self.assertEqual(len(upstream_calls), 2)
        self.assertEqual(sum("content" in chunk for chunk in chunks), 1)
        self.assertEqual(instance.inference_admission(), {})

    async def test_controller_job_is_cancelable_while_waiting_for_slot(self) -> None:
        instance = self.manager()
        target_container = container()
        held = await instance._acquire_inference_slot(
            target_container, "model [test]", None,
        )
        instance._resolve_vllm_target = mock.AsyncMock(
            return_value=target_container,
        )
        instance.jobs = {
            "job": {
                "id": "job", "model": "model", "container": None,
                "messages": [], "params": {}, "status": "admission_waiting",
                "requested_at": 0, "started_at": None, "completed_at": None,
                "result": None, "error": None, "attempts": 0,
                "_cancel_event": asyncio.Event(),
            },
        }
        instance.queue = __import__("collections").deque(["job"])
        instance.lock = asyncio.Lock()
        instance.settings = {"max_retries": 2}
        run = asyncio.create_task(
            instance._run_inference("job", target_container)
        )
        await asyncio.sleep(0)

        await instance.cancel_job("job")
        await run

        self.assertEqual(instance.jobs["job"]["status"], "canceled")
        instance._release_inference_slot(held)


if __name__ == "__main__":
    unittest.main()
