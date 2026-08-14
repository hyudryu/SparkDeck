import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

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
        second_stream = await instance._vllm_chat(
            "model", {"model": "model", "messages": [], "stream": True},
            stream=True,
        )

        async def consume(stream):
            return [chunk async for chunk in stream]

        first = asyncio.create_task(consume(first_stream))
        while len(upstream_calls) < 1:
            await asyncio.sleep(0)
        second = asyncio.create_task(consume(second_stream))
        await asyncio.sleep(0)

        self.assertEqual(len(upstream_calls), 1)
        self.assertEqual(
            instance.inference_admission()["deployment-a"]["queued"], 1,
        )
        releases[0].set()
        await first
        while len(upstream_calls) < 2:
            await asyncio.sleep(0)
        releases[1].set()
        await second
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
        stream = await instance._vllm_chat(
            "model", {"model": "model", "messages": [], "stream": True},
            stream=True, cancel=cancel,
        )
        consume = asyncio.create_task(
            anext(stream)
        )
        await entered.wait()
        cancel.set()

        with self.assertRaises(StopAsyncIteration):
            await consume
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
