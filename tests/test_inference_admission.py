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


if __name__ == "__main__":
    unittest.main()
