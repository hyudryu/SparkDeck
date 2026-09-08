import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sparkdeck.service import SparkDeckService


class PromptGateServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.service = SparkDeckService(SimpleNamespace(http=None), Path(self.directory.name))
        self.addAsyncCleanup(self.service.close)
        self.configure_target()

    def configure_target(self):
        self.deployment = {"id": "serving-target", "settings": {}}
        self.service._live_deployment_for_model_id = AsyncMock(return_value=self.deployment)

    async def close_stream(self, stream):
        await stream.aclose()

    async def wait_for_queue(self, count):
        async def wait():
            while len(self.service.prompt_gate.get(("deployment", self.deployment["id"]))._waiting) != count:
                await asyncio.sleep(0)
        await asyncio.wait_for(wait(), 2)

    async def test_target_aliases_endpoints_and_callers_share_fifo_until_first_generated_output(self):
        chunks = {name: asyncio.Queue() for name in ("first", "second", "third")}
        entered = []
        closed = []

        async def upstream(deployment, body, endpoint, cancel, **kwargs):
            name = body["model"]
            entered.append((name, endpoint, kwargs["caller_ip"]))

            async def stream():
                try:
                    while True:
                        chunk = await chunks[name].get()
                        if chunk is None:
                            return
                        yield chunk
                finally:
                    closed.append(name)
            return stream()

        self.service._proxy_registered_unlimited = AsyncMock(side_effect=upstream)
        first = await self.service.proxy(
            {"model": "first", "stream": True}, "chat/completions", caller_ip="192.0.2.1",
        )
        self.addAsyncCleanup(self.close_stream, first)
        second_task = asyncio.create_task(self.service.proxy(
            {"model": "second", "stream": True}, "completions", caller_ip="192.0.2.2",
        ))
        self.addCleanup(second_task.cancel)
        await self.wait_for_queue(1)
        third_task = asyncio.create_task(self.service.proxy(
            {"model": "third", "stream": True}, "chat/completions", caller_ip="192.0.2.3",
        ))
        self.addCleanup(third_task.cancel)
        await self.wait_for_queue(2)
        role = 'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
        await chunks["first"].put(role)
        self.assertEqual(await asyncio.wait_for(anext(first), 2), role)
        self.assertEqual(len(entered), 1)
        self.assertFalse(second_task.done())
        self.assertFalse(third_task.done())

        token = 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        await chunks["first"].put(token)
        self.assertEqual(await asyncio.wait_for(anext(first), 2), token)
        second = await asyncio.wait_for(second_task, 2)
        self.addAsyncCleanup(self.close_stream, second)
        self.assertNotIn("first", closed)
        self.assertFalse(third_task.done())
        await chunks["first"].put(token)
        self.assertEqual(await asyncio.wait_for(anext(first), 2), token)
        self.assertFalse(third_task.done())

        completion = 'data: {"choices":[{"text":"Hello"}]}\n\n'
        await chunks["second"].put(completion)
        self.assertEqual(await asyncio.wait_for(anext(second), 2), completion)
        third = await asyncio.wait_for(third_task, 2)
        self.addAsyncCleanup(self.close_stream, third)
        self.assertEqual(entered, [
            ("first", "chat/completions", "192.0.2.1"),
            ("second", "completions", "192.0.2.2"),
            ("third", "chat/completions", "192.0.2.3"),
        ])
        self.assertEqual(closed, [])

    async def test_different_serving_targets_process_prompts_concurrently(self):
        targets = {name: {"id": name, "settings": {}} for name in ('a', 'b')}
        self.service._live_deployment_for_model_id = AsyncMock(
            side_effect=lambda model: targets[model],
        )
        entered = {name: asyncio.Event() for name in targets}
        finish = asyncio.Event()

        async def upstream(deployment, body, endpoint, cancel, **kwargs):
            entered[deployment['id']].set()
            await finish.wait()
            return {'model': body['model']}

        self.service._proxy_registered_unlimited = AsyncMock(side_effect=upstream)
        tasks = [
            asyncio.create_task(self.service.proxy({'model': name}, 'completions'))
            for name in targets
        ]
        for task in tasks:
            self.addCleanup(task.cancel)
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered.values())), 2)
        self.assertTrue(all(not task.done() for task in tasks))
        finish.set()
        self.assertEqual(await asyncio.gather(*tasks), [{'model': 'a'}, {'model': 'b'}])

    async def test_saved_limit_survives_restart_and_bounds_nonstream_requests(self):
        self.service.store.set_setting("max_concurrent_prompt_processing", 2)
        await self.service.close()
        self.service = SparkDeckService(SimpleNamespace(http=None), Path(self.directory.name))
        self.addAsyncCleanup(self.service.close)
        self.configure_target()
        entered = asyncio.Queue()
        release = {name: asyncio.Event() for name in ("first", "second", "third")}

        async def upstream(deployment, body, endpoint, cancel, **kwargs):
            name = body["model"]
            await entered.put(name)
            await release[name].wait()
            return {"model": name}

        self.service._proxy_registered_unlimited = AsyncMock(side_effect=upstream)
        tasks = []
        for name in release:
            task = asyncio.create_task(self.service.proxy({"model": name}, "completions"))
            tasks.append(task)
            self.addCleanup(task.cancel)
        self.assertEqual(await asyncio.wait_for(entered.get(), 2), "first")
        self.assertEqual(await asyncio.wait_for(entered.get(), 2), "second")
        await self.wait_for_queue(1)
        self.assertTrue(entered.empty())
        release["first"].set()
        self.assertEqual(await asyncio.wait_for(tasks[0], 2), {"model": "first"})
        self.assertEqual(await asyncio.wait_for(entered.get(), 2), "third")
        for event in release.values():
            event.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        self.assertEqual(self.service.prompt_gate.get(("deployment", self.deployment["id"])).active, 0)
