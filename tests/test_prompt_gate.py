import asyncio
import unittest

from manager import ClientAbort
from sparkdeck.prompt_gate import PromptGate


async def turns():
    for _ in range(5):
        await asyncio.sleep(0)


class PromptGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_fifo_resize_and_idempotent_release(self):
        limit = [1]
        gate = PromptGate(lambda: limit[0])
        first = await gate.acquire()
        second = asyncio.create_task(gate.acquire())
        third = asyncio.create_task(gate.acquire())
        await turns()
        self.assertFalse(second.done())
        limit[0] = 2
        gate.refresh()
        await turns()
        self.assertTrue(second.done())
        self.assertFalse(third.done())
        limit[0] = 1
        second.result().release()
        second.result().release()
        self.assertFalse(third.done())
        first.release()
        (await third).release()
        self.assertEqual(gate.active, 0)

    async def test_cancelled_waiter_and_granted_race_do_not_leak(self):
        gate = PromptGate()
        first = await gate.acquire()
        event = asyncio.Event()
        waiter = asyncio.create_task(gate.acquire(event))
        await turns()
        event.set()
        first.release()
        with self.assertRaises(ClientAbort):
            await waiter
        self.assertEqual(gate.active, 0)
        first = await gate.acquire()
        waiter = asyncio.create_task(gate.acquire())
        await turns()
        first.release()
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertEqual(gate.active, 0)

    async def test_nonstream_and_factory_error_release(self):
        gate = PromptGate()
        async def response():
            return {"choices": []}
        self.assertEqual(await gate.run(response), {"choices": []})
        async def fail():
            raise ValueError("failed")
        with self.assertRaises(ValueError):
            await gate.run(fail)
        self.assertEqual(gate.active, 0)

    async def test_role_usage_and_fragmented_output(self):
        for field in ('content', 'reasoning_content', 'tool_calls'):
            with self.subTest(field=field):
                gate = PromptGate()
                output = asyncio.Event()
                finish = asyncio.Event()
                closed = asyncio.Event()
                async def upstream():
                    try:
                        yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
                        yield 'data: {"usage":{"prompt_tokens":5},"choices":[]}\n\n'
                        await output.wait()
                        yield b'data: {"choices":[{"delta":{'
                        yield ('"' + field + '":"generated"}}]}\n\n').encode()
                        await finish.wait()
                        yield 'data: [DONE]\n\n'
                    finally:
                        closed.set()
                async def factory():
                    return upstream()
                stream = await gate.run(factory)
                await anext(stream)
                await anext(stream)
                queued = asyncio.create_task(gate.acquire())
                await turns()
                self.assertFalse(queued.done())
                output.set()
                await anext(stream)
                await anext(stream)
                lease = await asyncio.wait_for(queued, 1)
                self.assertFalse(finish.is_set())
                lease.release()
                await stream.aclose()
                self.assertTrue(closed.is_set())
                self.assertEqual(gate.active, 0)

    async def test_disconnect_without_consumption_closes_upstream(self):
        gate = PromptGate()
        cancel = asyncio.Event()
        closed = asyncio.Event()
        async def upstream():
            try:
                await asyncio.Event().wait()
                yield ''
            finally:
                closed.set()
        async def factory():
            return upstream()
        stream = await gate.run(factory, cancel)
        await turns()
        cancel.set()
        await asyncio.wait_for(closed.wait(), 1)
        self.assertEqual(gate.active, 0)
        with self.assertRaises(ClientAbort):
            await anext(stream)

    async def test_stream_error_and_immediate_close_release(self):
        gate = PromptGate()
        async def upstream():
            raise ValueError('upstream failed')
            yield ''
        async def factory():
            return upstream()
        stream = await gate.run(factory)
        with self.assertRaises(ValueError):
            await anext(stream)
        self.assertEqual(gate.active, 0)
        stream = await gate.run(factory)
        await stream.aclose()
        self.assertEqual(gate.active, 0)

    async def test_cancel_during_factory_releases_and_cleans_up(self):
        gate = PromptGate()
        cancel = asyncio.Event()
        started = asyncio.Event()
        closed = asyncio.Event()
        async def factory():
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                closed.set()
        pending = asyncio.create_task(gate.run(factory, cancel))
        await started.wait()
        cancel.set()
        with self.assertRaises(ClientAbort):
            await pending
        self.assertTrue(closed.is_set())
        self.assertEqual(gate.active, 0)

    async def test_completed_empty_stream_releases_and_preserves_chunks(self):
        gate = PromptGate()
        chunks = ['data: {"choices":[]}\n\n', 'data: [DONE]\n\n']
        async def upstream():
            for chunk in chunks:
                yield chunk
        async def factory():
            return upstream()
        stream = await gate.run(factory)
        self.assertEqual([chunk async for chunk in stream], chunks)
        self.assertEqual(gate.active, 0)

    async def test_cancelled_consumer_closes_in_iteration_context(self):
        from contextvars import ContextVar
        marker = ContextVar('prompt_gate_test')
        gate = PromptGate()
        started = asyncio.Event()
        closed = asyncio.Event()
        async def upstream():
            token = marker.set('running')
            try:
                started.set()
                await asyncio.Event().wait()
                yield ''
            finally:
                marker.reset(token)
                closed.set()
        async def factory():
            return upstream()
        stream = await gate.run(factory)
        consumer = asyncio.create_task(anext(stream))
        await started.wait()
        consumer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await consumer
        self.assertTrue(closed.is_set())
        self.assertEqual(gate.active, 0)

    async def test_disconnect_and_repeated_close_wait_for_transport_cleanup(self):
        gate = PromptGate()
        cancel = asyncio.Event()
        started = asyncio.Event()
        cleanup_started = asyncio.Event()
        cleanup_finish = asyncio.Event()
        cleanup_completed = asyncio.Event()
        async def upstream():
            try:
                started.set()
                await asyncio.Event().wait()
                yield ''
            finally:
                cleanup_started.set()
                await cleanup_finish.wait()
                cleanup_completed.set()
        async def factory():
            return upstream()
        stream = await gate.run(factory, cancel)
        await started.wait()
        queued = asyncio.create_task(gate.acquire())
        cancel.set()
        await cleanup_started.wait()
        closing = asyncio.create_task(stream.aclose())
        await turns()
        closing.cancel()
        await turns()
        self.assertFalse(closing.done())
        self.assertFalse(queued.done())
        self.assertEqual(gate.active, 1)
        cleanup_finish.set()
        await asyncio.wait_for(closing, 1)
        self.assertTrue(cleanup_completed.is_set())
        (await queued).release()
        self.assertEqual(gate.active, 0)

    async def test_repeated_factory_task_cancellation_waits_for_cleanup(self):
        gate = PromptGate()
        started = asyncio.Event()
        cleanup_started = asyncio.Event()
        cleanup_finish = asyncio.Event()
        cleanup_completed = asyncio.Event()
        async def factory():
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await cleanup_finish.wait()
                cleanup_completed.set()
        pending = asyncio.create_task(gate.run(factory))
        await started.wait()
        pending.cancel()
        await cleanup_started.wait()
        pending.cancel()
        await turns()
        self.assertFalse(pending.done())
        self.assertEqual(gate.active, 1)
        cleanup_finish.set()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertTrue(cleanup_completed.is_set())
        self.assertEqual(gate.active, 0)
