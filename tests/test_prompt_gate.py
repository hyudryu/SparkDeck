import asyncio
import gc
import unittest
import weakref

from manager import ClientAbort
from sparkdeck.prompt_gate import PromptGate, PromptGates


async def turns():
    for _ in range(5):
        await asyncio.sleep(0)


class PromptGatesTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_group_runs_one_request_and_queues_its_next_request(self):
        gates = PromptGates()
        started = {key: asyncio.Event() for key in ('a1', 'a2', 'b1', 'b2')}
        finish = {key: asyncio.Event() for key in started}

        async def response(key):
            started[key].set()
            await finish[key].wait()
            return key

        tasks = {
            key: asyncio.create_task(gates.run(key[0], lambda key=key: response(key)))
            for key in started
        }
        await asyncio.wait_for(asyncio.gather(started['a1'].wait(), started['b1'].wait()), 1)
        self.assertFalse(started['a2'].is_set())
        self.assertFalse(started['b2'].is_set())
        finish['a1'].set()
        await asyncio.wait_for(started['a2'].wait(), 1)
        self.assertFalse(started['b2'].is_set())
        finish['b1'].set()
        await asyncio.wait_for(started['b2'].wait(), 1)
        for event in finish.values():
            event.set()
        self.assertEqual(await asyncio.gather(*tasks.values()), list(started))

    async def test_refresh_applies_live_limit_to_all_groups(self):
        limit = [1]
        gates = PromptGates(lambda: limit[0])
        group_gates = [gates.get(('recipe', key)) for key in ('a', 'b')]
        active = [await gate.acquire() for gate in group_gates]
        queued = [asyncio.create_task(gate.acquire()) for gate in group_gates]
        await turns()
        self.assertTrue(all(not task.done() for task in queued))
        limit[0] = 2
        gates.refresh()
        admitted = await asyncio.wait_for(asyncio.gather(*queued), 1)
        self.assertEqual([gate.active for gate in group_gates], [2, 2])
        for lease in active + admitted:
            lease.release()

    async def test_same_key_shares_gate_and_idle_gate_is_collectable(self):
        gates = PromptGates()
        gate = gates.get(('recipe', 'group'))
        self.assertIs(gate, gates.get(('recipe', 'group')))
        reference = weakref.ref(gate)
        lease = await gate.acquire()
        del gate
        gc.collect()
        self.assertIs(gates.get(('recipe', 'group')), lease.gate)
        lease.release()
        del lease
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(len(gates._gates), 0)


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
