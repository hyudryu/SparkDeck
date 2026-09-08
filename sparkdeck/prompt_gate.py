"""FIFO admission control for prompt processing, ending at first generated output."""
from __future__ import annotations

import asyncio
import codecs
import json

import anyio
from collections import deque
from contextlib import suppress
from typing import Callable

from sparkdeck.stream_cleanup import close_async_stream


def _abort():
    # Manager imports the service, so importing this at module scope is circular.
    from manager import ClientAbort
    return ClientAbort()


def _cancel_once(task):
    # Repeated cancel() interrupts asynchronous finally blocks already closing
    # the upstream transport. That work must finish before its lease is freed.
    if not task.done() and not task.cancelling():
        task.cancel()


async def _join_cleanup(task):
    # AnyIO cancellation is level-triggered; shield it as well as direct asyncio
    # cancellation so disconnect and response teardown cannot cancel cleanup twice.
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        return task.result()


class PromptLease:
    def __init__(self, gate: PromptGate):
        self.gate = gate
        self.released = False

    def release(self):
        if not self.released:
            self.released = True
            self.gate.active -= 1
            self.gate.refresh()


class PromptGate:
    """One event-loop-local gate shared by every inference route in a service."""
    def __init__(self, limit: Callable[[], int] = lambda: 1):
        self.limit = limit
        self.active = 0
        self._waiting = deque()

    def refresh(self):
        """Admit FIFO waiters after a setting increase or a lease release."""
        while self._waiting and self.active < max(1, int(self.limit())):
            future = self._waiting.popleft()
            if future.done():
                continue
            self.active += 1
            future.set_result(PromptLease(self))

    async def acquire(self, cancel=None):
        if cancel is not None and cancel.is_set():
            raise _abort()
        future = asyncio.get_running_loop().create_future()
        self._waiting.append(future)
        self.refresh()
        watcher = asyncio.create_task(cancel.wait()) if cancel is not None else None
        try:
            try:
                if watcher is not None:
                    await asyncio.wait((future, watcher), return_when=asyncio.FIRST_COMPLETED)
                    if cancel.is_set():
                        raise _abort()
                lease = await asyncio.shield(future)
            finally:
                if watcher is not None:
                    watcher.cancel()
            if cancel is not None and cancel.is_set():
                raise _abort()
            return lease
        except BaseException:
            if future.done() and not future.cancelled():
                future.result().release()
            else:
                future.cancel()
                with suppress(ValueError):
                    self._waiting.remove(future)
            raise

    async def run(self, factory, cancel=None):
        lease = await self.acquire(cancel)
        task = asyncio.create_task(factory())
        watcher = asyncio.create_task(cancel.wait()) if cancel is not None else None
        handed_off = False
        try:
            if watcher is not None:
                await asyncio.wait((task, watcher), return_when=asyncio.FIRST_COMPLETED)
                if cancel.is_set():
                    raise _abort()
            result = await asyncio.shield(task)
            if hasattr(result, "__aiter__"):
                result = _PromptStream(result, lease, cancel)
                handed_off = True
            return result
        finally:
            if watcher is not None:
                watcher.cancel()
            if not handed_off:
                _cancel_once(task)
                try:
                    result = await _join_cleanup(task)
                    await close_async_stream(result)
                except BaseException:
                    pass
                lease.release()


class _OutputDetector:
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.pending = ""

    def feed(self, chunk):
        self.pending += self.decoder.decode(chunk) if isinstance(chunk, bytes) else chunk
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if not line.startswith("data:"):
                continue
            try:
                payload = json.loads(line[5:].strip())
            except (ValueError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            for choice in payload.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                if choice.get("text"):
                    return True
                delta = choice.get("delta") or choice.get("message") or {}
                if isinstance(delta, dict) and any(delta.get(key) for key in (
                    "content", "reasoning_content", "reasoning", "tool_calls", "function_call",
                )):
                    return True
        # Do not retain unbounded invalid upstream data.
        if len(self.pending) > 1024 * 1024:
            self.pending = ""
        return False


class _PromptStream:
    """Bounded eager reader owns cleanup even if the consumer never starts."""
    def __init__(self, upstream, lease, cancel):
        self.upstream = upstream
        self.lease = lease
        self.cancel = cancel
        self.started = False
        self.queue = asyncio.Queue(maxsize=1)
        self.producer = asyncio.create_task(self._produce())
        self.watcher = asyncio.create_task(self._watch()) if cancel is not None else None
        self.producer.add_done_callback(self._finished)

    def _finished(self, task):
        self.lease.release()
        # A disconnected client may never consume the iterator or its exception.
        if not task.cancelled():
            task.exception()
        if self.watcher is not None:
            self.watcher.cancel()

    async def _watch(self):
        await self.cancel.wait()
        _cancel_once(self.producer)

    async def _produce(self):
        self.started = True
        detector = _OutputDetector()
        try:
            async for chunk in self.upstream:
                if not self.lease.released and detector.feed(chunk):
                    self.lease.release()
                await self.queue.put(chunk)
        finally:
            try:
                await close_async_stream(self.upstream)
            finally:
                self.lease.release()

    def __aiter__(self):
        return self

    async def __anext__(self):
        reader = None
        try:
            if self.cancel is not None and self.cancel.is_set():
                await self.aclose()
                raise _abort()
            if not self.queue.empty():
                return self.queue.get_nowait()
            if self.producer.done():
                await self.producer
                raise StopAsyncIteration
            reader = asyncio.create_task(self.queue.get())
            await asyncio.wait((reader, self.producer), return_when=asyncio.FIRST_COMPLETED)
            if self.cancel is not None and self.cancel.is_set():
                await self.aclose()
                raise _abort()
            if reader.done():
                return reader.result()
            await self.producer
            if not self.queue.empty():
                return self.queue.get_nowait()
            raise StopAsyncIteration
        except asyncio.CancelledError:
            await self.aclose()
            raise
        finally:
            if reader is not None:
                reader.cancel()
                with suppress(asyncio.CancelledError):
                    await reader

    async def aclose(self):
        _cancel_once(self.producer)
        try:
            await _join_cleanup(self.producer)
        except BaseException:
            pass
        finally:
            # A task cancelled before its first turn never executes its finally.
            try:
                if not self.started:
                    await close_async_stream(self.upstream)
            finally:
                self.lease.release()
            if self.watcher is not None:
                self.watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await self.watcher
