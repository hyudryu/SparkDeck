import asyncio
import contextvars
from unittest.mock import Mock, patch

import anyio
import pytest

with patch("docker.from_env", return_value=Mock()):
    from server import _guard_stream

from sparkdeck.service import SparkDeckService
from sparkdeck.stream_cleanup import close_async_stream


def service():
    result = SparkDeckService.__new__(SparkDeckService)
    result._community_observation = contextvars.ContextVar("observation", default=None)
    result._community_active_observations = {"test": {"id": "test"}}
    return result


def test_guard_closes_inner_stream_and_watcher_on_early_close():
    async def run():
        closed = []
        async def upstream():
            try:
                yield "token"
            finally:
                await asyncio.sleep(0)
                closed.append(True)
        watcher = asyncio.create_task(asyncio.sleep(3600))
        inner = upstream()
        wrapped = _guard_stream(inner, watcher)
        await anext(wrapped)
        await wrapped.aclose()
        await asyncio.gather(watcher, return_exceptions=True)
        assert closed == [True]
        assert watcher.cancelled()
    asyncio.run(run())


def test_guard_still_cancels_watcher_when_inner_close_raises():
    async def run():
        async def upstream():
            try:
                yield "token"
            finally:
                raise RuntimeError("close failed")
        watcher = asyncio.create_task(asyncio.sleep(3600))
        wrapped = _guard_stream(upstream(), watcher)
        await anext(wrapped)
        with pytest.raises(RuntimeError, match="close failed"):
            await wrapped.aclose()
        await asyncio.gather(watcher, return_exceptions=True)
        assert watcher.cancelled()
    asyncio.run(run())


def test_community_wrapper_closes_in_same_context_under_disconnect_scope():
    async def run():
        app = service()
        closed = []
        marker = contextvars.ContextVar("inner-context", default="outside")
        async def upstream():
            token = marker.set("inside")
            try:
                yield "token"
            finally:
                await anyio.sleep(0)
                marker.reset(token)
                closed.append(True)
        with anyio.CancelScope() as scope:
            wrapped = app._community_observed_stream(upstream(), {"id": "test"})
            await anext(wrapped)
            scope.cancel()
            await wrapped.aclose()
        assert closed == [True]
        assert marker.get() == "outside"
        assert not app._community_active_observations
        assert app._community_observation.get() is None
    asyncio.run(run())


def test_observation_producer_closes_upstream_on_transform_error():
    async def run():
        app = service()
        closed = []
        async def upstream():
            try:
                yield "token"
            finally:
                closed.append(True)
        stream = app._observe_stream(upstream(), None, "model", "vllm", {}, 0, response_model="alias")
        with patch("sparkdeck.service._rewrite_sse_model", side_effect=ValueError("rewrite failed")):
            with pytest.raises(ValueError, match="rewrite failed"):
                await asyncio.wait_for(anext(stream), 1)
        assert closed == [True]
    asyncio.run(run())


def test_observation_close_error_does_not_strand_consumer_waiting_for_finished():
    class Upstream:
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise StopAsyncIteration
        async def aclose(self):
            raise RuntimeError("close failed")
    async def run():
        app = service()
        stream = app._observe_stream(Upstream(), None, "model", "vllm", {}, 0)
        with pytest.raises(RuntimeError, match="close failed"):
            await asyncio.wait_for(anext(stream), 1)
    asyncio.run(run())


def test_registered_stream_finishes_on_done_without_waiting_for_transport_eof():
    async def run():
        closed = []
        class Response:
            async def aiter_lines(self):
                yield "data: [DONE]"
                await asyncio.Event().wait()
        class Context:
            async def __aexit__(self, *args):
                closed.append(True)
        stream = service()._consume_http_stream(Context(), Response(), {}, 0, None)
        assert await anext(stream) == "data: [DONE]\n\n"
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(stream), 1)
        assert closed == [True]
    asyncio.run(run())


def test_close_helper_allows_iterators_without_optional_aclose():
    asyncio.run(close_async_stream(object()))
