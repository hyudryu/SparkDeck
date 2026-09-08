import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sparkdeck.service import SparkDeckService


class ObservationStreamContextTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.service = SparkDeckService(SimpleNamespace(http=None), Path(directory.name))
        self.addAsyncCleanup(self.service.close)

    async def close_stream(self, stream):
        await stream.aclose()

    async def test_interleaved_streams_restore_caller_context_between_chunks(self):
        caller = {"id": "caller"}
        token = self.service._community_observation.set(caller)
        self.addCleanup(self.service._community_observation.reset, token)
        seen = []

        async def upstream(name):
            for chunk in ("first", "second"):
                seen.append((name, self.service._community_observation.get()))
                yield chunk

        observations = [self.service._community_observation_start() for _ in range(2)]
        streams = [
            self.service._community_observed_stream(upstream(index), observation)
            for index, observation in enumerate(observations)
        ]
        for stream in streams:
            self.addAsyncCleanup(self.close_stream, stream)
        for chunk in ("first", "second"):
            for stream in streams:
                self.assertEqual(await anext(stream), chunk)
                self.assertIs(self.service._community_observation.get(), caller)
        for stream in streams:
            with self.assertRaises(StopAsyncIteration):
                await anext(stream)
        self.assertEqual(seen, [
            (0, observations[0]), (1, observations[1]),
            (0, observations[0]), (1, observations[1]),
        ])
        self.assertEqual(self.service._community_active_observations, {})

    async def test_stream_can_advance_and_close_in_different_tasks(self):
        observation = self.service._community_observation_start()
        closed = []

        async def upstream():
            try:
                yield "first"
                yield "second"
            finally:
                closed.append(self.service._community_observation.get())

        stream = self.service._community_observed_stream(upstream(), observation)
        self.addAsyncCleanup(self.close_stream, stream)
        self.assertEqual(await asyncio.wait_for(anext(stream), 2), "first")
        self.assertEqual(await asyncio.wait_for(anext(stream), 2), "second")
        await stream.aclose()
        self.assertEqual(closed, [observation])
        self.assertIsNone(self.service._community_observation.get())
        self.assertEqual(self.service._community_active_observations, {})

    async def test_upstream_error_restores_context_and_ends_observation(self):
        observation = self.service._community_observation_start()

        async def upstream():
            yield "first"
            raise RuntimeError("upstream failed")

        stream = self.service._community_observed_stream(upstream(), observation)
        self.assertEqual(await anext(stream), "first")
        with self.assertRaisesRegex(RuntimeError, "upstream failed"):
            await anext(stream)
        self.assertIsNone(self.service._community_observation.get())
        self.assertEqual(self.service._community_active_observations, {})
