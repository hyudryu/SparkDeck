import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import httpx

from sparkdeck.service import SparkDeckService


class FakeManager:
    def __init__(self):
        self.http = httpx.AsyncClient()
        self.list_containers = AsyncMock(return_value=[])


class BenchmarkCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = FakeManager()
        self.service = SparkDeckService(self.manager, Path(self.temp.name))

    async def asyncTearDown(self):
        await self.manager.http.aclose()
        await self.service.close()
        self.temp.cleanup()

    async def test_response_metrics_persist_without_request_or_output_content(self):
        self.service.store.set_setting("community_consent", True)
        response = {
            "choices": [{"message": {"content": "generated secret"}}],
            "usage": {"prompt_tokens": 32, "completion_tokens": 24},
            "timings": {"predicted_per_second": 96.0, "prompt_per_second": 128.0},
        }
        self.service._record_response(
            None, "org/model", "vllm", {"context_size": 4096, "api_key": "secret"},
            time.monotonic() - 0.25, response,
        )
        items, total = self.service.store.benchmarks()
        self.assertEqual(total, 1)
        payload = str(items[0])
        self.assertNotIn("generated secret", payload)
        self.assertNotIn("api_key", payload)
        self.assertEqual(items[0]["output_tokens"], 24)
        self.assertTrue(items[0]["eligible_for_community"])
        self.assertEqual(
            self.service.store.sync_status()["outbox"]["waiting_for_account"], 1
        )

    async def test_short_sample_remains_local_and_is_not_queued(self):
        self.service.store.set_setting("community_consent", True)
        self.service._record_response(
            None, "org/model", "llama.cpp", {}, time.monotonic() - 0.1,
            {"usage": {"prompt_tokens": 4, "completion_tokens": 2}},
        )
        items, _ = self.service.store.benchmarks()
        self.assertFalse(items[0]["eligible_for_community"])
        self.assertEqual(
            self.service.store.sync_status()["outbox"]["waiting_for_account"], 0
        )
