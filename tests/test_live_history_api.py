"""The History panel's read endpoint.

The collector's own behaviour is covered by
:mod:`tests.test_live_history_metrics`; this checks the contract the panel polls:
the sampler is started on first use, and the payload carries the bucket cadence
and the trailing range the graph draws.
"""

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sparkdeck.live_metrics import BUCKET_SECONDS, LiveHistory


class LiveHistoryEndpointTests(unittest.TestCase):
    def test_first_request_starts_the_sampler_and_reports_the_window(self) -> None:
        app = FastAPI()
        manager = type("Manager", (), {"_active_reqs": {}})()
        history = LiveHistory(manager)

        @app.get("/api/v1/live-history")
        async def live_history():
            history.ensure_sampler()
            return history.snapshot()

        with TestClient(app) as client:
            payload = client.get("/api/v1/live-history").json()
            # The panel's first poll starts the sampler, and it keeps running so
            # every bucket is worth its full five seconds.
            self.assertIsNotNone(history._task)
            self.assertFalse(history._task.done())

        self.assertEqual(payload["bucket_seconds"], BUCKET_SECONDS)
        self.assertEqual(payload["range_seconds"], 720 * BUCKET_SECONDS)
        self.assertEqual(payload["sample_seconds"], 1.0)
        self.assertEqual(payload["series"], [])
        self.assertIsInstance(payload["generated_at"], float)

    def test_snapshot_never_exposes_manager_internals(self) -> None:
        manager = type("Manager", (), {"_active_reqs": {}})()
        snapshot = LiveHistory(manager).snapshot()

        self.assertEqual(
            set(snapshot),
            {"generated_at", "bucket_seconds", "range_seconds", "sample_seconds", "series"},
        )
