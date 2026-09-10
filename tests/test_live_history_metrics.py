"""Trailing live-inference history for the History panel.

The panel needs complete five-second buckets rather than instantaneous samples,
so the collector ticks once a second and accumulates deltas from the manager's
cumulative per-request counters.  These tests drive that accumulation directly
with a manual clock, so no sleeping or event loop is involved.
"""

from __future__ import annotations

from collections import deque
from unittest import TestCase

from sparkdeck.live_metrics import BUCKET_SECONDS, LiveHistory

GROUP_A = {
    "group_id": "deployment-a:0", "model": "model-a", "deployment_id": "deployment-a",
    "instance_id": 0, "node_names": ["node-a"],
}
GROUP_B = {
    "group_id": "deployment-b:0", "model": "model-b", "deployment_id": "deployment-b",
    "instance_id": 0, "node_names": ["node-b"],
}


class _Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Manager:
    """Stand-in for Manager: only the records the collector reads."""

    def __init__(self) -> None:
        self._active_reqs: dict[int, dict] = {}
        self._next = 0
        self._admission: dict[str, dict] = {}

    def _request_group(self, key: str) -> dict:
        return dict(GROUP_A, model=key)

    def inference_admission(self) -> dict:
        return dict(self._admission)

    def start(self, key: str = "model-a", group: dict | None = None, streaming: bool = True) -> int:
        self._next += 1
        rid = self._next
        self._active_reqs[rid] = {
            "key": key, "group": dict(group or GROUP_A), "streaming": streaming,
            "thinking": deque(), "output": deque(), "started_at": 0.0,
            "pp_tokens": 0, "pp_time_s": 0.0, "total_tokens": 0, "paused": False,
        }
        return rid

    def output(self, rid: int, tokens: int, kind: str = "output", now: float = 0.0) -> None:
        """Mirror Manager._track_output: a timestamp per token plus the total."""
        rec = self._active_reqs[rid]
        rec[kind].extend([now] * tokens)
        rec["total_tokens"] += tokens

    def prefill(self, rid: int, tokens: int, seconds: float) -> None:
        rec = self._active_reqs[rid]
        rec["pp_tokens"] = tokens
        rec["pp_time_s"] = seconds

    def finish(self, rid: int) -> None:
        self._active_reqs.pop(rid, None)


def _collector(manager: _Manager, clock: _Clock) -> LiveHistory:
    return LiveHistory(manager, clock=clock)


class BucketAccumulationTests(TestCase):
    def test_tokens_emitted_between_ticks_all_land_in_one_bucket(self) -> None:
        """A once-per-bucket sample would miss everything between samples."""
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock)
        rid = manager.start()
        history.start(GROUP_A)
        history.tick()
        # Fifty output tokens are observed by the second sample; the bucket that
        # closed in between would have reported nothing without the per-tick
        # cumulative deltas.
        manager.output(rid, 50)
        clock.advance(1.0)
        history.tick()
        clock.advance(BUCKET_SECONDS)
        history.tick()

        series = history.series()[0]
        self.assertEqual(len(series["buckets"]), 1)
        bucket = series["buckets"][0]
        self.assertAlmostEqual(bucket["output_tok_s"], 10.0, places=2)
        self.assertEqual(bucket["concurrent"], 1.0)

    def test_thinking_and_output_are_attributed_separately(self) -> None:
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock)
        rid = manager.start()
        history.start(GROUP_A)
        history.tick()
        manager.output(rid, 4, "thinking")
        clock.advance(1.0)
        history.tick()
        manager.output(rid, 6, "output")
        for _ in range(3):
            clock.advance(1.0)
            history.tick()
        clock.advance(BUCKET_SECONDS)
        history.tick()

        buckets = history.series()[0]["buckets"]
        self.assertAlmostEqual(buckets[0]["thinking_tok_s"], 4.0 / 5.0, places=2)
        self.assertAlmostEqual(buckets[0]["output_tok_s"], 6.0 / 5.0, places=2)

    def test_concurrency_is_the_mean_over_the_bucket(self) -> None:
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock)
        first = manager.start()
        history.start(GROUP_A)
        # One session for the first second, two for the middle three, one again
        # for the last second, then the bucket is closed.
        history.tick()
        for _ in range(2):
            clock.advance(1.0)
            history.tick()
        second = manager.start()
        history.start(GROUP_A)
        for _ in range(3):
            clock.advance(1.0)
            history.tick()
        history.end(GROUP_A)
        manager.finish(second)
        clock.advance(2.0)
        history.tick()

        bucket = history.series()[0]["buckets"][0]
        self.assertAlmostEqual(bucket["concurrent"], 1.5, places=2)
        self.assertEqual(bucket["concurrent_peak"], 2)
        self.assertIn(first, manager._active_reqs)

    def test_a_request_that_ends_mid_bucket_still_contributes(self) -> None:
        """Sampling once per bucket would miss this request entirely."""
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock)
        history.tick()
        rid = manager.start()
        history.start(GROUP_A)
        clock.advance(1.0)
        history.tick()
        # The tokens arrive between two samples and the request is gone by the
        # sample that closes the bucket.  Its final count is still attributed.
        manager.output(rid, 30)
        clock.advance(BUCKET_SECONDS)
        # Manager._track_end runs while the record is still in _active_reqs, so
        # the hook receives the live record itself.
        record = manager._active_reqs[rid]
        history.end(GROUP_A, record)
        manager.finish(rid)
        history.tick()

        bucket = history.series()[0]["buckets"][0]
        # Thirty tokens over the six seconds the bucket was open.
        self.assertAlmostEqual(bucket["output_tok_s"], 6.0, places=2)
        self.assertTrue(bucket["output_active"])

    def test_paused_replay_sessions_are_not_counted(self) -> None:
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock)
        rid = manager.start()
        manager._active_reqs[rid]["paused"] = True
        history.tick()
        clock.advance(BUCKET_SECONDS)
        history.tick()

        self.assertEqual(history.series(), [])

    def test_idle_serving_unit_records_zero_traffic_buckets(self) -> None:
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock)
        rid = manager.start()
        history.start(GROUP_A)
        history.tick()
        manager.output(rid, 5)
        manager.finish(rid)
        history.end(GROUP_A)
        clock.advance(BUCKET_SECONDS)
        history.tick()
        clock.advance(BUCKET_SECONDS)
        history.tick()

        buckets = history.series()[0]["buckets"]
        self.assertGreaterEqual(len(buckets), 2)
        self.assertEqual(buckets[-1]["output_tok_s"], 0.0)
        self.assertEqual(buckets[-1]["concurrent"], 0.0)


class PrefillEstimateTests(TestCase):
    def test_measured_prefill_rate_is_reported_after_completion(self) -> None:
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock)
        rid = manager.start()
        history.start(GROUP_A)
        history.tick()
        manager.prefill(rid, 1_000, 2.0)
        clock.advance(1.0)
        history.tick()
        clock.advance(BUCKET_SECONDS)
        history.tick()

        bucket = history.series()[0]["buckets"][0]
        self.assertTrue(bucket["prefill_measured"])
        self.assertAlmostEqual(bucket["prefill_tok_s"], 500.0, places=2)

    def test_held_prefill_without_token_counts_reports_pending(self) -> None:
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock)
        rid = manager.start()
        history.start(GROUP_A)
        manager._active_reqs[rid]["started_at"] = clock.now
        for _ in range(3):
            history.tick()
            clock.advance(1.0)
        clock.advance(BUCKET_SECONDS)
        history.tick()

        bucket = history.series()[0]["buckets"][0]
        self.assertFalse(bucket["prefill_measured"])
        self.assertIsNone(bucket["prefill_tok_s"])
        self.assertEqual(bucket["prefill_sessions"], 1)

    def test_in_flight_prefill_estimates_from_held_tokens(self) -> None:
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock)
        rid = manager.start()
        history.start(GROUP_A)
        # The engine reports the prompt token count only with the first output
        # token, so the estimate only exists once that count is known.  It is
        # the in-flight equivalent of the measured rate: prompt tokens over the
        # time the prefill has been held, which is short of the real prefill
        # speed because the prefill is not finished.
        manager._active_reqs[rid]["started_at"] = clock.now - 4.0
        manager.prefill(rid, 0, 0.0)
        manager._active_reqs[rid]["pp_tokens"] = 800
        history.tick()
        clock.advance(BUCKET_SECONDS)
        history.tick()

        bucket = history.series()[0]["buckets"][0]
        self.assertFalse(bucket["prefill_measured"])
        self.assertAlmostEqual(bucket["prefill_tok_s"], 800.0 / 9.0, places=1)

    def test_estimate_falls_back_to_the_last_measured_rate(self) -> None:
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock)
        completed = manager.start()
        history.start(GROUP_A)
        history.tick()
        manager.prefill(completed, 900, 1.5)
        clock.advance(1.0)
        history.tick()
        clock.advance(BUCKET_SECONDS)
        history.tick()
        manager.finish(completed)
        history.end(GROUP_A)

        held = manager.start()
        history.start(GROUP_A)
        history.tick()
        # No prompt token count is known for this prefill yet, so the panel
        # must report this serving unit's last measured rate rather than blank.
        manager._active_reqs[held]["started_at"] = clock.now
        clock.advance(1.0)
        history.tick()
        clock.advance(BUCKET_SECONDS)
        history.tick()

        measured = history.series()[0]["buckets"][0]
        fallback = history.series()[0]["buckets"][-1]
        self.assertAlmostEqual(measured["prefill_tok_s"], 600.0, places=1)
        self.assertAlmostEqual(fallback["prefill_tok_s"], 600.0, places=1)
        self.assertFalse(fallback["prefill_measured"])


class ManagerIntegrationTests(TestCase):
    """The manager's lifecycle hooks feed the collector, not just the tests."""

    def _manager(self):
        from manager import Manager

        instance = Manager.__new__(Manager)
        instance._active_reqs = {}
        instance._req_seq = 0
        instance._trailing_window = 5.0
        instance.deployments = []
        instance.inference_admission = lambda: {}
        instance._mark_deployment_used = lambda deployment_id: None
        return instance

    def test_tracked_requests_produce_buckets_through_the_hooks(self) -> None:
        manager = self._manager()
        clock = _Clock()
        manager.live_history = LiveHistory(manager, clock=clock)

        rid = manager._track_start("model-a", streaming=True)
        for _ in range(4):
            clock.advance(1.0)
            manager.live_history.tick()
        # A request that streams: one reasoning token, then visible output.
        manager._track_output(rid, clock.now, "thinking", 4)
        clock.advance(1.0)
        manager.live_history.tick()
        manager._track_output(rid, clock.now, "output", 46)
        manager._track_prompt_processing(rid, 1_200, 2.0)
        clock.advance(1.0)
        manager.live_history.tick()
        clock.advance(1.0)
        manager.live_history.tick()
        manager._track_end(rid)

        series = manager.live_history.series()[0]
        self.assertEqual(series["model"], "model-a")
        self.assertEqual(series["live_sessions"], 0)
        bucket = series["buckets"][0]
        self.assertAlmostEqual(bucket["thinking_tok_s"], 4.0 / 5.0, places=2)
        self.assertAlmostEqual(bucket["output_tok_s"], 46.0 / 5.0, places=2)
        self.assertTrue(bucket["prefill_measured"])
        self.assertAlmostEqual(bucket["prefill_tok_s"], 600.0, places=2)

    def test_a_finished_request_is_released_from_the_live_count(self) -> None:
        manager = self._manager()
        clock = _Clock()
        manager.live_history = LiveHistory(manager, clock=clock)

        rid = manager._track_start("model-a", streaming=True)
        clock.advance(BUCKET_SECONDS)
        manager.live_history.tick()
        self.assertEqual(manager.live_history.live_keys(), ("model-a",))
        manager._track_end(rid)
        clock.advance(BUCKET_SECONDS)
        # Enough idle samples that the newest bucket never saw the request.
        for _ in range(6):
            manager.live_history.tick()
            clock.advance(1.0)
        manager.live_history.tick()

        self.assertEqual(manager.live_history.live_keys(), ())
        entry = manager.live_history.series()[0]
        self.assertEqual(entry["live_sessions"], 0)
        self.assertEqual(entry["buckets"][-1]["concurrent"], 0.0)
        self.assertEqual(entry["buckets"][-1]["output_tok_s"], 0.0)

    def test_a_manager_without_a_collector_still_tracks_requests(self) -> None:
        """A bare manager must not need a collector to work."""
        manager = self._manager()
        rid = manager._track_start("model-a", streaming=True)
        manager._track_output(rid, 0.0, "output", 3)
        manager._track_prompt_processing(rid, 100, 1.0)
        manager._track_end(rid)

        self.assertEqual(manager._active_reqs, {})
        self.assertIsNone(getattr(manager, "live_history", None))


class SeriesShapeTests(TestCase):
    def test_series_are_split_per_serving_unit(self) -> None:
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock)
        first = manager.start(group=GROUP_A)
        second = manager.start(key="model-b", group=GROUP_B)
        history.start(GROUP_A)
        history.start(GROUP_B)
        history.tick()
        manager.output(first, 10, "output")
        manager.output(second, 20, "output")
        clock.advance(BUCKET_SECONDS)
        history.tick()
        clock.advance(BUCKET_SECONDS)
        history.tick()

        series = {item["key"]: item for item in history.series()}
        self.assertEqual(set(series), {"deployment-a:0", "deployment-b:0"})
        self.assertAlmostEqual(series["deployment-a:0"]["buckets"][0]["output_tok_s"], 2.0, places=2)
        self.assertAlmostEqual(series["deployment-b:0"]["buckets"][0]["output_tok_s"], 4.0, places=2)
        self.assertEqual(series["deployment-a:0"]["node_names"], ["node-a"])

    def test_admission_running_count_covers_untracked_work(self) -> None:
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock)
        manager.start()
        history.start(GROUP_A)
        # The engine is serving three sessions while only one has token
        # evidence; the authoritative count must win without double counting.
        manager._admission = {"target": {**GROUP_A, "running": 3, "limit": 8}}
        history.tick()
        clock.advance(BUCKET_SECONDS)
        history.tick()

        bucket = history.series()[0]["buckets"][0]
        self.assertEqual(bucket["concurrent"], 3.0)
        self.assertEqual(bucket["concurrent_peak"], 3)

    def test_admission_only_unit_appears_without_tracked_requests(self) -> None:
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock)
        manager._admission = {"target": {**GROUP_B, "running": 2, "limit": 4}}
        history.tick()
        clock.advance(BUCKET_SECONDS)
        history.tick()

        series = history.series()
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0]["key"], "deployment-b:0")
        self.assertEqual(series[0]["buckets"][0]["concurrent"], 2.0)

    def test_snapshot_describes_the_trailing_range(self) -> None:
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock)
        snapshot = history.snapshot()

        self.assertEqual(snapshot["bucket_seconds"], BUCKET_SECONDS)
        self.assertEqual(snapshot["range_seconds"], 720 * BUCKET_SECONDS)
        self.assertEqual(snapshot["series"], [])

    def test_live_session_counts_are_reported_per_unit(self) -> None:
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock)
        manager.start()
        history.start(GROUP_A)
        clock.advance(BUCKET_SECONDS)
        history.tick()
        clock.advance(BUCKET_SECONDS)
        history.tick()

        self.assertEqual(history.live_keys(), ("deployment-a:0",))
        self.assertEqual(history.series()[0]["live_sessions"], 1)
        history.end(GROUP_A)
        self.assertEqual(history.live_keys(), ())

    def test_state_record_matches_the_dashboard_classification(self) -> None:
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock)
        prefilling = manager.start()
        thinking = manager.start()
        generating = manager.start()
        manager.output(thinking, 3, "thinking")
        manager.output(generating, 5, "output")
        history.tick()
        clock.advance(BUCKET_SECONDS)
        history.tick()

        state = history.series()[0]["state"]
        self.assertEqual(state, {
            "output_sessions": 1, "thinking_sessions": 1, "prefill_sessions": 1,
        })
        self.assertEqual(manager._active_reqs[prefilling]["total_tokens"], 0)
