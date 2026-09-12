"""Trailing live-inference history for the History panel.

The panel needs complete buckets rather than instantaneous samples, so the
collector ticks at the configured cadence and accumulates deltas from the
manager's cumulative per-request counters.  One bucket spans one sampling
interval, so the graph's time resolution follows the setting.  These tests drive
that accumulation directly with a manual clock, so no sleeping or event loop is
involved.
"""

from __future__ import annotations

import time
from collections import deque
from types import SimpleNamespace
from unittest import TestCase

from manager import Manager
from sparkdeck.live_metrics import (
    DEFAULT_HISTORY_SAMPLE_SECONDS,
    KEEP_SECONDS,
    MAX_SAMPLE_SECONDS,
    MIN_SAMPLE_SECONDS,
    RETAIN_ACTIVE_SECONDS,
    LiveHistory,
    buckets_for,
    clamp_sample_seconds,
)

# The default cadence, and therefore the default bucket span.
BUCKET_SECONDS = DEFAULT_HISTORY_SAMPLE_SECONDS

GROUP_A = {
    "group_id": "deployment-a:0", "model": "model-a", "deployment_id": "deployment-a",
    "instance_id": 0, "node_names": ["node-a"],
}
GROUP_B = {
    "group_id": "deployment-b:0", "model": "model-b", "deployment_id": "deployment-b",
    "instance_id": 0, "node_names": ["node-b"],
}


# An arbitrary but realistic epoch: the panel renders the collector's timestamps
# as times, so the tests pin the wall clock instead of whatever today happens to
# be.  The offset from the manual monotonic clock is what a real controller sees,
# where a monotonic reading is seconds since boot and days away from epoch.
EPOCH_BASE = 1_700_000_000.0


class _Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _WallClock:
    """Wall clock running alongside a manual monotonic clock.

    Reading the monotonic clock plus a fixed offset keeps the timestamps the
    panel receives deterministic while still exercising the two-clock mapping.
    ``shift`` moves the wall clock on its own, which is how a clock correction
    reaches the collector without touching the monotonic timeline.
    """

    def __init__(self, clock: _Clock, epoch: float = EPOCH_BASE) -> None:
        self.clock = clock
        self.epoch = epoch
        self.started_at = clock.now

    def __call__(self) -> float:
        return self.at(self.clock.now)

    def at(self, monotonic: float) -> float:
        """The wall-clock second the manual monotonic clock reads *monotonic*."""
        return self.epoch + (monotonic - self.started_at)

    def shift(self, seconds: float) -> None:
        """Correct the wall clock by *seconds* without moving monotonic time."""
        self.epoch += seconds


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


def _collector(
    manager: _Manager,
    clock: _Clock,
    *,
    interval: object = None,
    enabled: object = None,
    wall: _WallClock | None = None,
) -> LiveHistory:
    return LiveHistory(
        manager,
        clock=clock,
        wall_clock=wall or _WallClock(clock),
        interval_provider=None if interval is None else (lambda: interval),
        enabled_provider=None if enabled is None else (lambda: enabled),
    )


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

    def test_a_stub_manager_without_the_hook_still_tracks_requests(self) -> None:
        """Some callers drive the tracking methods on a minimal stub object.

        Several suites pass a ``SimpleNamespace`` as the manager, so the
        lifecycle notification must tolerate an object that has neither the hook
        method nor a collector.
        """
        stub = SimpleNamespace(
            _active_reqs={}, _req_seq=0, _trailing_window=5.0,
            _mark_deployment_used=lambda deployment_id: None,
            _request_group=lambda key, deployment_id=None, container_name=None: {
                "group_id": key, "model": key, "node_ids": ["local"], "node_names": [],
            },
        )

        rid = Manager._track_start(stub, "model-a", streaming=True)
        Manager._track_output(stub, rid, 0.0, "output", 2)
        Manager._track_prompt_processing(stub, rid, 100, 1.0)
        Manager._track_end(stub, rid)

        self.assertEqual(stub._active_reqs, {})


class TimestampTests(TestCase):
    """Every timestamp the panel receives is a wall-clock time it can render."""

    def test_bucket_and_series_timestamps_are_epoch_seconds(self) -> None:
        manager, clock = _Manager(), _Clock()
        wall = _WallClock(clock)
        history = _collector(manager, clock, wall=wall)
        manager.start()
        history.start(GROUP_A)
        history.tick()
        clock.advance(BUCKET_SECONDS)
        history.tick()

        series = history.series()[0]
        # The bucket closed on the boundary between the two samples, so its
        # published time is the wall-clock instant of that boundary.
        self.assertAlmostEqual(
            series["buckets"][0]["at"], wall.at(1_000.0 + BUCKET_SECONDS), places=3,
        )
        self.assertAlmostEqual(series["last_at"], wall.at(clock.now), places=3)
        self.assertAlmostEqual(history.snapshot()["generated_at"], wall.at(clock.now), places=3)

    def test_a_bucket_that_closes_between_samples_keeps_its_own_time(self) -> None:
        manager, clock = _Manager(), _Clock()
        wall = _WallClock(clock)
        history = _collector(manager, clock, wall=wall)
        manager.start()
        history.start(GROUP_A)
        history.tick()
        # The boundary passed a second ago: the sample that closes the bucket
        # lands after it, and the bucket must not inherit the later time.
        clock.advance(BUCKET_SECONDS + 1.0)
        history.tick()

        bucket = history.series()[0]["buckets"][0]
        self.assertAlmostEqual(bucket["at"], wall.at(1_000.0 + BUCKET_SECONDS), places=3)

    def test_the_panel_receives_the_wall_clock_not_seconds_since_boot(self) -> None:
        """A monotonic reading renders as a 1970 date, so it cannot be published."""
        manager = _Manager()
        manager.start()
        history = LiveHistory(
            manager,
            interval_provider=lambda: MIN_SAMPLE_SECONDS,
            enabled_provider=lambda: True,
        )
        history.start(GROUP_A)
        history.tick()
        # Close a bucket without sleeping: the collector's instant is monotonic.
        history.tick(now=time.monotonic() + MIN_SAMPLE_SECONDS + 1)

        series = history.series()[0]
        published = {
            "bucket at": series["buckets"][0]["at"],
            "series last_at": series["last_at"],
            "generated_at": history.snapshot()["generated_at"],
        }
        for label, value in published.items():
            with self.subTest(label):
                self.assertLess(
                    abs(value - time.time()), 5,
                    f"{label} must be a wall-clock time, not a monotonic reading",
                )


    def test_a_forward_clock_correction_cannot_erase_a_running_graph(self) -> None:
        """A host clock jump must not drop the points already collected."""
        manager, clock = _Manager(), _Clock()
        wall = _WallClock(clock)
        history = _collector(manager, clock, wall=wall)
        manager.start()
        history.start(GROUP_A)
        history.tick()
        clock.advance(BUCKET_SECONDS)
        history.tick()
        before = history.series()[0]

        # The clock is stepped two hours forward while the release keeps serving.
        wall.shift(7_200.0)
        clock.advance(BUCKET_SECONDS)
        history.tick()
        after = history.series()[0]

        # The graph is one timeline: the earlier point keeps the time it was
        # published with, so nothing jumps out of the trailing window.
        self.assertEqual(
            [bucket["at"] for bucket in after["buckets"][:len(before["buckets"])]],
            [bucket["at"] for bucket in before["buckets"]],
        )
        self.assertEqual(len(after["buckets"]), 2)
        self.assertAlmostEqual(after["last_at"], wall.at(clock.now) - 7_200.0, places=3)

    def test_a_backward_clock_correction_cannot_reorder_a_running_graph(self) -> None:
        manager, clock = _Manager(), _Clock()
        wall = _WallClock(clock)
        history = _collector(manager, clock, wall=wall)
        manager.start()
        history.start(GROUP_A)
        for _ in range(3):
            history.tick()
            clock.advance(BUCKET_SECONDS)
        wall.shift(-900.0)
        history.tick()

        times = [bucket["at"] for bucket in history.series()[0]["buckets"]]
        self.assertEqual(times, sorted(times))
        self.assertEqual(len(set(times)), len(times))

    def test_a_graph_anchored_after_a_correction_uses_the_corrected_clock(self) -> None:
        manager, clock = _Manager(), _Clock()
        wall = _WallClock(clock)
        history = _collector(manager, clock, wall=wall)
        stale = manager.start()
        history.start(GROUP_A)
        history.tick()
        clock.advance(BUCKET_SECONDS)
        history.tick()

        # The corrected clock is picked up by the next graph: the serving unit
        # goes idle long enough to be retired, then serves again.
        manager.finish(stale)
        history.end(GROUP_A)
        wall.shift(7_200.0)
        clock.advance(RETAIN_ACTIVE_SECONDS + 1.0)
        history.tick()
        manager.start()
        history.start(GROUP_A)
        for _ in range(2):
            clock.advance(BUCKET_SECONDS)
            history.tick()

        series = history.series()[0]
        self.assertEqual(len(series["buckets"]), 1)
        self.assertAlmostEqual(series["buckets"][0]["at"], wall.at(clock.now), places=3)


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

        self.assertTrue(snapshot["enabled"])
        self.assertEqual(snapshot["bucket_seconds"], float(BUCKET_SECONDS))
        self.assertEqual(snapshot["sample_seconds"], float(BUCKET_SECONDS))
        self.assertEqual(snapshot["range_seconds"], KEEP_SECONDS)
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


class SampleIntervalTests(TestCase):
    """The sampling cadence is a setting between one and thirty seconds."""

    def test_interval_is_clamped_to_the_supported_range(self) -> None:
        self.assertEqual(clamp_sample_seconds(0), MIN_SAMPLE_SECONDS)
        self.assertEqual(clamp_sample_seconds(-4), MIN_SAMPLE_SECONDS)
        self.assertEqual(clamp_sample_seconds(1), 1)
        self.assertEqual(clamp_sample_seconds("12"), 12)
        self.assertEqual(clamp_sample_seconds(30), MAX_SAMPLE_SECONDS)
        self.assertEqual(clamp_sample_seconds(45), MAX_SAMPLE_SECONDS)
        self.assertEqual(clamp_sample_seconds(10_000), MAX_SAMPLE_SECONDS)

    def test_unusable_settings_fall_back_to_the_default(self) -> None:
        for value in (None, "abc", "", float("nan"), object()):
            with self.subTest(value=value):
                self.assertEqual(
                    clamp_sample_seconds(value), DEFAULT_HISTORY_SAMPLE_SECONDS,
                )

    def test_a_bucket_spans_one_sampling_interval(self) -> None:
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock, interval=10)
        rid = manager.start()
        history.start(GROUP_A)
        history.tick()
        manager.output(rid, 100)
        clock.advance(10)
        history.tick()

        series = history.series()[0]
        self.assertEqual(series["bucket_seconds"], 10.0)
        # One hundred tokens over the ten seconds the bucket was open.
        self.assertAlmostEqual(series["buckets"][0]["output_tok_s"], 10.0, places=2)

    def test_changing_the_interval_takes_effect_on_the_next_tick(self) -> None:
        manager, clock = _Manager(), _Clock()
        setting = [4]
        history = LiveHistory(manager, clock=clock, interval_provider=lambda: setting[0])
        rid = manager.start()
        history.start(GROUP_A)
        history.tick()
        manager.output(rid, 40)
        clock.advance(4)
        history.tick()
        self.assertEqual(len(history.series()[0]["buckets"]), 1)

        # The operator slows the cadence down; the open bucket now spans longer.
        setting[0] = 20
        manager.output(rid, 100)
        clock.advance(5)
        history.tick()
        self.assertEqual(len(history.series()[0]["buckets"]), 1)
        clock.advance(15)
        history.tick()
        self.assertEqual(len(history.series()[0]["buckets"]), 2)
        self.assertAlmostEqual(
            history.series()[0]["buckets"][1]["output_tok_s"], 100.0 / 20.0, places=2,
        )

    def test_retention_covers_a_full_hour_at_every_cadence(self) -> None:
        # At the fastest cadence the window needs the most buckets; the deque is
        # sized for it so a chosen interval can never truncate the trailing hour.
        for seconds in (MIN_SAMPLE_SECONDS, 5, 17, MAX_SAMPLE_SECONDS):
            with self.subTest(seconds=seconds):
                self.assertGreaterEqual(buckets_for(KEEP_SECONDS, seconds) * seconds, KEEP_SECONDS)

    def test_an_unreadable_interval_falls_back_without_breaking_recording(self) -> None:
        manager, clock = _Manager(), _Clock()

        def explode():
            raise RuntimeError("settings are unavailable")

        history = LiveHistory(manager, clock=clock, interval_provider=explode)
        rid = manager.start()
        history.start(GROUP_A)
        history.tick()
        manager.output(rid, 50)
        clock.advance(DEFAULT_HISTORY_SAMPLE_SECONDS)
        history.tick()

        self.assertEqual(history.interval_seconds, DEFAULT_HISTORY_SAMPLE_SECONDS)
        self.assertEqual(len(history.series()[0]["buckets"]), 1)


class HistoryToggleTests(TestCase):
    """Recording can be switched off so a disabled panel costs nothing."""

    def test_tick_and_hooks_do_nothing_while_disabled(self) -> None:
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock, enabled=False)
        manager.start()
        history.start(GROUP_A)
        clock.advance(BUCKET_SECONDS)
        history.tick()
        clock.advance(BUCKET_SECONDS)
        history.tick()

        self.assertEqual(history.live_keys(), ())
        self.assertEqual(history.series(), [])
        self.assertEqual(history.snapshot()["series"], [])

    def test_snapshot_reports_that_recording_is_off(self) -> None:
        manager, clock = _Manager(), _Clock()
        history = _collector(manager, clock, enabled=False)

        self.assertFalse(history.snapshot()["enabled"])
        self.assertFalse(history.enabled)

    def test_re_enabling_records_from_scratch(self) -> None:
        manager, clock = _Manager(), _Clock()
        setting = [False]
        history = LiveHistory(manager, clock=clock, enabled_provider=lambda: setting[0])
        rid = manager.start()
        history.start(GROUP_A)
        clock.advance(BUCKET_SECONDS)
        history.tick()
        self.assertEqual(history.series(), [])

        setting[0] = True
        history.start(GROUP_A)
        history.tick()
        manager.output(rid, 30)
        clock.advance(BUCKET_SECONDS)
        history.tick()

        series = history.series()
        self.assertEqual(len(series), 1)
        self.assertAlmostEqual(series[0]["buckets"][0]["output_tok_s"], 6.0, places=2)

    def test_forget_drops_recorded_buckets(self) -> None:
        manager, clock = _Manager(), _Clock()
        setting = [True]
        history = LiveHistory(manager, clock=clock, enabled_provider=lambda: setting[0])
        rid = manager.start()
        history.start(GROUP_A)
        history.tick()
        manager.output(rid, 30)
        clock.advance(BUCKET_SECONDS)
        history.tick()
        self.assertEqual(len(history.series()), 1)

        setting[0] = False
        history.forget()

        self.assertEqual(history.series(), [])
        self.assertEqual(history.live_keys(), ())

    def test_an_unreadable_toggle_keeps_recording(self) -> None:
        """A failed settings read must not silently stop recording."""

        def explode():
            raise RuntimeError("settings are unavailable")

        manager = _Manager()
        history = LiveHistory(manager, clock=_Clock(), enabled_provider=explode)

        self.assertTrue(history.enabled)
