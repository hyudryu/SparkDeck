"""Live session-state classification for the dashboard "Current inference" panel.

The dashboard reports how many live sessions are outputting, thinking, or still
prompt processing. Prompt processing has no measurable rate before the first
output token, so it is reported as a held-session count plus the longest wait.
"""

import time
from collections import deque
from unittest import TestCase, mock

import manager as manager_module
from manager import Manager

GROUP = {
    "group_id": "deployment:0", "model": "model", "deployment_id": "deployment",
    "instance_id": 0, "node_names": ["node-a"],
}


def _manager() -> Manager:
    instance = Manager.__new__(Manager)
    instance._req_seq = 0
    instance._active_reqs = {}
    instance._trailing_window = 5.0
    instance.inference_admission = lambda: {}
    return instance


class SessionStateCountTests(TestCase):
    def test_prefill_only_session_counts_as_prompt_processing(self) -> None:
        manager = _manager()
        with mock.patch.object(manager_module.time, "monotonic", return_value=100.0):
            manager._track_start("model", streaming=True)
            rates = manager.active_requests()["model"]

        self.assertEqual(rates["connections"], 1)
        self.assertEqual(rates["prefill_sessions"], 1)
        self.assertEqual(rates["output_sessions"], 0)
        self.assertEqual(rates["thinking_sessions"], 0)
        self.assertEqual(rates["prefill_seconds"], 0.0)

    def test_thinking_session_is_not_counted_as_output_or_prefill(self) -> None:
        manager = _manager()
        with mock.patch.object(manager_module.time, "monotonic", return_value=100.0):
            request_id = manager._track_start("model", streaming=True)
            manager._track_output(request_id, 100.0, "thinking", 4)
            rates = manager.active_requests()["model"]

        self.assertEqual(rates["thinking_sessions"], 1)
        self.assertEqual(rates["output_sessions"], 0)
        self.assertEqual(rates["prefill_sessions"], 0)

    def test_visible_output_wins_over_reasoning_in_the_same_session(self) -> None:
        manager = _manager()
        with mock.patch.object(manager_module.time, "monotonic", return_value=100.0):
            request_id = manager._track_start("model", streaming=True)
            manager._track_output(request_id, 100.0, "thinking", 3)
            manager._track_output(request_id, 100.0, "output", 2)
            rates = manager.active_requests()["model"]

        self.assertEqual(rates["output_sessions"], 1)
        self.assertEqual(rates["thinking_sessions"], 0)
        self.assertEqual(rates["prefill_sessions"], 0)

    def test_states_are_counted_across_sessions_in_one_entry(self) -> None:
        manager = _manager()
        with mock.patch.object(manager_module.time, "monotonic", return_value=100.0):
            prefilling = manager._track_start("model", streaming=True)
            thinking = manager._track_start("model", streaming=True)
            generating = manager._track_start("model", streaming=True)
            manager._track_output(thinking, 100.0, "thinking", 5)
            manager._track_output(generating, 100.0, "output", 7)
            # A completed prefill with decoded output is generating, not prefilling.
            manager._track_prompt_processing(generating, 900, 1.5)
            rates = manager.active_requests()["model"]

        self.assertEqual(rates["connections"], 3)
        self.assertEqual(rates["output_sessions"], 1)
        self.assertEqual(rates["thinking_sessions"], 1)
        self.assertEqual(rates["prefill_sessions"], 1)
        self.assertIn(prefilling, manager._active_reqs)

    def test_prefill_seconds_report_the_longest_held_session(self) -> None:
        manager = _manager()
        with mock.patch.object(manager_module.time, "monotonic", return_value=100.0):
            manager._track_start("model", streaming=True)
        with mock.patch.object(manager_module.time, "monotonic", return_value=112.5):
            manager._track_start("model", streaming=True)
            rates = manager.active_requests()["model"]

        self.assertEqual(rates["prefill_sessions"], 2)
        self.assertAlmostEqual(rates["prefill_seconds"], 12.5, places=3)

    def test_drained_window_never_reports_a_decoded_session_as_prefilling(self) -> None:
        """A request that already generated is never called prompt processing."""
        manager = _manager()
        with mock.patch.object(manager_module.time, "monotonic", return_value=100.0):
            request_id = manager._track_start("model", streaming=True)
            manager._track_output(request_id, 100.0, "output", 4)
            self.assertEqual(
                manager.active_requests()["model"]["output_sessions"], 1
            )
        # Long after the 5s trailing window drained the session is no longer
        # emitting tokens, but it is still never prefilling: it already decoded,
        # so it is left out of the prefill count rather than misreported.
        with mock.patch.object(manager_module.time, "monotonic", return_value=130.0):
            rates = manager.active_requests()["model"]

        self.assertEqual(rates["output_sessions"], 0)
        self.assertEqual(rates["thinking_sessions"], 0)
        self.assertEqual(rates["prefill_sessions"], 0)
        self.assertEqual(rates["connections"], 1)

    def test_paused_replay_session_is_not_counted_in_any_state(self) -> None:
        manager = _manager()
        with mock.patch.object(manager_module.time, "monotonic", return_value=100.0):
            running = manager._track_start("model", streaming=True)
            paused = manager._track_start("model", streaming=True)
            manager._active_reqs[paused]["paused"] = True
            rates = manager.active_requests()["model"]

        self.assertEqual(rates["connections"], 1)
        self.assertEqual(rates["prefill_sessions"], 1)
        self.assertEqual(rates["output_sessions"], 0)
        self.assertEqual(rates["thinking_sessions"], 0)
        self.assertIn(running, manager._active_reqs)

    def test_grouped_entries_carry_their_own_state_counts(self) -> None:
        manager = _manager()
        second_group = {
            **GROUP, "group_id": "deployment:1", "instance_id": 1,
            "node_names": ["node-b"],
        }
        with mock.patch.object(manager_module.time, "monotonic", return_value=100.0):
            first = manager._track_start("model", streaming=True)
            second = manager._track_start("model", streaming=True)
            manager._active_reqs[first]["group"] = GROUP
            manager._active_reqs[second]["group"] = second_group
            manager._track_output(first, 100.0, "output", 3)
            groups = manager.active_request_groups()

        self.assertEqual(groups["deployment:0"]["output_sessions"], 1)
        self.assertEqual(groups["deployment:0"]["prefill_sessions"], 0)
        self.assertEqual(groups["deployment:1"]["output_sessions"], 0)
        self.assertEqual(groups["deployment:1"]["prefill_sessions"], 1)

    def test_admission_created_entry_exposes_zeroed_state_counts(self) -> None:
        """A group known only through the queue must not omit the new fields."""
        manager = _manager()
        manager.inference_admission = lambda: {
            "target": {
                **GROUP, "limit": 1, "running": 0, "queued": 2,
            },
        }
        entry = manager.active_request_groups()["deployment:0"]

        self.assertEqual(entry["queued"], 2)
        self.assertEqual(entry["output_sessions"], 0)
        self.assertEqual(entry["thinking_sessions"], 0)
        self.assertEqual(entry["prefill_sessions"], 0)
        self.assertIsNone(entry["prefill_seconds"])

    def test_state_counts_never_exceed_reported_connections(self) -> None:
        manager = _manager()
        with mock.patch.object(manager_module.time, "monotonic", return_value=100.0):
            prefilling = manager._track_start("model", streaming=True)
            generating = manager._track_start("model", streaming=True)
            manager._track_output(generating, 100.0, "output", 2)
            rates = manager.active_requests()["model"]

        total = (
            rates["output_sessions"] + rates["thinking_sessions"]
            + rates["prefill_sessions"]
        )
        # Each session counts in at most one state, and a session whose tokens
        # drained is counted in none, so the total is a lower bound.
        self.assertLessEqual(total, rates["connections"])
        self.assertEqual(total, 2)
        self.assertIn(prefilling, manager._active_reqs)


class SessionPrefillTelemetryTests(TestCase):
    """The prefill-only record shape used by the queue telemetry suite."""

    def test_untracked_record_fields_default_to_prefill(self) -> None:
        manager = _manager()
        manager._active_reqs = {
            1: {
                "key": "model", "group": GROUP,
                "thinking": deque(), "output": deque(),
                "started_at": time.monotonic(),
            },
        }
        rates = manager.active_request_groups()["deployment:0"]

        self.assertEqual(rates["prefill_sessions"], 1)
        self.assertEqual(rates["output_sessions"], 0)
        self.assertIsNotNone(rates["prefill_seconds"])
