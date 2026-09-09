import time
from collections import deque
from unittest import TestCase

from manager import Manager


class GroupPromptTelemetryTests(TestCase):
    def setUp(self):
        self.manager = Manager.__new__(Manager)
        self.group = {
            "group_id": "deployment:0", "model": "model",
            "deployment_id": "deployment", "instance_id": 0,
            "node_names": ["node-a", "node-b"],
        }
        self.manager._active_reqs = {
            index: {
                "key": "model", "group": self.group,
                "thinking": deque(), "output": deque(),
                "pp_tokens": 100 if index else 0,
                "pp_time_s": 1.0 if index else 0.0,
            }
            for index in range(3)
        }
        self.manager._prompt_waiting_requests = {
            "fourth": {
                "model": "model", "group": self.group,
                "created_at": time.monotonic() - 2,
            },
        }

    def test_unlimited_total_concurrency_reports_three_active_one_queued(self):
        admission = list(self.manager.inference_admission().values())
        self.assertEqual(len(admission), 1)
        self.assertEqual(admission[0]["running"], 3)
        self.assertEqual(admission[0]["queued"], 1)
        self.assertEqual(admission[0]["group_id"], self.group["group_id"])
        self.assertGreaterEqual(admission[0]["oldest_wait_seconds"], 2)
        group = self.manager.active_request_groups()[self.group["group_id"]]
        self.assertEqual(group["connections"], 3)
        self.assertEqual(group["queued"], 1)
        self.assertEqual(group["pp_measuring"], 1)

    def test_total_admission_merges_prompt_queue_without_double_counting(self):
        self.manager._inference_admission = {
            "target": {
                "model": "model", "group": self.group,
                "limit": 4, "running": 3, "waiters": deque(),
            },
        }
        admission = self.manager.inference_admission()
        self.assertEqual(list(admission), ["target"])
        self.assertEqual(admission["target"]["running"], 3)
        self.assertEqual(admission["target"]["queued"], 1)
        group = self.manager.active_request_groups()[self.group["group_id"]]
        self.assertEqual(group["connections"], 3)
        self.assertEqual(group["queued"], 1)

    def test_waiter_removal_clears_queue_without_changing_active_count(self):
        self.manager._prompt_waiting_requests.clear()
        self.assertEqual(self.manager.inference_admission(), {})
        group = self.manager.active_request_groups()[self.group["group_id"]]
        self.assertEqual(group["connections"], 3)
        self.assertEqual(group.get("queued", 0), 0)

    def test_independent_groups_keep_their_own_queues(self):
        other = {**self.group, "group_id": "deployment:1", "instance_id": 1,
                 "node_names": ["node-c", "node-d"]}
        self.manager._prompt_waiting_requests["other"] = {
            "model": "model", "group": other, "created_at": time.monotonic(),
        }
        groups = self.manager.active_request_groups()
        self.assertEqual(groups[self.group["group_id"]]["connections"], 3)
        self.assertEqual(groups[self.group["group_id"]]["queued"], 1)
        self.assertEqual(groups[other["group_id"]]["connections"], 0)
        self.assertEqual(groups[other["group_id"]]["queued"], 1)
        model = self.manager.active_requests()["model"]
        self.assertEqual(model["connections"], 3)
        self.assertEqual(model["queued"], 2)
