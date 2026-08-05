import json
import unittest
from unittest import mock

import manager as manager_module
from manager import Manager


def sse(choice: dict) -> str:
    return f"data: {json.dumps({'choices': [choice]})}"


class StreamTokenRateTests(unittest.TestCase):
    def test_counts_all_mtp_token_ids_in_one_chunk(self) -> None:
        line = sse({
            "delta": {"content": " several tokens"},
            "token_ids": [10, 11, 12],
        })

        self.assertEqual(Manager._sse_chunk_token_counts(line), (0, 3))

    def test_recognizes_vllm_reasoning_field(self) -> None:
        line = sse({
            "delta": {"reasoning": "Thinking"},
            "token_ids": [20, 21],
        })

        self.assertEqual(Manager._sse_chunk_token_counts(line), (2, 0))

    def test_falls_back_to_one_token_when_ids_are_unavailable(self) -> None:
        line = sse({
            "delta": {"reasoning_content": "Thinking"},
            "token_ids": None,
        })

        self.assertEqual(Manager._sse_chunk_token_counts(line), (1, 0))

    def test_live_rate_uses_token_count_not_event_count(self) -> None:
        instance = Manager.__new__(Manager)
        instance._req_seq = 0
        instance._active_reqs = {}
        instance._trailing_window = 5.0

        with mock.patch.object(manager_module.time, "monotonic", return_value=100.0):
            request_id = instance._track_start("model", streaming=True)
            instance._track_output(request_id, 99.0, "output", count=3)
            rates = instance.active_requests()

        self.assertEqual(rates["model"]["output_tok_s"], 3.0)


if __name__ == "__main__":
    unittest.main()
