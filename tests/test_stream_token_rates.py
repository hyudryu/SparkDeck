import json
import unittest
from unittest import mock

import manager as manager_module
from manager import Manager


def sse(choice: dict) -> str:
    return f"data: {json.dumps({'choices': [choice]})}"


class StreamTokenRateTests(unittest.TestCase):
    @staticmethod
    def recording_manager() -> Manager:
        instance = Manager.__new__(Manager)
        instance.token_stats = {}
        instance.session_token_stats = {}
        instance.hourly_token_stats = {}
        instance.token_usage_sync = None
        instance._record_speed_sample = mock.Mock()
        instance._save_token_stats = mock.Mock()
        instance._save_hourly_token_stats = mock.Mock()
        instance._queue_speed_samples_save = mock.Mock()
        return instance

    def test_records_session_prompt_processing_speed_inputs(self) -> None:
        instance = self.recording_manager()

        instance._record_tokens(
            "model", 1_200, 100, gen_time_s=2.0, cached_tokens=1_000,
            pp_time_s=0.5,
        )

        session = instance.session_token_stats["model"]
        self.assertEqual(session["pp_tokens"], 200)
        self.assertEqual(session["pp_time_s"], 0.5)
        self.assertEqual(session["pp_tokens"] / session["pp_time_s"], 400)

    def test_records_vllm_prompt_token_detail_cache_hits(self) -> None:
        instance = self.recording_manager()

        instance._record_usage("model", {
            "prompt_tokens": 1_200,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 1_000},
        }, pp_time_s=0.5)

        session = instance.session_token_stats["model"]
        self.assertEqual(session["input"], 1_200)
        self.assertEqual(session["cached"], 1_000)
        self.assertEqual(session["pp_tokens"], 200)

    def test_cache_detail_must_be_explicit_for_prompt_speed(self) -> None:
        self.assertFalse(Manager._usage_has_cached_prompt_tokens({
            "prompt_tokens": 216_000,
        }))
        self.assertTrue(Manager._usage_has_cached_prompt_tokens({
            "prompt_tokens": 216_000,
            "prompt_tokens_details": {"cached_tokens": 214_000},
        }))

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

    def test_live_request_reports_pp_speed_after_first_token(self) -> None:
        instance = Manager.__new__(Manager)
        instance._req_seq = 0
        instance._active_reqs = {}
        instance._trailing_window = 5.0
        instance.inference_admission = lambda: {}

        with mock.patch.object(manager_module.time, "monotonic", return_value=100.0):
            request_id = instance._track_start("model", streaming=True)
            instance._track_prompt_processing(request_id, 1_200, 0.5)
            rates = instance.active_requests()

        self.assertEqual(rates["model"]["pp_tok_s"], 2_400.0)
        self.assertEqual(rates["model"]["pp_measuring"], 0)

    def test_live_request_marks_pp_as_measuring_during_prefill(self) -> None:
        instance = Manager.__new__(Manager)
        instance._req_seq = 0
        instance._active_reqs = {}
        instance._trailing_window = 5.0
        instance.inference_admission = lambda: {}

        instance._track_start("model", streaming=True)
        rates = instance.active_requests()

        self.assertEqual(rates["model"]["pp_measuring"], 1)
        self.assertIsNone(rates["model"]["pp_tok_s"])

    def test_non_streaming_request_remains_active_until_track_end(self) -> None:
        instance = Manager.__new__(Manager)
        instance._req_seq = 0
        instance._active_reqs = {}
        instance._trailing_window = 5.0
        instance.inference_admission = lambda: {}

        request_id = instance._track_start("model", streaming=False)

        self.assertEqual(instance.active_requests()["model"]["connections"], 1)
        self.assertIn(request_id, instance._active_reqs)

        instance._track_end(request_id)
        self.assertNotIn("model", instance.active_requests())

    def test_cluster_request_records_last_used_at_start_and_end(self) -> None:
        instance = Manager.__new__(Manager)
        instance._req_seq = 0
        instance._active_reqs = {}
        instance._trailing_window = 5.0
        instance.deployments = [{"id": "cluster-a", "members": []}]
        instance._deployment_last_used_saved_at = {}
        instance._save_deployments = mock.Mock()

        with mock.patch.object(manager_module.time, "time", side_effect=[100.0, 125.0]):
            request_id = instance._track_start(
                "model", streaming=True, deployment_id="cluster-a"
            )
            self.assertEqual(instance.deployments[0]["last_used_at"], 100.0)
            instance._track_end(request_id)

        self.assertEqual(instance.deployments[0]["last_used_at"], 125.0)
        self.assertEqual(instance._save_deployments.call_count, 2)

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
        self.assertEqual(rates["model"]["decoded_tokens"], 3)

    def test_live_decoded_tokens_include_reasoning_and_visible_output(self) -> None:
        instance = Manager.__new__(Manager)
        instance._req_seq = 0
        instance._active_reqs = {}
        instance._trailing_window = 5.0

        with mock.patch.object(manager_module.time, "monotonic", return_value=100.0):
            request_id = instance._track_start("model", streaming=True)
            instance._track_output(request_id, 99.0, "thinking", count=5)
            instance._track_output(request_id, 99.5, "output", count=2)
            rates = instance.active_requests()

        self.assertEqual(rates["model"]["decoded_tokens"], 7)

    def test_paused_replay_stream_is_not_reported_as_running(self) -> None:
        instance = Manager.__new__(Manager)
        instance._req_seq = 0
        instance._active_reqs = {}
        instance._trailing_window = 5.0

        running = instance._track_start("model", streaming=True)
        paused = instance._track_start("model", streaming=True)
        instance._active_reqs[paused]["paused"] = True

        rates = instance.active_requests()

        self.assertEqual(rates["model"]["connections"], 1)
        self.assertEqual(rates["model"]["decoded_tokens"], 0)
        self.assertIn(running, instance._active_reqs)
        self.assertIn(paused, instance._active_reqs)


class VllmStreamUsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_continuous_usage_populates_live_pp_and_records_only_final_snapshot(self) -> None:
        first_usage = {
            "prompt_tokens": 1_200,
            "completion_tokens": 1,
        }
        final_usage = {
            **first_usage,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 1_000},
        }
        lines = [
            f"data: {json.dumps({'choices': [{'delta': {'content': 'a'}, 'token_ids': [1]}], 'usage': first_usage})}",
            f"data: {json.dumps({'choices': [{'delta': {'content': 'b'}, 'token_ids': [2]}], 'usage': final_usage})}",
            "data: [DONE]",
        ]

        class Response:
            status_code = 200

            async def aiter_lines(self):
                for line in lines:
                    yield line

        class StreamContext:
            async def __aenter__(self):
                return Response()

            async def __aexit__(self, *_args):
                return None

        class Http:
            def __init__(self):
                self.body = None

            def stream(self, _method, _url, *, json, timeout):
                self.body = json
                return StreamContext()

        instance = Manager.__new__(Manager)
        instance.http = Http()
        instance._active_reqs = {}
        instance._track_start = mock.Mock(return_value=7)
        instance._track_output = mock.Mock()
        instance._track_prompt_processing = mock.Mock()
        instance._record_usage = mock.Mock()
        instance._track_end = mock.Mock()
        instance._release_inference_slot = mock.Mock()

        stream = instance._vllm_stream(
            "http://localhost/v1/chat/completions",
            {"model": "model", "stream": True},
            "model",
        )
        first_chunk = await stream.__anext__()
        self.assertIn('"completion_tokens": 1', first_chunk)
        # The real vLLM early snapshot has no cached-token detail. Treating
        # that as zero cache hits would briefly report all 1,200 tokens as
        # newly processed and wildly inflate live PP speed.
        instance._track_prompt_processing.assert_not_called()
        remaining = [chunk async for chunk in stream]

        self.assertTrue(remaining)
        self.assertTrue(
            instance.http.body["stream_options"]["continuous_usage_stats"]
        )
        pp_args = instance._track_prompt_processing.call_args.args
        self.assertEqual(pp_args[:2], (7, 200))
        self.assertGreater(pp_args[2], 0)
        instance._record_usage.assert_called_once()
        usage_args = instance._record_usage.call_args.args
        self.assertEqual(usage_args[:2], ("model", final_usage))
        self.assertGreaterEqual(usage_args[2], 0)
        self.assertGreater(usage_args[3], 0)


if __name__ == "__main__":
    unittest.main()
