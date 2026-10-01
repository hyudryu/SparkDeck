from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, mock


from manager import Manager


def metrics(computed, cached=10000, ttft_sum=None, ttft_count=None):
    ttft_sum = computed / 200 if ttft_sum is None else ttft_sum
    ttft_count = computed // 100 + 1 if ttft_count is None else ttft_count
    return (
        f'vllm:prompt_tokens_by_source_total{{engine="0",model_name="model",source="local_compute"}} {computed}\n'
        f'vllm:prompt_tokens_by_source_total{{engine="0",model_name="model",source="cache_hit"}} {cached}\n'
        f'vllm:prompt_tokens_cached_total{{engine="0",model_name="model"}} {cached}\n'
        f'vllm:time_to_first_token_seconds_sum{{engine="0",model_name="model"}} {ttft_sum}\n'
        f'vllm:time_to_first_token_seconds_count{{engine="0",model_name="model"}} {ttft_count}\n'
    )


class RuntimePromptRateTests(IsolatedAsyncioTestCase):
    async def test_runtime_prefill_rate_is_available_during_generation_and_resets(self):
        manager = Manager.__new__(Manager)
        container = SimpleNamespace(id="first", attrs={"State": {"StartedAt": "start"}})
        manager.client = SimpleNamespace(containers=SimpleNamespace(get=lambda _: container))
        response = SimpleNamespace(text=metrics(100), raise_for_status=lambda: None)
        manager.http = SimpleNamespace(get=mock.AsyncMock(return_value=response))
        manager._runtime_prompt_endpoint = lambda c: (c.id, c.attrs["State"]["StartedAt"], 8000)
        manager.inference_admission = lambda: {}
        manager._track_start("model", streaming=True, container_name="engine")
        request_id = manager._req_seq
        with mock.patch("manager.time.monotonic", return_value=100):
            await manager._sample_runtime_prompt_rates()
        response.text = metrics(500, cached=999999)
        with mock.patch("manager.time.monotonic", return_value=102):
            await manager._sample_runtime_prompt_rates()
            manager._track_output(request_id, 102, count=3)
            group = manager.active_request_groups()["engine"]
            model = manager.active_requests()["model"]
        assert group["pp_tok_s"] == 200
        assert model["pp_tok_s"] == 200
        assert group["pp_rate_source"] == "runtime_ttft"
        assert group["pp_sample_seconds"] == 2
        assert group["pp_tokens"] == 0  # Never claim a per-request measurement.
        with mock.patch("manager.time.monotonic", return_value=105):
            await manager._sample_runtime_prompt_rates()
            assert manager.active_request_groups()["engine"]["pp_tok_s"] == 200
        for timestamp in range(106, 145):
            with mock.patch("manager.time.monotonic", return_value=timestamp):
                await manager._sample_runtime_prompt_rates()
                assert manager.active_request_groups()["engine"]["pp_tok_s"] == 200
        response.text = metrics(10)
        with mock.patch("manager.time.monotonic", return_value=147):
            await manager._sample_runtime_prompt_rates()
            assert manager.active_request_groups()["engine"]["pp_tok_s"] is None
        response.text = metrics(110)
        with mock.patch("manager.time.monotonic", return_value=149):
            await manager._sample_runtime_prompt_rates()
            assert manager.active_request_groups()["engine"]["pp_tok_s"] == 200
        container.attrs["State"]["StartedAt"] = "restarted"
        with mock.patch("manager.time.monotonic", return_value=151):
            await manager._sample_runtime_prompt_rates()
            assert manager.active_request_groups()["engine"]["pp_tok_s"] is None


    async def test_runtime_samples_expire_and_failed_scrape_clears_baseline(self):
        manager = Manager.__new__(Manager)
        container = SimpleNamespace(id="first", attrs={"State": {"StartedAt": "start"}})
        manager.client = SimpleNamespace(containers=SimpleNamespace(get=lambda _: container))
        manager.http = SimpleNamespace(get=mock.AsyncMock(side_effect=RuntimeError("offline")))
        manager._runtime_prompt_endpoint = lambda c: (c.id, c.attrs["State"]["StartedAt"], 8000)
        manager.inference_admission = lambda: {}
        manager._track_start("model", streaming=True, container_name="engine")
        manager._runtime_prompt_samples = {
            "engine": {"model": "model", "rate": 200, "seconds": 1, "measured_at": 100, "refreshed_at": 100},
            "other": {"model": "different", "rate": 999, "seconds": 1, "measured_at": 100, "refreshed_at": 100},
        }
        manager._runtime_prompt_baselines = {"engine": ("old", 100, {}), "other": ("old", 100, {})}
        with mock.patch("manager.time.monotonic", return_value=131):
            assert manager.active_requests()["model"]["pp_tok_s"] is None
            await manager._sample_runtime_prompt_rates()
        assert manager._runtime_prompt_baselines == {}
        assert manager._runtime_prompt_samples == {}

    async def test_runtime_rate_does_not_leak_between_groups_of_same_model(self):
        manager = Manager.__new__(Manager)
        manager.inference_admission = lambda: {}
        manager._track_start("model", streaming=True, container_name="active")
        manager._runtime_prompt_samples = {
            "old": {"model": "model", "rate": 999, "seconds": 1, "refreshed_at": 100},
            "active": {"model": "model", "rate": 200, "seconds": 1, "refreshed_at": 100},
        }
        with mock.patch("manager.time.monotonic", return_value=102):
            assert manager.active_requests()["model"]["pp_tok_s"] == 200
            assert manager.active_request_groups()["active"]["pp_tok_s"] == 200

    async def test_completed_prefill_uses_native_duration_instead_of_scrape_burst(self):
        manager = Manager.__new__(Manager)
        container = SimpleNamespace(id="first", attrs={"State": {"StartedAt": "start"}})
        manager.client = SimpleNamespace(containers=SimpleNamespace(get=lambda _: container))
        response = SimpleNamespace(text=metrics(100, ttft_sum=10, ttft_count=1), raise_for_status=lambda: None)
        manager.http = SimpleNamespace(get=mock.AsyncMock(return_value=response))
        manager._runtime_prompt_endpoint = lambda _: ("first", "start", 8000)
        manager.inference_admission = lambda: {}
        manager._track_start("model", streaming=True, container_name="engine")
        with mock.patch("manager.time.monotonic", return_value=100):
            await manager._sample_runtime_prompt_rates()
        response.text = metrics(10100, cached=1000000, ttft_sum=110, ttft_count=2)
        with mock.patch("manager.time.monotonic", return_value=101):
            await manager._sample_runtime_prompt_rates()
            group = manager.active_request_groups()["engine"]
        assert group["pp_tok_s"] == 100  # 10,000 computed tokens / native 100s TTFT.
        assert group["pp_sample_seconds"] == 100
        # A new request must not inherit a previous generation's estimate,
        # even when both generations fall between adjacent dashboard polls.
        manager._track_end(manager._req_seq)
        manager._track_start("model", streaming=True, container_name="engine")
        with mock.patch("manager.time.monotonic", return_value=101.9):
            await manager._sample_runtime_prompt_rates()
            assert manager.active_request_groups()["engine"]["pp_tok_s"] is None
        response.text = metrics(20100, ttft_sum=110, ttft_count=2)
        with mock.patch("manager.time.monotonic", return_value=103):
            await manager._sample_runtime_prompt_rates()
            assert manager.active_request_groups()["engine"]["pp_tok_s"] is None


def test_runtime_parser_excludes_cache_and_nonfinite_counters():
    parsed = Manager._runtime_prompt_counters(metrics(123))
    assert sum(parsed.values()) == 123
    assert Manager._runtime_prompt_counters(metrics(float("nan"))) == {}
    assert Manager._runtime_prompt_counters(metrics(-1)) == {}
    reordered = 'vllm:prompt_tokens_by_source_total{source="local_compute",model_name="model",engine="0"} 123'
    assert Manager._runtime_prompt_counters(reordered) == parsed
    other_engine = 'vllm:prompt_tokens_by_source_total{source="local_compute",model_name="model",engine="1"} 12'
    assert sum(Manager._runtime_prompt_counters(metrics(123) + other_engine).values()) == 135


def test_runtime_endpoint_uses_exact_container_identity_and_port():
    container = SimpleNamespace(id="docker-id", attrs={
        "Config": {"Image": "vllm/vllm-openai:latest"},
        "State": {"StartedAt": "boot"},
        "NetworkSettings": {"Ports": {"8000/tcp": [{"HostPort": "8123"}]}},
    })
    assert Manager._runtime_prompt_endpoint(container) == ("docker-id", "boot", 8123)
    container.attrs["Config"]["Image"] = "lmsysorg/sglang:latest"
    assert Manager._runtime_prompt_endpoint(container) is None


def test_native_ttft_must_match_computed_token_engine_and_model():
    assert Manager._runtime_prompt_observations(metrics(123).replace(
        'time_to_first_token_seconds_sum{engine="0"',
        'time_to_first_token_seconds_sum{engine="1"',
    )) == {}
    assert Manager._runtime_prompt_observations(metrics(123).replace(
        'time_to_first_token_seconds_count{engine="0",model_name="model"}',
        'time_to_first_token_seconds_count{engine="0",model_name="other"}',
    )) == {}
