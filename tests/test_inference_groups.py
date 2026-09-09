import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from manager import Manager


def grouped_manager():
    manager = Manager.__new__(Manager)
    manager._active_reqs = {}
    manager._req_seq = 0
    manager._trailing_window = 5.0
    manager._save_deployments = Mock()
    manager.deployments = [{
        "id": "split", "mode": "grouped_sharded", "model": "model",
        "launch_settings": {}, "members": [
            {"node_id": f"node-{i}", "node_name": f"Node {i}",
             "container_name": f"engine-{i}", "instance_id": (i - 1) // 2,
             "rank": (i - 1) % 2}
            for i in range(1, 5)
        ],
    }]
    return manager


def test_group_sessions_rates_and_legacy_model_aggregation():
    manager = grouped_manager()
    first = manager._track_start("model", deployment_id="split", container_name="engine-1")
    second = manager._track_start("model", deployment_id="split", container_name="engine-3")
    manager._track_prompt_processing(first, 100, 2)
    manager._track_prompt_processing(second, 300, 3)
    groups = manager.active_request_groups()
    assert groups["split:instance:0"]["connections"] == 1
    assert groups["split:instance:1"]["connections"] == 1
    assert groups["split:instance:0"]["node_names"] == ["Node 1", "Node 2"]
    assert groups["split:instance:1"]["node_names"] == ["Node 3", "Node 4"]
    assert groups["split:instance:0"]["pp_tok_s"] == 50
    assert groups["split:instance:1"]["pp_tok_s"] == 100
    assert manager.active_requests()["model"]["connections"] == 2
    manager._active_reqs[first]["paused"] = True
    assert "split:instance:0" not in manager.active_request_groups()
    manager._track_end(first)
    manager._track_end(second)
    assert manager.active_request_groups() == {}


def test_remote_groups_track_unlimited_nonstream_requests_without_caller_ip():
    async def run():
        manager = grouped_manager()
        release = asyncio.Event()
        async def response(*args, **kwargs):
            await release.wait()
            return {"choices": []}
        manager.node_registry = Mock()
        manager.node_registry.request = AsyncMock(side_effect=response)
        deployment = manager.deployments[0]
        tasks = [asyncio.create_task(manager._proxy_cluster_member(
            deployment, deployment["members"][i], "model", {}, "chat/completions", None,
        )) for i in (0, 2)]
        for _ in range(5):
            await asyncio.sleep(0)
        try:
            assert {k: v["connections"] for k, v in manager.active_request_groups().items()} == {
                "split:instance:0": 1, "split:instance:1": 1,
            }
        finally:
            release.set()
            await asyncio.gather(*tasks)
        assert manager.active_request_groups() == {}
    asyncio.run(run())


def test_group_admission_queue_does_not_double_count_running_tracking():
    async def run():
        manager = grouped_manager()
        container = {"name": "engine-1", "load_settings": {"max_concurrency": 1}}
        target = await manager._acquire_inference_slot(container, "model", None)
        rid = manager._track_start("model", deployment_id="split", container_name="engine-1")
        pending = asyncio.create_task(manager._acquire_inference_slot(container, "model", None))
        await asyncio.sleep(0)
        group = manager.active_request_groups()["split:instance:0"]
        assert group["connections"] == 1
        assert group["queued"] == 1
        assert manager.inference_admission()[target]["group_id"] == "split:instance:0"
        manager._track_end(rid)
        manager._release_inference_slot(target)
        pending_lease = await pending
        manager._release_inference_slot(pending_lease)
        assert manager.active_request_groups() == {}
    asyncio.run(run())


def test_group_prompt_processing_rate_sums_concurrent_request_rates():
    manager = grouped_manager()
    first = manager._track_start("model", deployment_id="split", container_name="engine-1")
    second = manager._track_start("model", deployment_id="split", container_name="engine-1")
    manager._track_prompt_processing(first, 100, 2.0)
    manager._track_prompt_processing(second, 60, 2.0)
    group = manager.active_request_groups()["split:instance:0"]
    assert group["connections"] == 2
    assert group["pp_tok_s"] == 80.0
    assert manager.active_requests()["model"]["pp_tok_s"] == 40.0


def test_remote_stream_group_sessions_close_independently():
    async def run():
        manager = grouped_manager()
        class Response:
            status_code = 200
            async def aiter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"hello"},"token_ids":[1,2]}]}'
                yield "data: [DONE]"
            async def aclose(self):
                pass
        manager.node_registry = Mock()
        manager.node_registry.open_stream = AsyncMock(side_effect=lambda *a, **k: Response())
        deployment = manager.deployments[0]
        streams = [await manager._proxy_cluster_member(
            deployment, deployment["members"][i], "model", {"stream": True},
            "chat/completions", None,
        ) for i in (0, 2)]
        assert len(manager.active_request_groups()) == 2
        await streams[0].__anext__()
        assert manager.active_request_groups()["split:instance:0"]["decoded_tokens"] == 2
        await streams[0].aclose()
        assert list(manager.active_request_groups()) == ["split:instance:1"]
        async for _ in streams[1]:
            pass
        assert manager.active_request_groups() == {}
    asyncio.run(run())
