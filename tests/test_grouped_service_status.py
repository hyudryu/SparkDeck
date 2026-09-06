import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from sparkdeck.service import SparkDeckService, _deployment_launch_progress, _grouped_instance_summary
from manager import Manager
from cluster import NodeRegistry, AGENT_PROTOCOL_VERSION


def split_deployment(stopped_status):
    return {
        "mode": "grouped_sharded", "status": "degraded", "desired_state": "running",
        "members": [
            {
                "node_id": f"node-{i + 1}", "instance_id": i // 2, "rank": i % 2,
                "status": "running" if i < 2 else stopped_status,
                "desired_state": "running" if i < 2 else "stopped",
                "phase": {"phase": "ready" if i < 2 else stopped_status},
            }
            for i in range(4)
        ],
    }


@pytest.mark.parametrize("stopped_status", ["exited", "dead", "removed", "stopped"])
def test_stopped_peer_group_does_not_appear_to_be_starting(stopped_status):
    deployment = split_deployment(stopped_status)

    groups = _grouped_instance_summary(deployment)

    assert [(group["instance_id"], group["status"], group["desired_state"]) for group in groups] == [
        (0, "running", "running"), (1, "stopped", "stopped"),
    ]
    assert _deployment_launch_progress(deployment)["launch_phase"] == "ready"


def test_recreating_group_progress_remains_visible_despite_stopped_intent():
    deployment = split_deployment("exited")
    for member in deployment["members"][2:]:
        member.update({
            "status": "queued", "recreate_pending": True,
            "phase": {"phase": "pulling_image", "message": "Downloading group image"},
        })

    assert _grouped_instance_summary(deployment)[1]["status"] == "starting"
    assert _deployment_launch_progress(deployment) == {
        "launch_phase": "pulling_image", "launch_message": "Downloading group image",
    }


def test_group_stop_transition_stays_stopping_as_other_ranks_exit():
    deployment = split_deployment("exited")
    deployment["members"][2]["status"] = "stopping"

    assert _grouped_instance_summary(deployment)[1]["status"] == "stopping"


@pytest.mark.parametrize("phase", [None, "starting", "loading", "initializing"])
def test_running_containers_wait_for_their_own_group_engine_readiness(phase):
    deployment = split_deployment("running")
    for member in deployment["members"]:
        member["desired_state"] = "running"
    primary, worker = deployment["members"][2:]
    primary["phase"] = {"phase": phase} if phase else None
    worker["phase"] = {"phase": "initializing"}

    assert [group["status"] for group in _grouped_instance_summary(deployment)] == [
        "running", "starting",
    ]

    primary["phase"] = {"phase": "ready"}
    # Headless workers never expose the API; the primary's own readiness
    # signal is sufficient while all ranks' containers remain running.
    assert [group["status"] for group in _grouped_instance_summary(deployment)] == [
        "running", "running",
    ]

    worker["status"] = "restarting"
    assert _grouped_instance_summary(deployment)[1]["status"] == "starting"


def test_ready_peer_or_worker_cannot_substitute_for_missing_primary_readiness():
    deployment = split_deployment("running")
    for member in deployment["members"][2:]:
        member["desired_state"] = "running"
        member["phase"] = {"phase": "ready"}
    deployment["members"] = [member for member in deployment["members"]
                             if not (member["instance_id"] == 1 and member["rank"] == 0)]

    assert _grouped_instance_summary(deployment)[1]["status"] == "starting"


@pytest.mark.parametrize("mode", ["sharded", "grouped_sharded"])
@pytest.mark.parametrize("marker", ["Application startup complete", "Uvicorn running on http://0.0.0.0:8000"])
def test_group_api_rank_ignores_old_ready_logs_until_current_probe_succeeds(mode, marker):
    manager = Manager.__new__(Manager)
    manager._check_ready = AsyncMock(return_value=False)
    manager.get_logs = AsyncMock(return_value=marker)
    container = {"name": "group-api", "status": "running", "rank": 0, "deployment_mode": mode}

    phase = asyncio.run(manager._get_container_phase(container))
    assert phase["phase"] == "starting"
    assert "API" in phase["message"]

    manager._check_ready.return_value = True
    assert asyncio.run(manager._get_container_phase(container))["phase"] == "ready"


@pytest.mark.parametrize("prefix", ["", "Application startup complete\nUvicorn running on http://0.0.0.0:8000\n"])
def test_group_api_rank_preserves_weight_loading_progress_after_failed_probe(prefix):
    manager = Manager.__new__(Manager)
    manager._check_ready = AsyncMock(return_value=False)
    manager.get_logs = AsyncMock(return_value=prefix + "Loading checkpoint shards: 1/4")
    container = {"name": "group-api", "status": "running", "rank": 0, "deployment_mode": "grouped_sharded"}

    assert asyncio.run(manager._get_container_phase(container))["phase"] == "loading"


@pytest.mark.parametrize("current_log, expected_phase, expected_progress", [
    ("Starting process", "starting", None),
    ("Loading checkpoint shards: 1/4", "loading", 0.25),
])
def test_failed_probe_discards_loading_before_last_successful_startup(current_log, expected_phase, expected_progress):
    manager = Manager.__new__(Manager)
    manager._check_ready = AsyncMock(return_value=False)
    manager.get_logs = AsyncMock(return_value="\n".join([
        "Application startup complete", "Loading checkpoint shards: 4/4",
        "Uvicorn running on http://0.0.0.0:8000", current_log,
    ]))
    container = {"name": "group-api", "status": "running", "rank": 0, "deployment_mode": "grouped_sharded"}

    phase = asyncio.run(manager._get_container_phase(container))

    assert phase["phase"] == expected_phase
    assert phase["progress"] == expected_progress


@pytest.mark.parametrize("peer_status", ["starting", "stopped"])
def test_group_action_response_probes_ready_sibling_instead_of_saved_startup_phase(peer_status):
    saved = split_deployment(peer_status)
    for member in saved["members"][:2]:
        member["phase"] = {"phase": "starting"}
    saved.update(id="cluster-1", node_ids=["node-1", "node-2", "node-3", "node-4"])
    live = copy.deepcopy(saved)
    live["members"][0]["phase"] = {"phase": "ready"}
    service = SparkDeckService.__new__(SparkDeckService)
    service.manager = SimpleNamespace(deployments=[saved], get_state=AsyncMock(return_value={"deployments": [live]}))
    service._layout_contract = Mock(return_value={})

    response = asyncio.run(service._grouped_action_response({}, "cluster-1"))

    assert [group["status"] for group in response["instances"]] == ["running", peer_status]
    service.manager.get_state.assert_awaited_once()
    assert saved["members"][0]["phase"]["phase"] == "starting"


@pytest.mark.parametrize("mode, rank", [("single", 0), ("grouped_sharded", 1)])
def test_other_runtime_log_phase_behavior_is_unchanged(mode, rank):
    manager = Manager.__new__(Manager)
    manager._check_ready = AsyncMock(return_value=False)
    manager.get_logs = AsyncMock(return_value="Application startup complete")
    container = {"name": "runtime", "status": "running", "rank": rank, "deployment_mode": mode}

    assert asyncio.run(manager._get_container_phase(container))["phase"] == "ready"


@pytest.mark.parametrize("action", ["start", "stop", "remove"])
@pytest.mark.parametrize("fails", [False, True])
def test_remote_group_action_invalidates_cached_ready_inventory(action, fails):
    manager = Manager.__new__(Manager)
    cache = {"remote": (0, {"phase": "ready"}), "peer": (0, {"phase": "ready"})}
    manager.node_registry = SimpleNamespace(
        _status_cache=cache,
        invalidate_status=lambda node_id: cache.pop(node_id, None),
        request=AsyncMock(side_effect=RuntimeError("agent disconnected") if fails else None),
    )
    member = {"node_id": "remote", "container_name": "group-r0"}
    if fails:
        with pytest.raises(RuntimeError, match="agent disconnected"):
            asyncio.run(manager._member_action(member, action))
    else:
        asyncio.run(manager._member_action(member, action))
    assert "remote" not in cache
    assert "peer" in cache


def test_lifecycle_invalidation_discards_inflight_ready_probe(tmp_path):
    async def exercise():
        registry = NodeRegistry(tmp_path, None, "controller")
        node = {"id": "remote", "name": "Remote", "agent_url": "http://remote:7878", "enabled": True}
        old_probe = asyncio.Event()
        release = asyncio.Event()
        status_calls = 0

        async def request(node_id, method, path, **kwargs):
            nonlocal status_calls
            common = {"protocol_version": AGENT_PROTOCOL_VERSION, "name": "Remote", "docker_ready": True}
            if path == "/api/agent/status":
                status_calls += 1
                if status_calls == 1:
                    old_probe.set()
                    await release.wait()
                    return {**common, "containers": [{"phase": {"phase": "ready"}}]}
                return {**common, "containers": [{"phase": {"phase": "starting"}}]}
            return common

        registry.request = AsyncMock(side_effect=request)
        pending = asyncio.create_task(registry.probe(node))
        await old_probe.wait()
        registry.invalidate_status("remote")
        release.set()
        result = await pending
        assert result["containers"][0]["phase"]["phase"] == "starting"
        assert registry._status_cache["remote"][1] == result
        assert status_calls == 2

    asyncio.run(exercise())


def test_remote_group_start_reads_fresh_starting_inventory(tmp_path):
    async def exercise():
        registry = NodeRegistry(tmp_path, None, "controller")
        node = {"id": "remote", "name": "Remote", "agent_url": "http://remote:7878", "enabled": True}
        registry._status_cache["remote"] = (float("inf"), {
            "docker_ready": True, "containers": [{"phase": {"phase": "ready"}}],
        })
        registry.request = AsyncMock(return_value={
            "protocol_version": AGENT_PROTOCOL_VERSION, "name": "Remote", "docker_ready": True,
            "containers": [{"phase": {"phase": "starting"}}],
        })
        manager = Manager.__new__(Manager)
        manager.node_registry = registry
        await manager._member_action({"node_id": "remote", "container_name": "rank-0"}, "start")

        assert (await registry.probe(node))["containers"][0]["phase"]["phase"] == "starting"
        assert registry.request.await_count == 3

    asyncio.run(exercise())


@pytest.mark.parametrize("fails", [False, True])
def test_remote_group_create_replaces_preflight_empty_inventory(tmp_path, fails):
    async def exercise():
        registry = NodeRegistry(tmp_path, None, "controller")
        node = {"id": "remote", "name": "Remote", "agent_url": "http://remote:7878", "enabled": True}
        registry._status_cache["remote"] = (float("inf"), {"docker_ready": True, "containers": []})
        registry._status_cache["peer"] = (float("inf"), {"docker_ready": True, "containers": []})
        created = {"name": "new-rank-0", "status": "running", "phase": {"phase": "starting"}}

        async def request(node_id, method, path, **kwargs):
            if method == "POST":
                if fails:
                    raise RuntimeError("agent disconnected after creation")
                return created
            return {
                "protocol_version": AGENT_PROTOCOL_VERSION, "name": "Remote", "docker_ready": True,
                "containers": [created],
            }

        registry.request = AsyncMock(side_effect=request)
        manager = Manager.__new__(Manager)
        manager.node_registry = registry
        if fails:
            with pytest.raises(RuntimeError, match="agent disconnected"):
                await manager._create_member("remote", {"name": "new-rank-0"})
        else:
            await manager._create_member("remote", {"name": "new-rank-0"})

        assert (await registry.probe(node))["containers"] == [created]
        assert registry._status_generations["remote"] == 1
        assert "peer" in registry._status_cache

    asyncio.run(exercise())


def test_unexpected_peer_failure_still_surfaces_in_progress():
    deployment = split_deployment("dead")
    for member in deployment["members"][2:]:
        member["desired_state"] = "running"

    assert _deployment_launch_progress(deployment)["launch_phase"] == "error"


@pytest.mark.parametrize("linked", [True, False])
def test_group_action_returns_fresh_instances_without_inventory_reconcile(linked):
    cluster = split_deployment("exited")
    cluster.update({
        "id": "cluster-1", "node_ids": [f"node-{i}" for i in range(1, 5)],
        "launch_settings": {"deployment_mode": "grouped_sharded", "tensor_parallel_size": 2},
    })
    saved = {
        "id": "saved-1", "kind": "managed", "container_name": "rank-0",
        "settings": {"manager_deployment_id": "cluster-1"} if linked else {},
    }
    service = SparkDeckService.__new__(SparkDeckService)
    service.store = Mock()
    service.store.deployment.side_effect = lambda *args, **kwargs: dict(saved)
    service._owning_cluster_deployment = Mock(return_value=None if linked else cluster)

    async def start_group(*args, **kwargs):
        assert kwargs == {"instance": 1}
        for member in cluster["members"][2:]:
            member.update({
                "desired_state": "running", "status": "starting",
                "phase": {"phase": "loading", "message": "Loading weights"},
            })
        cluster["status"] = "running"
        cluster["last_deployed_at"] = 1700000000
        return {"ok": True, "status": "running"}

    service.manager = SimpleNamespace(
        deployments=[cluster], deployment_action=AsyncMock(side_effect=start_group),
        recipe_deployment_contract=lambda settings: {
            **settings, "required_node_count": 4,
        },
        list_containers=AsyncMock(side_effect=AssertionError("must not reconcile")),
    )

    result = asyncio.run(service._deployment_action_locked("saved-1", "start", instance=1))

    assert result["status"] == "running"
    assert result["desired_state"] == "running"
    assert result["deployment_mode"] == "grouped_sharded"
    assert result["instance_node_count"] == 2
    assert result["required_node_count"] == 4
    assert result["node_ids"] == cluster["node_ids"]
    assert result["instances"][0]["status"] == "running"
    assert result["instances"][1]["status"] == "starting"
    assert result["instances"][1]["desired_state"] == "running"
    assert result["launch_phase"] == "loading"
    assert result["launch_message"] == "Loading weights"
    assert result["last_deployed_at"] == "2023-11-14T22:13:20+00:00"
    service.manager.list_containers.assert_not_awaited()
