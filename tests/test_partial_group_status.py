import asyncio
import copy
from unittest.mock import AsyncMock

import pytest

from manager import Manager
import docker
from sparkdeck.service import _deployment_launch_progress, _grouped_instance_summary


def partial_deployment():
    return {
        "id": "partial", "model": "org/model", "engine": "vllm",
        "mode": "grouped_sharded", "status": "degraded", "desired_state": "running",
        "node_ids": ["node-1", "node-2", "local", "node-4"],
        "launch_settings": {
            "model": "org/model", "engine": "vllm", "deployment_mode": "grouped_sharded",
            "tensor_parallel_size": 2, "instances": 2,
            "node_ids": ["node-1", "node-2", "local", "node-4"],
        },
        "members": [
            {"node_id": node, "container_name": f"rank-{index}", "rank": index % 2,
             "instance_id": index // 2, "desired_state": "stopped" if index < 2 else "running",
             "status": "exited" if index < 2 else "running",
             "phase": {"phase": "ready" if index == 2 else "starting"}}
            for index, node in enumerate(["node-1", "node-2", "local", "node-4"])
        ],
    }


def test_partial_group_is_healthy_when_coordinator_ready_and_headless_worker_running():
    deployment = partial_deployment()
    assert Manager._grouped_deployment_status(deployment) == "running"
    assert _deployment_launch_progress(deployment)["launch_phase"] == "ready"


def test_partial_group_loading_tracks_its_coordinator():
    deployment = partial_deployment()
    deployment["members"][2]["phase"] = {"phase": "loading", "message": "Loading weights"}
    assert Manager._grouped_deployment_status(deployment) == "starting"
    assert _deployment_launch_progress(deployment) == {
        "launch_phase": "loading", "launch_message": "Loading weights",
    }


@pytest.mark.parametrize("phase", [None, "loading", "initializing", "ready"])
def test_running_group_count_requires_its_own_coordinator_readiness(phase):
    deployment = partial_deployment()
    for member in deployment["members"]:
        member.update(status="running", desired_state="running")
    deployment["members"][0]["phase"] = {"phase": phase}
    summary = _grouped_instance_summary(deployment)
    assert sum(group["status"] == "running" for group in summary) == (2 if phase == "ready" else 1)


def test_unapplied_topology_edit_does_not_change_existing_group_health():
    deployment = partial_deployment()
    deployment["settings_dirty"] = True
    deployment["launch_settings"]["tensor_parallel_size"] = 4
    assert Manager._grouped_deployment_status(deployment) == "running"
    assert _deployment_launch_progress(deployment)["launch_phase"] == "ready"


@pytest.mark.parametrize("failure", ["dead", "exited", "missing", "unreachable", "error"])
def test_unexpected_worker_failure_is_not_hidden_by_ready_coordinator(failure):
    deployment = partial_deployment()
    deployment["members"][3].update(status=failure, phase={"phase": "ready"})
    assert Manager._grouped_deployment_status(deployment) == "degraded"
    assert _deployment_launch_progress(deployment)["launch_phase"] == "error"


def test_missing_worker_record_is_not_healthy():
    deployment = partial_deployment()
    deployment["members"].pop()
    assert Manager._grouped_deployment_status(deployment) == "degraded"
    assert _deployment_launch_progress(deployment)["launch_phase"] == "error"


def test_disconnected_worker_is_not_hidden_by_stale_running_inventory():
    deployment = partial_deployment()
    deployment["members"][3]["node_status"] = "offline"
    assert Manager._grouped_deployment_status(deployment) == "degraded"
    assert _deployment_launch_progress(deployment)["launch_phase"] == "error"


def test_queued_group_recreation_remains_starting():
    deployment = partial_deployment()
    for member in deployment["members"][:2]:
        member.update(status="queued", recreate_pending=True, phase={"phase": "pulling_image"})
    assert Manager._grouped_deployment_status(deployment) == "starting"
    assert _deployment_launch_progress(deployment)["launch_phase"] == "pulling_image"


@pytest.mark.parametrize("active_online", [True, False])
def test_public_inventory_reconciles_partial_health_from_active_group(tmp_path, active_online):
    async def run():
        manager = Manager(tmp_path)
        deployment = partial_deployment()
        manager.deployments = [deployment]
        nodes = [
            {"id": member["node_id"], "online": active_online if index == 3 else index >= 2,
             "status": "online", "docker_ready": True, "containers": [{
                 "name": member["container_name"], "status": member["status"],
                 "phase": member["phase"],
             }]}
            for index, member in enumerate(deployment["members"])
        ]
        manager.list_containers = AsyncMock(return_value=copy.deepcopy(nodes[2]["containers"]))
        manager.list_images = AsyncMock(return_value=[])
        manager.get_stats = AsyncMock(return_value={})
        manager.cluster_nodes = AsyncMock(return_value=nodes)
        try:
            public = (await manager.get_state())["deployments"][0]
            assert public["status"] == ("running" if active_online else "degraded")
            assert _deployment_launch_progress(public)["launch_phase"] == ("ready" if active_online else "error")
        finally:
            await manager.http.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("all_stopped_intent", [False, True])
@pytest.mark.parametrize("failed_rank_online", [False, True])
def test_failed_group_stop_remains_degraded_during_reconciliation(tmp_path, all_stopped_intent, failed_rank_online):
    async def run():
        manager = Manager(tmp_path)
        deployment = partial_deployment()
        deployment["error"] = "Failed to stop rank 0: agent disconnected"
        deployment["members"][0]["failed_stop_error"] = deployment["error"]
        deployment["members"][0]["status"] = "running"
        if all_stopped_intent:
            for member in deployment["members"]:
                member["desired_state"] = "stopped"
        manager.deployments = [deployment]
        nodes = [{"id": member["node_id"], "online": failed_rank_online if index == 0 else True,
                  "status": "online", "docker_ready": True, "containers": [{
                      "name": member["container_name"], "status": member["status"], "phase": member["phase"],
                  }]} for index, member in enumerate(deployment["members"])]
        manager.list_containers = AsyncMock(return_value=nodes[2]["containers"])
        manager.list_images = AsyncMock(return_value=[])
        manager.get_stats = AsyncMock(return_value={})
        manager.cluster_nodes = AsyncMock(return_value=nodes)
        try:
            public = (await manager.get_state())["deployments"][0]
            assert public["status"] == "degraded"
            assert public["error"] == deployment["error"]
            assert _deployment_launch_progress(public) == {"launch_phase": "error", "launch_message": deployment["error"]}
        finally:
            await manager.http.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("status", ["stopped", "running", "error"])
def test_failed_recreation_remains_expected_after_cleanup(status):
    deployment = partial_deployment()
    for member in deployment["members"][:2]:
        member.update(recreate_pending=True, status=status, phase={"phase": "ready"})
    assert Manager._grouped_deployment_status(deployment) == "degraded"


@pytest.mark.parametrize("scenario", ["docker_unavailable", "stale_stopped", "recovery_error"])
def test_reconciliation_preserves_infrastructure_and_fresh_intent(tmp_path, scenario):
    async def run():
        manager = Manager(tmp_path)
        deployment = partial_deployment()
        if scenario == "stale_stopped":
            deployment["status"] = "stopped"
        if scenario == "recovery_error":
            deployment["error"] = "Automatic recovery could not stop ranks"
        manager.deployments = [deployment]
        nodes = [{"id": member["node_id"], "online": True, "status": "online", "docker_ready": True,
                  "containers": [{"name": member["container_name"], "status": member["status"], "phase": member["phase"]}]}
                 for member in deployment["members"]]
        manager.list_containers = AsyncMock(return_value=nodes[2]["containers"])
        if scenario == "docker_unavailable":
            manager.list_containers.side_effect = docker.errors.DockerException("socket down")
            nodes[2]["containers"] = []
        manager.list_images = AsyncMock(return_value=[])
        manager.get_stats = AsyncMock(return_value={})
        manager.cluster_nodes = AsyncMock(return_value=nodes)
        try:
            public = (await manager.get_state())["deployments"][0]
            if scenario == "docker_unavailable":
                assert public["status"] == "unknown"
                assert _deployment_launch_progress(public) == {"launch_phase": "unknown", "launch_message": "Docker is unavailable"}
            else:
                assert public["status"] == "running"
                assert not public.get("error")
        finally:
            await manager.http.aclose()
    asyncio.run(run())


def test_other_group_action_preserves_failed_stop_marker(tmp_path):
    async def run():
        manager = Manager(tmp_path)
        deployment = partial_deployment()
        deployment["members"][0]["failed_stop_error"] = "Node 1 did not stop"
        manager.deployments = [deployment]
        manager._member_action = AsyncMock(return_value={"ok": True})
        try:
            await manager._deployment_action_locked("partial", "stop", instance=1)
            assert deployment["members"][0]["failed_stop_error"] == "Node 1 did not stop"
            assert deployment["status"] == "degraded"
            assert deployment["error"] == "Node 1 did not stop"
            await manager._deployment_action_locked("partial", "stop", instance=0)
            assert all(not member.get("failed_stop_error") for member in deployment["members"])
        finally:
            await manager.http.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["sharded", "grouped_sharded"])
@pytest.mark.parametrize("scenario, expected", [
    ("completed_stop", "stopped"),
    ("legacy_completed_stop", "stopped"),
    ("running_intent", "stopped"),
    ("stop_intent_only", "stopped"),
    ("observed_running", "degraded"),
    ("observed_unknown", "degraded"),
    ("remote_docker_unavailable", "unknown"),
    ("all_offline", "stopped"),
    ("failed_stop", "error"),
])
def test_offline_peers_are_assumed_stopped_without_hiding_online_activity(tmp_path, mode, scenario, expected):
    async def run():
        manager = Manager(tmp_path)
        deployment = partial_deployment()
        deployment.update(mode=mode, status="stopped", desired_state="stopped")
        for member in deployment["members"]:
            member.update(status="stopped", desired_state="stopped")
        if scenario == "legacy_completed_stop":
            deployment.pop("desired_state")
        elif scenario in {"running_intent", "all_offline"}:
            deployment["desired_state"] = "running"
            for member in deployment["members"]:
                member["desired_state"] = "running"
            if scenario == "all_offline":
                deployment["status"] = "running"
        elif scenario == "stop_intent_only":
            deployment["status"] = "degraded"
        elif scenario == "failed_stop":
            deployment.update(status="error", error="Failed to stop offline ranks")
        manager.deployments = [deployment]
        original = copy.deepcopy(deployment)
        nodes = []
        for index, member in enumerate(deployment["members"]):
            # The local rank and one peer remain connected; the other pair
            # is unplugged after a previously completed Stop.
            online = index in {1, 2} and scenario != "all_offline"
            status = "exited"
            if index == 2 and scenario in {"observed_running", "observed_unknown"}:
                status = scenario.removeprefix("observed_")
            nodes.append({
                "id": member["node_id"], "online": online,
                "status": "online" if online else "offline", "docker_ready": True,
                "containers": [{"name": member["container_name"], "status": status}],
            })
            if index == 1 and scenario == "remote_docker_unavailable":
                nodes[-1].update(docker_ready=False, containers=[])
        manager.list_containers = AsyncMock(return_value=nodes[2]["containers"])
        manager.list_images = AsyncMock(return_value=[])
        manager.get_stats = AsyncMock(return_value={})
        manager.cluster_nodes = AsyncMock(return_value=nodes)
        try:
            public = (await manager.get_state())["deployments"][0]
            assert public["status"] == expected
            assert public["members"][0]["status"] == "stopped"
            assert public["members"][3]["status"] == "stopped"
            assert public["members"][0]["node_status"] == "offline"
            assert public["members"][3]["node_status"] == "offline"
            if scenario == "remote_docker_unavailable":
                assert public["members"][1]["status"] == "unknown"
                assert public["members"][1]["status_message"] == "Docker is unavailable"
            assert deployment == original
            if scenario == "failed_stop":
                assert public["error"] == "Failed to stop offline ranks"
        finally:
            await manager.http.aclose()
    asyncio.run(run())


def test_failed_user_stop_sets_marker_and_successful_whole_start_clears_it(tmp_path):
    async def run():
        manager = Manager(tmp_path)
        deployment = partial_deployment()
        manager.deployments = [deployment]
        manager._member_action = AsyncMock(side_effect=[RuntimeError("Node 1 did not stop"), {"ok": True}])
        manager._deployment_environment_drift = AsyncMock(return_value=None)
        try:
            result = await manager._deployment_action_locked("partial", "stop", instance=0)
            assert not result["ok"]
            assert deployment["members"][0]["failed_stop_error"] == "Node 1 did not stop"
            manager._member_action = AsyncMock(return_value={"ok": True})
            result = await manager._deployment_action_locked("partial", "start")
            assert result["ok"]
            assert all(not member.get("failed_stop_error") for member in deployment["members"])
            assert deployment["error"] is None
        finally:
            await manager.http.aclose()
    asyncio.run(run())
