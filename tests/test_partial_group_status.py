import asyncio
import copy
from unittest.mock import AsyncMock

import pytest

from manager import Manager
from sparkdeck.service import _deployment_launch_progress


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
