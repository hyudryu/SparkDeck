"""Runtime visibility and stop reconciliation require a successful inventory."""

import asyncio
import copy
from unittest.mock import AsyncMock, Mock

import docker
import pytest

from manager import Manager
from sparkdeck.service import _grouped_instance_summary


def deployment_fixture():
    return {
        "id": "interrupted", "name": "Interrupted", "model": "org/model",
        "engine": "vllm", "mode": "grouped_sharded", "status": "error",
        "desired_state": "stopped", "error": "agent disconnected",
        "members": [{
            "node_id": "remote-1", "node_name": "Worker", "rank": 0,
            "instance_id": 0, "container_name": "rank-0", "status": "starting",
            "desired_state": "stopped", "failed_stop_error": "agent disconnected",
            "phase": {"phase": "starting"},
        }],
    }


def test_agent_enumeration_failure_does_not_confirm_a_failed_stop(tmp_path):
    async def run():
        manager = Manager(tmp_path)
        manager.get_disk = AsyncMock(return_value={})
        manager._docker_runtime_status = AsyncMock(return_value=(True, None))
        manager.list_containers = AsyncMock(side_effect=docker.errors.DockerException("inventory failed"))
        manager._network_interfaces = Mock(return_value=[])
        manager.llama_rpc_status = Mock(return_value={})
        manager.agent_health = Mock(return_value={"online": True})
        manager._member_action = AsyncMock()
        manager._save_deployments = Mock()
        manager.deployments = [deployment_fixture()]
        before = copy.deepcopy(manager.deployments)
        try:
            report = await manager.agent_status(stats={})
            assert report["docker_ready"] is True
            assert report["containers"] == []
            assert report["inventory_available"] is False
            report["id"] = "remote-1"
            manager.cluster_nodes = AsyncMock(return_value=[report])
            await manager._reconcile_stopped_members()
            assert manager.deployments == before
            manager._member_action.assert_not_awaited()
            manager._save_deployments.assert_not_called()

            manager.list_containers = AsyncMock(return_value=[])
            assert (await manager.agent_status(stats={}))["inventory_available"] is True
        finally:
            await manager.http.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("observed_status, inventory_available, online, docker_ready, expected", [
    (None, True, True, True, False),
    ("running", True, True, True, True),
    ("restarting", True, True, True, True),
    ("paused", True, True, True, True),
    ("created", True, True, True, False),
    ("creating", True, True, True, False),
    ("exited", True, True, True, False),
    (None, False, True, True, False),
    ("running", False, True, True, False),
    ("running", True, False, True, False),
    ("running", True, True, False, False),
])
def test_error_group_visibility_uses_observed_containers(
    tmp_path, observed_status, inventory_available, online, docker_ready, expected,
):
    async def run():
        manager = Manager(tmp_path)
        manager.deployments = [deployment_fixture()]
        manager.list_containers = AsyncMock(return_value=[])
        manager.list_images = AsyncMock(return_value=[])
        manager.get_stats = AsyncMock(return_value={})
        manager.cluster_nodes = AsyncMock(return_value=[{
            "id": "remote-1", "online": online, "status": "online",
            "docker_ready": docker_ready, "inventory_available": inventory_available,
            "containers": [] if observed_status is None else [{
                "name": "rank-0", "status": observed_status,
                "phase": {"phase": "starting"},
            }],
        }])
        try:
            public = (await manager.get_state())["deployments"][0]
            assert public["members"][0]["has_live_container"] is expected
            assert _grouped_instance_summary(public)[0]["has_live_containers"] is expected
        finally:
            await manager.http.aclose()
    asyncio.run(run())
