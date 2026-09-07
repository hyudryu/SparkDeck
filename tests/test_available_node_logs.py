import asyncio
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, patch

from cluster import NodeRegistry
from manager import Manager
from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import SparkDeckService


class AvailableNodeLogsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.registry = NodeRegistry(Path(self.temp.name), None)
        self.registry.request = AsyncMock(return_value={"logs": "remote output"})
        self.manager = SimpleNamespace(
            http=None,
            node_registry=self.registry,
            get_cluster_member_logs=AsyncMock(return_value="local output"),
        )
        self.manager._member_action = MethodType(Manager._member_action, self.manager)
        self.manager._read_member_logs = MethodType(Manager._read_member_logs, self.manager)
        self.service = SparkDeckService(self.manager, Path(self.temp.name))

    async def asyncTearDown(self):
        await self.service.close()
        self.temp.cleanup()

    async def test_service_keeps_available_logs_when_other_nodes_are_offline_or_stall(self):
        members = [
            {"node_id": node, "container_name": f"rank-{rank}", "rank": rank}
            for rank, node in enumerate(["local", "online", "offline", "stalled"])
        ]
        self.manager.deployments = [{"id": "cluster", "members": members}]
        self.service.store.add_deployment(Deployment(
            id="deployment", alias="model", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
            settings={"manager_deployment_id": "cluster"},
        ))
        stalled_cancelled = asyncio.Event()

        async def request(node_id, method, path, **kwargs):
            if node_id == "stalled":
                try:
                    await asyncio.Event().wait()
                finally:
                    stalled_cancelled.set()
            return {"logs": "remote output"}

        self.registry.request.side_effect = request
        with patch("cluster.time", SimpleNamespace(monotonic=lambda: 100)):
            self.registry._status_cache["offline"] = (99, {"online": False})
            with patch("manager.MEMBER_LOG_TIMEOUT_SECONDS", 0.01):
                result = await asyncio.wait_for(
                    self.service.deployment_logs("deployment", 150), timeout=1,
                )

        by_node = {member["node_id"]: member for member in result["members"]}
        self.assertEqual(by_node["local"]["logs"], "local output")
        self.assertEqual(by_node["online"]["logs"], "remote output")
        self.assertNotIn("error", by_node["local"])
        self.assertNotIn("error", by_node["online"])
        self.assertIn("offline", by_node["offline"]["error"])
        self.assertIn("did not respond", by_node["stalled"]["error"])
        self.assertIn("local output", result["logs"])
        self.assertIn("remote output", result["logs"])
        self.assertTrue(stalled_cancelled.is_set())
        self.assertEqual(
            {call.args[0] for call in self.registry.request.await_args_list},
            {"online", "stalled"},
        )
        self.manager.get_cluster_member_logs.assert_awaited_once_with("rank-0", 150)

    async def test_expired_offline_observation_allows_reconnected_node_logs(self):
        member = {"node_id": "remote", "container_name": "rank-1"}
        self.registry._status_cache["remote"] = (100, {"online": False})
        with patch("cluster.time", SimpleNamespace(monotonic=lambda: 103.9)):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                await self.manager._member_action(member, "logs")
        self.registry.request.assert_not_awaited()
        with patch("cluster.time", SimpleNamespace(monotonic=lambda: 104)):
            result = await self.manager._member_action(member, "logs", log_tail=500)
        self.assertEqual(result, {"logs": "remote output"})
        self.registry.request.assert_awaited_once_with(
            "remote", "GET", "/api/agent/containers/rank-1/logs?tail=500", timeout=5.0,
        )

    async def test_cancelling_log_request_cancels_remote_read(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def request(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.registry.request.side_effect = request
        task = asyncio.create_task(self.manager._member_action(
            {"node_id": "remote", "container_name": "rank-1"}, "logs",
        ))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(cancelled.is_set())

    async def test_stalled_local_logs_also_have_a_deadline(self):
        async def local_logs(*args):
            await asyncio.Event().wait()

        self.manager.get_cluster_member_logs.side_effect = local_logs
        with patch("manager.MEMBER_LOG_TIMEOUT_SECONDS", 0.01):
            with self.assertRaisesRegex(RuntimeError, "did not respond"):
                await asyncio.wait_for(self.manager._member_action(
                    {"node_id": "local", "container_name": "rank-0"}, "logs",
                ), timeout=1)
