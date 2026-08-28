import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import httpx

from manager import Manager
from sparkdeck.service import SparkDeckService


class FailingClusterManager:
    def __init__(self):
        self.http = httpx.AsyncClient()
        self.deployments = []
        self.selected_cluster_nodes = AsyncMock(return_value=[{
            "id": "remote-1", "name": "Worker", "online": True,
            "docker_ready": True, "enabled": True,
        }])
        self.create_deployment = AsyncMock(side_effect=self._fail_launch)
        self.deployment_action = AsyncMock(side_effect=self._remove)

    async def _fail_launch(self, body):
        self.deployments.append({
            "id": "failed-manager", "managed_by": "sparkdeck",
            "sparkdeck_record_id": body["sparkdeck_record_id"],
            "status": "error", "api_port": 9010,
            "node_ids": body["node_ids"],
            "members": [{
                "node_id": "remote-1", "container_name": "failed-rank",
            }],
        })
        raise RuntimeError("worker launch failed")

    async def _remove(self, deployment_id, action):
        self.deployments = [
            item for item in self.deployments if item["id"] != deployment_id
        ]
        return {"ok": True, "errors": []}

    @staticmethod
    def public_target_node(node):
        return Manager.public_target_node(node)


class FailedClusterLaunchTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_manager_record_is_removed_before_sparkdeck_rethrows(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FailingClusterManager()
            service = SparkDeckService(manager, Path(directory))

            with self.assertRaisesRegex(RuntimeError, "worker launch failed"):
                await service.create_deployment({
                    "model": "org/model", "alias": "retryable", "runtime": "vllm",
                    "node_ids": ["remote-1"],
                }, launch=True)

            manager.deployment_action.assert_awaited_once_with(
                "failed-manager", "remove"
            )
            self.assertEqual(manager.deployments, [])
            self.assertIsNone(service.store.deployment("retryable"))
            await manager.http.aclose()
            await service.close()

    async def test_failed_cleanup_is_adopted_as_actionable_local_record(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FailingClusterManager()
            manager.deployment_action.side_effect = RuntimeError("worker offline")
            service = SparkDeckService(manager, Path(directory))

            with self.assertRaisesRegex(RuntimeError, "worker launch failed"):
                await service.create_deployment({
                    "model": "org/model", "alias": "adopted", "runtime": "vllm",
                    "node_ids": ["remote-1"],
                }, launch=True)

            stored = service.store.deployment("adopted")
            self.assertEqual(
                stored["settings"]["manager_deployment_id"], "failed-manager"
            )
            self.assertEqual(stored["container_name"], "failed-rank")
            await manager.http.aclose()
            await service.close()


class SelectedSglangSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_nodes_forward_data_parallel_size(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FailingClusterManager()
            manager.create_deployment.side_effect = None
            manager.create_deployment.return_value = {
                "id": "cluster-1", "status": "starting", "api_port": 9010,
                "node_ids": ["remote-1"],
                "members": [{"container_name": "rank-0", "node_id": "remote-1"}],
            }
            service = SparkDeckService(manager, Path(directory))

            await service.create_deployment({
                "model": "org/model", "alias": "sglang-dp", "runtime": "sglang",
                "node_ids": ["remote-1"],
                "settings": {"data_parallel_size": 4},
            }, launch=True)

            launch = manager.create_deployment.await_args.args[0]
            index = launch["extra_args"].index("--dp-size")
            self.assertEqual(launch["extra_args"][index + 1], "4")
            await manager.http.aclose()
            await service.close()


if __name__ == "__main__":
    unittest.main()
