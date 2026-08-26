import time
import unittest
from unittest.mock import AsyncMock

from manager import Manager


class SparkDeckIdleTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def manager_for(container):
        manager = Manager.__new__(Manager)
        manager.settings = {"idle_timeout_seconds": 1}
        manager.list_containers = AsyncMock(return_value=[container])
        manager._read_activity_counter = AsyncMock(return_value=5)
        manager.stop_container = AsyncMock(return_value={"ok": True})
        manager._activity = {
            container["name"]: {"counter": 5, "last_active": time.time() - 10}
        }
        return manager

    async def test_single_container_deployment_is_stopped_when_idle(self):
        manager = self.manager_for({
            "name": "single", "managed": True, "status": "running",
            "deployment_id": "dep-1", "deployment_mode": "single", "nnodes": 1,
            "phase": {"phase": "ready"}, "port": 8000,
        })

        await manager._idle_tick()

        manager.stop_container.assert_awaited_once_with("single")

    async def test_sharded_deployment_member_is_not_stopped_independently(self):
        manager = self.manager_for({
            "name": "rank-0", "managed": True, "status": "running",
            "deployment_id": "dep-2", "deployment_mode": "sharded", "nnodes": 2,
            "phase": {"phase": "ready"}, "port": 8000,
        })

        await manager._idle_tick()

        manager.stop_container.assert_not_awaited()
