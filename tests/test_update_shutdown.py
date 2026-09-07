import asyncio
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sparkdeck.updater import UpdateService
from sparkdeck.virtual_nas import VirtualNAS

with patch("docker.from_env", return_value=Mock()):
    import server


class UpdateShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.nas = VirtualNAS(root, lambda: root / "hub", Mock(), lambda: True)
        self.service = UpdateService(SimpleNamespace(virtual_nas=self.nas), root, root)
        self.service.runtime_revision = "a" * 40
        self.service.preflight_local = AsyncMock()
        self.service._launch_local_helper = Mock()

    async def asyncTearDown(self):
        await self.service.close()
        self.temp.cleanup()

    async def test_shutdown_cancels_update_before_queue_teardown_unblocks_it(self):
        self.nas._reserve_stream("org/model")
        await self.service.start_local("main", "b" * 40)
        await asyncio.sleep(0)
        events = []

        @asynccontextmanager
        async def session():
            yield

        async def stop_manager():
            events.append("manager")
            self.nas._release_stream("org/model")
            await asyncio.sleep(0)

        close_update = self.service.close

        async def stop_updater():
            await close_update()
            events.append("updater")

        with patch.object(server, "updater", self.service), \
                patch.object(self.service, "close", side_effect=stop_updater), \
                patch.object(server, "mcp_control", SimpleNamespace(
                    session_manager=SimpleNamespace(run=session))), \
                patch.object(server.manager, "start", new=AsyncMock()), \
                patch.object(server.manager, "stop", side_effect=stop_manager), \
                patch.object(server.sparkdeck, "close", new=AsyncMock()), \
                patch.object(server, "community_upload_loop", side_effect=asyncio.Event().wait):
            async with server.lifespan(server.app):
                pass
        self.assertEqual(events, ["updater", "manager"])
        self.service._launch_local_helper.assert_not_called()
        self.assertTrue(self.service._agent_task.done())
        self.assertEqual(self.service._read(self.service.agent_path)["phase"], "failed")
        self.assertFalse(self.nas._update_reserved)

    async def test_shutdown_before_pending_task_starts_releases_reservation(self):
        self.nas._reserve_stream("org/model")
        await self.service.start_local("main", "b" * 40)
        await self.service.close()
        self.assertTrue(self.service._agent_task.done())
        self.assertFalse(self.nas._update_reserved)
        self.service._launch_local_helper.assert_not_called()
        self.assertEqual(self.service._read(self.service.agent_path)["phase"], "failed")

    async def test_shutdown_cancels_second_preflight_without_spawning_helper(self):
        second_preflight = asyncio.Event()
        calls = 0

        async def preflight(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                second_preflight.set()
                await asyncio.Event().wait()

        self.service.preflight_local = AsyncMock(side_effect=preflight)
        await self.service.start_local("main", "b" * 40)
        await asyncio.wait_for(second_preflight.wait(), 1)
        await asyncio.wait_for(self.service.close(), 1)
        self.service._launch_local_helper.assert_not_called()
        self.assertFalse(self.nas._update_reserved)
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            await self.service.start_local("main", "b" * 40)
