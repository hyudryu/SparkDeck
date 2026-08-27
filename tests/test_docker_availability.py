import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import docker
import httpx

from manager import Manager


with patch("docker.from_env", return_value=Mock()):
    import server


class DockerAvailabilityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    async def asyncTearDown(self):
        manager = getattr(self, "manager", None)
        if manager is not None:
            await manager.http.aclose()
        self.temp.cleanup()

    def manager_without_docker(self) -> Manager:
        unavailable = docker.errors.DockerException("daemon is not running")
        with patch("manager.docker.from_env", side_effect=unavailable):
            self.manager = Manager(Path(self.temp.name))
        return self.manager

    async def test_controller_starts_with_degraded_empty_docker_inventory(self):
        manager = self.manager_without_docker()
        unavailable = docker.errors.DockerException("daemon is not running")

        with patch("manager.docker.from_env", side_effect=unavailable):
            self.assertEqual(await manager.list_containers(), [])
            self.assertEqual(await manager.list_images(), [])
            manager.get_disk = AsyncMock(return_value={})
            status = await manager.agent_status(stats={})

        self.assertEqual(status["status"], "degraded")
        self.assertEqual(status["status_message"], "Docker is unavailable")
        self.assertFalse(status["docker_ready"])

    async def test_docker_inventory_recovers_without_restarting_manager(self):
        manager = self.manager_without_docker()
        container = SimpleNamespace(
            name="external-container",
            labels={},
            status="running",
            attrs={"Config": {}, "NetworkSettings": {}, "State": {}},
        )
        client = SimpleNamespace(
            containers=SimpleNamespace(list=Mock(return_value=[container])),
            images=SimpleNamespace(list=Mock(return_value=[])),
        )

        with patch("manager.docker.from_env", return_value=client) as reconnect:
            inventory = await manager.list_containers()
            self.assertEqual(inventory, [])
            await manager.list_images()

        reconnect.assert_called_once_with()
        client.containers.list.assert_called_once_with(all=True)
        client.images.list.assert_called_once_with()

    async def test_docker_mutation_fails_explicitly_when_daemon_is_unavailable(self):
        manager = self.manager_without_docker()
        unavailable = docker.errors.DockerException("daemon is not running")

        with (
            patch("manager.docker.from_env", side_effect=unavailable),
            self.assertRaisesRegex(
                docker.errors.DockerException, "Docker is unavailable"
            ),
        ):
            await manager.remove_container("missing")

    async def test_state_api_remains_available_without_local_docker(self):
        manager = self.manager_without_docker()
        unavailable = docker.errors.DockerException("daemon is not running")
        manager.get_stats = AsyncMock(return_value={})
        manager.get_disk = AsyncMock(return_value={})

        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            with (
                patch.object(server, "manager", manager),
                patch("manager.docker.from_env", side_effect=unavailable),
            ):
                response = await client.get("/api/state")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["containers"], [])
        self.assertEqual(response.json()["images"], [])
