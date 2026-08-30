import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import docker
import httpx

from manager import Manager, _RetryingDockerClient


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

    async def test_manager_construction_does_not_negotiate_with_docker(self):
        with patch(
            "manager.docker.from_env",
            side_effect=AssertionError("Docker was negotiated eagerly"),
        ) as negotiate:
            self.manager = Manager(Path(self.temp.name))

        negotiate.assert_not_called()

    async def test_agent_health_does_not_touch_docker_or_telemetry(self):
        manager = self.manager_without_docker()
        manager.app_revision = "a" * 40

        health = manager.agent_health()

        self.assertTrue(health["online"])
        self.assertEqual(health["app_revision"], "a" * 40)
        self.assertNotIn("docker_ready", health)
        self.assertNotIn("stats", health)

    async def test_controller_starts_with_explicitly_unavailable_docker_inventory(self):
        manager = self.manager_without_docker()
        unavailable = docker.errors.DockerException("daemon is not running")

        with patch("manager.docker.from_env", side_effect=unavailable):
            with self.assertRaisesRegex(
                docker.errors.DockerException, "Docker is unavailable"
            ):
                await manager.list_containers()
            with self.assertRaisesRegex(
                docker.errors.DockerException, "Docker is unavailable"
            ):
                await manager.list_images()
            manager.get_disk = AsyncMock(return_value={})
            status = await manager.agent_status(stats={})

        self.assertEqual(status["status"], "degraded")
        self.assertEqual(status["status_message"], "Docker is unavailable")
        self.assertFalse(status["docker_ready"])

    async def test_agent_status_reports_the_revision_pinned_at_manager_startup(self):
        manager = self.manager_without_docker()
        manager.app_revision = "a" * 40
        manager.get_disk = AsyncMock(return_value={})
        unavailable = docker.errors.DockerException("daemon is not running")

        with patch("manager.current_revision", return_value="b" * 40), \
             patch("manager.docker.from_env", side_effect=unavailable):
            status = await manager.agent_status(stats={})

        self.assertEqual(status["app_revision"], "a" * 40)

    async def test_agent_info_reports_the_revision_pinned_at_server_startup(self):
        with patch.object(server.updater, "runtime_revision", "a" * 40):
            info = await server.agent_info()

        self.assertEqual(info["app_revision"], "a" * 40)

    async def test_agent_status_rejects_windows_container_daemon(self):
        manager = self.manager_without_docker()
        client = Mock()
        client.ping.return_value = True
        client.info.return_value = {"OSType": "windows"}
        client.containers.list.return_value = []
        manager.client._client = client
        manager.get_disk = AsyncMock(return_value={})

        status = await manager.agent_status(stats={})

        self.assertFalse(status["docker_ready"])
        self.assertEqual(status["status"], "degraded")
        self.assertEqual(
            status["status_message"],
            "Docker must be configured to use Linux containers",
        )

    async def test_agent_status_explains_service_user_docker_permissions(self):
        manager = self.manager_without_docker()
        manager.client._retry_after = 0
        with patch(
            "manager.docker.from_env",
            side_effect=docker.errors.DockerException(
                "PermissionError(13, 'Permission denied')"
            ),
        ):
            manager.get_disk = AsyncMock(return_value={})
            status = await manager.agent_status(stats={})

        self.assertFalse(status["docker_ready"])
        self.assertIn("service user cannot access Docker", status["status_message"])

    async def test_linux_container_daemon_is_ready(self):
        manager = self.manager_without_docker()
        client = Mock()
        client.ping.return_value = True
        client.info.return_value = {"OSType": "linux"}
        client.containers.list.return_value = []
        manager.client._client = client
        manager.get_disk = AsyncMock(return_value={})

        status = await manager.agent_status(stats={})

        self.assertTrue(status["docker_ready"])
        self.assertEqual(status["status"], "online")
        self.assertIsNone(status["status_message"])

    async def test_state_rejects_windows_container_daemon(self):
        manager = self.manager_without_docker()
        client = Mock()
        client.ping.return_value = True
        client.info.return_value = {"OSType": "windows"}
        client.containers.list.return_value = []
        client.images.list.return_value = []
        manager.client._client = client
        manager.get_stats = AsyncMock(return_value={})
        manager.get_disk = AsyncMock(return_value={})

        state = await manager.get_state()

        local = next(node for node in state["nodes"] if node.get("local"))
        self.assertFalse(local["docker_ready"])
        self.assertFalse(state["docker_ready"])

    async def test_state_reuses_local_container_inventory_for_node_status(self):
        manager = self.manager_without_docker()
        manager.list_containers = AsyncMock(return_value=[])
        manager.list_images = AsyncMock(return_value=[])
        manager.get_stats = AsyncMock(return_value={})
        manager.get_disk = AsyncMock(return_value={})
        manager._docker_runtime_status = AsyncMock(return_value=(True, None))

        await manager.get_state()

        manager.list_containers.assert_awaited_once_with()

    async def test_docker_inventory_recovers_without_restarting_manager(self):
        manager = self.manager_without_docker()
        manager.client._retry_after = 0.0
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
        self.assertFalse(response.json()["docker_ready"])

    async def test_state_marks_local_deployment_unknown_during_docker_outage(self):
        manager = self.manager_without_docker()
        manager.deployments = [{
            "id": "deployment-1",
            "name": "Existing deployment",
            "model": "org/model",
            "status": "ready",
            "members": [{
                "node_id": "local",
                "rank": 0,
                "container_name": "sparkdeck-existing",
            }],
        }]
        manager.get_stats = AsyncMock(return_value={})
        manager.get_disk = AsyncMock(return_value={})
        unavailable = docker.errors.DockerException("daemon is not running")

        with patch("manager.docker.from_env", side_effect=unavailable):
            state = await manager.get_state()

        deployment = state["deployments"][0]
        self.assertEqual(deployment["status"], "unknown")
        self.assertEqual(deployment["status_message"], "Docker is unavailable")
        self.assertEqual(deployment["members"][0]["status"], "unknown")
        self.assertNotEqual(deployment["members"][0]["status"], "missing")
        self.assertFalse(state["docker_ready"])

    async def test_state_replaces_stale_queued_phase_after_member_creation(self):
        manager = self.manager_without_docker()
        manager.deployments = [{
            "id": "deployment-1", "name": "Interrupted", "model": "org/model",
            "engine": "vllm", "mode": "replicated", "status": "starting",
            "desired_state": "running", "node_ids": ["local", "remote-1"],
            "members": [
                {
                    "node_id": "local", "node_name": "Controller", "rank": 0,
                    "container_name": "missing-local", "status": "starting",
                    "phase": {"phase": "queued", "message": "Stale queue phase"},
                },
                {
                    "node_id": "remote-1", "node_name": "Worker", "rank": 1,
                    "container_name": "missing-remote", "status": "starting",
                    "phase": {"phase": "queued", "message": "Stale queue phase"},
                },
            ],
            "launch_settings": {
                "model": "org/model", "engine": "vllm",
                "deployment_mode": "replicated",
                "node_ids": ["local", "remote-1"], "extra_args": [],
            },
        }]
        manager.list_containers = AsyncMock(return_value=[])
        manager.list_images = AsyncMock(return_value=[])
        manager.get_stats = AsyncMock(return_value={})
        manager.cluster_nodes = AsyncMock(return_value=[
            {
                "id": "local", "name": "Controller", "local": True,
                "online": True, "docker_ready": True, "containers": [],
            },
            {
                "id": "remote-1", "name": "Worker", "online": False,
                "docker_ready": False, "containers": [],
            },
        ])

        state = await manager.get_state()

        members = state["deployments"][0]["members"]
        self.assertEqual(members[0]["status"], "missing")
        self.assertEqual(members[0]["phase"]["phase"], "missing")
        self.assertEqual(members[1]["status"], "unreachable")
        self.assertEqual(members[1]["phase"]["phase"], "unreachable")
        self.assertNotIn("queued", str(members))

        manager.deployments[0]["status"] = "recovering"
        manager.deployments[0]["status_message"] = (
            "Waiting for selected nodes to reconnect"
        )
        recovering_state = await manager.get_state()
        recovering = recovering_state["deployments"][0]
        self.assertEqual(recovering["status"], "recovering")
        self.assertEqual(
            recovering["members"][1]["phase"]["phase"], "recovering",
        )

    async def test_liveness_api_does_not_query_docker_or_cluster_state(self):
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            with patch.object(
                server.manager,
                "get_state",
                AsyncMock(side_effect=AssertionError("aggregate state was queried")),
            ) as get_state:
                response = await client.get("/healthz")

        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.content, b"")
        get_state.assert_not_awaited()


class RetryingDockerClientTests(unittest.TestCase):
    def test_failed_reconnect_is_cached_then_eventually_recovers(self):
        now = [100.0]
        unavailable = docker.errors.DockerException("daemon is not running")
        proxy = _RetryingDockerClient(
            initial_error=unavailable,
            retry_interval=5,
            clock=lambda: now[0],
        )

        with patch("manager.docker.from_env", side_effect=unavailable) as reconnect:
            with self.assertRaisesRegex(docker.errors.DockerException, "unavailable"):
                proxy.ping()
            self.assertEqual(reconnect.call_count, 0)

            now[0] = 106.0
            with self.assertRaisesRegex(docker.errors.DockerException, "unavailable"):
                proxy.ping()
            with self.assertRaisesRegex(docker.errors.DockerException, "unavailable"):
                proxy.ping()
            self.assertEqual(reconnect.call_count, 1)

        client = Mock()
        client.ping.return_value = True
        now[0] = 112.0
        with patch("manager.docker.from_env", return_value=client) as reconnect:
            self.assertTrue(proxy.ping())
            self.assertTrue(proxy.ping())

        reconnect.assert_called_once_with()
        self.assertEqual(client.ping.call_count, 2)

    def test_concurrent_caller_shares_first_successful_connection(self):
        entered = threading.Event()
        release = threading.Event()
        waiter_entered = threading.Event()
        client = Mock()
        client.ping.return_value = True

        def connect():
            entered.set()
            release.wait(timeout=2)
            return client

        proxy = _RetryingDockerClient(connect_wait_timeout=1)
        original_wait_for = proxy._condition.wait_for

        def observed_wait_for(predicate, timeout=None):
            waiter_entered.set()
            return original_wait_for(predicate, timeout)

        proxy._condition.wait_for = observed_wait_for
        results = []

        def call_ping():
            try:
                results.append(proxy.ping())
            except Exception as exc:
                results.append(exc)

        with patch("manager.docker.from_env", side_effect=connect) as reconnect:
            first = threading.Thread(target=call_ping)
            second = threading.Thread(target=call_ping)
            first.start()
            self.assertTrue(entered.wait(timeout=1))
            second.start()
            self.assertTrue(waiter_entered.wait(timeout=1))
            release.set()
            first.join(timeout=1)
            second.join(timeout=1)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(results, [True, True])
        reconnect.assert_called_once_with()
        self.assertEqual(client.ping.call_count, 2)

    def test_hung_reconnect_only_blocks_another_caller_for_bounded_wait(self):
        entered = threading.Event()
        release = threading.Event()
        first_error = []

        def hung_connect():
            entered.set()
            release.wait(timeout=2)
            raise docker.errors.DockerException("daemon timed out")

        proxy = _RetryingDockerClient(
            retry_interval=5,
            connect_wait_timeout=0.05,
        )

        def first_caller():
            try:
                proxy.ping()
            except docker.errors.DockerException as exc:
                first_error.append(exc)

        with patch("manager.docker.from_env", side_effect=hung_connect) as reconnect:
            thread = threading.Thread(target=first_caller)
            thread.start()
            self.assertTrue(entered.wait(timeout=1))
            started = time.monotonic()
            with self.assertRaisesRegex(
                docker.errors.DockerException, "reconnect is already in progress"
            ):
                proxy.ping()
            elapsed = time.monotonic() - started
            release.set()
            thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertGreaterEqual(elapsed, 0.04)
        self.assertLess(elapsed, 0.5)
        self.assertEqual(reconnect.call_count, 1)
        self.assertEqual(len(first_error), 1)
