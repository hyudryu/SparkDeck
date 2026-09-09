import unittest
from unittest.mock import AsyncMock, patch

import httpx


class AgentRoutingHealthStateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import server

        self.server = server
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_requires_agent_auth_before_inspecting_container(self):
        with patch.object(
            self.server.manager, "is_managed_container", AsyncMock(),
        ) as managed:
            response = await self.client.get("/api/agent/containers/pinned/state")
        self.assertEqual(response.status_code, 401)
        managed.assert_not_awaited()

    async def test_unmanaged_container_is_hidden(self):
        with patch.object(self.server, "_require_agent"), patch.object(
            self.server.manager, "is_managed_container", AsyncMock(return_value=False),
        ), patch.object(
            self.server.manager, "_container_by_name", AsyncMock(),
        ) as inspect:
            response = await self.client.get("/api/agent/containers/unmanaged/state")
        self.assertEqual(response.status_code, 404)
        inspect.assert_not_awaited()

    async def test_worker_observation_is_exact_and_redacted_without_inventory(self):
        container = {
            "name": "pinned", "status": "running", "environment": {"SECRET": "private"},
            "agent_token": "private", "load_settings": {"private_path": "/secret"},
        }
        with patch.object(self.server, "_require_agent"), patch.object(
            self.server.manager, "is_managed_container", AsyncMock(return_value=True),
        ), patch.object(
            self.server.manager, "_container_by_name", AsyncMock(return_value=container),
        ) as inspect, patch.object(
            self.server.manager, "_check_ready", AsyncMock(),
        ) as ready, patch.object(
            self.server.manager, "get_state", AsyncMock(),
        ) as inventory, patch.object(
            self.server.manager, "list_containers", AsyncMock(),
        ) as containers:
            response = await self.client.get("/api/agent/containers/pinned/state")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"name": "pinned", "status": "running", "ready": None})
        inspect.assert_awaited_once_with("pinned")
        ready.assert_not_awaited()
        inventory.assert_not_awaited()
        containers.assert_not_awaited()

    async def test_coordinator_requires_strict_readiness_and_never_checks_stopped_container(self):
        for status, healthy in (("running", True), ("running", False), ("exited", False)):
            with self.subTest(status=status, healthy=healthy):
                container = {"name": "pinned", "status": status}
                with patch.object(self.server, "_require_agent"), patch.object(
                    self.server.manager, "is_managed_container", AsyncMock(return_value=True),
                ), patch.object(
                    self.server.manager, "_container_by_name", AsyncMock(return_value=container),
                ), patch.object(
                    self.server.manager, "_check_ready", AsyncMock(return_value=healthy),
                ) as ready:
                    response = await self.client.get(
                        "/api/agent/containers/pinned/state", params={"check_ready": "true"},
                    )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["ready"], healthy)
                if status == "running":
                    ready.assert_awaited_once_with(container, strict_health=True)
                else:
                    ready.assert_not_awaited()

    async def test_container_disappearing_after_ownership_check_returns_not_found(self):
        with patch.object(self.server, "_require_agent"), patch.object(
            self.server.manager, "is_managed_container", AsyncMock(return_value=True),
        ), patch.object(
            self.server.manager, "_container_by_name", AsyncMock(return_value=None),
        ):
            response = await self.client.get("/api/agent/containers/pinned/state")
        self.assertEqual(response.status_code, 404)
