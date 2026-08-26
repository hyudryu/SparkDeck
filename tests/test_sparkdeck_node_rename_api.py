import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx


with patch("docker.from_env", return_value=Mock()):
    import server


class NodeRenameApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_versioned_patch_delegates_and_returns_public_node(self):
        rename = AsyncMock(return_value={
            "id": "remote-1", "name": "Compute", "online": True,
            "name_sync": "synchronized",
        })
        with patch.object(server.manager, "rename_cluster_node", rename):
            response = await self.client.patch(
                "/api/v1/nodes/remote-1", json={"name": " Compute "},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["name"], "Compute")
        rename.assert_awaited_once_with("remote-1", " Compute ")

    async def test_versioned_patch_maps_validation_and_missing_node(self):
        rename = AsyncMock(side_effect=ValueError("name must not be empty"))
        with patch.object(server.manager, "rename_cluster_node", rename):
            invalid = await self.client.patch("/api/v1/nodes/local", json={"name": ""})
        rename.side_effect = LookupError("node not found")
        with patch.object(server.manager, "rename_cluster_node", rename):
            missing = await self.client.patch(
                "/api/v1/nodes/missing", json={"name": "Worker"},
            )

        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(missing.status_code, 404)

    async def test_agent_patch_requires_auth_and_updates_only_local_node(self):
        rename = AsyncMock(return_value={"id": "local", "name": "Worker"})
        with (
            patch.object(server.manager.agent_credentials, "authorize_controller", return_value=False),
            patch.object(server.manager, "rename_cluster_node", rename),
        ):
            unauthorized = await self.client.patch(
                "/api/agent/node", json={"name": "Worker"},
            )
        with (
            patch.object(server.manager.agent_credentials, "authorize_controller", return_value=True),
            patch.object(server.manager, "rename_cluster_node", rename),
        ):
            accepted = await self.client.patch(
                "/api/agent/node",
                headers={"authorization": "Bearer paired"},
                json={"name": "Worker"},
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(accepted.status_code, 200)
        rename.assert_awaited_once_with("local", "Worker")

    async def test_legacy_patch_preserves_general_registry_updates(self):
        update = Mock(return_value={"id": "remote-1", "enabled": False})
        with patch.object(server.manager.node_registry, "update", update):
            response = await self.client.patch(
                "/api/nodes/remote-1", json={"enabled": False},
            )

        self.assertEqual(response.status_code, 200)
        update.assert_called_once_with("remote-1", {"enabled": False})


if __name__ == "__main__":
    unittest.main()
