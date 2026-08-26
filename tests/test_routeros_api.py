import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx


with patch("docker.from_env", return_value=Mock()):
    import server


class RouterOSApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_presence_and_overview_delegate_to_cluster_aggregates(self) -> None:
        presence = AsyncMock(return_value={"detected": True, "nodes": []})
        overview = AsyncMock(return_value={"detected": True, "nodes": [{"node_id": "local"}]})
        with (
            patch.object(server.manager, "routeros_cluster_presence", presence),
            patch.object(server.manager, "routeros_cluster_overview", overview),
        ):
            presence_response = await self.client.get("/api/v1/routeros/presence")
            overview_response = await self.client.get("/api/v1/routeros")

        self.assertEqual(presence_response.status_code, 200)
        self.assertTrue(presence_response.json()["detected"])
        self.assertEqual(overview_response.json()["nodes"][0]["node_id"], "local")

    async def test_connection_write_is_same_origin_and_delegates_without_url_credentials(self) -> None:
        connect = AsyncMock(return_value={"connected": True, "node_id": "local"})
        body = {
            "base_url": "https://192.168.88.1", "username": "sparkdeck",
            "password": "secret", "verify_tls": True,
        }
        with patch.object(server.manager, "connect_routeros", connect):
            rejected = await self.client.put(
                "/api/v1/routeros/nodes/local/connection",
                headers={"origin": "https://evil.example"}, json=body,
            )
            accepted = await self.client.put(
                "/api/v1/routeros/nodes/local/connection", json=body,
            )

        self.assertEqual(rejected.status_code, 403)
        self.assertEqual(accepted.status_code, 200)
        self.assertNotIn("secret", accepted.text)
        connect.assert_awaited_once_with("local", body)

    async def test_fan_validation_errors_map_to_bad_request(self) -> None:
        update = AsyncMock(side_effect=ValueError("unsupported fan setting(s): script"))
        with patch.object(server.manager, "update_routeros_fan_settings", update):
            response = await self.client.patch(
                "/api/v1/routeros/nodes/local/fan-settings", json={"script": "bad"},
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported fan setting", response.text)

    async def test_agent_routeros_routes_require_paired_controller_auth(self) -> None:
        overview = AsyncMock(return_value={"detected": True, "connected": True})
        with (
            patch.object(server.manager.agent_credentials, "authorize_controller", return_value=False),
            patch.object(server.manager.routeros, "overview", overview),
        ):
            unauthorized = await self.client.get("/api/agent/routeros")
        with (
            patch.object(server.manager.agent_credentials, "authorize_controller", return_value=True),
            patch.object(server.manager.routeros, "overview", overview),
        ):
            accepted = await self.client.get(
                "/api/agent/routeros", headers={"authorization": "Bearer paired"},
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(accepted.status_code, 200)
        overview.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
