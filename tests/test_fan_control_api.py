import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from manager import FanSettingsConflict


with patch("docker.from_env", return_value=Mock()):
    import server


class FanControlApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_cluster_overview_delegates_to_capability_aggregate(self) -> None:
        overview = AsyncMock(return_value={
            "available": True,
            "nodes": [{"node_id": "local", "settings": {"mode": "curve"}}],
        })
        with patch.object(server.manager, "fan_control_cluster_overview", overview):
            response = await self.client.get("/api/v1/fan-control")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["available"])
        overview.assert_awaited_once_with()

    async def test_node_toggle_requires_exact_boolean_payload(self) -> None:
        update = AsyncMock(return_value={"node_id": "local", "enabled": True})
        with patch.object(server.manager, "set_node_fan_max_speed", update):
            string_response = await self.client.patch(
                "/api/v1/fan-control/nodes/local/max-speed",
                json={"enabled": "true"},
            )
            extra_response = await self.client.patch(
                "/api/v1/fan-control/nodes/local/max-speed",
                json={"enabled": True, "mode": "manual"},
            )
            accepted = await self.client.patch(
                "/api/v1/fan-control/nodes/local/max-speed",
                json={"enabled": True},
            )

        self.assertEqual(string_response.status_code, 400)
        self.assertEqual(extra_response.status_code, 400)
        self.assertEqual(accepted.status_code, 200)
        update.assert_awaited_once_with("local", True)

    async def test_node_toggle_requires_same_origin(self) -> None:
        update = AsyncMock(return_value={"node_id": "local", "enabled": True})
        with patch.object(server.manager, "set_node_fan_max_speed", update):
            response = await self.client.patch(
                "/api/v1/fan-control/nodes/local/max-speed",
                headers={"origin": "https://evil.example"},
                json={"enabled": True},
            )

        self.assertEqual(response.status_code, 403)
        update.assert_not_awaited()

    async def test_node_settings_update_requires_exact_same_origin_payload(self) -> None:
        curve = {
            "curve_points": [[30, 20], [60, 55], [80, 100]],
            "curve_min_temp": 30,
            "curve_max_temp": 80,
            "min_floor_pct": 20,
        }
        update = AsyncMock(return_value={
            "node_id": "node-1", "mode": "curve", "previous_mode": "pid",
            "active_settings": curve,
        })
        with patch.object(server.manager, "update_node_fan_settings", update):
            rejected = await self.client.patch(
                "/api/v1/fan-control/nodes/node-1/settings",
                headers={"origin": "https://evil.example"},
                json={"mode": "curve", "active_settings": curve, "expected_mode": "pid"},
            )
            extra = await self.client.patch(
                "/api/v1/fan-control/nodes/node-1/settings",
                json={
                    "mode": "curve", "active_settings": curve,
                    "expected_mode": "pid", "enabled": True,
                },
            )
            invalid_expected_mode = await self.client.patch(
                "/api/v1/fan-control/nodes/node-1/settings",
                json={"mode": "curve", "active_settings": curve, "expected_mode": []},
            )
            accepted = await self.client.patch(
                "/api/v1/fan-control/nodes/node-1/settings",
                json={"mode": "curve", "active_settings": curve, "expected_mode": "pid"},
            )

        self.assertEqual(rejected.status_code, 403)
        self.assertEqual(extra.status_code, 400)
        self.assertEqual(invalid_expected_mode.status_code, 400)
        self.assertEqual(accepted.status_code, 200)
        update.assert_awaited_once_with("node-1", "curve", curve, "pid")

    async def test_node_settings_preserve_mode_conflicts_as_http_409(self) -> None:
        curve = {
            "curve_points": [[30, 20], [60, 55]],
            "curve_min_temp": 30,
            "curve_max_temp": 80,
            "min_floor_pct": 20,
        }
        update = AsyncMock(side_effect=FanSettingsConflict(
            "fan mode changed; refresh and try again",
        ))
        with patch.object(server.manager, "update_node_fan_settings", update):
            response = await self.client.patch(
                "/api/v1/fan-control/nodes/node-1/settings",
                json={"mode": "curve", "active_settings": curve, "expected_mode": "pid"},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("fan mode changed", response.text)

    async def test_agent_routes_require_auth_and_report_stale_state(self) -> None:
        local = Mock(side_effect=FanSettingsConflict("FanController state is unavailable"))
        with patch.object(
            server.manager.agent_credentials, "authorize_controller", return_value=False,
        ):
            unauthorized = await self.client.get("/api/agent/fan-control")
        with (
            patch.object(
                server.manager.agent_credentials,
                "authorize_controller",
                return_value=True,
            ),
            patch.object(server.manager, "local_fan_control_overview", local),
        ):
            stale = await self.client.get(
                "/api/agent/fan-control",
                headers={"authorization": "Bearer paired"},
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(stale.status_code, 409)
        self.assertIn("unavailable", stale.text)

    async def test_agent_toggle_requires_auth_and_exact_boolean(self) -> None:
        update = Mock(return_value={"enabled": False})
        with (
            patch.object(
                server.manager.agent_credentials,
                "authorize_controller",
                return_value=True,
            ),
            patch.object(server.manager, "set_fan_max_speed", update),
        ):
            invalid = await self.client.patch(
                "/api/agent/fan-control/max-speed",
                headers={"authorization": "Bearer paired"},
                json={"enabled": 0},
            )
            accepted = await self.client.patch(
                "/api/agent/fan-control/max-speed",
                headers={"authorization": "Bearer paired"},
                json={"enabled": False},
            )

        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(accepted.status_code, 200)
        update.assert_called_once_with(False)

    async def test_agent_temperature_override_requires_auth_and_exact_payload(self) -> None:
        override = {
            "temperature_c": 81.0,
            "source": "vllm-cluster-max",
            "sensor": "gpu:0",
            "node_id": "worker-1",
            "node_name": "Hot worker",
            "observed_at": 1_000.0,
            "expires_at": 1_012.0,
        }
        update = Mock(return_value={"applied": True})
        with patch.object(
            server.manager.agent_credentials, "authorize_controller", return_value=False,
        ):
            unauthorized = await self.client.patch(
                "/api/agent/fan-control/temperature-override",
                json={"temperature_override": override},
            )
        with (
            patch.object(
                server.manager.agent_credentials,
                "authorize_controller",
                return_value=True,
            ),
            patch.object(server.manager, "set_fan_temperature_override", update),
        ):
            invalid = await self.client.patch(
                "/api/agent/fan-control/temperature-override",
                headers={"authorization": "Bearer paired"},
                json={"temperature_override": override, "extra": True},
            )
            accepted = await self.client.patch(
                "/api/agent/fan-control/temperature-override",
                headers={"authorization": "Bearer paired"},
                json={"temperature_override": override},
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(accepted.status_code, 200)
        update.assert_called_once_with(override)

    async def test_agent_settings_update_requires_auth_and_forwards_exact_fields(self) -> None:
        curve = {
            "curve_points": [[30, 20], [60, 55]],
            "curve_min_temp": 30,
            "curve_max_temp": 80,
            "min_floor_pct": 20,
        }
        update = Mock(return_value={
            "mode": "curve", "previous_mode": "curve", "active_settings": curve,
        })
        with patch.object(
            server.manager.agent_credentials, "authorize_controller", return_value=False,
        ):
            unauthorized = await self.client.patch(
                "/api/agent/fan-control/settings",
                json={"mode": "curve", "active_settings": curve, "expected_mode": "curve"},
            )
        with (
            patch.object(
                server.manager.agent_credentials, "authorize_controller", return_value=True,
            ),
            patch.object(server.manager, "update_fan_settings", update),
        ):
            invalid_expected_mode = await self.client.patch(
                "/api/agent/fan-control/settings",
                headers={"authorization": "Bearer paired"},
                json={"mode": "curve", "active_settings": curve, "expected_mode": {}},
            )
            accepted = await self.client.patch(
                "/api/agent/fan-control/settings",
                headers={"authorization": "Bearer paired"},
                json={"mode": "curve", "active_settings": curve, "expected_mode": "curve"},
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(invalid_expected_mode.status_code, 400)
        self.assertEqual(accepted.status_code, 200)
        update.assert_called_once_with("curve", curve, "curve")


if __name__ == "__main__":
    unittest.main()
