import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import manager as manager_module
from manager import FanSettingsConflict, Manager
from cluster import NodeAgentResponseError


CURVE = {
    "curve_points": [[40, 0], [60, 35], [90, 100]],
    "curve_min_temp": 30,
    "curve_max_temp": 100,
    "min_floor_pct": 10,
}

PID = {
    "setpoint": 65,
    "kp": 4,
    "ki": 0.2,
    "kd": 1,
    "min_floor_pct": 5,
}


class FanSettingsTests(unittest.TestCase):
    def manager_for(self, config_path: Path, mode: str = "curve") -> Manager:
        instance = Manager.__new__(Manager)
        instance._fan_settings_lock = threading.Lock()
        instance._fan_control_lock = threading.Lock()
        instance._fan_config_path = lambda: config_path
        instance._read_fan_state = lambda: {
            "mode": mode,
            "active_settings": CURVE if mode == "curve" else PID,
        }
        return instance

    def test_curve_and_pid_fields_are_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown: kp"):
            Manager._validate_fan_settings("curve", {**CURVE, "kp": 2})
        with self.assertRaisesRegex(ValueError, "unknown: curve_points"):
            Manager._validate_fan_settings("pid", {**PID, "curve_points": []})
        with self.assertRaisesRegex(ValueError, "unknown fan mode"):
            Manager._validate_fan_settings([], {})

    def test_unhashable_expected_mode_is_rejected_before_state_access(self) -> None:
        instance = Manager.__new__(Manager)

        with self.assertRaisesRegex(ValueError, "unknown expected fan mode"):
            instance.update_fan_settings("curve", CURVE, expected_mode=[])

    def test_curve_validation_requires_sorted_unique_points(self) -> None:
        invalid = {**CURVE, "curve_points": [[60, 20], [60, 30]]}
        with self.assertRaisesRegex(ValueError, "unique and increasing"):
            Manager._validate_fan_settings("curve", invalid)

    def test_stale_mode_is_rejected_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"mode": "pid", **PID}), encoding="utf-8")
            instance = self.manager_for(path, mode="pid")

            with self.assertRaisesRegex(FanSettingsConflict, "mode changed"):
                instance.update_fan_settings("curve", CURVE)

            self.assertEqual(json.loads(path.read_text()), {"mode": "pid", **PID})

    def test_update_preserves_other_fields_and_atomically_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            original = {
                "mode": "curve",
                "serial_port": "/dev/ttyACM0",
                "kp": 7.5,
                "manual_duty_pct": 42,
                **CURVE,
            }
            path.write_text(json.dumps(original), encoding="utf-8")
            instance = self.manager_for(path)
            updated = {**CURVE, "min_floor_pct": 15}
            real_replace = os.replace

            with mock.patch.object(manager_module.os, "replace", wraps=real_replace) as replace:
                result = instance.update_fan_settings("curve", updated)

            saved = json.loads(path.read_text())
            self.assertEqual(saved["serial_port"], "/dev/ttyACM0")
            self.assertEqual(saved["kp"], 7.5)
            self.assertEqual(saved["manual_duty_pct"], 42)
            self.assertEqual(saved["min_floor_pct"], 15.0)
            self.assertEqual(result["active_settings"], Manager._validate_fan_settings("curve", updated))
            replace.assert_called_once_with(path.with_suffix(".json.tmp"), path)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_mode_can_be_changed_when_live_mode_matches_expectation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"mode": "pid", **PID, **CURVE}), encoding="utf-8")
            instance = self.manager_for(path, mode="pid")

            result = instance.update_fan_settings(
                "manual", {"manual_duty_pct": 55}, expected_mode="pid",
            )

            saved = json.loads(path.read_text())
            self.assertEqual(saved["mode"], "manual")
            self.assertEqual(saved["manual_duty_pct"], 55.0)
            self.assertEqual(saved["curve_points"], CURVE["curve_points"])
            self.assertEqual(result["previous_mode"], "pid")

    def test_get_settings_returns_all_mode_panes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "mode": "pid",
                "setpoint": 72,
                "serial_port": "/dev/ttyACM0",
            }), encoding="utf-8")
            instance = self.manager_for(path, mode="pid")

            result = instance.get_fan_settings()

            self.assertEqual(result["mode"], "pid")
            self.assertEqual(result["settings"]["pid"]["setpoint"], 72)
            self.assertIn("curve_points", result["settings"]["curve"])
            self.assertNotIn("serial_port", result["settings"])

    def test_state_reader_honors_xdg_state_home(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "fancontroller" / "state.json"
            state_path.parent.mkdir()
            state_path.write_text(json.dumps({
                "rpm": 1800,
                "duty_byte": 128,
                "duty_pct": 50.2,
                "temp": 71.5,
                "mode": "curve",
                "active_settings": CURVE,
                "status": "connected",
                "max_speed": False,
                "ts": time.time(),
            }), encoding="utf-8")
            instance = Manager.__new__(Manager)

            with mock.patch.dict(os.environ, {"XDG_STATE_HOME": directory}):
                state = instance._read_fan_state()

            self.assertEqual(state["active_settings"], CURVE)
            self.assertEqual(state["mode"], "curve")

            stale_mtime = time.time() - manager_module.FAN_STATE_MAX_AGE_SECONDS - 1
            os.utime(state_path, (stale_mtime, stale_mtime))
            with mock.patch.dict(os.environ, {"XDG_STATE_HOME": directory}):
                self.assertIsNone(instance._read_fan_state())

    def test_state_sanitizer_rejects_stale_future_and_malformed_files(self) -> None:
        now = 10_000.0
        state = {
            "rpm": 1800,
            "duty_byte": 128,
            "duty_pct": 50.2,
            "temp": 71.5,
            "mode": "curve",
            "active_settings": CURVE,
            "status": "connected",
            "max_speed": False,
            "ts": now,
        }

        self.assertIsNotNone(Manager._sanitize_fan_state(state, now=now))
        self.assertIsNone(Manager._sanitize_fan_state({
            **state, "ts": now - manager_module.FAN_STATE_MAX_AGE_SECONDS - 0.1,
        }, now=now))
        self.assertIsNone(Manager._sanitize_fan_state({
            **state, "ts": now + manager_module.FAN_STATE_MAX_FUTURE_SKEW_SECONDS + 0.1,
        }, now=now))
        self.assertIsNone(Manager._sanitize_fan_state({
            **state, "duty_pct": "100",
        }, now=now))
        self.assertIsNone(Manager._sanitize_fan_state({
            **state, "active_settings": {**CURVE, "script": "unsafe"},
        }, now=now))

    def test_cluster_temperature_uses_hottest_fresh_cpu_or_gpu(self) -> None:
        now = 1_000.0
        result = Manager._cluster_temperature_override([
            {
                "id": "node-1",
                "name": "gx10-node-1",
                "online": True,
                "stats": {
                    "ts": now - 1,
                    "cpu_temp_c": 48.0,
                    "gpus": [{"temp_c": 52.0}],
                },
            },
            {
                "id": "node-3",
                "name": "gx10-node-3",
                "online": True,
                "stats": {
                    "ts": now - 2,
                    "cpu_temp_c": 71.5,
                    "gpus": [{"temp_c": None}],
                },
            },
            {
                "id": "stale",
                "name": "stale-hot-node",
                "online": True,
                "stats": {"ts": now - 30, "cpu_temp_c": 99.0},
            },
            {
                "id": "offline",
                "name": "offline-hot-node",
                "online": False,
                "stats": {"ts": now, "cpu_temp_c": 100.0},
            },
        ], now=now)

        self.assertEqual(result["temperature_c"], 71.5)
        self.assertEqual(result["node_id"], "node-3")
        self.assertEqual(result["node_name"], "gx10-node-3")
        self.assertEqual(result["sensor"], "cpu")
        self.assertGreater(result["expires_at"], now)

    def test_fan_control_updates_preserve_independent_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fancontroller" / "control.json"
            instance = Manager.__new__(Manager)
            instance._fan_control_lock = threading.Lock()
            instance._fan_control_path = lambda: path
            instance._read_fan_state = lambda: {
                "mode": "curve", "active_settings": CURVE,
            }
            override = {
                "temperature_c": 75.0,
                "expires_at": time.time() + 10,
            }

            instance._set_fan_temperature_override(override)
            instance.set_fan_max_speed(True)
            self.assertEqual(json.loads(path.read_text()), {
                "temperature_override": override,
                "max_speed": True,
            })

            instance.set_fan_max_speed(False)
            self.assertEqual(json.loads(path.read_text()), {
                "temperature_override": override,
            })


class FanControlClusterTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def live_fan(now: float | None = None) -> dict:
        return {
            "rpm": 1800,
            "duty_byte": 128,
            "duty_pct": 50.2,
            "temp": 71.5,
            "mode": "curve",
            "active_settings": CURVE,
            "status": "connected",
            "max_speed": False,
            "ts": time.time() if now is None else now,
        }

    @staticmethod
    def settings() -> dict:
        return {
            "mode": "curve",
            "settings": {
                mode: {key: value for key, value in values.items()}
                for mode, values in manager_module.FAN_MODE_DEFAULTS.items()
            },
        }

    async def test_cluster_overview_only_returns_revalidated_capable_nodes(self) -> None:
        instance = Manager.__new__(Manager)
        local_fan = self.live_fan()
        remote_fan = self.live_fan()
        stale_fan = self.live_fan(time.time() - 60)
        instance.cluster_nodes = mock.AsyncMock(return_value=[
            {"id": "local", "name": "Controller", "online": True,
             "stats": {"fan": local_fan}},
            {"id": "worker-1", "name": "Rack", "online": True,
             "stats": {"fan": remote_fan}},
            {"id": "stale", "name": "Old", "online": True,
             "stats": {"fan": stale_fan}},
            {"id": "plain", "name": "No fan", "online": True,
             "stats": {"fan": None}},
        ])
        instance.local_fan_control_overview = mock.Mock(return_value={
            "fan": local_fan, "settings": self.settings(),
        })
        instance.node_registry = mock.Mock()
        instance.node_registry.request = mock.AsyncMock(return_value={
            "fan": remote_fan, "settings": self.settings(),
        })

        result = await instance.fan_control_cluster_overview()

        self.assertTrue(result["available"])
        self.assertEqual(
            [node["node_id"] for node in result["nodes"]],
            ["local", "worker-1"],
        )
        self.assertIn("curve", result["nodes"][1]["settings"]["settings"])
        instance.node_registry.request.assert_awaited_once_with(
            "worker-1", "GET", "/api/agent/fan-control",
            timeout=manager_module.FAN_CONTROL_AGENT_TIMEOUT_SECONDS,
        )

    async def test_remote_max_speed_uses_authenticated_agent_contract(self) -> None:
        instance = Manager.__new__(Manager)
        instance.cluster_nodes = mock.AsyncMock(return_value=[{
            "id": "worker-1", "name": "Rack", "online": True,
            "stats": {"fan": self.live_fan()},
        }])
        instance.node_registry = mock.Mock()
        instance.node_registry.request = mock.AsyncMock(return_value={"enabled": True})

        result = await instance.set_node_fan_max_speed("worker-1", True)

        self.assertEqual(result, {"node_id": "worker-1", "enabled": True})
        instance.node_registry.request.assert_awaited_once_with(
            "worker-1", "PATCH", "/api/agent/fan-control/max-speed",
            json_body={"enabled": True},
            timeout=manager_module.FAN_CONTROL_AGENT_TIMEOUT_SECONDS,
        )

    async def test_remote_settings_update_uses_authenticated_agent_contract(self) -> None:
        instance = Manager.__new__(Manager)
        instance.cluster_nodes = mock.AsyncMock(return_value=[{
            "id": "worker-1", "name": "Rack", "online": True,
            "stats": {"fan": self.live_fan()},
        }])
        instance.node_registry = mock.Mock()
        instance.node_registry.request = mock.AsyncMock(return_value={
            "mode": "curve", "previous_mode": "pid", "active_settings": CURVE,
        })

        result = await instance.update_node_fan_settings(
            "worker-1", "curve", CURVE, "pid",
        )

        self.assertEqual(result["node_id"], "worker-1")
        self.assertEqual(result["active_settings"], CURVE)
        instance.node_registry.request.assert_awaited_once_with(
            "worker-1", "PATCH", "/api/agent/fan-control/settings",
            json_body={
                "mode": "curve", "active_settings": CURVE, "expected_mode": "pid",
            },
            timeout=manager_module.FAN_CONTROL_AGENT_TIMEOUT_SECONDS,
        )

    async def test_remote_settings_preserve_agent_mode_conflict(self) -> None:
        instance = Manager.__new__(Manager)
        instance.cluster_nodes = mock.AsyncMock(return_value=[{
            "id": "worker-1", "name": "Rack", "online": True,
            "stats": {"fan": self.live_fan()},
        }])
        instance.node_registry = mock.Mock()
        instance.node_registry.request = mock.AsyncMock(side_effect=NodeAgentResponseError(
            "Rack", 409, '{"detail":"fan mode changed; refresh and try again"}',
        ))

        with self.assertRaisesRegex(FanSettingsConflict, "fan mode changed"):
            await instance.update_node_fan_settings(
                "worker-1", "curve", CURVE, "pid",
            )

    async def test_remote_settings_rethrow_non_conflict_agent_errors(self) -> None:
        instance = Manager.__new__(Manager)
        instance.cluster_nodes = mock.AsyncMock(return_value=[{
            "id": "worker-1", "name": "Rack", "online": True,
            "stats": {"fan": self.live_fan()},
        }])
        instance.node_registry = mock.Mock()
        instance.node_registry.request = mock.AsyncMock(side_effect=NodeAgentResponseError(
            "Rack", 500, '{"detail":"write failed"}',
        ))

        with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
            await instance.update_node_fan_settings(
                "worker-1", "curve", CURVE, "pid",
            )

    async def test_node_settings_reject_unhashable_expected_mode_before_routing(self) -> None:
        instance = Manager.__new__(Manager)
        instance.cluster_nodes = mock.AsyncMock()

        with self.assertRaisesRegex(ValueError, "unknown expected fan mode"):
            await instance.update_node_fan_settings(
                "worker-1", "curve", CURVE, {},
            )

        instance.cluster_nodes.assert_not_awaited()

    async def test_max_speed_rejects_non_boolean_before_routing(self) -> None:
        instance = Manager.__new__(Manager)
        instance.cluster_nodes = mock.AsyncMock()

        with self.assertRaisesRegex(ValueError, "enabled must be a boolean"):
            await instance.set_node_fan_max_speed("local", 1)

        instance.cluster_nodes.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
