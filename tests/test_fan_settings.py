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


if __name__ == "__main__":
    unittest.main()
