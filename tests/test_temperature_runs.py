import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from manager import Manager


class TemperatureRunTests(unittest.IsolatedAsyncioTestCase):
    def manager_for_test(self, directory: str) -> Manager:
        instance = Manager.__new__(Manager)
        instance.temperature_runs_path = Path(directory) / "temperature_runs.json"
        instance.temperature_runs = {}
        instance._active_temperature_run_id = None
        instance._temperature_recording_lock = asyncio.Lock()
        instance._temperature_runs_last_saved_at = 0.0
        instance.temperature_recording_task = None
        instance.settings = {"cluster_node_name": "Spark One"}
        instance.node_registry = mock.Mock()
        return instance

    def test_hysteresis_starts_at_margin_and_stops_below_target(self) -> None:
        run = {
            "status": "armed",
            "target_temp_c": 60.0,
            "trigger_temp_c": 63.0,
            "samples": [],
        }
        instance = Manager.__new__(Manager)

        self.assertEqual(instance._process_temperature_run_sample(
            run, {"cpu_temp_c": 62.9, "gpus": [{"temp": 61.0}]}, 100.0,
        ), "waiting")
        self.assertEqual(run["samples"], [])

        self.assertEqual(instance._process_temperature_run_sample(
            run, {"cpu_temp_c": 61.0, "gpus": [{"temp": 63.0}]}, 101.0,
        ), "recording")
        self.assertEqual(run["started_at"], 101.0)
        self.assertEqual(run["samples"][0]["elapsed_seconds"], 0.0)

        self.assertEqual(instance._process_temperature_run_sample(
            run, {"cpu_temp_c": 60.0, "gpus": [{"temp": 59.0}]}, 102.0,
        ), "recording")
        self.assertEqual(instance._process_temperature_run_sample(
            run, {"cpu_temp_c": 59.9, "gpus": [{"temp": 58.0}]}, 103.0,
        ), "complete")
        self.assertEqual(run["status"], "complete")
        self.assertEqual(run["stopped_at"], 103.0)
        self.assertEqual(len(run["samples"]), 3)

    async def test_arm_cancel_rename_and_persist_waiting_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = self.manager_for_test(directory)
            instance.get_stats = mock.AsyncMock(return_value={
                "cpu_temp_c": 45.0,
                "gpus": [{"temp": 50.0}],
            })

            run = await instance.arm_temperature_recording("local", 60, 5)

            self.assertEqual(run["status"], "armed")
            self.assertEqual(run["trigger_temp_c"], 63.0)
            self.assertEqual(instance.temperature_runs_state()["active_run_id"], run["id"])

            renamed = instance.rename_temperature_run(run["id"], "Cool-down A")
            self.assertEqual(renamed["name"], "Cool-down A")

            cancelled = await instance.cancel_temperature_recording()
            self.assertEqual(cancelled["status"], "cancelled")
            self.assertIsNone(instance.temperature_runs_state()["active_run_id"])

            persisted = json.loads(instance.temperature_runs_path.read_text())
            self.assertEqual(persisted["runs"][0]["name"], "Cool-down A")
            self.assertEqual(persisted["runs"][0]["status"], "cancelled")

            reloaded = self.manager_for_test(directory)
            reloaded.temperature_runs = reloaded._load_temperature_runs()
            self.assertEqual(reloaded.temperature_run(run["id"])["name"], "Cool-down A")

    async def test_remote_recording_uses_authenticated_agent_stats_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = self.manager_for_test(directory)
            instance.node_registry.get.return_value = {
                "id": "node-2", "name": "Spark Two", "enabled": True,
            }
            instance.node_registry.request = mock.AsyncMock(return_value={
                "cpu_temp_c": 40.0, "gpus": [{"temp": 42.0}],
            })

            run = await instance.arm_temperature_recording("node-2", 60, 5)

            self.assertEqual(run["node_name"], "Spark Two")
            instance.node_registry.request.assert_awaited_once_with(
                "node-2", "GET", "/api/agent/stats", timeout=5,
            )
            await instance.cancel_temperature_recording()

    async def test_recording_can_be_manually_cancelled_after_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = self.manager_for_test(directory)
            instance.get_stats = mock.AsyncMock(return_value={
                "cpu_temp_c": 64.0, "gpus": [{"temp": 65.0}],
            })
            run = await instance.arm_temperature_recording("local", 60, 5)
            self.assertEqual(run["status"], "recording")

            cancelled = await instance.cancel_temperature_recording()

            self.assertEqual(cancelled["status"], "cancelled")
            self.assertIsNone(instance._active_temperature_run_id)

    def test_sustained_telemetry_failure_interrupts_run(self) -> None:
        instance = Manager.__new__(Manager)
        run = {"status": "recording", "samples": []}

        for attempt in range(4):
            self.assertFalse(instance._record_temperature_telemetry_failure(
                run, "node unavailable", 100.0 + attempt,
            ))
        self.assertTrue(instance._record_temperature_telemetry_failure(
            run, "node unavailable", 104.0,
        ))

        self.assertEqual(run["status"], "interrupted")
        self.assertEqual(run["stopped_at"], 104.0)
        self.assertEqual(run["samples"], [])

    def test_valid_telemetry_resets_failure_streak_while_armed(self) -> None:
        instance = Manager.__new__(Manager)
        run = {
            "status": "armed",
            "target_temp_c": 60.0,
            "trigger_temp_c": 63.0,
            "telemetry_failures": 4,
            "last_error": "node unavailable",
            "samples": [],
        }

        result = instance._process_temperature_run_sample(
            run, {"cpu_temp_c": 45.0, "gpus": []}, 100.0,
        )

        self.assertEqual(result, "waiting")
        self.assertNotIn("telemetry_failures", run)
        self.assertNotIn("last_error", run)
