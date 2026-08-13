import asyncio
import unittest
from collections import deque
from unittest import mock

from manager import (
    Manager,
    TEMPERATURE_HISTORY_INTERVAL_SECONDS,
    TEMPERATURE_HISTORY_MAX_SAMPLES,
    TEMPERATURE_HISTORY_WINDOW_SECONDS,
)


class TemperatureHistoryTests(unittest.IsolatedAsyncioTestCase):
    def manager_for_test(self) -> Manager:
        instance = Manager.__new__(Manager)
        instance._temperature_history = deque(maxlen=TEMPERATURE_HISTORY_MAX_SAMPLES)
        return instance

    def test_samples_are_downsampled_and_invalid_values_become_gaps(self) -> None:
        instance = self.manager_for_test()

        self.assertTrue(instance._record_temperature_sample({
            "cpu_temp_c": 51.23,
            "gpus": [{"temp": 62.18}],
        }, 1000.0))
        self.assertFalse(instance._record_temperature_sample({
            "cpu_temp_c": 99,
            "gpus": [{"temp": 99}],
        }, 1000.0 + TEMPERATURE_HISTORY_INTERVAL_SECONDS - 1))
        self.assertTrue(instance._record_temperature_sample({
            "cpu_temp_c": float("nan"),
            "gpus": [{"temp": None}],
        }, 1000.0 + TEMPERATURE_HISTORY_INTERVAL_SECONDS))

        self.assertEqual(instance.temperature_history()["samples"], [
            {"ts": 1000.0, "cpu_temp_c": 51.2, "gpu_temp_c": 62.2},
            {"ts": 1030.0, "cpu_temp_c": None, "gpu_temp_c": None},
        ])

    def test_samples_older_than_two_hours_are_removed(self) -> None:
        instance = self.manager_for_test()
        instance._record_temperature_sample({"gpus": []}, 10.0)
        now = 10.0 + TEMPERATURE_HISTORY_WINDOW_SECONDS + 1

        instance._record_temperature_sample({"cpu_temp_c": 55, "gpus": []}, now)

        self.assertEqual(len(instance._temperature_history), 1)
        self.assertEqual(instance._temperature_history[0]["ts"], now)

    async def test_remote_history_is_requested_from_selected_agent(self) -> None:
        instance = self.manager_for_test()
        instance.node_registry = mock.Mock()
        instance.node_registry.request = mock.AsyncMock(return_value={
            "window_seconds": 7200,
            "samples": [{"ts": 10, "cpu_temp_c": 50}],
        })

        result = await instance.temperature_history_for_node("node-2")

        self.assertEqual(result["node_id"], "node-2")
        instance.node_registry.request.assert_awaited_once_with(
            "node-2", "GET", "/api/agent/temperature-history", timeout=5,
        )

    async def test_monitor_samples_immediately_and_stops_cleanly(self) -> None:
        instance = self.manager_for_test()
        instance.get_stats = mock.AsyncMock(return_value={})

        task = asyncio.create_task(instance._temperature_history_monitor_loop())
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        instance.get_stats.assert_awaited_once()
