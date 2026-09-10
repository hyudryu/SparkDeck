"""The History panel's endpoints and its settings.

The collector's own behaviour is covered by
:mod:`tests.test_live_history_metrics`.  This exercises the real route handlers
in ``server.py`` rather than a copy of them, calling the async view functions
directly so no cluster lifespan and no real settings database are needed:

* the sampler starts on first use, and the payload carries the cadence and the
  trailing range the graph draws;
* the settings route turns recording on and off and clamps the sampling
  interval, rejecting unusable values instead of silently rewriting the cadence.
"""

import asyncio
import unittest
from types import SimpleNamespace

import server
from sparkdeck.live_metrics import (
    KEEP_SECONDS,
    MAX_SAMPLE_SECONDS,
    MIN_SAMPLE_SECONDS,
    LiveHistory,
)


class _Store:
    """In-memory stand-in for the settings store."""

    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def get_setting(self, key: str, default=None):
        return self.values.get(key, default)

    def set_setting(self, key: str, value) -> None:
        self.values[key] = value


class _Request:
    def __init__(self, body) -> None:
        self._body = body

    async def json(self):
        return self._body


class _Harness:
    """Swaps the service's history collector and store for test doubles."""

    def __init__(self) -> None:
        self.store = _Store()
        self.history = LiveHistory(
            SimpleNamespace(_active_reqs={}),
            interval_provider=lambda: self.store.get_setting("history_sample_seconds", 5),
            enabled_provider=lambda: bool(self.store.get_setting("history_enabled", True)),
        )

    def __enter__(self):
        self._store = server.sparkdeck.store
        self._history = server.sparkdeck.history
        server.sparkdeck.store = self.store
        server.sparkdeck.history = self.history
        return self

    def __exit__(self, *exc):
        server.sparkdeck.store = self._store
        server.sparkdeck.history = self._history
        return False


def _settings(body):
    return asyncio.run(server.update_live_history_settings(_Request(body)))


def _read():
    return asyncio.run(server.get_live_history())


class LiveHistoryEndpointTests(unittest.TestCase):
    def test_first_read_starts_the_sampler_and_reports_the_window(self) -> None:
        async def run() -> tuple[dict, LiveHistory]:
            payload = await server.get_live_history()
            history = server.sparkdeck.history
            # The panel's first poll starts the sampler, and it stays up so the
            # trailing hour keeps filling in between polls.
            self.assertIsNotNone(history._task)
            self.assertFalse(history._task.done())
            await history.stop()
            return payload, history

        with _Harness():
            payload, _ = asyncio.run(run())

        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["range_seconds"], KEEP_SECONDS)
        self.assertEqual(payload["sample_seconds"], 5.0)
        self.assertEqual(payload["series"], [])
        self.assertIsInstance(payload["generated_at"], float)

    def test_snapshot_never_exposes_manager_internals(self) -> None:
        snapshot = LiveHistory(SimpleNamespace(_active_reqs={})).snapshot()

        self.assertEqual(
            set(snapshot),
            {
                "generated_at", "enabled", "bucket_seconds",
                "range_seconds", "sample_seconds", "series",
            },
        )


class LiveHistorySettingsTests(unittest.TestCase):
    def test_recording_can_be_switched_off_and_back_on(self) -> None:
        with _Harness() as harness:
            off = _settings({"enabled": False})
            self.assertFalse(off["enabled"])
            self.assertEqual(off["series"], [])
            self.assertIs(harness.store.values["history_enabled"], False)

            on = _settings({"enabled": True})
            self.assertTrue(on["enabled"])
            asyncio.run(harness.history.stop())

    def test_a_disabled_panel_reads_back_no_series(self) -> None:
        with _Harness() as harness:
            _settings({"enabled": False})
            payload = _read()

            self.assertFalse(payload["enabled"])
            self.assertEqual(payload["series"], [])
            # Nothing is kept in memory while recording is off.
            self.assertEqual(harness.history.series(), [])
            asyncio.run(harness.history.stop())

    def test_the_interval_is_saved_and_the_snapshot_follows_it(self) -> None:
        with _Harness() as harness:
            payload = _settings({"sample_seconds": 12})

            self.assertEqual(harness.store.values["history_sample_seconds"], 12)
            self.assertEqual(payload["sample_seconds"], 12.0)
            self.assertEqual(payload["bucket_seconds"], 12.0)
            asyncio.run(harness.history.stop())

    def test_an_out_of_range_interval_is_clamped_rather_than_rejected(self) -> None:
        with _Harness() as harness:
            low = _settings({"sample_seconds": 0})
            high = _settings({"sample_seconds": 600})

            self.assertEqual(low["sample_seconds"], float(MIN_SAMPLE_SECONDS))
            self.assertEqual(high["sample_seconds"], float(MAX_SAMPLE_SECONDS))
            self.assertEqual(harness.store.values["history_sample_seconds"], MAX_SAMPLE_SECONDS)
            asyncio.run(harness.history.stop())

    def test_junk_settings_are_rejected_without_changing_anything(self) -> None:
        from fastapi import HTTPException

        with _Harness() as harness:
            for body in (
                {"enabled": "yes"},
                {"sample_seconds": "soon"},
                {"sample_seconds": True},
                {"sample_seconds": None},
            ):
                with self.subTest(body=body):
                    with self.assertRaises(HTTPException) as raised:
                        _settings(body)
                    self.assertEqual(raised.exception.status_code, 400)

            self.assertEqual(harness.store.values, {})

    def test_a_partial_update_leaves_the_other_setting_alone(self) -> None:
        with _Harness() as harness:
            _settings({"enabled": False})
            _settings({"sample_seconds": 20})

            self.assertIs(harness.store.values["history_enabled"], False)
            self.assertEqual(harness.store.values["history_sample_seconds"], 20)
            asyncio.run(harness.history.stop())

    def test_settings_are_bounded_to_the_documented_range(self) -> None:
        with _Harness() as harness:
            for seconds in (1, 2, 5, 10, 15, 30):
                with self.subTest(seconds=seconds):
                    payload = _settings({"sample_seconds": seconds})
                    self.assertEqual(payload["sample_seconds"], float(seconds))
            asyncio.run(harness.history.stop())
