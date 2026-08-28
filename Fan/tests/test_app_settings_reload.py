import argparse
import importlib
import sys
import types
import unittest
from unittest import mock

from fancontroller.settings import Settings


def load_app_module():
    gi = types.ModuleType("gi")
    gi.require_version = lambda *_args: None
    repository = types.ModuleType("gi.repository")
    repository.Gtk = types.SimpleNamespace(
        RadioButton=object,
        CheckButton=object,
        ComboBoxText=object,
        Widget=object,
    )
    repository.GLib = types.SimpleNamespace(idle_add=mock.Mock())
    repository.Gdk = types.SimpleNamespace()
    gi.repository = repository

    fcntl = types.ModuleType("fcntl")
    fcntl.LOCK_EX = 1
    fcntl.LOCK_NB = 2
    fcntl.flock = lambda *_args: None

    curve_widget = types.ModuleType("fancontroller.curve_widget")
    curve_widget.FanCurveEditor = object
    plot_widget = types.ModuleType("fancontroller.plot_widget")
    plot_widget.HistoryPlot = object

    stubs = {
        "gi": gi,
        "gi.repository": repository,
        "fcntl": fcntl,
        "fancontroller.curve_widget": curve_widget,
        "fancontroller.plot_widget": plot_widget,
    }
    sys.modules.pop("fancontroller.app", None)
    with mock.patch.dict(sys.modules, stubs):
        return importlib.import_module("fancontroller.app")


class ForegroundSettingsReloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app_module = load_app_module()

    @classmethod
    def tearDownClass(cls) -> None:
        sys.modules.pop("fancontroller.app", None)

    def setUp(self) -> None:
        self.app_module.GLib.idle_add.reset_mock()

    def test_external_config_rebuilds_publishing_controllers(self) -> None:
        instance = self.app_module.FanApp.__new__(self.app_module.FanApp)
        instance.settings = Settings(mode="pid", poll_interval_s=1.0)
        instance.settings_mtime_ns = 100
        instance._config_mtime = mock.Mock(return_value=200)
        existing_source = types.SimpleNamespace(key="gpu")
        instance.sources = [existing_source]
        instance.source_map = {"gpu": existing_source}
        instance.temp_smoother = mock.Mock()
        instance.slew = mock.Mock()
        instance.current_duty_byte = 128
        instance.link = None
        instance._replace_serial_link = mock.Mock()
        instance._sync_settings_widgets = mock.Mock()
        instance._schedule_poll = mock.Mock()

        updated = Settings(
            mode="curve",
            curve_points=[[35.5, 15.25], [70.5, 88.75]],
            min_floor_pct=12.5,
            poll_interval_s=2.0,
            sources=["gpu"],
        )
        with (
            mock.patch.object(self.app_module.Settings, "load", return_value=updated),
            mock.patch.object(self.app_module, "discover", return_value=[]),
            mock.patch.object(self.app_module.time, "monotonic", return_value=50.0),
        ):
            instance._reload_settings_if_changed()

        self.assertIs(instance.settings, updated)
        self.assertEqual(instance.settings_mtime_ns, 200)
        self.assertEqual(instance.curve.points, [(35.5, 15.25), (70.5, 88.75)])
        self.assertEqual(instance.curve.min_floor_pct, 12.5)
        self.assertEqual(instance.pid.min_floor_pct, 12.5)
        self.assertEqual(instance.sources, [existing_source])
        self.assertEqual(instance.source_map, {"gpu": existing_source})
        self.assertEqual(instance.settings.sources, ["gpu"])
        instance.temp_smoother.reset.assert_called_once_with()
        instance.slew.sync.assert_called_once_with(
            self.app_module.byte_to_pct(128), 50.0,
        )
        instance.slew.reset.assert_not_called()
        instance._sync_settings_widgets.assert_called_once_with(False, False)
        self.app_module.GLib.idle_add.assert_called_with(instance._schedule_poll)

    def test_same_mode_hysteresis_and_new_source_survive_reload(self) -> None:
        instance = self.app_module.FanApp.__new__(self.app_module.FanApp)
        instance.settings = Settings(mode="hysteresis", poll_interval_s=1.0)
        instance.settings_mtime_ns = 100
        instance._config_mtime = mock.Mock(return_value=200)
        instance.sources = []
        instance.source_map = {}
        instance.hyst = self.app_module.Hysteresis(75.0, 65.0)
        instance.hyst._on = True
        instance.temp_smoother = mock.Mock()
        instance.slew = mock.Mock()
        instance.current_duty_byte = 180
        instance.link = None
        instance._replace_serial_link = mock.Mock()
        instance._sync_settings_widgets = mock.Mock()
        instance._schedule_poll = mock.Mock()

        new_source = types.SimpleNamespace(key="gpu")
        updated = Settings(
            mode="hysteresis",
            hyst_on_temp=78.0,
            hyst_off_temp=62.0,
            poll_interval_s=1.0,
            sources=["gpu"],
        )
        with (
            mock.patch.object(self.app_module.Settings, "load", return_value=updated),
            mock.patch.object(self.app_module, "discover", return_value=[new_source]),
            mock.patch.object(self.app_module.time, "monotonic", return_value=75.0),
        ):
            instance._reload_settings_if_changed()

        self.assertTrue(instance.hyst._on)
        self.assertEqual(instance.source_map, {"gpu": new_source})
        instance._sync_settings_widgets.assert_called_once_with(False, True)

    def test_startup_write_race_marks_the_fingerprint_stale(self) -> None:
        instance = self.app_module.FanApp.__new__(self.app_module.FanApp)
        instance._config_mtime = mock.Mock(side_effect=[100, 200])
        loaded = Settings(mode="pid")

        with mock.patch.object(
            self.app_module.Settings, "load", return_value=loaded,
        ):
            self.assertIs(instance._load_initial_settings(), loaded)

        self.assertEqual(
            instance.settings_mtime_ns,
            self.app_module.STALE_CONFIG_MTIME_NS,
        )

    def test_ui_only_poll_reloads_before_reading_daemon_state(self) -> None:
        instance = self.app_module.FanApp.__new__(self.app_module.FanApp)
        instance.args = argparse.Namespace(ui_only=True)
        instance._reload_settings_if_changed = mock.Mock()
        instance.link_status = "waiting"
        instance._update_status_label = mock.Mock()

        with mock.patch.object(self.app_module, "read_fan_state", return_value=None):
            self.assertTrue(instance._poll_tick_impl())

        instance._reload_settings_if_changed.assert_called_once_with()

    def test_local_save_advances_the_config_fingerprint(self) -> None:
        instance = self.app_module.FanApp.__new__(self.app_module.FanApp)
        instance.settings = mock.Mock()
        instance.settings_mtime_ns = 100
        instance._config_mtime = mock.Mock(side_effect=[300, 300])

        with mock.patch.object(
            self.app_module.Settings, "load", return_value=instance.settings,
        ):
            instance._save()

        instance.settings.save.assert_called_once_with()
        self.assertEqual(instance.settings_mtime_ns, 300)

    def test_racing_external_write_forces_a_reload_after_local_save(self) -> None:
        instance = self.app_module.FanApp.__new__(self.app_module.FanApp)
        instance.settings = mock.Mock()
        instance.settings_mtime_ns = 100
        instance._config_mtime = mock.Mock(side_effect=[300, 400])

        with mock.patch.object(
            self.app_module.Settings, "load", return_value=instance.settings,
        ):
            instance._save()

        self.assertEqual(
            instance.settings_mtime_ns,
            self.app_module.STALE_CONFIG_MTIME_NS,
        )


if __name__ == "__main__":
    unittest.main()
