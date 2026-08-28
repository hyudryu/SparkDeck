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

    def test_external_config_rebuilds_publishing_controllers(self) -> None:
        instance = self.app_module.FanApp.__new__(self.app_module.FanApp)
        instance.settings = Settings(mode="pid", poll_interval_s=1.0)
        instance.settings_mtime_ns = 100
        instance._config_mtime = mock.Mock(return_value=200)
        instance.sources = []
        instance.source_map = {}
        instance.temp_smoother = mock.Mock()
        instance.slew = mock.Mock()
        instance.link = None
        instance._replace_serial_link = mock.Mock()
        instance._sync_settings_widgets = mock.Mock()
        instance._schedule_poll = mock.Mock()

        updated = Settings(
            mode="curve",
            curve_points=[[35.5, 15.25], [70.5, 88.75]],
            min_floor_pct=12.5,
            poll_interval_s=2.0,
        )
        with (
            mock.patch.object(self.app_module.Settings, "load", return_value=updated),
            mock.patch.object(self.app_module, "discover", return_value=[]),
        ):
            instance._reload_settings_if_changed()

        self.assertIs(instance.settings, updated)
        self.assertEqual(instance.settings_mtime_ns, 200)
        self.assertEqual(instance.curve.points, [(35.5, 15.25), (70.5, 88.75)])
        self.assertEqual(instance.curve.min_floor_pct, 12.5)
        self.assertEqual(instance.pid.min_floor_pct, 12.5)
        instance.temp_smoother.reset.assert_called_once_with()
        instance.slew.reset.assert_called_once_with()
        instance._sync_settings_widgets.assert_called_once_with(False)
        self.app_module.GLib.idle_add.assert_called_with(instance._schedule_poll)

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
        instance._config_mtime = mock.Mock(return_value=300)

        instance._save()

        instance.settings.save.assert_called_once_with()
        self.assertEqual(instance.settings_mtime_ns, 300)


if __name__ == "__main__":
    unittest.main()
