"""Read control overrides written by external dashboards.

VLLMController (or any other tool) can drop a JSON file at
~/.local/state/fancontroller/control.json with {"max_speed": true} to force
the fan to 100 % duty.  FanController checks this file every poll tick.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


CONTROL_DIR = Path(os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")) / "fancontroller"
CONTROL_PATH = CONTROL_DIR / "control.json"

_cached_max_speed: bool | None = None
_cached_mtime: float | None = None


def read_max_speed() -> bool:
    """Return True if an external dashboard requested max fan speed.

    Results are cached keyed on the file's mtime so we don't hit disk on
    every poll tick (default 1 Hz) when the file hasn't changed.
    """
    global _cached_max_speed, _cached_mtime
    try:
        mtime = CONTROL_PATH.stat().st_mtime
    except OSError:
        _cached_max_speed = False
        _cached_mtime = None
        return False
    if _cached_mtime == mtime:
        return _cached_max_speed or False
    try:
        data = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))
        _cached_max_speed = bool(data.get("max_speed", False))
    except Exception:
        _cached_max_speed = False
    _cached_mtime = mtime
    return _cached_max_speed or False


def clear_max_speed() -> None:
    """Remove the control file (used on clean shutdown / reset)."""
    global _cached_max_speed, _cached_mtime
    try:
        CONTROL_PATH.unlink(missing_ok=True)
    except OSError:
        pass
    _cached_max_speed = False
    _cached_mtime = None
