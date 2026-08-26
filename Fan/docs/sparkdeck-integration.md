# FanController integration for SparkDeck

## State transport

FanController publishes its current state as JSON after every control-loop
update (normally once per second).

- Preferred path: `$XDG_STATE_HOME/fancontroller/state.json`
- Default path when `XDG_STATE_HOME` is unset:
  `~/.local/state/fancontroller/state.json`

The write is atomic: FanController writes `state.json.tmp` and then replaces
`state.json`. Consumers may read `state.json` without taking a lock.

Example while curve mode is active:

```json
{
  "rpm": 1800,
  "duty_byte": 128,
  "duty_pct": 50.19607843137255,
  "temp": 71.5,
  "mode": "curve",
  "active_settings": {
    "curve_points": [
      [40.0, 0.0],
      [60.0, 30.0],
      [75.0, 60.0],
      [90.0, 100.0]
    ],
    "curve_min_temp": 30.0,
    "curve_max_temp": 100.0,
    "min_floor_pct": 0.0
  },
  "status": "connected",
  "max_speed": false,
  "ts": 1785243015.5749884
}
```

### Common fields

| Field | Type | Meaning |
| --- | --- | --- |
| `rpm` | integer | Display RPM. When `max_rpm > 0`, this is estimated from duty; otherwise it is tachometer RPM. |
| `duty_byte` | integer | PWM output in the inclusive range 0–255. |
| `duty_pct` | number | The same PWM output expressed as a percentage. |
| `temp` | number or null | Maximum temperature across selected sensors, in degrees Celsius. |
| `mode` | string | `curve`, `pid`, `hysteresis`, or `manual`. |
| `active_settings` | object | Only the settings applicable to `mode`; schemas are below. |
| `status` | string | Fan serial-link status for display. |
| `max_speed` | boolean | Whether the external full-speed override is active. |
| `ts` | number | Unix timestamp in seconds. |

Treat state as unavailable if it is missing, invalid JSON, or more than 30
seconds old. Unknown fields should be ignored for forward compatibility.

### `active_settings` schemas

Only one of these schemas is emitted at a time. Curve fields are not emitted
in PID mode, and PID fields are not emitted in curve mode.

#### Curve mode

```json
{
  "curve_points": [[40.0, 0.0], [60.0, 30.0], [90.0, 100.0]],
  "curve_min_temp": 30.0,
  "curve_max_temp": 100.0,
  "min_floor_pct": 0.0
}
```

- Each `curve_points` entry is `[temperature_celsius, duty_percent]`.
- Points should be sorted by temperature with unique temperatures.
- Use at least two points for the editor.
- Duty and `min_floor_pct` must be between 0 and 100.
- The existing FanController UI permits `curve_min_temp` from 0–90 °C and
  `curve_max_temp` from 40–120 °C, with max greater than min.
- The range fields control the editor's X axis. The curve points control the
  fan output.

#### PID mode

```json
{
  "setpoint": 65.0,
  "kp": 4.0,
  "ki": 0.2,
  "kd": 1.0,
  "min_floor_pct": 0.0
}
```

The existing UI ranges are: `setpoint` 30–100 °C, `kp` 0–50, `ki` 0–10,
`kd` 0–20, and `min_floor_pct` 0–100.

#### Hysteresis mode

```json
{
  "hyst_on_temp": 75.0,
  "hyst_off_temp": 65.0
}
```

The existing UI permits `hyst_on_temp` from 30–110 °C and
`hyst_off_temp` from 20–100 °C. The off temperature should be lower than the
on temperature.

#### Manual mode

```json
{
  "manual_duty_pct": 100.0
}
```

`manual_duty_pct` must be between 0 and 100.

An unknown mode emits an empty `active_settings` object.

## Current SparkDeck read path

SparkDeck already reads the state file in
`Manager._read_fan_state()` and exposes the result as `fan` from:

```http
GET /api/stats
```

At the time of this document, `_read_fan_state()` reconstructs the object
field by field and therefore drops `active_settings`. Add this field to its
return value:

```python
"active_settings": data.get("active_settings", {}),
```

SparkDeck currently hardcodes
`~/.local/state/fancontroller/state.json`. It should use `XDG_STATE_HOME` when
set so its lookup matches FanController.

## Updating settings

There is no FanController HTTP endpoint for changing controller settings yet.
The existing SparkDeck endpoints only control the full-speed override:

```http
GET  /api/fan/max-speed
POST /api/fan/max-speed   {"enabled": true}
```

The control file supports independent, short-lived runtime overrides. It is
not a persistent settings channel, and adding curve or PID keys there has no
effect. SparkDeck preserves unrelated fields when updating either
override:

```json
{
  "max_speed": false,
  "temperature_override": {
    "temperature_c": 74.5,
    "source": "vllm-cluster-max",
    "sensor": "cpu",
    "node_id": "node-3",
    "node_name": "gx10-node-3",
    "observed_at": 1710000000.0,
    "expires_at": 1710000012.0
  }
}
```

FanController validates the temperature and expiry on every control tick and
feeds `max(local_temperature, temperature_override)` into the existing active
curve, PID, or hysteresis controller. An expired or malformed override is
ignored. If neither local nor external telemetry is available, the existing
fail-safe behavior still requests full fan speed.

FanController loads persistent settings from:

```text
~/.config/fancontroller/config.json
```

The headless daemon watches this file's nanosecond modification time and
reloads it on its next poll, normally within one second. SparkDeck can
therefore provide its own endpoint, for example:

```http
POST /api/fan/settings
Content-Type: application/json

{
  "mode": "curve",
  "active_settings": {
    "curve_points": [[40, 0], [55, 25], [70, 60], [85, 100]],
    "curve_min_temp": 30,
    "curve_max_temp": 100,
    "min_floor_pct": 15
  }
}
```

Recommended endpoint behavior:

1. Require `mode` to equal the currently published mode. This prevents a stale
   editor from writing settings for the wrong controller.
2. Validate the request against the matching schema and reject unknown fields.
3. Read the existing config as a JSON object.
4. Merge the validated active fields into that object. Do not replace the
   config with the partial request: omitted settings would revert to defaults.
5. In the same directory, write the complete merged object to
   `config.json.tmp`, flush it, and atomically replace `config.json` with
   `os.replace()`.
6. Return the accepted values. The UI should use a later `/api/stats` response
   as confirmation that FanController loaded and republished them.

Illustrative backend logic (validation omitted here intentionally):

```python
def update_fan_settings(mode: str, updates: dict) -> dict:
    path = Path.home() / ".config" / "fancontroller" / "config.json"
    current = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(current, dict):
        raise ValueError("FanController config is not an object")

    current_mode = manager._read_fan_state().get("mode")
    if mode != current_mode:
        raise ValueError("fan mode changed; refresh and try again")

    allowed = {
        "curve": {
            "curve_points", "curve_min_temp", "curve_max_temp",
            "min_floor_pct",
        },
        "pid": {"setpoint", "kp", "ki", "kd", "min_floor_pct"},
        "hysteresis": {"hyst_on_temp", "hyst_off_temp"},
        "manual": {"manual_duty_pct"},
    }[mode]
    if updates.keys() - allowed:
        raise ValueError("unknown setting")

    current.update(updates)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(current, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return {"mode": mode, "active_settings": updates}
```

Production code should use FastAPI request models or equivalent validation,
map invalid input to HTTP 400/409 responses, and serialize concurrent writes
with the manager's lock. Reject an update if the config file is absent or
invalid rather than constructing a partial config without the hidden inactive
mode settings.

Because the GTK FanController UI keeps an in-memory copy of settings, avoid
editing the same fan settings simultaneously in both UIs. A later GTK save can
overwrite an external update with its older in-memory values.

## Suggested SparkDeck work

1. Pass `active_settings` through `Manager._read_fan_state()`.
2. Add validated `POST /api/fan/settings` manager and server methods.
3. Add an editor that selects its controls from `fan.mode` and initializes
   them from `fan.active_settings`.
4. Submit the complete active-mode object, disable save while pending, and
   surface validation/write errors.
5. Refresh from `/api/stats` after saving and show the republished values as
   the authoritative applied state.
6. Add tests for curve/PID field exclusivity, stale mode rejection, validation,
   preservation of inactive/general config fields, and atomic file replacement.
