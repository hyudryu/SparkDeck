# RP2040-Zero firmware for OwlTree fan controller using an X9C103
# 10K digital potentiometer.
#
# Wiring:
#   X9C103 VCC -> RP2040 5V
#   X9C103 GND -> RP2040 GND
#   X9C103 INC -> RP2040 GP29
#   X9C103 U/D -> RP2040 GP28
#   X9C103 CS  -> RP2040 GP27
#
#   X9C103 VH -> one outside OwlTree B10K pad
#   X9C103 VW -> center OwlTree B10K pad
#   X9C103 VL -> other outside OwlTree B10K pad
#
# Serial protocol:
#   DUTY 0..255
#   PING
#
# DUTY 0   = minimum requested fan speed
# DUTY 255 = maximum requested fan speed
#
# Watchdog:
#   No valid command for WATCHDOG_S seconds forces DUTY 255.

import sys
import select
import time
from machine import Pin


# ---------------------------------------------------------------------
# Pin configuration
# ---------------------------------------------------------------------

INC_PIN = 29
UD_PIN = 28
CS_PIN = 27

WATCHDOG_S = 5
RPM_PERIOD_MS = 1000

# Start with True.
#
# After flashing:
#   DUTY 255 should produce maximum fan speed.
#   DUTY 0 should produce minimum fan speed.
#
# If operation is reversed, change this to False.
MAX_SPEED_AT_HIGH_TAP = True


# ---------------------------------------------------------------------
# X9C103 setup
# ---------------------------------------------------------------------

inc = Pin(INC_PIN, Pin.OUT, value=1)
ud = Pin(UD_PIN, Pin.OUT, value=0)
cs = Pin(CS_PIN, Pin.OUT, value=1)

current_tap = 0


def move_wiper(up, steps):
    """
    Move the X9C103 by a specified number of positions.

    The X9C103 moves on each HIGH-to-LOW transition of INC while
    CS is LOW. INC is kept LOW when CS rises so the position is
    not written to the X9C103's nonvolatile memory each time.
    """
    steps = int(steps)

    if steps <= 0:
        return

    if steps > 100:
        steps = 100

    # INC must begin HIGH.
    inc.value(1)

    # U/D must settle before the first INC falling edge.
    ud.value(1 if up else 0)
    time.sleep_us(5)

    # Select the digital potentiometer.
    cs.value(0)
    time.sleep_us(5)

    for step in range(steps):
        # Falling edge moves the wiper one position.
        inc.value(0)
        time.sleep_us(5)

        # Raise INC for the next falling edge, except after the
        # final step. Leaving it LOW prevents an EEPROM store
        # when CS rises.
        if step < steps - 1:
            inc.value(1)
            time.sleep_us(5)

    # Deselect while INC is LOW: no nonvolatile-memory write.
    cs.value(1)
    time.sleep_us(5)

    # Restore the normal idle state.
    inc.value(1)


def home_wiper_low():
    """
    Establish a known starting position.

    The X9C103 does not wrap around at its endpoints, so sending
    100 downward steps guarantees that it reaches tap zero.
    """
    global current_tap

    move_wiper(False, 100)
    current_tap = 0


def set_tap(target):
    """Set the X9C103 to a tap position from 0 through 99."""
    global current_tap

    target = int(target)

    if target < 0:
        target = 0
    elif target > 99:
        target = 99

    if target > current_tap:
        move_wiper(True, target - current_tap)
    elif target < current_tap:
        move_wiper(False, current_tap - target)

    current_tap = target


def set_duty_byte(duty):
    """
    Convert the existing 0..255 DUTY command into one of the
    X9C103's 100 wiper positions.
    """
    duty = int(duty)

    if duty < 0:
        duty = 0
    elif duty > 255:
        duty = 255

    logical_tap = round(duty * 99 / 255)

    if MAX_SPEED_AT_HIGH_TAP:
        target_tap = logical_tap
    else:
        target_tap = 99 - logical_tap

    set_tap(target_tap)


# ---------------------------------------------------------------------
# Safe startup
# ---------------------------------------------------------------------

# Find a known endpoint first.
home_wiper_low()

# Full cooling until the host begins sending commands.
set_duty_byte(255)

print("BOOT")


# ---------------------------------------------------------------------
# Serial command loop
# ---------------------------------------------------------------------

last_cmd_ms = time.ticks_ms()
last_rpm_ms = time.ticks_ms()
watchdog_active = False
buf = ""

poll = select.poll()
poll.register(sys.stdin, select.POLLIN)

while True:
    if poll.poll(0):
        ch = sys.stdin.read(1)

        if ch:
            if ch == "\n" or ch == "\r":
                line = buf.strip()
                buf = ""

                if line.startswith("DUTY"):
                    parts = line.split()

                    if len(parts) == 2:
                        try:
                            duty = int(parts[1])

                            if duty < 0 or duty > 255:
                                print("ERR duty-range")
                            else:
                                set_duty_byte(duty)
                                last_cmd_ms = time.ticks_ms()
                                watchdog_active = False
                                print("OK")

                        except ValueError:
                            print("ERR bad-duty")
                    else:
                        print("ERR bad-duty")

                elif line == "PING":
                    last_cmd_ms = time.ticks_ms()
                    watchdog_active = False
                    print("OK")

                elif line == "":
                    pass

                else:
                    print("ERR unknown")

            else:
                buf += ch

                if len(buf) > 64:
                    buf = ""
                    print("ERR line-too-long")

    now = time.ticks_ms()

    # Loss of communication forces maximum cooling.
    if time.ticks_diff(now, last_cmd_ms) > WATCHDOG_S * 1000:
        if not watchdog_active:
            set_duty_byte(255)
            watchdog_active = True

    # Tach remains disconnected.
    if time.ticks_diff(now, last_rpm_ms) >= RPM_PERIOD_MS:
        last_rpm_ms = now
        print("RPM", 0)

    time.sleep_ms(2)