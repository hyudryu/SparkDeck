"""Token-bucket download bandwidth limiting for SparkDeck-managed downloads.

The controller's ``max_download_bandwidth_mbps`` setting (megabits per second,
0 disables limiting) is converted with :func:`parse_bandwidth_mbps` and
:func:`bandwidth_bytes_per_second`, then carried to the process that actually
reads from the network. Only the download-worker subprocess installs
:func:`install_socket_throttle`, so the limit bounds Hub traffic without
touching Virtual NAS transfers or the agent's other sockets.

The bucket is a shared virtual-time schedule: every reader reserves its byte
cost up front and sleeps until its own reservation comes due, so several
download threads together stay under one global rate without holding a lock
across the sleep.
"""

from __future__ import annotations

import math
import socket
import ssl
import threading
import time

# Upper sanity bound: 1 Tbit/s. A configured value above this is a typo, not a
# real uplink, and would otherwise pass through as an effective no-limit.
MAX_BANDWIDTH_MBPS = 1_000_000.0

_BITS_PER_MEGABIT = 1_000_000


def parse_bandwidth_mbps(value: object) -> float:
    """Coerce a settings/request value to megabits-per-second (0 = unlimited).

    Raises ValueError for anything that is not a finite, non-negative number
    within the sanity bound. Booleans are rejected: they are numbers to
    ``isinstance`` but never a bandwidth.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("max_download_bandwidth_mbps must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("max_download_bandwidth_mbps must be a finite number")
    if result < 0:
        raise ValueError("max_download_bandwidth_mbps must not be negative")
    if result > MAX_BANDWIDTH_MBPS:
        raise ValueError(
            "max_download_bandwidth_mbps must not exceed "
            f"{MAX_BANDWIDTH_MBPS:.0f} megabits per second"
        )
    return result


def bandwidth_bytes_per_second(megabits_per_second: object) -> float:
    """Convert megabits per second to bytes per second; 0 means unlimited."""
    return parse_bandwidth_mbps(megabits_per_second) * _BITS_PER_MEGABIT / 8.0


# Ceiling for any derived transfer deadline. Beyond this a stalled peer is
# indistinguishable from a slow capped pull, so requests stop extending.
CAPPED_TRANSFER_CEILING_SECONDS = 90 * 24 * 3600.0
# Allowance used when the byte count of a capped transfer is unknown: sized
# for the largest container images SparkDeck launches, so the deadline only
# ever expires after a transfer that could not plausibly finish.
CAPPED_UNKNOWN_BYTES_ALLOWANCE = 32 * 1024 ** 3


def capped_transfer_timeout_seconds(
    megabits_per_second: object,
    expected_bytes: object = 0,
    default: float = 1800.0,
    floor: float = 1800.0,
) -> float:
    """HTTP deadline for a request whose work is bounded by the cap.

    Unlimited requests keep ``default``. Capped requests get a deadline
    derived from the rate: four times the theoretical transfer time when
    ``expected_bytes`` is known, otherwise the unknown-bytes allowance. The
    result is clamped between ``floor`` and the ceiling, so a low cap gets
    proportionally more runway instead of being killed by a fixed timeout.
    """
    try:
        mbps = parse_bandwidth_mbps(megabits_per_second)
    except ValueError:
        return default
    if mbps <= 0:
        return default
    try:
        expected = int(expected_bytes or 0)
    except (TypeError, ValueError):
        expected = 0
    rate = bandwidth_bytes_per_second(mbps)
    seconds = (max(expected, 0) * 4 or CAPPED_UNKNOWN_BYTES_ALLOWANCE) / rate
    return min(CAPPED_TRANSFER_CEILING_SECONDS, max(floor, seconds))


class TokenBucket:
    """A thread-safe fixed-rate byte schedule; ``clock`` is injectable for tests."""

    def __init__(self, bytes_per_second: float, clock=time.monotonic):
        if bytes_per_second <= 0:
            raise ValueError("token bucket rate must be positive")
        self._rate = float(bytes_per_second)
        self._clock = clock
        self._lock = threading.Lock()
        self._due = 0.0

    def acquire(self, count: int) -> None:
        """Reserve ``count`` bytes, sleeping until the reservation comes due."""
        if count <= 0:
            return
        with self._lock:
            now = self._clock()
            due = max(now, self._due) + count / self._rate
            self._due = due
            wait = due - now
        if wait > 0:
            time.sleep(wait)


_bucket: TokenBucket | None = None
_installed = False


def _result_byte_count(result: object) -> int:
    if isinstance(result, tuple):
        result = result[0] if result else b""
    if isinstance(result, int):
        return result
    if isinstance(result, (bytes, bytearray)):
        return len(result)
    return 0


def _wrap_reads(cls: type, bucket: TokenBucket) -> None:
    """Charge every buffered read on ``cls`` to ``bucket``.

    ``ssl.SSLSocket`` overrides these methods, so patching only
    ``socket.socket`` would leave HTTPS transfers (what the Hub actually
    serves) unbilled; both classes must be patched explicitly. ``read`` and
    ``read_into`` on SSLSocket are deliberately left alone: its ``recv``
    implementations call them internally, and wrapping both would bill the
    same bytes twice.
    """
    for name in ("recv", "recv_into", "recvfrom", "recvfrom_into"):
        original = getattr(cls, name, None)
        if original is None or getattr(original, "_sparkdeck_throttled", False):
            # Missing, or already charged by a patch through inheritance.
            continue

        def charge(self, *args, _original=original, **kwargs):  # type: ignore[no-untyped-def]
            result = _original(self, *args, **kwargs)
            count = _result_byte_count(result)
            if count > 0:
                bucket.acquire(count)
            return result

        charge.__name__ = name
        charge._sparkdeck_throttled = True
        setattr(cls, name, charge)  # type: ignore[method-assign]


def install_socket_throttle(bytes_per_second: float) -> bool:
    """Throttle every socket read in this process to the given byte rate.

    Returns True when the patch was installed. Idempotent. The Hugging Face
    download worker calls this before importing huggingface_hub: Hub HTTP
    reads then pay for their bytes from the shared bucket, including HTTPS,
    because ``ssl.SSLSocket`` overrides the recv methods and would otherwise
    bypass the base-socket patch. Xet-backed storage is downloaded by a Rust
    client that bypasses Python sockets, so callers must disable Xet whenever
    this returns True.
    """
    global _bucket, _installed
    if _installed:
        return True
    if bytes_per_second <= 0:
        return False
    bucket = TokenBucket(bytes_per_second)
    _wrap_reads(socket.socket, bucket)
    ssl_socket = getattr(ssl, "SSLSocket", None)
    if ssl_socket is not None:
        _wrap_reads(ssl_socket, bucket)
    _bucket = bucket
    _installed = True
    return True
