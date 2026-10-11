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


def install_socket_throttle(bytes_per_second: float) -> bool:
    """Throttle every socket read in this process to the given byte rate.

    Returns True when the patch was installed. Idempotent. The Hugging Face
    download worker calls this before importing huggingface_hub: Hub HTTP
    reads then pay for their bytes from the shared bucket. Xet-backed storage
    is downloaded by a Rust client that bypasses Python sockets, so callers
    must disable Xet whenever this returns True.
    """
    global _bucket, _installed
    if _installed:
        return True
    if bytes_per_second <= 0:
        return False
    bucket = TokenBucket(bytes_per_second)
    original_recv = socket.socket.recv
    original_recv_into = socket.socket.recv_into
    original_recvfrom = socket.socket.recvfrom
    original_recvfrom_into = socket.socket.recvfrom_into

    def recv(self, bufsize, *args):  # type: ignore[no-untyped-def]
        data = original_recv(self, bufsize, *args)
        if data:
            bucket.acquire(len(data))
        return data

    def recv_into(self, buffer, nbytes=0, *args):  # type: ignore[no-untyped-def]
        read = original_recv_into(self, buffer, nbytes, *args)
        if read:
            bucket.acquire(read)
        return read

    def recvfrom(self, bufsize, *args):  # type: ignore[no-untyped-def]
        data, address = original_recvfrom(self, bufsize, *args)
        if data:
            bucket.acquire(len(data))
        return data, address

    def recvfrom_into(self, buffer, nbytes=0, *args):  # type: ignore[no-untyped-def]
        read, address = original_recvfrom_into(self, buffer, nbytes, *args)
        if read:
            bucket.acquire(read)
        return read, address

    socket.socket.recv = recv  # type: ignore[method-assign]
    socket.socket.recv_into = recv_into  # type: ignore[method-assign]
    socket.socket.recvfrom = recvfrom  # type: ignore[method-assign]
    socket.socket.recvfrom_into = recvfrom_into  # type: ignore[method-assign]
    _bucket = bucket
    _installed = True
    return True
