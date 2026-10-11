import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import sparkdeck.download_throttle as throttle
from sparkdeck.download_throttle import (
    MAX_BANDWIDTH_MBPS,
    TokenBucket,
    bandwidth_bytes_per_second,
    parse_bandwidth_mbps,
)


class ParseBandwidthTests(unittest.TestCase):
    def test_accepts_finite_non_negative_numbers(self):
        self.assertEqual(parse_bandwidth_mbps(0), 0.0)
        self.assertEqual(parse_bandwidth_mbps(12), 12.0)
        self.assertEqual(parse_bandwidth_mbps(0.5), 0.5)
        self.assertEqual(parse_bandwidth_mbps(MAX_BANDWIDTH_MBPS), MAX_BANDWIDTH_MBPS)

    def test_rejects_non_numbers_non_finite_negative_and_oversized(self):
        for value in (True, False, "10", None, float("nan"), float("inf"), -1,
                      MAX_BANDWIDTH_MBPS * 2):
            with self.assertRaises(ValueError, msg=repr(value)):
                parse_bandwidth_mbps(value)


class BandwidthConversionTests(unittest.TestCase):
    def test_megabits_per_second_to_bytes_per_second(self):
        self.assertEqual(bandwidth_bytes_per_second(8), 1_000_000.0)
        self.assertEqual(bandwidth_bytes_per_second(1.5), 187_500.0)
        self.assertEqual(bandwidth_bytes_per_second(0), 0.0)

    def test_conversion_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            bandwidth_bytes_per_second("fast")
        with self.assertRaises(ValueError):
            bandwidth_bytes_per_second(-2)


class TokenBucketTests(unittest.TestCase):
    def test_rejects_non_positive_rate(self):
        for rate in (0, 0.0, -5.0):
            with self.assertRaises(ValueError):
                TokenBucket(rate)

    def test_reserves_virtual_time_and_sleeps_to_schedule(self):
        now = [0.0]
        bucket = TokenBucket(100.0, clock=lambda: now[0])
        with patch.object(throttle.time, "sleep", side_effect=lambda s: None) as slept:
            bucket.acquire(100)  # due at 1.0 s
            bucket.acquire(50)   # due at 1.5 s
        self.assertEqual(slept.call_args_list[0].args[0], 1.0)
        self.assertEqual(slept.call_args_list[1].args[0], 1.5)
        # Once the clock has caught up past the schedule, acquires no longer wait.
        now[0] = 2.0
        with patch.object(throttle.time, "sleep") as slept_late:
            bucket.acquire(25)   # due at 2.25 s, wait .25 s
        self.assertAlmostEqual(slept_late.call_args.args[0], 0.25)

    def test_zero_and_negative_acquires_are_free(self):
        bucket = TokenBucket(100.0)
        with patch.object(throttle.time, "sleep") as slept:
            bucket.acquire(0)
            bucket.acquire(-10)
        slept.assert_not_called()

    def test_real_clock_acquires_wait_without_stalling(self):
        # 1e9 B/s makes the one-byte reservation wait nanoseconds; the sleep
        # call proves the default clock is the real monotonic one.
        bucket = TokenBucket(1_000_000_000.0)
        with patch.object(throttle.time, "sleep") as slept:
            bucket.acquire(100)
        slept.assert_called_once()


class InstallSocketThrottleTests(unittest.TestCase):
    """Installed in a throwaway process: the patch is global and permanent."""

    REPO_ROOT = Path(__file__).resolve().parents[1]

    def run_worker(self, script: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, cwd=str(self.REPO_ROOT), timeout=60,
        )

    def test_zero_rate_is_not_installed_and_positive_rate_throttles_reads(self):
        result = self.run_worker(
            "import socket, ssl\n"
            "from sparkdeck.download_throttle import install_socket_throttle\n"
            "assert install_socket_throttle(0) is False\n"
            "assert install_socket_throttle(10 ** 9) is True\n"
            "assert install_socket_throttle(10 ** 9) is True\n"
            "a, b = socket.socketpair()\n"
            "a.sendall(b'hello')\n"
            "assert b.recv(5) == b'hello'\n"
            "a.sendall(b'world')\n"
            "buf = bytearray(5)\n"
            "assert b.recv_into(buf) == 5 and bytes(buf) == b'world'\n"
            # HTTPS reads go through ssl.SSLSocket's own overrides of the
            # recv methods; a base-socket-only patch would miss them.
            "assert getattr(ssl.SSLSocket.recv, '_sparkdeck_throttled', False)\n"
            "assert getattr(ssl.SSLSocket.recv_into, '_sparkdeck_throttled', False)\n"
            "assert ssl.SSLSocket.recv is not socket.socket.recv\n"
            "print('ok')\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)


class WrapReadsTests(unittest.TestCase):
    """Wrapper charging logic against a synthetic class (no global patch)."""

    def test_charges_each_read_shape(self):
        class Synthetic:
            def recv(self, bufsize):
                return b"abcd"

            def recv_into(self, buffer, nbytes=0):
                return 3

            def recvfrom(self, bufsize):
                return b"xy", ("127.0.0.1", 53)

            def recvfrom_into(self, buffer, nbytes=0):
                return 2, ("127.0.0.1", 53)

        bucket = Mock()
        throttle._wrap_reads(Synthetic, bucket)
        instance = Synthetic()
        instance.recv(4096)
        instance.recv_into(bytearray(10))
        instance.recvfrom(4096)
        instance.recvfrom_into(bytearray(10))
        self.assertEqual(
            [call.args[0] for call in bucket.acquire.call_args_list], [4, 3, 2, 2],
        )

    def test_already_charged_methods_are_not_wrapped_twice(self):
        class Base:
            def recv(self, bufsize):
                return b"data"

        class Child(Base):
            pass

        base_bucket, child_bucket = Mock(), Mock()
        throttle._wrap_reads(Base, base_bucket)
        throttle._wrap_reads(Child, child_bucket)
        Child().recv(4096)
        self.assertEqual([c.args[0] for c in base_bucket.acquire.call_args_list], [4])
        child_bucket.acquire.assert_not_called()

    def test_missing_methods_are_skipped(self):
        class Bare:
            pass

        throttle._wrap_reads(Bare, Mock())
        self.assertFalse(hasattr(Bare, "recv"))


if __name__ == "__main__":
    unittest.main()
