import io
import json
import os
import unittest
from unittest.mock import Mock, patch

from sparkdeck import hf_download_worker


class WorkerMainTests(unittest.TestCase):
    """main() consumes the stdin payload; the socket throttle itself is
    exercised in test_download_throttle (in a subprocess, since it patches
    the process-wide socket module)."""

    KEYS = ("HF_HUB_DISABLE_XET", "HF_XET_HIGH_PERFORMANCE")

    def run_main(self, payload, *, throttle_installed):
        hub = Mock()
        saved = {key: os.environ.pop(key, None) for key in self.KEYS}
        try:
            with (
                patch.object(hf_download_worker, "_guard_parent_lifetime"),
                patch.object(hf_download_worker, "install_socket_throttle",
                             return_value=throttle_installed) as install,
                patch.dict("sys.modules", {"huggingface_hub": hub}),
                patch("sys.stdin", io.StringIO(json.dumps(payload))),
            ):
                hf_download_worker.main()
            # Capture before the finally block restores the caller's env.
            captured = {key: os.environ.get(key) for key in self.KEYS}
            return hub, install, captured
        finally:
            for key in self.KEYS:
                os.environ.pop(key, None)
                if saved[key] is not None:
                    os.environ[key] = saved[key]

    def payload(self, **overrides):
        base = {
            "model_id": "org/model", "revision": "a" * 40,
            "parent_pid": 1234, "cache_dir": "/tmp/hub", "token": "",
        }
        base.update(overrides)
        return base

    def test_full_snapshot_calls_snapshot_download(self):
        hub, install, env = self.run_main(self.payload(), throttle_installed=False)
        hub.snapshot_download.assert_called_once_with(
            repo_id="org/model", revision="a" * 40,
            cache_dir="/tmp/hub", token=None,
        )
        hub.hf_hub_download.assert_not_called()
        # An unlimited download never touches the throttle (short-circuit).
        install.assert_not_called()
        # The unlimited path keeps SparkDeck's high-throughput Xet default.
        self.assertEqual(env["HF_XET_HIGH_PERFORMANCE"], "1")
        self.assertIsNone(env["HF_HUB_DISABLE_XET"])

    def test_files_payload_downloads_each_file(self):
        hub, _, _ = self.run_main(
            self.payload(files=["a.gguf", "b.gguf"], token="hf_t"),
            throttle_installed=False,
        )
        self.assertEqual(hub.hf_hub_download.call_count, 2)
        self.assertEqual(hub.hf_hub_download.call_args_list[0].kwargs, {
            "repo_id": "org/model", "filename": "a.gguf",
            "revision": "a" * 40, "cache_dir": "/tmp/hub", "token": "hf_t",
        })
        hub.snapshot_download.assert_not_called()

    def test_capped_download_installs_throttle_and_disables_xet(self):
        # The Xet client streams through Rust sockets that the Python
        # throttle cannot govern; a capped download must fall back to Hub HTTP.
        hub, install, env = self.run_main(
            self.payload(files=["a.gguf"], bytes_per_second=125_000.0),
            throttle_installed=True,
        )
        install.assert_called_once_with(125_000.0, None)
        self.assertEqual(env["HF_HUB_DISABLE_XET"], "1")
        self.assertIsNone(env["HF_XET_HIGH_PERFORMANCE"])
        hub.hf_hub_download.assert_called_once()

    def test_rate_file_is_forwarded_for_live_reeating(self):
        # The queue registers a rate file alongside a capped full-snapshot
        # worker so a settings change re-rates the transfer in flight.
        hub, install, _ = self.run_main(
            self.payload(bytes_per_second=125_000.0, rate_file="/tmp/rate.txt"),
            throttle_installed=True,
        )
        install.assert_called_once_with(125_000.0, "/tmp/rate.txt")


if __name__ == "__main__":
    unittest.main()
