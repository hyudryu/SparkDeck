"""Node/cluster enforcement of the max_download_bandwidth_mbps setting.

These cover the pieces that keep the configured cap honest as a ceiling:
the launch path must use the throttled helper (not a daemon pull), the node
lock must serialize capped transfers against one another, and controller
fan-outs must divide the cap instead of multiplying it.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import docker

from manager import Manager
from sparkdeck.virtual_nas import VirtualNAS


def bare_manager(cap_mbps=0.0):
    """A Manager instance with only the members these paths touch."""
    m = Manager.__new__(Manager)
    m.settings = {"max_download_bandwidth_mbps": cap_mbps}
    m.client = Mock()
    m.internet_transfer_lock = asyncio.Lock()
    return m


class ImagePullShareTests(unittest.TestCase):
    def test_fanout_divides_the_cap(self):
        m = bare_manager(100.0)
        self.assertEqual(m._cluster_image_pull_share_mbps(1), 100.0)
        self.assertEqual(m._cluster_image_pull_share_mbps(4), 25.0)

    def test_unlimited_or_invalid_cap_stays_zero(self):
        self.assertEqual(bare_manager(0.0)._cluster_image_pull_share_mbps(4), 0.0)
        self.assertEqual(bare_manager("fast")._cluster_image_pull_share_mbps(4), 0.0)

    def test_stamped_share_is_read_from_the_member_payload(self):
        m = bare_manager(100.0)
        self.assertEqual(
            m._image_pull_share_mbps({"image_pull_bandwidth_mbps": 25.0}), 25.0,
        )
        self.assertEqual(m._image_pull_share_mbps({"image_pull_bandwidth_mbps": 0}), 0.0)
        # Missing or garbage shares fall back to the node's own setting.
        self.assertIsNone(m._image_pull_share_mbps({}))
        self.assertIsNone(m._image_pull_share_mbps(None))
        self.assertIsNone(m._image_pull_share_mbps({"image_pull_bandwidth_mbps": "25"}))
        self.assertIsNone(m._image_pull_share_mbps({"image_pull_bandwidth_mbps": True}))


class EnsureRuntimeImageTests(unittest.IsolatedAsyncioTestCase):
    async def test_present_image_does_not_pull(self):
        m = bare_manager(100.0)
        m.client.images.get.return_value = Mock()
        m.pull_image_result = AsyncMock()
        await m._ensure_runtime_image("img:1", runtime="llama")
        m.pull_image_result.assert_not_awaited()

    async def test_missing_image_pulls_through_the_capped_path(self):
        m = bare_manager(100.0)
        m.client.images.get.side_effect = docker.errors.ImageNotFound("gone")
        m.pull_image_result = AsyncMock(return_value={"ok": True})
        await m._ensure_runtime_image(
            "img:1", cluster_member={"image_pull_bandwidth_mbps": 25.0},
            runtime="llama",
        )
        m.pull_image_result.assert_awaited_once_with("img:1", 25.0)

    async def test_missing_image_without_a_share_uses_the_node_setting(self):
        m = bare_manager(100.0)
        m.client.images.get.side_effect = docker.errors.ImageNotFound("gone")
        m.pull_image_result = AsyncMock(return_value={"ok": True})
        await m._ensure_runtime_image("img:1", runtime="vllm")
        m.pull_image_result.assert_awaited_once_with("img:1", None)

    async def test_other_docker_errors_propagate(self):
        m = bare_manager(100.0)
        m.client.images.get.side_effect = docker.errors.APIError("daemon down")
        m.pull_image_result = AsyncMock()
        with self.assertRaises(docker.errors.APIError):
            await m._ensure_runtime_image("img:1", runtime="vllm")


class PullImageStreamPathTests(unittest.IsolatedAsyncioTestCase):
    async def collect(self, m, image, bandwidth_mbps=None):
        out = []
        async for chunk in m.pull_image_stream(image, bandwidth_mbps):
            out.append(chunk)
        return out

    async def test_capped_pull_uses_the_helper_under_the_lock(self):
        m = bare_manager(100.0)
        held = []

        async def fake_events(client, image, bps):
            held.append(m.internet_transfer_lock.locked())
            self.assertIs(client, m.client)
            self.assertEqual(image, "ghcr.io/org/model:1")
            self.assertEqual(bps, 12_500_000.0)  # 100 Mbps
            yield {"status": "Downloading"}

        with patch("sparkdeck.throttled_image_pull.throttled_image_pull_events",
                   fake_events):
            chunks = await self.collect(m, "ghcr.io/org/model:1")
        self.assertEqual(held, [True])
        self.assertFalse(m.internet_transfer_lock.locked())
        self.assertIn("data: {\"status\": \"Downloading\"}\n\n", chunks)
        self.assertEqual(chunks[-1], "data: {\"done\": true}\n\n")
        m.client.api.pull.assert_not_called()

    async def test_helper_failure_still_releases_the_lock(self):
        m = bare_manager(100.0)

        async def exploding_events(client, image, bps):
            raise RuntimeError("registry unreachable")
            yield  # pragma: no cover — makes this an async generator

        with patch("sparkdeck.throttled_image_pull.throttled_image_pull_events",
                   exploding_events):
            chunks = await self.collect(m, "ghcr.io/org/model:1")
        self.assertFalse(m.internet_transfer_lock.locked())
        self.assertIn("registry unreachable", chunks[-2])

    async def test_loopback_registry_uses_the_daemon_despite_the_cap(self):
        m = bare_manager(100.0)
        m.client.api.pull.return_value = [{"status": "pulled locally"}]
        chunks = await self.collect(m, "localhost:5000/sparkdeck/vllm:latest")
        m.client.api.pull.assert_called_once_with(
            "localhost:5000/sparkdeck/vllm:latest", stream=True, decode=True,
        )
        self.assertEqual(chunks[-1], "data: {\"done\": true}\n\n")

    async def test_uncapped_pull_uses_the_daemon(self):
        m = bare_manager(0.0)
        m.client.api.pull.return_value = [{"status": "downloading"}]
        chunks = await self.collect(m, "ghcr.io/org/model:1")
        m.client.api.pull.assert_called_once()
        self.assertEqual(chunks[-1], "data: {\"done\": true}\n\n")

    async def test_daemon_insecure_registry_uses_the_daemon(self):
        m = bare_manager(100.0)
        m.client.api.info.return_value = {"RegistryConfig": {
            "InsecureRegistryCIDRs": ["10.0.0.0/8"],
        }}
        m.client.api.pull.return_value = [{"status": "ok"}]
        chunks = await self.collect(m, "10.1.2.3:5000/team/img:1")
        m.client.api.pull.assert_called_once()
        self.assertEqual(chunks[-1], "data: {\"done\": true}\n\n")

    async def test_daemon_info_failure_still_uses_the_capped_helper(self):
        m = bare_manager(100.0)
        m.client.api.info.side_effect = RuntimeError("daemon unreachable")
        used = []

        async def fake_events(client, image, bps):
            used.append(image)
            yield {"status": "Downloading"}

        with patch("sparkdeck.throttled_image_pull.throttled_image_pull_events",
                   fake_events):
            chunks = await self.collect(m, "ghcr.io/org/model:1")
        self.assertEqual(used, ["ghcr.io/org/model:1"])
        m.client.api.pull.assert_not_called()


class PullImageStreamLockTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_queued_pull_does_not_release_the_holders_lock(
        self,
    ):
        m = bare_manager(100.0)
        await m.internet_transfer_lock.acquire()  # an active transfer

        async def fake_events(client, image, bps):
            yield {"status": "x"}

        with patch("sparkdeck.throttled_image_pull.throttled_image_pull_events",
                   fake_events):
            stream = m.pull_image_stream("ghcr.io/org/model:1")
            task = asyncio.create_task(stream.__anext__())
            await asyncio.sleep(0.05)  # let the queued pull reach acquire
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await stream.aclose()
        # The holder still owns the lock; the cancelled waiter must not
        # have freed it, and the holder can still release it cleanly.
        self.assertTrue(m.internet_transfer_lock.locked())
        m.internet_transfer_lock.release()
        self.assertFalse(m.internet_transfer_lock.locked())


def bare_vnas(internet_transfer_lock):
    directory = Path(tempfile.mkdtemp())
    return VirtualNAS(
        data_dir=directory,
        hub_path_provider=lambda: directory / "hub",
        node_registry=Mock(),
        enabled_provider=lambda: True,
        internet_transfer_lock=internet_transfer_lock,
    )


class PrefetchRuntimeWeightsTests(unittest.IsolatedAsyncioTestCase):
    SHA = "a" * 40

    def harness(self, cap_mbps, cluster_member=None):
        m = bare_manager(cap_mbps)
        m._resolved_hf_token = Mock(return_value="tok")
        m._cluster_launch_update = Mock()
        nas = Mock()
        nas.inventory = Mock(return_value=[])
        nas._has_revision = Mock(return_value=False)
        nas._download_model_process = AsyncMock(return_value={"ok": True})
        m.virtual_nas = nas
        return m

    def hf(self, sha=None, error=None):
        api = Mock()
        if error is not None:
            api.model_info.side_effect = error
        else:
            api.model_info.return_value.sha = sha
        return Mock(return_value=api)

    async def test_uncapped_skips_the_worker(self):
        m = self.harness(0.0)
        with patch("huggingface_hub.HfApi", self.hf(self.SHA)):
            await m._prefetch_runtime_weights(
                "org/model", runtime="vllm", name="launch-1",
            )
        m.virtual_nas._download_model_process.assert_not_awaited()

    async def test_capped_uncached_model_downloads_through_the_worker(self):
        m = self.harness(100.0)
        with patch("huggingface_hub.HfApi", self.hf(self.SHA)):
            await m._prefetch_runtime_weights(
                "org/model", runtime="vllm", name="launch-1",
            )
        m.virtual_nas._download_model_process.assert_awaited_once_with(
            "org/model", self.SHA, "tok", "main", bandwidth_mbps=100.0,
        )
        phases = [call.args[1] for call in m._cluster_launch_update.call_args_list]
        self.assertIn("downloading_weights", phases)

    async def test_member_share_overrides_the_local_setting(self):
        m = self.harness(0.0)
        with patch("huggingface_hub.HfApi", self.hf(self.SHA)):
            await m._prefetch_runtime_weights(
                "org/model", cluster_member={"image_pull_bandwidth_mbps": 25.0},
                runtime="sglang",
            )
        m.virtual_nas._download_model_process.assert_awaited_once_with(
            "org/model", self.SHA, "tok", "main", bandwidth_mbps=25.0,
        )

    async def test_cached_model_skips_the_worker(self):
        m = self.harness(100.0)
        m.virtual_nas.inventory.return_value = [{"model_id": "org/model"}]
        m.virtual_nas._has_revision.return_value = True
        with patch("huggingface_hub.HfApi", self.hf(self.SHA)):
            await m._prefetch_runtime_weights("org/model", runtime="vllm")
        m.virtual_nas._download_model_process.assert_not_awaited()

    async def test_local_path_model_skips_the_worker(self):
        m = self.harness(100.0)
        await m._prefetch_runtime_weights("/models/local-weights", runtime="vllm")
        m.virtual_nas._download_model_process.assert_not_awaited()

    async def test_resolution_failure_aborts_instead_of_uncapped_launch(self):
        m = self.harness(100.0)
        with patch("huggingface_hub.HfApi", self.hf(error=ConnectionError("offline"))):
            with self.assertRaisesRegex(RuntimeError, "download limit is active"):
                await m._prefetch_runtime_weights("org/model", runtime="vllm")
        m.virtual_nas._download_model_process.assert_not_awaited()

    async def test_requested_revision_is_resolved_and_downloaded(self):
        m = self.harness(100.0)
        api = Mock()
        api.model_info.return_value.sha = self.SHA
        with patch("huggingface_hub.HfApi", Mock(return_value=api)):
            await m._prefetch_runtime_weights(
                "org/model", runtime="vllm", revision="feature-branch",
            )
        api.model_info.assert_called_once_with(
            "org/model", revision="feature-branch",
        )
        m.virtual_nas._download_model_process.assert_awaited_once_with(
            "org/model", self.SHA, "tok", "feature-branch", bandwidth_mbps=100.0,
        )

    async def test_pinned_commit_sha_skips_resolution(self):
        m = self.harness(100.0)
        factory = Mock()
        with patch("huggingface_hub.HfApi", factory):
            await m._prefetch_runtime_weights(
                "org/model", runtime="vllm", revision="b" * 40,
            )
        factory.assert_not_called()
        m.virtual_nas._download_model_process.assert_awaited_once_with(
            "org/model", "b" * 40, "tok", "b" * 40, bandwidth_mbps=100.0,
        )


class CacheExternalRuntimeWiringTests(unittest.IsolatedAsyncioTestCase):
    """Laya and TensorFold must prefetch weights before the container starts.

    Both resolve the model inside the container through the mounted shared
    HF cache, so the capped pre-launch prefetch has to run between the image
    gate and container creation, and must forward the launch's --revision.
    """

    def launcher_manager(self, cap_mbps=100.0):
        m = bare_manager(cap_mbps)
        m.settings["hf_cache"] = "/cache"
        m.settings["shm_size"] = "8g"
        m._ensure_runtime_image = AsyncMock()
        m._prefetch_runtime_weights = AsyncMock()
        m._allocate_port = AsyncMock(return_value=3123)
        m._cluster_launch_update = Mock()
        m._build_volumes = Mock(return_value={})
        m._container_hf_environment = Mock(return_value={})
        m._container_summary = Mock(return_value=None)
        m._created_container_model_source = Mock(return_value="")
        m._run_managed_container = Mock(return_value=Mock())
        m.evict_other_backends = AsyncMock()
        return m

    async def test_laya_prefetches_with_the_flag_revision(self):
        m = self.launcher_manager()
        with (
            patch("manager._node_has_nvidia_driver", return_value=False),
            patch("manager._laya_gpu_preference", return_value=False),
        ):
            await m._create_laya_container(
                "org/model", 3000, "laya:1", None,
                ["--revision", "v1.2"], "laya-1", None, "tok", None,
            )
        m._prefetch_runtime_weights.assert_awaited_once()
        kwargs = m._prefetch_runtime_weights.await_args.kwargs
        self.assertEqual(kwargs["runtime"], "laya")
        self.assertEqual(kwargs["revision"], "v1.2")
        self.assertEqual(kwargs["hf_token"], "tok")

    async def test_tensorfold_prefetches_without_a_revision_flag(self):
        m = self.launcher_manager()
        with patch("manager._node_has_nvidia_driver", return_value=True):
            await m._create_tensorfold_container(
                "org/model", 3000, "tf:1", None, [], "tf-1", None, "tok", None,
            )
        kwargs = m._prefetch_runtime_weights.await_args.kwargs
        self.assertEqual(kwargs["runtime"], "tensorfold")
        self.assertNotIn("revision", kwargs)

    async def test_prefetch_runs_between_the_image_gate_and_container_start(self):
        m = self.launcher_manager()
        order = []
        m._ensure_runtime_image = AsyncMock(
            side_effect=lambda *a, **k: order.append("image"),
        )
        m._prefetch_runtime_weights = AsyncMock(
            side_effect=lambda *a, **k: order.append("prefetch"),
        )
        m._run_managed_container = Mock(
            side_effect=lambda opts: order.append("container") or Mock(),
        )
        with (
            patch("manager._node_has_nvidia_driver", return_value=False),
            patch("manager._laya_gpu_preference", return_value=False),
        ):
            await m._create_laya_container(
                "org/model", 3000, "laya:1", None, [], "laya-1", None, None, None,
            )
        self.assertEqual(order, ["image", "prefetch", "container"])


class StrataCappedLaunchGuardTests(unittest.TestCase):
    def test_capped_first_launch_is_refused(self):
        m = bare_manager(100.0)
        m.client.volumes.get.side_effect = docker.errors.NotFound("gone")
        with self.assertRaisesRegex(RuntimeError, "cannot cover"):
            m._require_strata_weights_prepared("dep-1", {"MODEL": "IQ2_XS"}, "imgid")

    def test_prepared_volume_allows_a_capped_relaunch(self):
        m = bare_manager(100.0)
        m.client.volumes.get.return_value = Mock()
        m._require_strata_weights_prepared("dep-1", {"MODEL": "IQ2_XS"}, "imgid")
        m.client.volumes.get.assert_called_once()

    def test_uncapped_never_probes(self):
        m = bare_manager(0.0)
        m._require_strata_weights_prepared("dep-1", {}, "imgid")
        m.client.volumes.get.assert_not_called()


class HelperPrefetchOnSettingsTests(unittest.IsolatedAsyncioTestCase):
    def settings_manager(self, cap_mbps):
        m = bare_manager(cap_mbps)
        m.lock = asyncio.Lock()
        m._save_settings = Mock()
        m.is_joined_worker = Mock(return_value=True)
        m.virtual_nas = Mock(start=Mock(), stop=AsyncMock())
        m.public_settings = Mock(return_value={})
        m._prefetch_pull_helper_images = Mock()
        return m

    async def test_enabling_the_cap_preinstalls_helpers(self):
        m = self.settings_manager(0.0)
        await m.update_settings({"max_download_bandwidth_mbps": 50})
        self.assertEqual(m.settings["max_download_bandwidth_mbps"], 50.0)
        m._prefetch_pull_helper_images.assert_called_once_with()

    async def test_disabling_the_cap_does_not_prefetch(self):
        m = self.settings_manager(50.0)
        await m.update_settings({"max_download_bandwidth_mbps": 0})
        m._prefetch_pull_helper_images.assert_not_called()

    async def test_prefetch_task_invokes_ensure_helper_images(self):
        m = bare_manager(0.0)
        with patch(
            "sparkdeck.throttled_image_pull.ensure_helper_images", AsyncMock(),
        ) as ensure:
            m._prefetch_pull_helper_images()
            await m._helper_prefetch_task
        ensure.assert_awaited_once_with(m.client)


class CappedInternetTransferGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_capped_sections_serialize_behind_one_lock(self):
        order = []
        nas = bare_vnas(asyncio.Lock())

        async def section(tag):
            async with nas._capped_internet_transfer(1_000_000.0):
                order.append(f"start {tag}")
                await asyncio.sleep(0.05)
                order.append(f"end {tag}")

        await asyncio.gather(section("a"), section("b"))
        self.assertEqual(order, ["start a", "end a", "start b", "end b"])

    async def test_uncapped_sections_run_concurrently(self):
        order = []
        nas = bare_vnas(asyncio.Lock())

        async def section(tag):
            async with nas._capped_internet_transfer(0):
                order.append(f"start {tag}")
                await asyncio.sleep(0.05)
                order.append(f"end {tag}")

        await asyncio.gather(section("a"), section("b"))
        self.assertEqual(order, ["start a", "start b", "end a", "end b"])

    async def test_guard_releases_the_lock_after_an_error(self):
        lock = asyncio.Lock()
        nas = bare_vnas(lock)
        with self.assertRaises(RuntimeError):
            async with nas._capped_internet_transfer(1.0):
                raise RuntimeError("boom")
        self.assertFalse(lock.locked())

    async def test_no_lock_configured_allows_the_transfer(self):
        nas = bare_vnas(None)
        async with nas._capped_internet_transfer(1_000_000.0):
            pass

    async def test_manager_and_vnas_share_one_lock(self):
        # Manager construction wires manager.internet_transfer_lock into the
        # VirtualNAS instance; a concurrent image pull and weight download
        # must therefore queue against each other.
        m = bare_manager(100.0)
        nas = bare_vnas(m.internet_transfer_lock)

        async def image_pull():
            async with m.internet_transfer_lock:
                await asyncio.sleep(0.05)
                return "image"

        async def weight_download():
            async with nas._capped_internet_transfer(1_000_000.0):
                await asyncio.sleep(0.05)
                return "weights"

        first, second = await asyncio.gather(image_pull(), weight_download())
        self.assertEqual((first, second), ("image", "weights"))
        self.assertFalse(m.internet_transfer_lock.locked())


class CappedTimeoutTests(unittest.TestCase):
    def test_dispatch_timeout_floor_and_allowance(self):
        from sparkdeck.download_throttle import (
            CAPPED_TRANSFER_CEILING_SECONDS,
            capped_transfer_timeout_seconds,
        )
        day = 24 * 60 * 60
        # Uncapped keeps the legacy one-day request timeout.
        self.assertEqual(
            capped_transfer_timeout_seconds(0, 10 ** 9, default=float(day),
                                            floor=float(day)),
            float(day),
        )
        # Capped with size unknown: the 32 GiB allowance at 100 Mbps is
        # ~2749 s, so the one-day floor governs.
        self.assertEqual(
            capped_transfer_timeout_seconds(100, 0, default=float(day),
                                            floor=float(day)),
            float(day),
        )
        # The default floor is 30 minutes, so the allowance applies there.
        self.assertAlmostEqual(
            capped_transfer_timeout_seconds(100, 0),
            32 * 1024 ** 3 / 12_500_000.0,
        )
        # A capped 10 GiB download at 100 Mbps needs ~819 s x 4 attempts.
        self.assertEqual(
            capped_transfer_timeout_seconds(100, 10 * 1024 ** 3),
            max(1800.0, 4 * 10 * 1024 ** 3 / 12_500_000.0),
        )
        self.assertLessEqual(
            capped_transfer_timeout_seconds(0.0001, 0),
            CAPPED_TRANSFER_CEILING_SECONDS,
        )


if __name__ == "__main__":
    unittest.main()
