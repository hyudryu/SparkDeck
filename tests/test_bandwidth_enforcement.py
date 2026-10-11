"""Node/cluster enforcement of the max_download_bandwidth_mbps setting.

These cover the pieces that keep the configured cap honest as a ceiling:
the launch path must use the throttled helper (not a daemon pull), the node
lock must serialize capped transfers against one another, and controller
fan-outs must divide the cap instead of multiplying it.
"""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import docker

from manager import Manager
from sparkdeck.virtual_nas import TransferCanceled, VirtualNAS


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
        m.pull_image_result.assert_awaited_once_with("img:1", 25.0, None)

    async def test_missing_image_without_a_share_uses_the_node_setting(self):
        m = bare_manager(100.0)
        m.client.images.get.side_effect = docker.errors.ImageNotFound("gone")
        m.pull_image_result = AsyncMock(return_value={"ok": True})
        await m._ensure_runtime_image("img:1", runtime="vllm")
        m.pull_image_result.assert_awaited_once_with("img:1", None, None)

    async def test_other_docker_errors_propagate(self):
        m = bare_manager(100.0)
        m.client.images.get.side_effect = docker.errors.APIError("daemon down")
        m.pull_image_result = AsyncMock()
        with self.assertRaises(docker.errors.APIError):
            await m._ensure_runtime_image("img:1", runtime="vllm")

    async def test_member_launch_pull_forwards_the_cluster_anchor(self):
        # The launch payload stamps the controller's full cap alongside the
        # member share so a live settings change re-rates the pull by its
        # share of THAT value, not of the agent's stale node-local setting.
        m = bare_manager(100.0)
        m.client.images.get.side_effect = docker.errors.ImageNotFound("gone")
        m.pull_image_result = AsyncMock(return_value={"ok": True})
        await m._ensure_runtime_image(
            "img:1", cluster_member={
                "image_pull_bandwidth_mbps": 25.0,
                "cluster_pull_bandwidth_mbps": 100.0,
            },
            runtime="llama",
        )
        m.pull_image_result.assert_awaited_once_with("img:1", 25.0, 100.0)

    async def test_capped_pull_holds_the_cluster_dispatch_slot(self):
        # The launch-time image pull is a controller-dispatched capped
        # transfer: running it beside a capped NAS or member operation on
        # another node would move two full rates through one capped uplink.
        m = bare_manager(100.0)
        m.dispatched_download_lock = lock = asyncio.Lock()
        m.client.images.get.side_effect = docker.errors.ImageNotFound("gone")
        observed = []

        async def fake_pull(image, mbps, cluster_mbps=None):
            observed.append(lock.locked())
            return {"ok": True}

        m.pull_image_result = fake_pull
        await m._ensure_runtime_image("img:1", runtime="vllm")
        self.assertEqual(observed, [True])
        self.assertFalse(lock.locked())

    async def test_uncapped_pull_skips_the_slot(self):
        m = bare_manager(0.0)
        m.dispatched_download_lock = lock = asyncio.Lock()
        m.client.images.get.side_effect = docker.errors.ImageNotFound("gone")
        observed = []

        async def fake_pull(image, mbps, cluster_mbps=None):
            observed.append(lock.locked())

        m.pull_image_result = fake_pull
        await m._ensure_runtime_image("img:1", runtime="vllm")
        self.assertEqual(observed, [False])
        self.assertFalse(lock.locked())


class PullImageStreamPathTests(unittest.IsolatedAsyncioTestCase):
    async def collect(self, m, image, bandwidth_mbps=None):
        out = []
        async for chunk in m.pull_image_stream(image, bandwidth_mbps):
            out.append(chunk)
        return out

    async def test_capped_pull_uses_the_helper_under_the_lock(self):
        m = bare_manager(100.0)
        held = []

        async def fake_events(client, image, bps, cluster_bytes_per_second=None):
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

        async def exploding_events(client, image, bps, cluster_bytes_per_second=None):
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

    async def test_daemon_insecure_registry_stays_behind_the_helper(self):
        # An insecure-registries entry is transport trust, not uplink
        # locality: falling back to the unthrottled daemon would exempt every
        # pull from a private registry from the cap. The helper keeps the
        # pull and turns its own TLS verification off (asserted in
        # test_throttled_image_pull flow tests).
        m = bare_manager(100.0)
        m.client.api.info.return_value = {"RegistryConfig": {
            "InsecureRegistryCIDRs": ["10.0.0.0/8"],
        }}
        used = []

        async def fake_events(client, image, bps, cluster_bytes_per_second=None):
            used.append(image)
            yield {"status": "ok"}

        with patch("sparkdeck.throttled_image_pull.throttled_image_pull_events",
                   fake_events):
            await self.collect(m, "10.1.2.3:5000/team/img:1")
        self.assertEqual(used, ["10.1.2.3:5000/team/img:1"])
        m.client.api.pull.assert_not_called()

    async def test_dispatched_cluster_anchor_overrides_the_node_reading(self):
        # The node's own setting is stale (100 Mbps here) while the
        # dispatch carries the controller's real cap (50 Mbps): the helper
        # must register the share against the ANCHOR, so a live re-rate
        # lands at new_cap/2 instead of a fraction of the stale value.
        m = bare_manager(100.0)
        seen = {}

        async def fake_events(client, image, bps, cluster_bytes_per_second=None):
            seen["bps"] = bps
            seen["cluster"] = cluster_bytes_per_second
            yield {"status": "ok"}

        with patch("sparkdeck.throttled_image_pull.throttled_image_pull_events",
                   fake_events):
            out = []
            async for chunk in m.pull_image_stream(
                "ghcr.io/org/model:1", 12.5, 50.0,
            ):
                out.append(chunk)
        self.assertEqual(seen["bps"], 1_562_500.0)       # 12.5 Mbps share
        self.assertEqual(seen["cluster"], 6_250_000.0)   # dispatched 50 Mbps

    async def test_fanout_member_stream_forwards_the_cluster_reference(self):
        # A member dispatched cap/4 must re-rate to new_cap/4 on a settings
        # change: the stream hands the helper the allocated rate AND the full
        # cluster cap so the share factor is exact.
        m = bare_manager(100.0)
        seen = {}

        async def fake_events(client, image, bps, cluster_bytes_per_second=None):
            seen["bps"] = bps
            seen["cluster"] = cluster_bytes_per_second
            yield {"status": "ok"}

        with patch("sparkdeck.throttled_image_pull.throttled_image_pull_events",
                   fake_events):
            await self.collect(m, "ghcr.io/org/model:1", bandwidth_mbps=25.0)
        self.assertEqual(seen["bps"], 3_125_000.0)       # 25 Mbps share
        self.assertEqual(seen["cluster"], 12_500_000.0)  # 100 Mbps cap

    async def test_no_proxy_registries_stay_behind_the_capped_helper(self):
        # NO_PROXY is proxy policy, not an uplink exemption: exempting its
        # hosts would silently disable the cap for every pull through the
        # unthrottled daemon path. The limiter dials those hosts directly
        # (covered in test_throttled_image_pull.LimiterProxyTests).
        m = bare_manager(100.0)
        used = []

        async def fake_events(client, image, bps, cluster_bytes_per_second=None):
            used.append(image)
            yield {"status": "Downloading"}

        with patch.dict(os.environ, {"NO_PROXY": "*"}), patch(
            "sparkdeck.throttled_image_pull.throttled_image_pull_events",
            fake_events,
        ):
            await self.collect(m, "registry.corp/img:1")
        self.assertEqual(used, ["registry.corp/img:1"])
        m.client.api.pull.assert_not_called()

    async def test_daemon_info_failure_still_uses_the_capped_helper(self):
        m = bare_manager(100.0)
        m.client.api.info.side_effect = RuntimeError("daemon unreachable")
        used = []

        async def fake_events(client, image, bps, cluster_bytes_per_second=None):
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

        async def fake_events(client, image, bps, cluster_bytes_per_second=None):
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


def bare_vnas(internet_transfer_lock, dispatched_download_lock=None):
    directory = Path(tempfile.mkdtemp())
    return VirtualNAS(
        data_dir=directory,
        hub_path_provider=lambda: directory / "hub",
        node_registry=Mock(),
        enabled_provider=lambda: True,
        internet_transfer_lock=internet_transfer_lock,
        dispatched_download_lock=dispatched_download_lock,
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
            cluster_bandwidth_mbps=None,
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
            cluster_bandwidth_mbps=None,
        )

    async def test_member_cluster_stamp_becomes_the_rerate_anchor(self):
        # A launched member carries the controller's cap on its payload;
        # the weights worker registers its share against that anchor so a
        # settings change mid-launch re-rates member shares of the new cap.
        m = self.harness(0.0)
        with patch("huggingface_hub.HfApi", self.hf(self.SHA)):
            await m._prefetch_runtime_weights(
                "org/model", cluster_member={
                    "image_pull_bandwidth_mbps": 25.0,
                    "cluster_pull_bandwidth_mbps": 100.0,
                }, runtime="sglang",
            )
        m.virtual_nas._download_model_process.assert_awaited_once_with(
            "org/model", self.SHA, "tok", "main",
            bandwidth_mbps=25.0, cluster_bandwidth_mbps=100.0,
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
            "org/model", self.SHA, "tok", "feature-branch",
            bandwidth_mbps=100.0, cluster_bandwidth_mbps=None,
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
            "org/model", "b" * 40, "tok", "b" * 40,
            bandwidth_mbps=100.0, cluster_bandwidth_mbps=None,
        )

    async def test_secondary_repos_ride_the_same_capped_worker(self):
        # A vLLM --tokenizer copy or a speculative-config draft model is
        # resolved by the online container: while capped it must be prefetched
        # through the same worker (default branch), or the container saturates
        # the uplink past the limit. A local-path tokenizer is not a repo.
        m = self.harness(100.0)
        m.dispatched_download_lock = lock = asyncio.Lock()
        second_sha = "b" * 40

        def model_info(repo_id, revision="main"):
            return Mock(sha=self.SHA if repo_id == "org/model" else second_sha)

        api = Mock()
        api.model_info.side_effect = model_info
        observed = []

        async def fake_download(model_id, revision, token, requested,
                                bandwidth_mbps=None,
                                cluster_bandwidth_mbps=None):
            observed.append(m.dispatched_download_lock.locked())
            return {"ok": True}

        m.virtual_nas._download_model_process = AsyncMock(
            side_effect=fake_download,
        )
        with patch("huggingface_hub.HfApi", Mock(return_value=api)):
            await m._prefetch_runtime_weights(
                "org/model", runtime="vllm", name="launch-1", extra_args=[
                    "--tokenizer", "org/tok-copy",
                    "--tokenizer-path", "/models/local-tokenizer",
                    "--speculative-config",
                    json.dumps({"method": "draft_model",
                                "model": "org/draft"}),
                ],
            )
        self.assertEqual(
            [call.args for call in
             m.virtual_nas._download_model_process.await_args_list],
            [
                ("org/model", self.SHA, "tok", "main"),
                ("org/tok-copy", second_sha, "tok", "main"),
                ("org/draft", second_sha, "tok", "main"),
            ],
        )
        # The whole prefetch ran under the cluster dispatch slot (QrNR): a
        # standalone local launch must not race a capped remote transfer.
        self.assertEqual(observed, [True, True, True])
        self.assertFalse(lock.locked())

    async def test_secondary_repo_resolution_failure_aborts_the_launch(self):
        # Proceeding would let the container download the draft model
        # uncapped, so the same fail-loud rule as the primary model applies.
        m = self.harness(100.0)
        def model_info(repo_id, revision="main"):
            if repo_id == "org/model":
                return Mock(sha=self.SHA)
            raise ConnectionError("offline")
        api = Mock()
        api.model_info.side_effect = model_info
        with patch("huggingface_hub.HfApi", Mock(return_value=api)):
            with self.assertRaisesRegex(RuntimeError, "org/tok-copy"):
                await m._prefetch_runtime_weights(
                    "org/model", runtime="vllm",
                    extra_args=["--tokenizer", "org/tok-copy"],
                )

    async def test_hf_cache_or_endpoint_override_is_refused_while_capped(self):
        # The runtime environment reaches the container, so a silent
        # HF_HOME/HF_ENDPOINT override would make the online container
        # re-download past the cap from a cache the prefetch never wrote.
        for key in (
            "HF_HOME", "HF_HUB_CACHE", "HF_ENDPOINT",
            "HUGGINGFACE_HUB_CACHE", "hf_endpoint",
        ):
            m = self.harness(100.0)
            with patch("huggingface_hub.HfApi", self.hf(self.SHA)):
                with self.assertRaisesRegex(
                    RuntimeError, "download bandwidth limit cannot be enforced",
                ) as ctx:
                    await m._prefetch_runtime_weights(
                        "org/model", runtime="vllm",
                        environment={key: "https://hf.internal"},
                    )
            self.assertIn(key.upper(), str(ctx.exception))
            m.virtual_nas._download_model_process.assert_not_awaited()

    async def test_conflict_refusal_lets_uncapped_launch_proceed(self):
        # Only the capped path refuses: unlimited launches use the daemon
        # and the container's own endpoint/cache, which the override governs
        # legitimately.
        m = self.harness(0.0)
        with patch("huggingface_hub.HfApi", self.hf(self.SHA)):
            await m._prefetch_runtime_weights(
                "org/model", runtime="vllm",
                environment={"HF_HOME": "/data/hf"},
            )
        m.virtual_nas._download_model_process.assert_not_awaited()

    async def test_unrelated_environment_does_not_trigger_the_guard(self):
        m = self.harness(100.0)
        with patch("huggingface_hub.HfApi", self.hf(self.SHA)):
            await m._prefetch_runtime_weights(
                "org/model", runtime="vllm",
                environment={"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"},
            )
        m.virtual_nas._download_model_process.assert_awaited_once()

    async def test_env_backed_speculative_config_resolves_to_the_draft_repo(self):
        # The vLLM launcher passes the RESOLVED launch args (--speculative-
        # config ${ENV_VAR} expanded from the runtime environment); raw
        # references would skip the draft repository in the scan and the
        # container would download it past the cap. This mirrors the call
        # site: resolve, then prefetch with the resolved args.
        m = self.harness(100.0)
        environment = {"SPEC_CONFIG": json.dumps({"model": "org/draft"})}
        resolved = m._resolve_environment_backed_speculative_args(
            ["--speculative-config", "${SPEC_CONFIG}"], environment,
        )
        self.assertNotIn("${SPEC_CONFIG}", resolved)
        second_sha = "b" * 40
        api = Mock()
        api.model_info.side_effect = lambda repo_id, revision="main": Mock(
            sha=self.SHA if repo_id == "org/model" else second_sha,
        )
        with patch("huggingface_hub.HfApi", Mock(return_value=api)):
            await m._prefetch_runtime_weights(
                "org/model", runtime="vllm",
                extra_args=resolved, environment=environment,
            )
        self.assertEqual(
            [call.args for call in
             m.virtual_nas._download_model_process.await_args_list],
            [
                ("org/model", self.SHA, "tok", "main"),
                ("org/draft", second_sha, "tok", "main"),
            ],
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

    def test_completion_marker_allows_a_capped_relaunch(self):
        m = bare_manager(100.0)
        m.client.volumes.get.return_value = Mock()
        m.client.containers.run.return_value = b""  # test -f exits 0
        m._require_strata_weights_prepared("dep-1", {"MODEL": "IQ2_XS"}, "imgid")
        kwargs = m.client.containers.run.call_args.kwargs
        self.assertEqual(kwargs["entrypoint"], ["sh"])
        self.assertIn("test -f /data/.sparkdeck-prepared", kwargs["command"][1])
        self.assertEqual(
            list(kwargs["volumes"].values()), [{"bind": "/data", "mode": "ro"}],
        )

    def test_volume_without_the_marker_is_refused(self):
        # Volume existence alone is not proof: _strata_data_volume creates
        # the volume before the container ever runs, so an interrupted
        # launch leaves an empty one behind.
        m = bare_manager(100.0)
        m.client.volumes.get.return_value = Mock()
        m.client.containers.run.side_effect = RuntimeError("exit code 1")
        with self.assertRaisesRegex(RuntimeError, "finished preparing"):
            m._require_strata_weights_prepared("dep-1", {"MODEL": "IQ2_XS"}, "imgid")

    def test_marker_probe_errors_fail_closed(self):
        m = bare_manager(100.0)
        m.client.volumes.get.return_value = Mock()
        m.client.containers.run.side_effect = docker.errors.APIError("no sh")
        with self.assertRaisesRegex(RuntimeError, "finished preparing"):
            m._require_strata_weights_prepared("dep-1", {}, "imgid")

    def test_stamped_share_caps_an_agent_whose_local_setting_is_unlimited(self):
        m = bare_manager(0.0)
        m.client.volumes.get.side_effect = docker.errors.NotFound("gone")
        with self.assertRaisesRegex(RuntimeError, "cannot cover"):
            m._require_strata_weights_prepared(
                "dep-1", {"MODEL": "IQ2_XS"}, "imgid",
                {"image_pull_bandwidth_mbps": 25.0},
            )

    def test_stamped_zero_overrides_a_capped_local_setting(self):
        m = bare_manager(100.0)
        m._require_strata_weights_prepared(
            "dep-1", {}, "imgid", {"image_pull_bandwidth_mbps": 0.0},
        )
        m.client.volumes.get.assert_not_called()

    def test_uncapped_never_probes(self):
        m = bare_manager(0.0)
        m._require_strata_weights_prepared("dep-1", {}, "imgid")
        m.client.volumes.get.assert_not_called()


_REAL_SLEEP = asyncio.sleep


class StrataPreparedMarkerTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def _yield_sleep(_seconds):
        # Replaces the watcher's 30-second poll without recursing into the
        # patched module attribute: the still-real sleep is captured above.
        await _REAL_SLEEP(0)

    async def _drain(self, task):
        for _ in range(80):
            await _REAL_SLEEP(0)
        await asyncio.gather(task, return_exceptions=True)

    async def test_marker_written_once_the_engine_serves(self):
        m = bare_manager(0.0)
        m.inference_target_health = AsyncMock(side_effect=[False, True])
        m.client.api.exec_create = Mock(return_value={"Id": "exec-1"})
        m.client.api.exec_start = Mock()
        m.client.api.exec_inspect = Mock(return_value={"ExitCode": 0})
        with patch("manager.asyncio.sleep", new=self._yield_sleep):
            m._begin_strata_prepared_marker("strata-1", "org/model", "vol-1")
            await self._drain(next(iter(m._strata_marker_tasks)))
        m.client.api.exec_create.assert_called_once_with(
            "strata-1", ["sh", "-c", "touch /data/.sparkdeck-prepared"],
        )

    async def test_never_serving_writes_nothing(self):
        m = bare_manager(0.0)
        m.inference_target_health = AsyncMock(return_value=False)
        m.client.api.exec_create = Mock()
        with patch("manager.asyncio.sleep", new=self._yield_sleep):
            m._begin_strata_prepared_marker("strata-1", "org/model", "vol-1")
            task = next(iter(m._strata_marker_tasks))
            for _ in range(50):
                await _REAL_SLEEP(0)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        m.client.api.exec_create.assert_not_called()

    async def test_failed_exec_is_swallowed(self):
        # A broken marker write must not break the launch: it only means
        # the next capped launch asks for one uncapped run instead of
        # guessing that setup finished.
        m = bare_manager(0.0)
        m.inference_target_health = AsyncMock(return_value=True)
        m.client.api.exec_create = Mock(side_effect=RuntimeError("no sh"))
        with patch("manager.asyncio.sleep", new=self._yield_sleep):
            m._begin_strata_prepared_marker("strata-1", "org/model", "vol-1")
            await self._drain(next(iter(m._strata_marker_tasks)))
        self.assertTrue(m.client.api.exec_create.called)


class CreateMemberLaunchTimeoutTests(unittest.IsolatedAsyncioTestCase):
    def _manager(self):
        m = bare_manager(100.0)
        m.node_registry = Mock()
        m.node_registry.request = AsyncMock(return_value={"ok": True})
        m.virtual_nas = Mock()
        m.virtual_nas.estimate_download_size = AsyncMock(
            return_value=70 * 1024 ** 3,
        )
        m._resolved_hf_token = Mock(return_value="tok")
        return m

    async def test_capped_member_deadline_covers_the_model_bytes(self):
        from sparkdeck.download_throttle import (
            CAPPED_UNKNOWN_BYTES_ALLOWANCE,
            capped_transfer_timeout_seconds,
        )

        m = self._manager()
        await m._create_member("ws1", {
            "model": "org/model",
            "extra_args": ["--revision", "v2.0"],
            "cluster_member": {"image_pull_bandwidth_mbps": 25.0},
        })
        m.virtual_nas.estimate_download_size.assert_awaited_once_with(
            "org/model", "v2.0", explicit_token="tok",
        )
        # The launch also pulls a missing runtime image, whose size is not
        # known yet, so the image allowance rides on top of the model bytes.
        self.assertEqual(
            m.node_registry.request.await_args.kwargs["timeout"],
            capped_transfer_timeout_seconds(
                25.0,
                70 * 1024 ** 3 + CAPPED_UNKNOWN_BYTES_ALLOWANCE,
                default=1800,
            ),
        )

    async def test_uncapped_member_skips_model_sizing(self):
        m = self._manager()
        await m._create_member("ws1", {
            "model": "org/model", "extra_args": [],
            "cluster_member": {"image_pull_bandwidth_mbps": 0.0},
        })
        m.virtual_nas.estimate_download_size.assert_not_awaited()
        self.assertEqual(
            m.node_registry.request.await_args.kwargs["timeout"], 1800,
        )

    async def test_capped_member_deadline_budgets_secondary_repos(self):
        # The agent prefetches secondary repositories before answering, and
        # it resolves environment-backed --speculative-config itself: the
        # deadline must budget those bytes (draft resolved from ${SPEC},
        # tokenizer copy) or a healthy draft download past the allowance is
        # reported as a failed launch.
        from sparkdeck.download_throttle import (
            CAPPED_UNKNOWN_BYTES_ALLOWANCE,
            capped_transfer_timeout_seconds,
        )

        m = self._manager()
        sizes = {
            "org/model": 70 * 1024 ** 3,
            "org/tok-copy": 2 * 1024 ** 3,
            "org/draft": 8 * 1024 ** 3,
        }
        m.virtual_nas.estimate_download_size = AsyncMock(
            side_effect=lambda repo, revision, explicit_token=None: sizes[repo],
        )
        await m._create_member("ws1", {
            "model": "org/model",
            "extra_args": [
                "--tokenizer", "org/tok-copy",
                "--speculative-config", "${SPEC_CONFIG}",
            ],
            "environment": {"SPEC_CONFIG": json.dumps({"model": "org/draft"})},
            "cluster_member": {"image_pull_bandwidth_mbps": 25.0},
        })
        self.assertEqual(
            {c.args[0] for c in m.virtual_nas.estimate_download_size.await_args_list},
            {"org/model", "org/tok-copy", "org/draft"},
        )
        self.assertEqual(
            m.node_registry.request.await_args.kwargs["timeout"],
            capped_transfer_timeout_seconds(
                25.0,
                80 * 1024 ** 3 + CAPPED_UNKNOWN_BYTES_ALLOWANCE,
                default=1800,
            ),
        )

    async def test_unknown_size_falls_back_to_the_allowance(self):
        from sparkdeck.download_throttle import (
            CAPPED_UNKNOWN_BYTES_ALLOWANCE,
            capped_transfer_timeout_seconds,
        )

        m = self._manager()
        m.virtual_nas.estimate_download_size = AsyncMock(
            side_effect=RuntimeError("hub unreachable"),
        )
        await m._create_member("ws1", {
            "model": "org/model", "extra_args": [],
            "cluster_member": {"image_pull_bandwidth_mbps": 25.0},
        })
        self.assertEqual(
            m.node_registry.request.await_args.kwargs["timeout"],
            capped_transfer_timeout_seconds(
                25.0, CAPPED_UNKNOWN_BYTES_ALLOWANCE, default=1800,
            ),
        )


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


class NodeDownloadModelFilesTests(unittest.IsolatedAsyncioTestCase):
    """manager.node_download_model_files: the GGUF seed dispatch."""

    class _NasStub:
        def __init__(self, cap_mbps):
            self._cap = cap_mbps

        def _download_bandwidth_mbps(self):
            return self._cap

        def transfer_operation(self, model_id):
            import contextlib
            return contextlib.nullcontext()

        async def _await_uncancelable(self, awaitable):
            return await awaitable

    def _manager(self, cap_mbps, supports_bandwidth):
        m = bare_manager(cap_mbps)
        m.node_supports_selective_downloads = AsyncMock(return_value=True)
        m.node_supports_download_bandwidth = AsyncMock(
            return_value=supports_bandwidth,
        )
        m._resolved_hf_token = Mock(return_value="tok")
        m.node_registry = Mock()
        m.node_registry.request = AsyncMock(return_value={"ok": True})
        m.virtual_nas = self._NasStub(float(cap_mbps))
        return m

    async def test_legacy_agent_refused_when_a_cap_is_set(self):
        m = self._manager(50.0, supports_bandwidth=False)
        with self.assertRaisesRegex(
            RuntimeError, "cannot enforce the download bandwidth",
        ):
            await m.node_download_model_files(
                "ws1", "org/model", "r" * 40, ["q.gguf"],
            )
        m.node_registry.request.assert_not_awaited()

    async def test_uncapped_dispatch_to_legacy_agent_is_allowed(self):
        m = self._manager(0.0, supports_bandwidth=False)
        await m.node_download_model_files("ws1", "org/model", "r" * 40, ["q.gguf"])
        kwargs = m.node_registry.request.await_args.kwargs
        self.assertNotIn("max_download_bandwidth_mbps", kwargs["json_body"])
        # Unlimited keeps the historical fixed day timeout.
        self.assertEqual(kwargs["timeout"], 24 * 60 * 60)

    async def test_capped_dispatch_holds_the_cluster_dispatch_slot(self):
        m = self._manager(50.0, supports_bandwidth=True)
        m.dispatched_download_lock = lock = asyncio.Lock()

        async def request(*args, **kwargs):
            observed.append(lock.locked())
            return {"ok": True}

        observed = []
        m.node_registry.request = request
        await m.node_download_model_files("ws1", "org/model", "r" * 40, ["q.gguf"])
        self.assertEqual(observed, [True])
        self.assertFalse(lock.locked())

    async def test_uncapped_dispatch_skips_the_slot_entirely(self):
        # An unlimited dispatch does not move the full budget, so it must
        # not queue behind (or block) capped dispatches.
        m = self._manager(0.0, supports_bandwidth=True)
        m.dispatched_download_lock = lock = asyncio.Lock()

        async def request(*args, **kwargs):
            observed.append(lock.locked())
            return {"ok": True}

        observed = []
        m.node_registry.request = request
        await m.node_download_model_files("ws1", "org/model", "r" * 40, ["q.gguf"])
        self.assertEqual(observed, [False])

    async def test_capped_dispatch_stamps_cap_and_scales_request_timeout(self):
        from sparkdeck.download_throttle import capped_transfer_timeout_seconds

        m = self._manager(50.0, supports_bandwidth=True)
        await m.node_download_model_files("ws1", "org/model", "r" * 40, ["q.gguf"])
        kwargs = m.node_registry.request.await_args.kwargs
        self.assertEqual(kwargs["json_body"]["max_download_bandwidth_mbps"], 50.0)
        # A capped download can legitimately run far past the fixed day
        # timeout; the request deadline must follow the rate, not stay fixed.
        self.assertEqual(
            kwargs["timeout"],
            capped_transfer_timeout_seconds(
                50.0, default=24 * 60 * 60, floor=24 * 60 * 60,
            ),
        )
        self.assertGreater(kwargs["timeout"], 24 * 60 * 60 - 1)


class BandwidthSettingsLiveUpdateTests(unittest.IsolatedAsyncioTestCase):
    """A cap change must reach transfers already running, not only new ones."""

    def fixture(self, cap_mbps):
        m = bare_manager(cap_mbps)
        m.lock = asyncio.Lock()
        m._save_settings = Mock()
        m._prefetch_pull_helper_images = Mock()
        m.public_settings = Mock(return_value={})
        m.virtual_nas = Mock()
        m.virtual_nas.start = Mock()
        m.virtual_nas.stop = AsyncMock()
        m.virtual_nas.list_transfers = Mock(return_value={"items": []})
        m.node_registry = Mock()
        m.node_registry.nodes = [
            {"id": "local", "name": "This node", "online": True},
            {"id": "ws1", "name": "WS1", "online": True},
            {"id": "ws2", "name": "WS2", "online": False},
        ]
        m.node_registry.request = AsyncMock(return_value={"ok": True})
        return m

    async def test_cap_change_re_rates_locally_and_pushes_agents(self):
        m = self.fixture(0.0)
        m.apply_bandwidth_cap_live = Mock()
        await m.update_settings({"max_download_bandwidth_mbps": 50})
        m.apply_bandwidth_cap_live.assert_called_once_with(50.0)
        await m._bandwidth_notify_task
        # Local node skipped, offline node skipped, online agent notified.
        m.node_registry.request.assert_awaited_once()
        args = m.node_registry.request.await_args
        self.assertEqual(args.args[0], "ws1")
        self.assertEqual(args.args[1], "POST")
        self.assertEqual(args.args[2], "/api/agent/bandwidth")
        self.assertEqual(
            args.kwargs["json_body"], {"max_download_bandwidth_mbps": 50.0},
        )

    async def test_rapid_saves_serialize_agent_broadcasts_last_wins(self):
        # Two quick saves must not race as two independent broadcast tasks
        # where the OLDER value could land on the agents last: the newest
        # value is parked in a pending slot and a single delivery loop owns
        # the wire, so broadcasts arrive in save order.
        m = self.fixture(0.0)
        m.apply_bandwidth_cap_live = Mock()
        calls = []
        gate = asyncio.Event()

        async def slow_notify(value):
            calls.append(value)
            await gate.wait()

        m._notify_agents_bandwidth_cap = slow_notify
        await m.update_settings({"max_download_bandwidth_mbps": 50})
        first_task = m._bandwidth_notify_task
        await asyncio.sleep(0)  # let the loop deliver 50 and block
        await m.update_settings({"max_download_bandwidth_mbps": 60})
        # The second save must reuse the running loop, not race alongside.
        self.assertIs(m._bandwidth_notify_task, first_task)
        gate.set()
        await first_task
        self.assertEqual(calls, [50.0, 60.0])

    async def test_enabling_cap_notes_uncapped_in_flight_transfers(self):
        m = self.fixture(0.0)
        m.apply_bandwidth_cap_live = Mock()
        m.virtual_nas.list_transfers = Mock(return_value={"items": [
            {"status": "running"}, {"status": "completed"},
        ]})
        result = await m.update_settings({"max_download_bandwidth_mbps": 40})
        self.assertIn("bandwidth_running_note", result)
        self.assertIn("1 transfer", result["bandwidth_running_note"])

    async def test_changing_an_existing_cap_leaves_no_note(self):
        # Capped transfers adopt a changed cap live through their rate file,
        # so nothing about them needs deferring or disclosing.
        m = self.fixture(100.0)
        m.apply_bandwidth_cap_live = Mock()
        m.virtual_nas.list_transfers = Mock(return_value={"items": [
            {"status": "running"},
        ]})
        result = await m.update_settings({"max_download_bandwidth_mbps": 20})
        m.apply_bandwidth_cap_live.assert_called_once_with(20.0)
        self.assertNotIn("bandwidth_running_note", result)

    async def test_unchanged_cap_does_not_touch_transfers(self):
        m = self.fixture(25.0)
        m.apply_bandwidth_cap_live = Mock()
        await m.update_settings({"max_download_bandwidth_mbps": 25.0})
        m.apply_bandwidth_cap_live.assert_not_called()

    def test_apply_bandwidth_cap_live_writes_both_listeners(self):
        m = self.fixture(0.0)
        m.virtual_nas.apply_download_bandwidth_cap = Mock()
        with patch(
            "sparkdeck.throttled_image_pull.apply_bandwidth_cap",
        ) as pull_apply:
            m.apply_bandwidth_cap_live(50.0)
        m.virtual_nas.apply_download_bandwidth_cap.assert_called_once_with(
            6_250_000.0,
        )
        pull_apply.assert_called_once_with(6_250_000.0)


class DispatchedDownloadSlotTests(unittest.IsolatedAsyncioTestCase):
    """One cluster-wide budget for every full-cap dispatch.

    Each dispatch sends the FULL node-wide cap to its target agent (the
    cap is not divided across simultaneous dispatches), so the manager's
    fan-out paths and the Virtual NAS queue must hold one shared slot.
    """

    async def test_manager_takes_the_slot_virtual_nas_was_given(self):
        lock = asyncio.Lock()
        m = bare_manager(100.0)
        # Manager.__init__ wires manager.dispatched_download_lock into the
        # VirtualNAS instance; bare fixtures simulate that wiring.
        m.dispatched_download_lock = lock
        nas = bare_vnas(m.internet_transfer_lock, lock)
        self.assertIs(nas._dispatched_download_lock, lock)

        order = []

        async def manager_fanout():
            async with m._dispatched_download_slot(True):
                order.append("fanout start")
                await asyncio.sleep(0.05)
                order.append("fanout end")

        async def queued_download():
            async with nas.dispatched_download_slot(True):
                order.append("queued start")
                await asyncio.sleep(0.05)
                order.append("queued end")

        await asyncio.gather(manager_fanout(), queued_download())
        self.assertEqual(
            order,
            ["fanout start", "fanout end", "queued start", "queued end"],
        )
        self.assertFalse(lock.locked())

    async def test_unlimited_dispatches_keep_full_parallelism(self):
        lock = asyncio.Lock()
        m = bare_manager(0.0)
        m.dispatched_download_lock = lock
        nas = bare_vnas(m.internet_transfer_lock, lock)
        order = []

        async def manager_fanout():
            async with m._dispatched_download_slot(False):
                self.assertFalse(lock.locked())
                order.append("start")
                await asyncio.sleep(0.05)
                order.append("end")

        async def queued_download():
            async with nas.dispatched_download_slot(False):
                await asyncio.sleep(0.05)

        await asyncio.gather(manager_fanout(), queued_download(), manager_fanout())
        self.assertEqual(order[:2], ["start", "start"])

    async def test_slot_releases_after_an_error(self):
        m = bare_manager(100.0)
        m.dispatched_download_lock = lock = asyncio.Lock()
        with self.assertRaises(RuntimeError):
            async with m._dispatched_download_slot(True):
                raise RuntimeError("boom")
        self.assertFalse(lock.locked())

    async def test_missing_lock_fixture_still_runs_the_dispatch(self):
        # Bare test Managers built with Manager.__new__ lack the lock; the
        # slot must no-op instead of failing the call.
        m = bare_manager(100.0)
        async with m._dispatched_download_slot(True):
            pass

    async def test_slot_is_reentrant_inside_the_holder_task_tree(self):
        # A deployment fan-out holds the slot across its member launches;
        # every member prefetch acquires the same slot too. Members are
        # children of the holder, so they must pass through (the budget is
        # already reserved) instead of deadlocking on the outer hold.
        m = bare_manager(100.0)
        m.dispatched_download_lock = lock = asyncio.Lock()

        async def member(seed=True):
            async with m._dispatched_download_slot(seed):
                return True

        async with m._dispatched_download_slot(True):
            nested = await asyncio.wait_for(
                asyncio.gather(member(), member(seed=False)), 1.0,
            )
        self.assertEqual(nested, [True, True])
        self.assertFalse(lock.locked())

    async def test_holder_flag_does_not_leak_to_sibling_dispatches(self):
        # The reentrancy flag is per task tree: a sibling started after the
        # holder's reservation must still wait for the slot.
        m = bare_manager(100.0)
        m.dispatched_download_lock = lock = asyncio.Lock()
        order = []

        async def holder():
            async with m._dispatched_download_slot(True):
                order.append("holder")
                await asyncio.sleep(0.05)

        async def sibling():
            async with m._dispatched_download_slot(True):
                order.append("sibling")

        outer = asyncio.create_task(holder())
        await asyncio.sleep(0)  # sibling's context copies the default flag
        other = asyncio.create_task(sibling())
        await asyncio.gather(outer, other)
        self.assertEqual(order, ["holder", "sibling"])

    async def test_cancellable_slot_races_the_wait_against_cancellation(self):
        # A cancelable job waiting for the slot has dispatched nothing yet.
        # Pressing Stop during that wait must end the job immediately instead
        # of parking it in "canceling" (with its node held busy) for the
        # duration of the holder's transfer, and the abort must not disturb
        # the holder's reservation.
        lock = asyncio.Lock()
        await lock.acquire()
        nas = bare_vnas(asyncio.Lock(), lock)
        event = asyncio.Event()

        async def waiter():
            async with nas._cancellable_dispatch_slot(True, event):
                self.fail("must not enter while the slot is held")

        task = asyncio.ensure_future(waiter())
        await asyncio.sleep(0.05)
        self.assertFalse(task.done())
        event.set()
        with self.assertRaises(TransferCanceled):
            await asyncio.wait_for(task, 1.0)
        self.assertTrue(lock.locked())
        lock.release()
        self.assertFalse(lock.locked())

    async def test_owner_cancellation_after_handover_releases_the_lock(self):
        # The owner (VirtualNAS shutdown, job eviction) can be cancelled
        # while waiting for the slot. If the current holder's release had
        # just delivered the lock to this waiter's acquire task, exiting
        # without a release consumes the handover permanently: every later
        # capped dispatch then queues behind a ghost.
        lock = asyncio.Lock()
        await lock.acquire()
        nas = bare_vnas(asyncio.Lock(), lock)
        event = asyncio.Event()

        async def waiter():
            async with nas._cancellable_dispatch_slot(True, event):
                self.fail("must not enter the body after cancellation")

        task = asyncio.ensure_future(waiter())
        await asyncio.sleep(0.05)  # waiter parks in the lock queue
        lock.release()  # handover: the acquire task is given the lock
        await asyncio.sleep(0)  # let the acquire task take delivery
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(lock.locked())
        # A later dispatch must be able to reserve the slot.
        async with lock:
            pass

    async def test_owner_cancellation_before_delivery_orphans_nothing(self):
        # Cancelling while still queued must abandon the pending
        # acquisition cleanly: no release for a lock never taken, and the
        # holder's reservation untouched.
        lock = asyncio.Lock()
        await lock.acquire()
        nas = bare_vnas(asyncio.Lock(), lock)
        event = asyncio.Event()

        async def waiter():
            async with nas._cancellable_dispatch_slot(True, event):
                self.fail("must not enter the body after cancellation")

        task = asyncio.ensure_future(waiter())
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        # The holder's reservation is untouched: the waiter never obtained
        # the lock, so it must not (and did not) release it either — only
        # the explicit release below frees the slot.
        self.assertTrue(lock.locked())
        lock.release()
        async with lock:
            pass

    async def test_cancellable_slot_aborts_when_cancel_lands_with_the_slot(self):
        # The event may fire while the lock is being handed over; the slot is
        # then given straight back so the next dispatch is not blocked.
        lock = asyncio.Lock()
        nas = bare_vnas(asyncio.Lock(), lock)
        event = asyncio.Event()
        event.set()
        with self.assertRaises(TransferCanceled):
            async with nas._cancellable_dispatch_slot(True, event):
                self.fail("a canceled job must not enter its dispatch body")
        self.assertFalse(lock.locked())

    async def test_cancellable_slot_acquires_normally_when_not_canceled(self):
        lock = asyncio.Lock()
        nas = bare_vnas(asyncio.Lock(), lock)
        event = asyncio.Event()
        async with nas._cancellable_dispatch_slot(True, event):
            self.assertTrue(lock.locked())
        self.assertFalse(lock.locked())

    async def test_cancellable_slot_passes_through_for_uncapped_and_holders(self):
        # Unlimited dispatches never take the budget, and a task inside an
        # existing holder's task tree passes through (same reentrancy as
        # dispatched_download_slot).
        lock = asyncio.Lock()
        m = bare_manager(100.0)
        m.dispatched_download_lock = lock
        nas = bare_vnas(m.internet_transfer_lock, lock)
        event = asyncio.Event()
        async with nas._cancellable_dispatch_slot(False, event):
            self.assertFalse(lock.locked())
        async with m._dispatched_download_slot(True):
            async with nas._cancellable_dispatch_slot(True, event):
                pass


class DeploymentLaunchDispatchSlotTests(unittest.IsolatedAsyncioTestCase):
    """A capped member fan-out reserves the cluster budget (QCHh)."""

    def launcher(self, cap_mbps=100.0):
        from manager import IMAGE_PULL_BANDWIDTH_CAPABILITY
        self.capability = IMAGE_PULL_BANDWIDTH_CAPABILITY
        m = bare_manager(cap_mbps)
        m.dispatched_download_lock = asyncio.Lock()
        m.deployments = []
        m._save_deployments = Mock()
        m._resolved_hf_token = Mock(return_value=None)
        m._deployment_launch_settings = Mock(return_value={})
        m.public_target_node = Mock(side_effect=lambda node: node)
        return m

    async def launch(self, m, observed):
        async def fake_member(node_id, payload):
            observed[node_id] = m.dispatched_download_lock.locked()
            # The member launch's own capped seed must pass through the
            # fan-out reservation rather than deadlock on it.
            async def seed():
                async with m._dispatched_download_slot(True):
                    return True
            observed[node_id + ":seed"] = await asyncio.wait_for(seed(), 1.0)
            return {"status": "starting", "id": "c", "port": 1,
                    "model_source": "unknown"}

        m._create_member = AsyncMock(side_effect=fake_member)
        m._preflight_deployment_launch = AsyncMock(return_value={
            "body": {"model": "org/model", "node_ids": ["local", "ws1"],
                     "extra_args": []},
            "engine": "sglang", "mode": "single",
            "node_ids": ["local", "ws1"],
            "available": {
                "local": {"id": "local", "name": "Coordinator"},
                "ws1": {"id": "ws1", "name": "WS1",
                        "capabilities": [self.capability]},
            },
            "model": "org/model", "vllm_parallel_layout": None,
            "requested_port": None, "local_port": None,
            "master_ip": None,
            "fabrics": {"local": (None, None), "ws1": (None, None)},
        })
        return await m._create_deployment({"model": "org/model"})

    async def test_capped_member_fanout_holds_the_dispatch_slot(self):
        m = self.launcher(100.0)
        observed = {}
        await self.launch(m, observed)
        # Both member launches run under one reservation, and their nested
        # seed requests pass through the holder's task tree.
        self.assertEqual(
            [observed["local"], observed["ws1"]], [True, True],
        )
        self.assertEqual(
            [observed["local:seed"], observed["ws1:seed"]], [True, True],
        )
        self.assertFalse(m.dispatched_download_lock.locked())

    async def test_uncapped_member_fanout_never_holds_the_slot(self):
        m = self.launcher(0.0)
        observed = {}
        await self.launch(m, observed)
        self.assertEqual(
            [observed["local"], observed["ws1"]], [False, False],
        )
        self.assertEqual(
            [observed["local:seed"], observed["ws1:seed"]], [True, True],
        )


class RequireBandwidthCapableNodesTests(unittest.TestCase):
    """Launch-time gate for members whose agents cannot enforce the cap."""

    def test_uncapped_launch_ignores_missing_capability(self):
        m = bare_manager(0.0)
        m._require_bandwidth_capable_nodes(
            ["ws1"], {"ws1": {"name": "WS1", "capabilities": []}},
        )

    def test_capped_launch_refuses_agent_without_the_capability(self):
        m = bare_manager(100.0)
        with self.assertRaisesRegex(
            ValueError, "cannot enforce the download bandwidth limit",
        ):
            m._require_bandwidth_capable_nodes(["local", "ws1"], {
                "local": {"name": "Coordinator", "capabilities": []},
                "ws1": {"name": "WS1", "capabilities": []},
            })

    def test_capped_launch_passes_with_capable_nodes_and_ignores_local(self):
        from manager import IMAGE_PULL_BANDWIDTH_CAPABILITY

        m = bare_manager(100.0)
        # The local node enforces the cap in-process, so its advertised
        # capabilities are irrelevant here.
        m._require_bandwidth_capable_nodes(["local", "ws1"], {
            "local": {"name": "Coordinator", "capabilities": []},
            "ws1": {"name": "WS1",
                    "capabilities": [IMAGE_PULL_BANDWIDTH_CAPABILITY]},
        })


if __name__ == "__main__":
    unittest.main()
