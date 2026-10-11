import base64
import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import sparkdeck.throttled_image_pull as tip
from sparkdeck.throttled_image_pull import (
    PROGRESS_NAME,
    PROXY_IMAGE,
    RATE_NAME,
    SKOPEO_IMAGE,
    ThrottledPullError,
    _auth_entry,
    _read_counter,
    daemon_proxy_config,
    helper_credentials,
    insecure_registry_entries,
    is_loopback_registry_image,
    matches_insecure_registry,
    normalize_image_reference,
    registry_ca_binds,
    registry_mirrors_config,
    select_docker_config,
    split_authority,
    skopeo_copy_args,
    staging_parent,
    throttled_image_pull_events,
)

_USAGE = namedtuple("usage", "total used free")


def copy_kwargs(api):
    """The real copy container kwargs, not the preflight inspect container."""
    return next(
        c for c in api.created
        if c.get("image") == SKOPEO_IMAGE
        and list(c.get("command") or [])[:1] == ["copy"]
    )


class NormalizeImageReferenceTests(unittest.TestCase):
    def test_docker_hub_expansions_and_default_tag(self):
        self.assertEqual(normalize_image_reference("nginx"), (
            "docker://docker.io/library/nginx:latest",
            "docker.io/library/nginx:latest",
        ))
        self.assertEqual(normalize_image_reference("org/app"), (
            "docker://docker.io/org/app:latest",
            "docker.io/org/app:latest",
        ))
        self.assertEqual(normalize_image_reference("org/app:1.2"), (
            "docker://docker.io/org/app:1.2",
            "docker.io/org/app:1.2",
        ))

    def test_tag_colon_in_unqualified_reference_is_not_a_port(self):
        # A colon only marks a registry port when a slash follows the first
        # segment; "nginx:1.27" is a tag and keeps the Hub expansion.
        self.assertEqual(normalize_image_reference("nginx:1.27"), (
            "docker://docker.io/library/nginx:1.27",
            "docker.io/library/nginx:1.27",
        ))

    def test_registry_hosted_references_keep_their_host(self):
        self.assertEqual(normalize_image_reference("ghcr.io/sparkdeck/x"), (
            "docker://ghcr.io/sparkdeck/x:latest",
            "ghcr.io/sparkdeck/x:latest",
        ))
        source, repo_tag = normalize_image_reference("localhost:5000/x")
        self.assertEqual(source, "docker://localhost:5000/x:latest")
        self.assertEqual(repo_tag, "localhost:5000/x:latest")
        source, repo_tag = normalize_image_reference("registry.local:5000/a/b:v1")
        self.assertEqual(source, "docker://registry.local:5000/a/b:v1")
        self.assertEqual(repo_tag, "registry.local:5000/a/b:v1")

    def test_digest_references_are_rejected(self):
        # docker load registers RepoTags only, so a digest-pinned reference
        # could never resolve after an archive load.
        for pinned in ("nginx@sha256:" + "a" * 64, "nginx:1.27@sha256:" + "a" * 64):
            with self.assertRaisesRegex(ThrottledPullError, "digest", msg=pinned):
                normalize_image_reference(pinned)

    def test_invalid_references_are_rejected(self):
        for bad in ("", "  ", "two words", "docker://nginx", "nginx@"):
            with self.assertRaises(ThrottledPullError, msg=bad):
                normalize_image_reference(bad)


class SkopeoCopyArgsTests(unittest.TestCase):
    def test_base_copy_invocation_uses_supported_flags_only(self):
        args = skopeo_copy_args(
            "docker://docker.io/library/nginx:latest",
            "docker-archive:/archive/image.tar:docker.io/library/nginx:latest",
            "amd64", has_authfile=False,
        )
        self.assertEqual(args[0], "copy")
        self.assertIn("--src-tls-verify=true", args)
        self.assertEqual(args[args.index("--override-os") + 1], "linux")
        self.assertEqual(args[args.index("--override-arch") + 1], "amd64")
        # Neither flag exists in skopeo's copy; passing them fails parsing.
        self.assertNotIn("--format-event-data", args)
        self.assertNotIn("--event-file-path", args)
        self.assertNotIn("--authfile", args)
        self.assertEqual(args[-2:], [
            "docker://docker.io/library/nginx:latest",
            "docker-archive:/archive/image.tar:docker.io/library/nginx:latest",
        ])

    def test_authfile_and_missing_arch(self):
        args = skopeo_copy_args("docker://a", "docker-archive:/archive/image.tar",
                                "", has_authfile=True)
        self.assertNotIn("--override-arch", args)
        self.assertEqual(args[args.index("--authfile") + 1], "/auth/config.json")


class SelectDockerConfigTests(unittest.TestCase):
    def test_first_existing_candidate_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            second = Path(directory) / "config.json"
            second.write_text("{}", encoding="utf-8")
            self.assertEqual(
                select_docker_config([str(Path(directory) / "missing"), str(second)]),
                str(second),
            )

    def test_no_candidates_returns_none(self):
        self.assertIsNone(select_docker_config(["", "/nope/config.json"]))


class AuthEntryTests(unittest.TestCase):
    def test_hub_login_key_matches_the_expanded_registry(self):
        config = {"auths": {"https://index.docker.io/v1/": {"auth": "eA=="}}}
        self.assertEqual(_auth_entry(config, "docker.io"), {"auth": "eA=="})
        self.assertIsNone(_auth_entry(config, "ghcr.io"))

    def test_identitytoken_only_login_is_returned(self):
        # The daemon can pull with an identity token alone; the helper must
        # not downgrade such a login to anonymous just because ``auth`` is
        # absent.
        entry = {"auths": {"ghcr.io": {"identitytoken": "gho_token"}}}
        self.assertEqual(
            _auth_entry(entry, "ghcr.io"), {"identitytoken": "gho_token"},
        )
        self.assertIsNone(_auth_entry({"auths": {"ghcr.io": {}}}, "ghcr.io"))

    def test_registry_port_key_matches_and_empty_auth_ignored(self):
        # Auth keys are host[:port], sometimes with a trailing path like
        # index.docker.io/v1/; the port must not be lost when comparing.
        entry = {"auths": {"registry:5000/v2/": {"auth": "eQ=="}}}
        self.assertEqual(_auth_entry(entry, "registry:5000"), {"auth": "eQ=="})
        self.assertIsNone(_auth_entry({"auths": {"other:5000": {"auth": "eQ=="}}}, "registry:5000"))
        self.assertIsNone(_auth_entry({"auths": {"docker.io": {}}}, "docker.io"))


class HelperCredentialsTests(unittest.TestCase):
    def test_no_store_configured_is_anonymous(self):
        self.assertIsNone(helper_credentials({}, "docker.io"))

    def test_missing_helper_binary_is_anonymous(self):
        with patch.object(tip.shutil, "which", return_value=None):
            self.assertIsNone(
                helper_credentials({"credsStore": "desktop"}, "docker.io"),
            )

    def test_helper_result_becomes_inline_auth(self):
        done = subprocess.CompletedProcess([], 0, stdout=b'{"Username":"u","Secret":"s"}')
        with (
            patch.object(tip.shutil, "which", return_value="/usr/bin/docker-credential-desktop"),
            patch.object(tip.subprocess, "run", return_value=done) as run,
        ):
            entry = helper_credentials({"credsStore": "desktop"}, "docker.io")
        self.assertIsNotNone(entry)
        import base64
        self.assertEqual(
            base64.b64decode(entry["auth"]).decode(), "u:s",
        )
        self.assertEqual(
            run.call_args.args[0], ["/usr/bin/docker-credential-desktop", "get"],
        )

    def test_helper_failure_falls_back_to_anonymous(self):
        done = subprocess.CompletedProcess([], 1, stdout=b"")
        with (
            patch.object(tip.shutil, "which", return_value="/usr/bin/docker-credential-x"),
            patch.object(tip.subprocess, "run", return_value=done),
        ):
            self.assertIsNone(helper_credentials({"credsStore": "x"}, "ghcr.io"))

    def test_unsafe_store_name_never_becomes_a_command(self):
        with patch.object(tip.shutil, "which") as which:
            self.assertIsNone(helper_credentials({"credsStore": "../evil"}, "docker.io"))
            self.assertIsNone(helper_credentials({"credsStore": "a b"}, "docker.io"))
            which.assert_not_called()


class ReadCounterTests(unittest.TestCase):
    def test_parses_counter_and_tolerates_absent_or_racing_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proxy_bytes"
            self.assertEqual(_read_counter(path), 0)
            path.write_text("500")
            self.assertEqual(_read_counter(path), 500)
            path.write_text("")  # mid-rotation empty read
            self.assertEqual(_read_counter(path), 0)


class FakeDockerApi:
    """Minimal docker-py APIClient stand-in for the helper orchestration."""

    def __init__(self, *, copy_exit=0, proxy_dies=False, stderr_tail=b"fatal: copy failed",
                 load_events=None, verify_fails=False, inspect_size=100,
                 helpers_missing=False, security_options=None,
                 copy_inspect_fails=False, registry_config=None):
        self.security_options = security_options or []
        self.registry_config = registry_config
        self.copy_exit = copy_exit
        self.proxy_dies = proxy_dies
        self.copy_inspect_fails = copy_inspect_fails
        self.stderr_tail = stderr_tail
        self.load_events = load_events if load_events is not None else [
            {"stream": "Loaded image: docker.io/library/nginx:latest"},
        ]
        self.verify_fails = verify_fails
        self.inspect_size = inspect_size
        self.helpers_missing = helpers_missing
        self.loaded = None
        self.created: list[dict] = []
        self.removed: list[str] = []
        self.removed_networks: list[str] = []
        self.pulls: list[str] = []
        self.networks: list[dict] = []
        self.on_start = None
        self._containers: dict[str, str] = {}
        self._copy_polls = 0
        self._exec_counter = 0
        self._execs: dict[str, tuple[str, list[str]]] = {}

    # -- API surface used by the module (dispatched via asyncio.to_thread) --
    def version(self):
        return {"Arch": "amd64"}

    def info(self):
        payload = {"SecurityOptions": list(self.security_options)}
        if self.registry_config is not None:
            payload["RegistryConfig"] = self.registry_config
        return payload

    def inspect_image(self, ref):
        # Helper images always exist (no pull); the requested image resolves
        # only after a successful load unless verify_fails is set.
        if ref in (PROXY_IMAGE, SKOPEO_IMAGE):
            if self.helpers_missing:
                raise RuntimeError(f"helper image {ref} not found")
            return {"Id": f"sha256:{ref}"}
        if self.verify_fails or self.loaded is None:
            raise RuntimeError(f"image {ref} not found")
        return {"Id": "sha256:loaded"}

    def pull(self, ref):
        self.pulls.append(ref)

    def create_network(self, name, driver):
        self.networks.append({"name": name, "driver": driver})
        return {"Id": "net-1"}

    def remove_network(self, net_id):
        self.removed_networks.append(net_id)

    def create_container(self, **kwargs):
        self.created.append(kwargs)
        image = kwargs.get("image")
        command = [str(part) for part in (kwargs.get("command") or [])]
        if image == PROXY_IMAGE:
            role = "proxy"
        elif image == SKOPEO_IMAGE and command[:1] == ["inspect"]:
            role = "inspect"
        elif image == SKOPEO_IMAGE:
            role = "copy"
        else:
            role = "other"
        self._containers[f"{role}-1"] = role
        return {"Id": f"{role}-1"}

    def start_container(self, container_id):
        if self.on_start is not None:
            self.on_start(container_id)

    def exec_create(self, container, cmd):
        self._exec_counter += 1
        exec_id = f"exec-{self._exec_counter}"
        self._execs[exec_id] = (container, list(cmd))
        return {"Id": exec_id}

    def exec_start(self, exec_id, demux=False):
        container, cmd = self._execs[exec_id]
        if demux:
            if container == "copy-1" and "inspect" in cmd:
                return (json.dumps({"Size": self.inspect_size}).encode(), None)
            return (b"", None)
        return b""

    def exec_inspect(self, exec_id):
        return {"ExitCode": 0}

    def inspect_container(self, container_id):
        role = self._containers.get(container_id)
        if role == "proxy":
            running = not self.proxy_dies
            return {"State": {"Running": running, "ExitCode": 0 if running else 137}}
        if role == "inspect":
            # The preflight inspect always finishes on its first poll.
            return {"State": {"Running": False, "ExitCode": 0}}
        if self.copy_inspect_fails:
            raise RuntimeError(f"no such container: {container_id}")
        self._copy_polls += 1
        if self._copy_polls < 2:
            return {"State": {"Running": True, "ExitCode": 0}}
        return {"State": {"Running": False, "ExitCode": self.copy_exit}}

    def container_logs(self, container_id, stderr=True, stdout=False, tail=12):
        if (
            self._containers.get(container_id) == "inspect"
            and stdout and not stderr
        ):
            # Registry images report size per layer, not at the top level.
            return json.dumps({"LayersData": [{"Size": self.inspect_size}]}).encode()
        return self.stderr_tail

    def load_image(self, stream):
        self.loaded = stream.read()
        events = self.load_events
        if isinstance(events, list):
            events = list(events)
        return events

    def remove_container(self, container_id, v=False, force=False):
        self.removed.append(container_id)


class ThrottledPullFlowTests(unittest.IsolatedAsyncioTestCase):
    @contextlib.contextmanager
    def proxy_env(self, **values):
        # The flow reads proxy policy from the process environment. Windows
        # collapses env-key case, so setting HTTPS_PROXY and https_proxy as
        # separate entries would collide (the empty one wins); strip every
        # spelling variant first, then write the intended values.
        variants = ("HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy")
        saved = {key: os.environ.pop(key, None) for key in variants}
        os.environ.update({key: str(value) for key, value in values.items()})
        try:
            yield
        finally:
            for key in variants:
                os.environ.pop(key, None)
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value

    async def collect(self, api, image="nginx", bps=1_000_000.0,
                      staging=None, authfile=None, auth_dir=None,
                      disk_free=10 ** 15, cluster_bps=None, mirror_dir=None):
        client = Mock()
        client.api = api
        staging = staging or Path(tempfile.mkdtemp())
        # The flow creates staging, the optional single-registry auth dir,
        # the always-created colon-safe CA staging dir, and (Hub mirrors
        # configured) the registries.conf.d dir, in that order.
        ca_dir = tempfile.mkdtemp(prefix="sparkdeck-test-cas-")
        mkdtemp_values = (
            [str(staging)]
            + ([str(auth_dir)] if auth_dir else [])
            + [ca_dir]
            + ([str(mirror_dir)] if mirror_dir else [])
        )
        events = []
        with (
            # The flow tests own registry auth explicitly; the developer
            # machine's own ~/.docker/config.json must not leak in.
            patch.object(tip, "select_docker_config", return_value=authfile),
            patch.object(tip.tempfile, "mkdtemp", side_effect=mkdtemp_values),
            # Capacity depends on the CI host disk; make it deterministic.
            patch.object(
                tip.shutil, "disk_usage",
                return_value=_USAGE(10 ** 15, 0, disk_free),
            ),
        ):
            async for event in throttled_image_pull_events(
                client, image, bps, cluster_bytes_per_second=cluster_bps,
            ):
                events.append(event)
        return events, staging

    async def test_success_stream_maps_progress_and_loads_archive(self):
        api = FakeDockerApi()
        staging = Path(tempfile.mkdtemp())

        def on_start(container_id):
            if container_id != "copy-1":
                return
            # Stand in for the proxy writing its byte counter and the skopeo
            # container writing the archive into their shared /archive bind.
            staging.joinpath(PROGRESS_NAME).write_bytes(b"50")
            staging.joinpath("image.tar").write_bytes(b"tarbytes")

        api.on_start = on_start
        events, _ = await self.collect(api, staging=staging)

        self.assertEqual(events[0]["status"],
                         "Preparing bandwidth-limited pull of nginx")
        # The total comes from the preflight inspect (100), fixed for the
        # whole transfer; the counter is the proxy's capped byte tally.
        self.assertIn({"id": "nginx:latest", "status": "Downloading",
                       "progressDetail": {"current": 50, "total": 100}}, events)
        self.assertEqual(events[-1], {"status": "Successfully pulled nginx"})
        self.assertNotIn("error", json.dumps(events))
        self.assertEqual(api.loaded, b"tarbytes")
        self.assertEqual(sorted(api.removed), ["copy-1", "inspect-1", "proxy-1"])
        self.assertEqual(api.removed_networks, ["net-1"])

    async def test_progress_without_total_when_inspect_fails(self):
        api = FakeDockerApi(inspect_size=0)
        staging = Path(tempfile.mkdtemp())

        def on_start(container_id):
            if container_id == "copy-1":
                staging.joinpath(PROGRESS_NAME).write_bytes(b"40")
                staging.joinpath("image.tar").write_bytes(b"tar")

        api.on_start = on_start
        events, _ = await self.collect(api, staging=staging)
        self.assertIn({"id": "nginx:latest", "status": "Downloading",
                      "progressDetail": {"current": 40}}, events)

    async def test_load_stream_errors_are_not_reported_as_success(self):
        api = FakeDockerApi(load_events=[{"error": "no space left on device"}])
        staging = Path(tempfile.mkdtemp())

        def on_start(container_id):
            if container_id == "copy-1":
                staging.joinpath("image.tar").write_bytes(b"tar")

        api.on_start = on_start
        events, _ = await self.collect(api, staging=staging)
        self.assertIn("no space left on device", events[-1]["error"])
        self.assertNotIn("Successfully", json.dumps(events))

    async def test_unregistered_image_after_load_is_not_success(self):
        api = FakeDockerApi(verify_fails=True)
        staging = Path(tempfile.mkdtemp())

        def on_start(container_id):
            if container_id == "copy-1":
                staging.joinpath("image.tar").write_bytes(b"tar")

        api.on_start = on_start
        events, _ = await self.collect(api, staging=staging)
        self.assertIn("does not register", events[-1]["error"])
        self.assertNotIn("Successfully", json.dumps(events))

    async def test_copy_failure_emits_error_with_container_stderr(self):
        api = FakeDockerApi(copy_exit=1)
        events, _ = await self.collect(api)
        self.assertIsNone(api.loaded)
        self.assertIn("error", events[-1])
        self.assertIn("fatal: copy failed", events[-1]["error"])

    async def test_proxy_death_mid_pull_is_reported(self):
        api = FakeDockerApi(proxy_dies=True)
        events, _ = await self.collect(api)
        self.assertIsNone(api.loaded)
        self.assertIn("limiter exited", events[-1]["error"])

    async def test_invalid_image_reference_fails_before_docker_calls(self):
        api = FakeDockerApi()
        events, _ = await self.collect(api, image="not valid!")
        self.assertEqual(len(events), 1)
        self.assertIn("error", events[0])
        self.assertEqual(api.created, [])

    async def test_digest_reference_fails_before_docker_calls(self):
        api = FakeDockerApi()
        events, _ = await self.collect(api, image="nginx@sha256:" + "a" * 64)
        self.assertEqual(len(events), 1)
        self.assertIn("digest", events[0]["error"])
        self.assertEqual(api.created, [])

    async def test_credential_store_without_helper_proceeds_anonymously(self):
        api = FakeDockerApi()
        staging = Path(tempfile.mkdtemp())

        def on_start(container_id):
            if container_id == "copy-1":
                staging.joinpath("image.tar").write_bytes(b"tar")

        api.on_start = on_start
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"credsStore": "desktop"}), encoding="utf-8")
            with patch.object(tip.shutil, "which", return_value=None):
                events, _ = await self.collect(
                    api, image="nginx", bps=1000.0,
                    staging=staging, authfile=str(config),
                )
        self.assertEqual(events[-1], {"status": "Successfully pulled nginx"})
        copy = copy_kwargs(api)
        self.assertFalse(any("auth" in bind for bind in copy["host_config"]["Binds"]))
        self.assertNotIn("--authfile", copy["command"])

    async def test_resolved_helper_credentials_get_a_mounted_authfile(self):
        api = FakeDockerApi()
        staging = Path(tempfile.mkdtemp())
        auth_dir = Path(tempfile.mkdtemp())
        captured: dict = {}

        def on_start(container_id):
            if container_id != "copy-1":
                return
            written = auth_dir / "config.json"
            captured["auth"] = json.loads(written.read_text()) if written.exists() else None
            staging.joinpath("image.tar").write_bytes(b"tar")

        api.on_start = on_start
        done = subprocess.CompletedProcess([], 0, stdout=b'{"Username":"u","Secret":"s"}')
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"credsStore": "desktop"}), encoding="utf-8")
            with (
                patch.object(tip.shutil, "which",
                             return_value="/usr/bin/docker-credential-desktop"),
                patch.object(tip.subprocess, "run", return_value=done),
            ):
                events, _ = await self.collect(
                    api, image="nginx", bps=1000.0,
                    staging=staging, authfile=str(config), auth_dir=auth_dir,
                )
        self.assertEqual(events[-1], {"status": "Successfully pulled nginx"})
        import base64
        self.assertEqual(
            base64.b64decode(captured["auth"]["auths"]["docker.io"]["auth"]).decode(),
            "u:s",
        )
        copy = copy_kwargs(api)
        self.assertTrue(any(bind.endswith(":/auth/config.json:ro,z")
                            for bind in copy["host_config"]["Binds"]))
        self.assertIn("--authfile", copy["command"])

    async def test_helper_containers_are_wired_to_the_proxy(self):
        api = FakeDockerApi()
        await self.collect(api, bps=1_000_000.0)
        proxy = next(c for c in api.created if c.get("image") == PROXY_IMAGE)
        copy = copy_kwargs(api)
        self.assertEqual(proxy["environment"]["SPARKDECK_DOWNLOAD_BPS"], "1000000.0")
        self.assertEqual(proxy["environment"]["SPARKDECK_BANDWIDTH_PROGRESS"],
                         f"/archive/{PROGRESS_NAME}")
        self.assertEqual(proxy["host_config"]["NetworkMode"], "net-1")
        self.assertEqual(
            copy["environment"]["HTTPS_PROXY"],
            f"http://{proxy['name']}:8899",
        )
        self.assertEqual(copy["host_config"]["NetworkMode"], "net-1")
        # The counter file lives on the shared staging volume the host reads.
        self.assertEqual(proxy["host_config"]["Binds"],
                         copy["host_config"]["Binds"][:1])
        self.assertTrue(any("image.tar:docker.io/library/nginx:latest" in " ".join(c["command"])
                            for c in api.created if c.get("image") == SKOPEO_IMAGE))
        # The preflight inspect runs on the proxy network too, so its
        # manifest request never escapes the capped path.
        inspect_kw = next(
            c for c in api.created
            if c.get("image") == SKOPEO_IMAGE
            and list(c.get("command") or [])[:1] == ["inspect"]
        )
        self.assertEqual(inspect_kw["host_config"]["NetworkMode"], "net-1")
        self.assertEqual(inspect_kw["host_config"]["Binds"], [])

    async def test_staging_capacity_refused_before_bytes_move(self):
        api = FakeDockerApi(inspect_size=2 * 1024 ** 3)
        staging = Path(tempfile.mkdtemp())

        def on_start(container_id):
            if container_id == "copy-1":
                staging.joinpath("image.tar").write_bytes(b"tar")

        api.on_start = on_start
        events, _ = await self.collect(
            api, staging=staging, disk_free=600 * 1024 ** 2,
        )
        self.assertIsNone(api.loaded)
        self.assertIsNone(next(
            (c for c in api.created
             if list(c.get("command") or [])[:1] == ["copy"]), None,
        ))
        self.assertIn("SPARKDECK_IMAGE_STAGING_DIR", events[-1]["error"])
        self.assertIn("download limit to 0", events[-1]["error"])

    async def test_rate_file_carries_the_initial_cap_for_live_updates(self):
        api = FakeDockerApi()
        staging = Path(tempfile.mkdtemp())
        captured = {}

        def on_start(container_id):
            if container_id != "copy-1":
                return
            captured["rate"] = (staging / RATE_NAME).read_text(encoding="utf-8")
            staging.joinpath(PROGRESS_NAME).write_bytes(b"50")
            staging.joinpath("image.tar").write_bytes(b"tar")

        api.on_start = on_start
        events, _ = await self.collect(api, staging=staging, bps=1_000_000.0)
        self.assertEqual(captured["rate"], repr(1_000_000.0))
        proxy = next(
            c for c in api.created if c.get("image") == PROXY_IMAGE
        )
        self.assertEqual(
            proxy["environment"]["SPARKDECK_BANDWIDTH_RATE_FILE"],
            f"/archive/{RATE_NAME}",
        )
        # After the flow the file is gone with staging and no longer
        # registered for live re-rating (the registry keys are plain strings).
        rate_key = os.path.join(str(staging), RATE_NAME)
        self.assertNotIn(rate_key, tip._ACTIVE_RATE_FILES)
        self.assertFalse(os.path.exists(rate_key))

    async def test_fanout_member_registers_its_share_for_live_rerating(self):
        # A member of a 4-node fan-out runs at cap/4; the live re-rate must
        # scale that share, so the registration carries the factor, not a
        # flag that would re-apply the full cluster cap on every member.
        api = FakeDockerApi()
        staging = Path(tempfile.mkdtemp())
        captured = {}

        def on_start(container_id):
            if container_id != "copy-1":
                return
            rate_key = os.path.join(str(staging), RATE_NAME)
            captured["factor"] = tip._ACTIVE_RATE_FILES.get(rate_key)
            captured["initial"] = (staging / RATE_NAME).read_text(encoding="utf-8")
            staging.joinpath(PROGRESS_NAME).write_bytes(b"50")
            staging.joinpath("image.tar").write_bytes(b"tar")

        api.on_start = on_start
        # 25 Mbps out of a 100 Mbps cluster cap.
        await self.collect(
            api, staging=staging, bps=3_125_000.0, cluster_bps=12_500_000.0,
        )
        self.assertEqual(captured["initial"], repr(3_125_000.0))
        self.assertAlmostEqual(captured["factor"], 0.25)
        # Cleanup unregisters the pull like before.
        self.assertNotIn(os.path.join(str(staging), RATE_NAME), tip._ACTIVE_RATE_FILES)

    async def test_insecure_registry_stays_capped_with_verification_off(self):
        # insecure-registries is transport trust, not locality: the capped
        # flow keeps the pull and mirrors the daemon's TLS behavior instead
        # of falling back to the unthrottled daemon pull.
        api = FakeDockerApi(registry_config={
            "InsecureRegistryCIDRs": ["10.0.0.0/8", "reg.corp"],
        })
        staging = Path(tempfile.mkdtemp())

        def on_start(container_id):
            if container_id == "copy-1":
                staging.joinpath("image.tar").write_bytes(b"tar")

        api.on_start = on_start
        await self.collect(api, image="10.1.2.3:5000/team/img:1", staging=staging)
        self.assertIsNotNone(api.loaded)
        copy = next(
            c for c in api.created if list(c.get("command") or [])[:1] == ["copy"]
        )
        self.assertIn("--src-tls-verify=false", copy["command"])
        inspect = next(
            c for c in api.created
            if list(c.get("command") or [])[:1] == ["inspect"]
        )
        self.assertIn("--tls-verify=false", inspect["command"])

    async def test_indexconfig_insecure_named_host_turns_verification_off(self):
        # Named insecure hosts live in IndexConfigs (Secure:false), not in
        # InsecureRegistryCIDRs: a helper consulting only the CIDRs keeps
        # TLS verification on and fails a pull the daemon itself completes.
        api = FakeDockerApi(registry_config={
            "InsecureRegistryCIDRs": [],
            "IndexConfigs": {
                "reg.corp:5000": {"Name": "reg.corp:5000", "Secure": False},
                "other.corp": {"Name": "other.corp", "Secure": True},
            },
        })
        staging = Path(tempfile.mkdtemp())

        def on_start(container_id):
            if container_id == "copy-1":
                staging.joinpath("image.tar").write_bytes(b"tar")

        api.on_start = on_start
        await self.collect(api, image="reg.corp:5000/team/img:1", staging=staging)
        self.assertIsNotNone(api.loaded)
        copy = next(
            c for c in api.created if list(c.get("command") or [])[:1] == ["copy"]
        )
        self.assertIn("--src-tls-verify=false", copy["command"])

    async def test_secure_pull_still_verifies_tls(self):
        api = FakeDockerApi(registry_config={
            "InsecureRegistryCIDRs": ["10.0.0.0/8"],
        })
        staging = Path(tempfile.mkdtemp())

        def on_start(container_id):
            if container_id == "copy-1":
                staging.joinpath("image.tar").write_bytes(b"tar")

        api.on_start = on_start
        await self.collect(api, image="ghcr.io/org/model:1", staging=staging)
        copy = next(
            c for c in api.created if list(c.get("command") or [])[:1] == ["copy"]
        )
        self.assertIn("--src-tls-verify=true", copy["command"])

    async def test_daemon_registry_mirrors_are_replayed_into_the_helpers(self):
        # Isolated skopeo ignores dockerd's registry-mirrors: without the
        # drop-in a mirror-only LAN fails the capped pull entirely, and
        # elsewhere the internal pull-through cache is silently skipped.
        api = FakeDockerApi(registry_config={
            "Mirrors": ["https://mirror.corp:5443/"],
        })
        staging = Path(tempfile.mkdtemp())
        mirror_dir = Path(tempfile.mkdtemp())
        captured = {}

        def on_start(container_id):
            if container_id == "copy-1":
                # The drop-in is removed with staging after the flow.
                captured["conf"] = (
                    mirror_dir / "sparkdeck.conf"
                ).read_text(encoding="utf-8")
                staging.joinpath("image.tar").write_bytes(b"tar")

        api.on_start = on_start
        await self.collect(
            api, image="nginx", staging=staging, mirror_dir=mirror_dir,
        )
        conf = captured["conf"]
        self.assertIn('prefix = "docker.io"', conf)
        self.assertIn('location = "mirror.corp:5443"', conf)
        copy = next(
            c for c in api.created if list(c.get("command") or [])[:1] == ["copy"]
        )
        self.assertIn(
            f"{mirror_dir}:/etc/containers/registries.conf.d:ro,z",
            copy["host_config"]["Binds"],
        )
        # The preflight inspect must see the mirrors too, or it inspects the
        # Hub directly while the copy goes through the mirror.
        inspect = next(
            c for c in api.created
            if list(c.get("command") or [])[:1] == ["inspect"]
        )
        self.assertTrue(
            any(
                str(bind).endswith(":/etc/containers/registries.conf.d:ro,z")
                for bind in inspect["host_config"]["Binds"]
            ),
            inspect["host_config"]["Binds"],
        )

    async def test_registry_mirrors_are_not_invented_for_foreign_registries(self):
        # registry-mirrors applies to Docker Hub only; a ghcr pull must not
        # even allocate the drop-in directory.
        api = FakeDockerApi(registry_config={
            "Mirrors": ["https://mirror.corp/"],
        })
        staging = Path(tempfile.mkdtemp())

        def on_start(container_id):
            if container_id == "copy-1":
                staging.joinpath("image.tar").write_bytes(b"tar")

        api.on_start = on_start
        await self.collect(api, image="ghcr.io/org/model:1", staging=staging)
        self.assertIsNotNone(api.loaded)

    async def test_daemon_proxy_settings_reach_the_limiter_as_fallback(self):
        # Proxy-only hosts configure the proxy in daemon.json only: the
        # process env is empty and the daemon's own proxy/no-proxy must be
        # handed to the limiter, or every capped pull dials a network it
        # cannot reach directly. The process env keeps the primary slot.
        api = FakeDockerApi(registry_config={
            "HTTPSProxy": "http://proxy.corp:8080",
            "NoProxy": ["localhost", ".intra", ""],
        })
        staging = Path(tempfile.mkdtemp())

        def on_start(container_id):
            if container_id == "copy-1":
                staging.joinpath("image.tar").write_bytes(b"tar")

        api.on_start = on_start
        with self.proxy_env():
            await self.collect(api, staging=staging)
        proxy = next(c for c in api.created if c.get("image") == PROXY_IMAGE)
        env = proxy["environment"]
        self.assertEqual(env["SPARKDECK_DAEMON_PROXY"], "http://proxy.corp:8080")
        self.assertEqual(env["SPARKDECK_DAEMON_NO_PROXY"], "localhost,.intra")
        self.assertEqual(env["SPARKDECK_UPSTREAM_PROXY"], "")
        self.assertEqual(env["NO_PROXY"], "")

    async def test_process_proxy_env_wins_over_the_daemon_fallback(self):
        # The daemon value still rides along as SPARKDECK_DAEMON_PROXY; the
        # script prefers the non-empty process value (covered in
        # LimiterProxyTests). The flow only has to keep the two slots
        # distinct.
        api = FakeDockerApi(registry_config={
            "HTTPSProxy": "http://daemon.proxy:8080",
        })
        staging = Path(tempfile.mkdtemp())

        def on_start(container_id):
            if container_id == "copy-1":
                staging.joinpath("image.tar").write_bytes(b"tar")

        api.on_start = on_start
        with self.proxy_env(HTTPS_PROXY="http://host.proxy:3128"):
            await self.collect(api, staging=staging)
        proxy = next(c for c in api.created if c.get("image") == PROXY_IMAGE)
        env = proxy["environment"]
        self.assertEqual(env["SPARKDECK_UPSTREAM_PROXY"], "http://host.proxy:3128")
        self.assertEqual(env["SPARKDECK_DAEMON_PROXY"], "http://daemon.proxy:8080")

    async def test_persistent_copy_inspect_failure_aborts_the_wait(self):
        # A copy container removed externally must end the poll loop: it
        # holds the node and cluster download budgets, so spinning on a
        # phantom would block every later capped transfer.
        api = FakeDockerApi(copy_inspect_fails=True)
        staging = Path(tempfile.mkdtemp())

        def on_start(container_id):
            if container_id == "copy-1":
                staging.joinpath("image.tar").write_bytes(b"tar")

        api.on_start = on_start
        with patch("sparkdeck.throttled_image_pull.asyncio.sleep",
                   new=AsyncMock()):
            events, _ = await self.collect(api, staging=staging)
        self.assertIsNone(api.loaded)
        self.assertIn("could no longer be inspected", events[-1]["error"])
        self.assertNotIn("Successfully", json.dumps(events))
        # The budgets and helpers are released like any other failure.
        self.assertEqual(sorted(api.removed), ["copy-1", "inspect-1", "proxy-1"])

    def test_apply_bandwidth_cap_rewrites_registered_rate_files(self):
        with tempfile.TemporaryDirectory() as directory:
            key = os.path.join(directory, RATE_NAME)
            with open(key, "w", encoding="utf-8") as handle:
                handle.write("1000.0")
            tip._ACTIVE_RATE_FILES[key] = 1.0
            try:
                tip.apply_bandwidth_cap(500_000.0)
            finally:
                tip._ACTIVE_RATE_FILES.pop(key, None)
            with open(key, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "500000.0")
            # The atomic replace leaves no temp file behind.
            self.assertFalse(os.path.exists(key + ".tmp"))

    def test_apply_bandwidth_cap_scales_registered_rate_by_share(self):
        # A fan-out member registered its share (cap/4); the new CLUSTER cap
        # must land at one quarter per member, not the full value.
        with tempfile.TemporaryDirectory() as directory:
            key = os.path.join(directory, RATE_NAME)
            with open(key, "w", encoding="utf-8") as handle:
                handle.write("12500000.0")
            tip._ACTIVE_RATE_FILES[key] = 0.25
            try:
                tip.apply_bandwidth_cap(40_000_000.0)
            finally:
                tip._ACTIVE_RATE_FILES.pop(key, None)
            with open(key, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "10000000.0")

    def test_apply_bandwidth_cap_survives_a_vanished_file(self):
        tip._ACTIVE_RATE_FILES[
            os.path.join(tempfile.gettempdir(), "sparkdeck-no-such-dir", RATE_NAME)
        ] = 1.0
        tip.apply_bandwidth_cap(10.0)  # must not raise

    async def test_helper_bootstrap_is_surfaced_as_uncapped_one_time_pull(self):
        api = FakeDockerApi(helpers_missing=True)
        staging = Path(tempfile.mkdtemp())

        def on_start(container_id):
            if container_id == "copy-1":
                staging.joinpath("image.tar").write_bytes(b"tar")

        api.on_start = on_start
        events, _ = await self.collect(api, staging=staging)
        notices = [
            str(event.get("status")) for event in events
            if "not covered by the bandwidth limit" in str(event.get("status", ""))
        ]
        self.assertEqual(len(notices), 2)
        self.assertIn(PROXY_IMAGE, notices[0])
        self.assertIn(SKOPEO_IMAGE, notices[1])
        self.assertEqual(api.pulls, [PROXY_IMAGE, SKOPEO_IMAGE])
        self.assertIsNotNone(api.loaded)

    async def test_staging_binds_carry_the_shared_selinux_label(self):
        api = FakeDockerApi()
        await self.collect(api)
        proxy = next(c for c in api.created if c.get("image") == PROXY_IMAGE)
        copy = copy_kwargs(api)
        # The proxy and copy containers are different containers sharing one
        # staging directory, so both mounts need the shared ":z" label; a
        # private ":Z" would relabel away the peer's access on SELinux hosts.
        self.assertTrue(
            copy["host_config"]["Binds"][0].endswith(":/archive:z"),
            copy["host_config"]["Binds"],
        )
        self.assertEqual(
            proxy["host_config"]["Binds"], [copy["host_config"]["Binds"][0]],
        )

    async def test_rootless_daemon_omits_the_host_uid_user_override(self):
        # Rootless Docker runs the daemon as this user already, so container
        # root maps straight to it. The host-UID override instead maps to an
        # unprivileged subuid that cannot write the 0700 staging directory or
        # read the 0600 auth file.
        api = FakeDockerApi(security_options=[
            "name=rootless", "name=seccomp,profile=default",
        ])
        with (
            patch.object(tip.os, "name", "posix"),
            patch.object(tip.os, "getuid", return_value=1000, create=True),
            patch.object(tip.os, "getgid", return_value=1000, create=True),
        ):
            await self.collect(api)
        self.assertIsNone(copy_kwargs(api).get("user"))

    async def test_rootful_daemon_keeps_the_host_uid_user_override(self):
        api = FakeDockerApi(security_options=["name=seccomp,profile=default"])
        with (
            patch.object(tip.os, "name", "posix"),
            patch.object(tip.os, "getuid", return_value=1000, create=True),
            patch.object(tip.os, "getgid", return_value=1000, create=True),
        ):
            await self.collect(api)
        self.assertEqual(copy_kwargs(api).get("user"), "1000:1000")

    async def test_daemon_info_failure_falls_back_to_the_uid_override(self):
        api = FakeDockerApi()

        def broken_info():
            raise RuntimeError("daemon unreachable")

        api.info = broken_info
        with (
            patch.object(tip.os, "name", "posix"),
            patch.object(tip.os, "getuid", return_value=4242, create=True),
            patch.object(tip.os, "getgid", return_value=4242, create=True),
        ):
            await self.collect(api)
        # Unknown daemon mode must behave exactly like today's rootful path.
        self.assertEqual(copy_kwargs(api).get("user"), "4242:4242")

    async def test_userns_remap_daemon_refuses_the_capped_pull(self):
        # Rootful userns-remap maps container root to a subordinate UID, and
        # bind mounts keep host permissions, so the 0700 staging directory
        # and 0600 auth file are unusable under any container user. Refuse
        # up front with the actionable knob instead of failing mid-pull.
        api = FakeDockerApi(security_options=[
            "name=name=userns", "name=seccomp,profile=default",
        ])
        with (
            patch.object(tip.os, "name", "posix"),
            patch.object(tip.os, "getuid", return_value=1000, create=True),
            patch.object(tip.os, "getgid", return_value=1000, create=True),
        ):
            events, _ = await self.collect(api)
        errors = [event for event in events if "error" in event]
        self.assertEqual(len(errors), 1)
        self.assertIn("userns-remap", errors[0]["error"])
        self.assertFalse(
            [
                created for created in api.created
                if created.get("image") == SKOPEO_IMAGE
                and list(created.get("command") or [])[:1] == ["copy"]
            ],
        )

    async def test_current_userns_marker_also_refuses_the_capped_pull(self):
        # Moby fixed the doubled SecurityOptions spelling, so current daemons
        # report the remap as plain "name=userns". Missing that marker would
        # run the copy as the host UID and fail mid-pull on the 0700 staging
        # directory instead of refusing with the actionable knob.
        api = FakeDockerApi(security_options=[
            "name=userns", "name=seccomp,profile=default",
        ])
        with (
            patch.object(tip.os, "name", "posix"),
            patch.object(tip.os, "getuid", return_value=1000, create=True),
            patch.object(tip.os, "getgid", return_value=1000, create=True),
        ):
            events, _ = await self.collect(api)
        errors = [event for event in events if "error" in event]
        self.assertEqual(len(errors), 1)
        self.assertIn("userns-remap", errors[0]["error"])


class ProxyTrustWiringTests(unittest.IsolatedAsyncioTestCase):
    """SSL_CERT_FILE forwarding into the limiter container (PozT/PozP)."""

    def proxy_kwargs(self, api):
        return next(
            c for c in api.created if c.get("image") == PROXY_IMAGE
        )

    async def test_host_ca_file_is_mounted_and_pointed_to(self):
        api = FakeDockerApi()
        ca_file = "/etc/ssl/certs/corp-proxy-ca.pem"
        with (
            patch.dict(os.environ, {"SSL_CERT_FILE": ca_file}),
            patch.object(
                tip.os.path, "isfile",
                side_effect=lambda path: path == ca_file,
            ),
        ):
            await ThrottledPullFlowTests.collect(self, api)
        proxy = self.proxy_kwargs(api)
        self.assertEqual(proxy["environment"]["SPARKDECK_UPSTREAM_PROXY_CA"], ca_file)
        self.assertIn(
            f"{ca_file}:{ca_file}:ro,z", proxy["host_config"]["Binds"],
        )

    async def test_no_host_ca_leaves_the_override_empty(self):
        api = FakeDockerApi()
        env = {k: v for k, v in os.environ.items() if k != "SSL_CERT_FILE"}
        with patch.dict(os.environ, env, clear=True):
            await ThrottledPullFlowTests.collect(self, api)
        proxy = self.proxy_kwargs(api)
        self.assertEqual(proxy["environment"]["SPARKDECK_UPSTREAM_PROXY_CA"], "")
        self.assertTrue(all(
            ":/etc/ssl" not in bind for bind in proxy["host_config"]["Binds"]
        ))


class HelperBootstrapTests(unittest.IsolatedAsyncioTestCase):
    async def test_preinstall_pulls_only_missing_helpers(self):
        client = Mock()
        client.api = FakeDockerApi()
        await tip.ensure_helper_images(client)
        self.assertEqual(client.api.pulls, [])
        api = FakeDockerApi(helpers_missing=True)
        client = Mock()
        client.api = api
        await tip.ensure_helper_images(client)
        self.assertEqual(api.pulls, [PROXY_IMAGE, SKOPEO_IMAGE])


class LimiterProxyTests(unittest.TestCase):
    """End-to-end CONNECT tests against the embedded limiter script."""

    def free_port(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        return port

    def start_proxy(self, port, env_extra=None):
        env = os.environ.copy()
        # The developer machine's own proxy policy must not decide whether
        # these scripted tests chain or dial directly. Windows collapses
        # env-key case, so every spelling variant present in the inherited
        # environment is stripped before the test-controlled NO_PROXY is
        # written: a coexisting lowercase no_proxy would clobber NO_PROXY
        # in the child.
        for stale in [k for k in env if k.upper() == "NO_PROXY"]:
            env.pop(stale)
        env.update({
            "SPARKDECK_DOWNLOAD_BPS": "10000000",
            "SPARKDECK_BANDWIDTH_PROGRESS": "",
            "SPARKDECK_PROXY_PORT": str(port),
            "NO_PROXY": "",
        })
        env.update(env_extra or {})
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", tip._PROXY_SCRIPT], env=env,
        )
        try:
            for _ in range(100):
                try:
                    probe = socket.create_connection(("127.0.0.1", port), 0.1)
                    probe.close()
                    return process
                except OSError:
                    time.sleep(0.05)
            raise AssertionError("limiter proxy never started")
        except Exception:
            process.kill()
            raise

    def echo_server(self, host):
        listener = socket.socket(
            socket.AF_INET6 if ":" in host else socket.AF_INET,
        )
        listener.bind((host, 0))
        listener.listen(8)
        port = listener.getsockname()[1]

        def serve():
            while True:
                try:
                    connection, _ = listener.accept()
                except OSError:
                    return
                def echo(sock=connection):
                    try:
                        while True:
                            data = sock.recv(4096)
                            if not data:
                                break
                            sock.sendall(data)
                    except OSError:
                        pass
                    finally:
                        sock.close()
                threading.Thread(target=echo, daemon=True).start()
        threading.Thread(target=serve, daemon=True).start()
        return listener, port

    def connect(self, proxy_port, authority):
        client = socket.create_connection(("127.0.0.1", proxy_port), 5)
        request = f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n\r\n"
        client.sendall(request.encode())
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = client.recv(4096)
            if not chunk:
                break
            response += chunk
        return client, response

    def test_ipv6_bracketed_authority_is_dialed_correctly(self):
        listener, echo_port = self.echo_server("::1")
        proxy_port = self.free_port()
        process = self.start_proxy(proxy_port)
        try:
            client, response = self.connect(proxy_port, f"[::1]:{echo_port}")
            self.assertIn(b"200", response.split(b"\r\n", 1)[0])
            client.sendall(b"ping-v6")
            self.assertEqual(client.recv(64), b"ping-v6")
            client.close()
        finally:
            process.kill()
            listener.close()

    def test_plain_http_absolute_form_request_is_forwarded(self):
        # A genuinely insecure registry or plain-HTTP mirror sends its
        # traffic as absolute-form requests to HTTP_PROXY, not a CONNECT
        # tunnel. The limiter must dial the URL endpoint, rewrite to
        # origin-form, and serve it through the same charged pump.
        listener, echo_port = self.echo_server("127.0.0.1")
        proxy_port = self.free_port()
        process = self.start_proxy(proxy_port)
        try:
            client = socket.create_connection(("127.0.0.1", proxy_port), 5)
            request = (
                "GET http://127.0.0.1:%d/v2/_catalog HTTP/1.1\r\n"
                "Host: 127.0.0.1:%d\r\n\r\n" % (echo_port, echo_port)
            )
            client.sendall(request.encode())
            echoed = b""
            deadline = time.monotonic() + 5
            while b"GET" not in echoed and time.monotonic() < deadline:
                chunk = client.recv(4096)
                if not chunk:
                    break
                echoed += chunk
            self.assertIn(b"GET /v2/_catalog HTTP/1.1", echoed)
            client.close()
        finally:
            process.kill()
            listener.close()

    def test_unsupported_proxy_request_shapes_are_rejected(self):
        # Origin-form requests are not proxy traffic, and https-form GETs
        # must arrive as CONNECT instead; both get an explicit refusal
        # rather than a mis-forward.
        proxy_port = self.free_port()
        process = self.start_proxy(proxy_port)
        try:
            for request in (
                "GET /plain HTTP/1.1\r\nHost: x\r\n\r\n",
                "GET https://registry.example/v2/ HTTP/1.1\r\n"
                "Host: registry.example\r\n\r\n",
            ):
                client = socket.create_connection(("127.0.0.1", proxy_port), 5)
                client.sendall(request.encode())
                response = b""
                while b"\r\n\r\n" not in response:
                    chunk = client.recv(4096)
                    if not chunk:
                        break
                    response += chunk
                self.assertIn(b"405", response.split(b"\r\n", 1)[0])
                client.close()
        finally:
            process.kill()

    def test_wildcard_no_proxy_entries_match_hostnames(self):
        # Go's proxy matcher (Docker's own rule) turns a leading "*.domain"
        # wildcard into a suffix rule. The matcher runs inside the proxy
        # container, so execute the script prefix (before the listener loop)
        # and drive it with the real normalization.
        script = tip._PROXY_SCRIPT.split("listener = socket.socket()")[0]
        namespace = {}
        with patch.dict(os.environ, {
            "SPARKDECK_DOWNLOAD_BPS": "1000",
            "SPARKDECK_BANDWIDTH_PROGRESS": "",
            "SPARKDECK_BANDWIDTH_RATE_FILE": "",
            "SPARKDECK_UPSTREAM_PROXY": "",
            "SPARKDECK_DAEMON_PROXY": "",
            "SPARKDECK_DAEMON_NO_PROXY": "",
            "SPARKDECK_UPSTREAM_PROXY_CA": "",
            "NO_PROXY": "*.test.example.com,.internal.corp",
        }, clear=False):
            exec(compile(script, "<proxy-prefix>", "exec"), namespace)
        self.assertTrue(namespace["no_proxy_direct"](
            "registry-1.test.example.com:443",
        ))
        self.assertTrue(namespace["no_proxy_direct"]("a.internal.corp:5000"))
        self.assertFalse(namespace["no_proxy_direct"]("ghcr.io:443"))

    def test_rate_file_change_re_rates_the_running_limiter(self):
        # A settings change rewrites the staged rate file mid-transfer; the
        # running proxy must throttle at the old rate first, then move at the
        # new one without a restart.
        listener, echo_port = self.echo_server("127.0.0.1")
        proxy_port = self.free_port()
        payload = b"x" * 30_000
        with tempfile.TemporaryDirectory() as directory:
            rate_path = os.path.join(directory, "rate.txt")
            with open(rate_path, "w") as handle:
                handle.write("50000")
            process = self.start_proxy(proxy_port, env_extra={
                "SPARKDECK_DOWNLOAD_BPS": "50000",
                "SPARKDECK_BANDWIDTH_RATE_FILE": rate_path,
            })
            client = None
            try:
                client, response = self.connect(
                    proxy_port, f"127.0.0.1:{echo_port}",
                )
                self.assertIn(b"200", response.split(b"\r\n", 1)[0])

                def transfer():
                    start = time.monotonic()
                    client.sendall(payload)
                    received = 0
                    while received < len(payload):
                        chunk = client.recv(65536)
                        if not chunk:
                            break
                        received += len(chunk)
                    self.assertEqual(received, len(payload))
                    return time.monotonic() - start

                throttled_seconds = transfer()
                with open(rate_path, "w") as handle:
                    handle.write("0")
                time.sleep(0.7)  # outlast one 0.5 s rate poll
                unlimited_seconds = transfer()
            finally:
                if client is not None:
                    client.close()
                process.kill()
                listener.close()
        # 30 KB at 50 KB/s needs at least ~0.6 s; unlimited has no sleeps at
        # all, so the bounds only have to separate the two clearly.
        self.assertGreaterEqual(throttled_seconds, 0.35)
        self.assertLess(unlimited_seconds, 1.0)

    def test_upstream_proxy_chaining_with_authorization(self):
        listener, echo_port = self.echo_server("127.0.0.1")
        upstream = socket.socket()
        upstream.bind(("127.0.0.1", 0))
        upstream.listen(8)
        upstream_port = upstream.getsockname()[1]
        captured = {}

        def upstream_serve():
            connection, _ = upstream.accept()
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                request += chunk
            captured["request"] = request
            connection.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            target = socket.create_connection(("127.0.0.1", echo_port), 5)
            def pipe(src, dst):
                try:
                    while True:
                        data = src.recv(4096)
                        if not data:
                            break
                        dst.sendall(data)
                except OSError:
                    pass
            threading.Thread(target=pipe, args=(connection, target), daemon=True).start()
            pipe(target, connection)

        threading.Thread(target=upstream_serve, daemon=True).start()
        proxy_port = self.free_port()
        process = self.start_proxy(proxy_port, {
            "SPARKDECK_UPSTREAM_PROXY": f"http://user:pass@127.0.0.1:{upstream_port}",
        })
        try:
            client, response = self.connect(proxy_port, f"127.0.0.1:{echo_port}")
            self.assertIn(b"200", response.split(b"\r\n", 1)[0])
            client.sendall(b"ping-chain")
            self.assertEqual(client.recv(64), b"ping-chain")
            client.close()
            request = captured.get("request", b"")
            self.assertIn(b"CONNECT 127.0.0.1:%d HTTP/1.1" % echo_port, request)
            expected = base64.b64encode(b"user:pass")
            self.assertIn(b"Proxy-Authorization: Basic " + expected, request)
        finally:
            process.kill()
            listener.close()
            upstream.close()

    def test_daemon_proxy_is_the_fallback_when_upstream_is_empty(self):
        # daemon.json-only proxy: the flow sends the daemon value in
        # SPARKDECK_DAEMON_PROXY with an empty upstream, and the limiter
        # must still chain CONNECT through it.
        listener, echo_port = self.echo_server("127.0.0.1")
        upstream = socket.socket()
        upstream.bind(("127.0.0.1", 0))
        upstream.listen(8)
        upstream_port = upstream.getsockname()[1]

        def upstream_serve():
            connection, _ = upstream.accept()
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                request += chunk
            self.assertIn(b"CONNECT", request)
            connection.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            target = socket.create_connection(("127.0.0.1", echo_port), 5)
            def pipe(src, dst):
                try:
                    while True:
                        data = src.recv(4096)
                        if not data:
                            break
                        dst.sendall(data)
                except OSError:
                    pass
            threading.Thread(target=pipe, args=(connection, target), daemon=True).start()
            pipe(target, connection)

        threading.Thread(target=upstream_serve, daemon=True).start()
        proxy_port = self.free_port()
        process = self.start_proxy(proxy_port, {
            "SPARKDECK_UPSTREAM_PROXY": "",
            "SPARKDECK_DAEMON_PROXY": f"http://127.0.0.1:{upstream_port}",
        })
        try:
            client, response = self.connect(proxy_port, f"127.0.0.1:{echo_port}")
            self.assertIn(b"200", response.split(b"\r\n", 1)[0])
            client.sendall(b"ping-daemon")
            self.assertEqual(client.recv(64), b"ping-daemon")
            client.close()
        finally:
            process.kill()
            listener.close()
            upstream.close()

    def test_process_upstream_wins_over_the_daemon_fallback(self):
        # The daemon proxy points at a dead port; only correct precedence
        # (process env first) can complete the CONNECT.
        listener, echo_port = self.echo_server("127.0.0.1")
        upstream = socket.socket()
        upstream.bind(("127.0.0.1", 0))
        upstream.listen(8)
        upstream_port = upstream.getsockname()[1]

        def upstream_serve():
            connection, _ = upstream.accept()
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                request += chunk
            connection.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            target = socket.create_connection(("127.0.0.1", echo_port), 5)
            def pipe(src, dst):
                try:
                    while True:
                        data = src.recv(4096)
                        if not data:
                            break
                        dst.sendall(data)
                except OSError:
                    pass
            threading.Thread(target=pipe, args=(connection, target), daemon=True).start()
            pipe(target, connection)

        threading.Thread(target=upstream_serve, daemon=True).start()
        dead_port = self.free_port()  # never bound
        proxy_port = self.free_port()
        process = self.start_proxy(proxy_port, {
            "SPARKDECK_UPSTREAM_PROXY": f"http://127.0.0.1:{upstream_port}",
            "SPARKDECK_DAEMON_PROXY": f"http://127.0.0.1:{dead_port}",
        })
        try:
            client, response = self.connect(proxy_port, f"127.0.0.1:{echo_port}")
            self.assertIn(b"200", response.split(b"\r\n", 1)[0])
            client.sendall(b"ping-precedence")
            self.assertEqual(client.recv(64), b"ping-precedence")
            client.close()
        finally:
            process.kill()
            listener.close()
            upstream.close()

    def test_daemon_no_proxy_fallback_dials_the_target_directly(self):
        # The daemon's no-proxy list is proxy policy: the matching target is
        # dialed directly (still capped), proven by the daemon proxy sitting
        # on a dead port the chain would have to use otherwise.
        listener, echo_port = self.echo_server("127.0.0.1")
        dead_port = self.free_port()  # never bound
        proxy_port = self.free_port()
        process = self.start_proxy(proxy_port, {
            "SPARKDECK_UPSTREAM_PROXY": "",
            "SPARKDECK_DAEMON_PROXY": f"http://127.0.0.1:{dead_port}",
            "SPARKDECK_DAEMON_NO_PROXY": "127.0.0.1",
        })
        try:
            client, response = self.connect(proxy_port, f"127.0.0.1:{echo_port}")
            self.assertIn(b"200", response.split(b"\r\n", 1)[0])
            client.sendall(b"ping-nop")
            self.assertEqual(client.recv(64), b"ping-nop")
            client.close()
        finally:
            process.kill()
            listener.close()

    def test_upstream_url_with_trailing_slash_and_path_is_parsed(self):
        # Proxy settings copied from browser/system configs often carry a
        # trailing slash or PAC path (http://proxy.corp:8080/). Splitting the
        # authority on ':' turns the port into "8080/" and every capped pull
        # would die on int(); the value must parse as a URL.
        listener, echo_port = self.echo_server("127.0.0.1")
        upstream = socket.socket()
        upstream.bind(("127.0.0.1", 0))
        upstream.listen(8)
        upstream_port = upstream.getsockname()[1]

        def upstream_serve():
            connection, _ = upstream.accept()
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                request += chunk
            connection.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            target = socket.create_connection(("127.0.0.1", echo_port), 5)
            def pipe(src, dst):
                try:
                    while True:
                        data = src.recv(4096)
                        if not data:
                            break
                        dst.sendall(data)
                except OSError:
                    pass
            threading.Thread(target=pipe, args=(connection, target), daemon=True).start()
            pipe(target, connection)

        threading.Thread(target=upstream_serve, daemon=True).start()
        proxy_port = self.free_port()
        process = self.start_proxy(proxy_port, {
            "SPARKDECK_UPSTREAM_PROXY":
                f"http://127.0.0.1:{upstream_port}/pac/proxy.pac",
        })
        try:
            client, response = self.connect(proxy_port, f"127.0.0.1:{echo_port}")
            self.assertIn(b"200", response.split(b"\r\n", 1)[0])
            client.sendall(b"ping-path")
            self.assertEqual(client.recv(64), b"ping-path")
            client.close()
        finally:
            process.kill()
            listener.close()
            upstream.close()

    def test_no_proxy_cidr_entry_matches_the_target_ip(self):
        # NO_PROXY entries commonly name a subnet (127.0.0.0/8, 10.0.0.0/8).
        # The hostname suffix rules cannot see networks, so the CIDR must be
        # parsed and matched by containment; a hit dials the target directly
        # (still capped), never chains through the excluded upstream.
        listener, echo_port = self.echo_server("127.0.0.1")
        proxy_port = self.free_port()
        process = self.start_proxy(proxy_port, {
            # A dead upstream: if the CIDR entry fails to match, the proxy
            # chains there and the CONNECT cannot succeed.
            "SPARKDECK_UPSTREAM_PROXY": f"http://127.0.0.1:{self.free_port()}",
            "NO_PROXY": "127.0.0.0/8, example.internal",
        })
        try:
            client, response = self.connect(proxy_port, f"127.0.0.1:{echo_port}")
            self.assertIn(b"200", response.split(b"\r\n", 1)[0])
            client.sendall(b"ping-cidr")
            self.assertEqual(client.recv(64), b"ping-cidr")
            client.close()
        finally:
            process.kill()
            listener.close()

    def test_no_proxy_cidr_miss_still_chains_through_upstream(self):
        # Containment, not resemblance: 127.0.0.1 is outside 10.0.0.0/8, so
        # the corporate upstream must still serve the CONNECT.
        listener, echo_port = self.echo_server("127.0.0.1")
        upstream = socket.socket()
        upstream.bind(("127.0.0.1", 0))
        upstream.listen(8)
        upstream_port = upstream.getsockname()[1]

        def upstream_serve():
            connection, _ = upstream.accept()
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                request += chunk
            connection.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            target = socket.create_connection(("127.0.0.1", echo_port), 5)
            def pipe(src, dst):
                try:
                    while True:
                        data = src.recv(4096)
                        if not data:
                            break
                        dst.sendall(data)
                except OSError:
                    pass
            threading.Thread(target=pipe, args=(connection, target), daemon=True).start()
            pipe(target, connection)

        threading.Thread(target=upstream_serve, daemon=True).start()
        proxy_port = self.free_port()
        process = self.start_proxy(proxy_port, {
            "SPARKDECK_UPSTREAM_PROXY": f"http://127.0.0.1:{upstream_port}",
            "NO_PROXY": "10.0.0.0/8",
        })
        try:
            client, response = self.connect(proxy_port, f"127.0.0.1:{echo_port}")
            self.assertIn(b"200", response.split(b"\r\n", 1)[0])
            client.sendall(b"ping-miss")
            self.assertEqual(client.recv(64), b"ping-miss")
            client.close()
        finally:
            process.kill()
            listener.close()
            upstream.close()

    def tls_proxy_listener(self, cert_file, key_file):
        """A TLS-terminated CONNECT proxy for the HTTPS upstream tests."""
        import ssl

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_file, key_file)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        port = listener.getsockname()[1]
        captured = {}

        def serve():
            try:
                connection, _ = listener.accept()
            except OSError:
                return
            try:
                tls = context.wrap_socket(connection, server_side=True)
            except OSError:
                captured["tls"] = "handshake-failed"
                connection.close()
                return
            captured["tls"] = "handshake-ok"
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = tls.recv(4096)
                if not chunk:
                    captured["request"] = request
                    return
                request += chunk
            captured["request"] = request
            tls.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")

            def pipe():
                target = None
                try:
                    while True:
                        data = tls.recv(4096)
                        if not data:
                            break
                        if target is None:
                            authority = request.split(b" ")[1].decode("latin-1")
                            host, _, port_text = authority.rpartition(":")
                            target = socket.create_connection(
                                (host, int(port_text)), 5,
                            )
                        target.sendall(data)
                        reply = target.recv(4096)
                        if reply:
                            tls.sendall(reply)
                except OSError:
                    pass
                finally:
                    if target is not None:
                        target.close()
                    tls.close()
            pipe()

        threading.Thread(target=serve, daemon=True).start()
        return listener, port, captured

    def make_localhost_cert(self):
        import ipaddress
        from datetime import datetime, timedelta, timezone

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
        now = datetime.now(timezone.utc)
        certificate = (
            x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1))
            .not_valid_after(now + timedelta(days=1))
            .add_extension(
                x509.SubjectAlternativeName(
                    [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))],
                ), critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        directory = tempfile.mkdtemp()
        cert_file = Path(directory) / "proxy.crt"
        key_file = Path(directory) / "proxy.key"
        cert_file.write_bytes(certificate.public_bytes(
            serialization.Encoding.PEM,
        ))
        key_file.write_bytes(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
        return str(cert_file), str(key_file)

    def test_https_upstream_proxy_gets_tls_before_connect(self):
        listener, echo_port = self.echo_server("127.0.0.1")
        cert_file, key_file = self.make_localhost_cert()
        tls_listener, tls_port, captured = self.tls_proxy_listener(
            cert_file, key_file,
        )
        proxy_port = self.free_port()
        process = self.start_proxy(proxy_port, {
            "SPARKDECK_UPSTREAM_PROXY": f"https://127.0.0.1:{tls_port}",
            "SPARKDECK_UPSTREAM_PROXY_CA": cert_file,
        })
        try:
            client, response = self.connect(proxy_port, f"127.0.0.1:{echo_port}")
            self.assertIn(b"200", response.split(b"\r\n", 1)[0])
            client.sendall(b"ping-tls")
            self.assertEqual(client.recv(64), b"ping-tls")
            client.close()
            self.assertEqual(captured.get("tls"), "handshake-ok")
            self.assertIn(b"CONNECT 127.0.0.1:%d" % echo_port, captured["request"])
        finally:
            process.kill()
            listener.close()
            tls_listener.close()

    def test_https_upstream_without_trust_is_refused_not_plaintext(self):
        # Without the CA file, the default trust store cannot verify the
        # private proxy certificate: the limiter must fail closed instead of
        # falling back to a plaintext CONNECT that a TLS listener rejects.
        listener, echo_port = self.echo_server("127.0.0.1")
        cert_file, key_file = self.make_localhost_cert()
        tls_listener, tls_port, captured = self.tls_proxy_listener(
            cert_file, key_file,
        )
        proxy_port = self.free_port()
        process = self.start_proxy(proxy_port, {
            "SPARKDECK_UPSTREAM_PROXY": f"https://127.0.0.1:{tls_port}",
        })
        try:
            _client, response = self.connect(proxy_port, f"127.0.0.1:{echo_port}")
            self.assertNotIn(b"200", response.split(b"\r\n", 1)[0])
        finally:
            process.kill()
            listener.close()
            tls_listener.close()

    def test_no_proxy_host_is_dialed_directly_but_still_capped(self):
        # NO_PROXY is proxy policy: the limiter dials the excluded host
        # itself (bytes still move through its token bucket) instead of
        # asking the corporate proxy that refused to serve it.
        listener, echo_port = self.echo_server("127.0.0.1")
        upstream = socket.socket()
        upstream.bind(("127.0.0.1", 0))
        upstream.listen(8)
        upstream_port = upstream.getsockname()[1]
        upstream_used = []

        def watch_upstream():
            try:
                connection, _ = upstream.accept()
            except OSError:
                return
            upstream_used.append("touched")
            connection.close()

        threading.Thread(target=watch_upstream, daemon=True).start()
        proxy_port = self.free_port()
        process = self.start_proxy(proxy_port, {
            "SPARKDECK_UPSTREAM_PROXY": f"http://127.0.0.1:{upstream_port}",
            "SPARKDECK_DOWNLOAD_BPS": "4096",
            "NO_PROXY": "other.example, 127.0.0.1",
        })
        try:
            client, response = self.connect(proxy_port, f"127.0.0.1:{echo_port}")
            self.assertIn(b"200", response.split(b"\r\n", 1)[0])
            client.sendall(b"ping-direct")
            self.assertEqual(client.recv(64), b"ping-direct")
            client.close()
        finally:
            process.kill()
            listener.close()
            upstream.close()
        self.assertEqual(upstream_used, [])

    def test_no_proxy_mismatch_still_chains_through_upstream(self):
        listener, echo_port = self.echo_server("127.0.0.1")
        upstream = socket.socket()
        upstream.bind(("127.0.0.1", 0))
        upstream.listen(8)
        upstream_port = upstream.getsockname()[1]
        captured = {}

        def upstream_serve():
            connection, _ = upstream.accept()
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                request += chunk
            captured["request"] = request
            connection.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            target = socket.create_connection(("127.0.0.1", echo_port), 5)

            def pipe(src, dst):
                try:
                    while True:
                        data = src.recv(4096)
                        if not data:
                            break
                        dst.sendall(data)
                except OSError:
                    pass
            threading.Thread(target=pipe, args=(connection, target), daemon=True).start()
            pipe(target, connection)

        threading.Thread(target=upstream_serve, daemon=True).start()
        proxy_port = self.free_port()
        process = self.start_proxy(proxy_port, {
            "SPARKDECK_UPSTREAM_PROXY": f"http://127.0.0.1:{upstream_port}",
            "NO_PROXY": "internal.example",
        })
        try:
            client, response = self.connect(proxy_port, f"127.0.0.1:{echo_port}")
            self.assertIn(b"200", response.split(b"\r\n", 1)[0])
            client.sendall(b"ping-chain")
            self.assertEqual(client.recv(64), b"ping-chain")
            client.close()
            self.assertIn(b"CONNECT", captured.get("request", b""))
        finally:
            process.kill()
            listener.close()
            upstream.close()

    def test_upstream_proxy_credentials_are_percent_decoded(self):
        # HTTPS_PROXY percent-encodes userinfo delimiters (@ : %); the Basic
        # credential must carry the decoded username and password.
        listener, echo_port = self.echo_server("127.0.0.1")
        upstream = socket.socket()
        upstream.bind(("127.0.0.1", 0))
        upstream.listen(8)
        upstream_port = upstream.getsockname()[1]
        captured = {}

        def upstream_serve():
            connection, _ = upstream.accept()
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                request += chunk
            captured["request"] = request
            connection.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            target = socket.create_connection(("127.0.0.1", echo_port), 5)

            def pipe(src, dst):
                try:
                    while True:
                        data = src.recv(4096)
                        if not data:
                            break
                        dst.sendall(data)
                except OSError:
                    pass
            threading.Thread(target=pipe, args=(connection, target), daemon=True).start()
            pipe(target, connection)

        threading.Thread(target=upstream_serve, daemon=True).start()
        proxy_port = self.free_port()
        process = self.start_proxy(proxy_port, {
            "SPARKDECK_UPSTREAM_PROXY": (
                f"http://user%40corp%3Apw%25@127.0.0.1:{upstream_port}"
            ),
        })
        try:
            client, response = self.connect(proxy_port, f"127.0.0.1:{echo_port}")
            self.assertIn(b"200", response.split(b"\r\n", 1)[0])
            client.sendall(b"ping-auth")
            self.assertEqual(client.recv(64), b"ping-auth")
            client.close()
            expected = base64.b64encode(b"user@corp:pw%")
            self.assertIn(
                b"Proxy-Authorization: Basic " + expected,
                captured.get("request", b""),
            )
        finally:
            process.kill()
            listener.close()
            upstream.close()


class RegistryTrustTests(unittest.TestCase):
    @patch("socket.getaddrinfo", side_effect=OSError)
    def test_insecure_entries_match_cidr_hosts_and_ports(self, resolver):
        # CIDR entries now consult DNS for hostname references (dockerd
        # matches registries resolving INTO the subnet); these fixtures test
        # name/port matching, so resolution stays unreachable and any host
        # the test does not intend must not accidentally match via DNS.
        insecure = ["127.0.0.0/8", "::1/128", "private.registry:5000", "lan.registry"]
        self.assertTrue(
            matches_insecure_registry("private.registry:5000/team/img:1", insecure),
        )
        self.assertTrue(matches_insecure_registry("lan.registry/team/img", insecure))
        self.assertTrue(matches_insecure_registry("10.1.2.3:5000/img", ["10.0.0.0/8"]))
        self.assertTrue(
            matches_insecure_registry("[fd00::2]:5000/img", ["fd00::/64"]),
        )
        self.assertFalse(matches_insecure_registry("ghcr.io/org/model:1", insecure))
        self.assertFalse(matches_insecure_registry("nginx:latest", insecure))
        self.assertFalse(matches_insecure_registry("bad:99999/x", ["::1/128"]))

    def test_endpoint_port_entries_stay_scoped_to_their_port(self):
        # dockerd documents "--insecure-registry reg.corp:5000" as that
        # host-and-port endpoint. Matching a port-less reference would
        # disable certificate verification for the same host's default-443
        # registry the daemon still trusts, admitting a forged image.
        self.assertFalse(matches_insecure_registry(
            "reg.corp/team/image", ["reg.corp:5000"],
        ))
        self.assertTrue(matches_insecure_registry(
            "reg.corp:5000/team/image", ["reg.corp:5000"],
        ))
        # A port-less reference implies 443, so an explicit-443 entry is its
        # equivalent; a port-less entry still covers every port on the host.
        self.assertTrue(matches_insecure_registry(
            "reg.corp/team/image", ["reg.corp:443"],
        ))
        self.assertTrue(matches_insecure_registry(
            "reg.corp:5000/team/image", ["reg.corp"],
        ))

    def test_insecure_cidrs_apply_to_hostnames_resolving_inside_them(self):
        # dockerd applies a CIDR entry to registries whose domain resolves
        # into the subnet; ignoring non-literal hosts would leave TLS
        # verification on against a plaintext or privately signed endpoint
        # the daemon pulls freely.
        with patch(
            "socket.getaddrinfo",
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 0)),
            ],
        ) as resolver:
            self.assertTrue(matches_insecure_registry(
                "reg.corp/team/image", ["10.0.0.0/8"],
            ))
        resolver.assert_called_once_with("reg.corp", None)

    def test_insecure_cidrs_stay_off_when_resolution_fails(self):
        with patch("socket.getaddrinfo", side_effect=OSError):
            self.assertFalse(matches_insecure_registry(
                "reg.corp/team/image", ["10.0.0.0/8"],
            ))

    @patch("socket.getaddrinfo", side_effect=OSError)
    def test_insecure_registry_entries_merge_indexconfigs(self, resolver):
        # Docker records named insecure hosts only under IndexConfigs with
        # Secure:false; consulting the CIDRs alone misses them entirely.
        # Named hosts stay unresolved here so only the name match decides.
        entries = insecure_registry_entries({
            "InsecureRegistryCIDRs": ["10.0.0.0/8"],
            "IndexConfigs": {
                "reg.corp:5000": {"Name": "reg.corp:5000", "Secure": False},
                "plain.corp": {
                    "Name": "https://plain.corp/v1/", "Secure": False,
                },
                "secure.corp": {"Name": "secure.corp", "Secure": True},
            },
        })
        self.assertIn("10.0.0.0/8", entries)
        self.assertIn("reg.corp:5000", entries)
        self.assertIn("plain.corp", entries)
        self.assertNotIn("https://plain.corp/v1/", entries)
        self.assertNotIn("secure.corp", entries)
        # The named entry drives the same verification-off match as a CIDR.
        self.assertTrue(matches_insecure_registry("plain.corp/team/img", entries))
        self.assertFalse(matches_insecure_registry("secure.corp/team/img", entries))

    def test_http_hub_mirror_is_marked_insecure_in_the_dropin(self):
        # The scheme is transport configuration, not decoration: a plain
        # HTTP mirror only works through containers/image when the entry is
        # marked insecure; dropping the scheme without the mark turns the
        # working mirror into a failing TLS handshake.
        text = registry_mirrors_config(
            {"RegistryConfig": {"Mirrors": [
                "http://mirror.lan:5000/", "https://cache.lan",
            ]}},
            "docker.io",
        )
        self.assertIn('location = "mirror.lan:5000"', text)
        self.assertIn('location = "cache.lan"', text)
        # Only the plain-HTTP mirror carries the insecure mark.
        self.assertEqual(text.count("insecure = true"), 1)
        self.assertLess(
            text.index('location = "mirror.lan:5000"'),
            text.index("insecure = true"),
        )

    def test_all_empty_mirrors_produce_no_dropin(self):
        self.assertEqual(registry_mirrors_config(
            {"RegistryConfig": {"Mirrors": ["/", "   "]}}, "docker.io",
        ), "")

    def test_split_authority_handles_ipv6_brackets(self):
        self.assertEqual(split_authority("[2001:db8::1]:5000"), ("2001:db8::1", "5000"))
        self.assertEqual(split_authority("[::1]"), ("::1", ""))
        self.assertEqual(split_authority("host:5000"), ("host", "5000"))
        self.assertEqual(split_authority("host"), ("host", ""))


class DockerConfigSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_docker_config_env_is_the_first_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory) / "docker-config"
            config_dir.mkdir()
            config = config_dir / "config.json"
            config.write_text(
                json.dumps({"auths": {"docker.io": {"auth": "dTpz"}}}),
                encoding="utf-8",
            )
            api = FakeDockerApi()
            staging = Path(tempfile.mkdtemp())
            auth_dir = Path(tempfile.mkdtemp())
            captured: dict = {}

            def on_start(container_id):
                if container_id != "copy-1":
                    return
                written = auth_dir / "config.json"
                captured["auth"] = (
                    json.loads(written.read_text()) if written.exists() else None
                )
                staging.joinpath("image.tar").write_bytes(b"tar")

            api.on_start = on_start
            client = Mock()
            client.api = api
            events = []
            with (
                patch.object(
                    tip.tempfile, "mkdtemp",
                    side_effect=[str(staging), str(auth_dir),
                                 tempfile.mkdtemp(prefix="sparkdeck-test-cas-")],
                ),
                patch.object(
                    tip.shutil, "disk_usage",
                    return_value=_USAGE(10 ** 15, 0, 10 ** 15),
                ),
                patch.dict(os.environ, {"DOCKER_CONFIG": str(config_dir)}),
            ):
                async for event in throttled_image_pull_events(
                    client, "nginx", 1000.0,
                ):
                    events.append(event)
            self.assertEqual(events[-1], {"status": "Successfully pulled nginx"})
            copy = copy_kwargs(api)
            # Only the selected registry's credential is mounted: the flow
            # materializes a single-entry copy instead of the whole config.
            self.assertTrue(any(
                bind.startswith(f"{auth_dir / 'config.json'}:")
                and bind.endswith(":/auth/config.json:ro,z")
                for bind in copy["host_config"]["Binds"]
            ), copy["host_config"]["Binds"])
            self.assertFalse(any(
                str(config) in bind for bind in copy["host_config"]["Binds"]
            ), copy["host_config"]["Binds"])
            self.assertEqual(
                captured["auth"], {"auths": {"docker.io": {"auth": "dTpz"}}},
            )
            self.assertIn("--authfile", copy["command"])

    async def test_only_the_pulled_registry_credential_is_materialized(self):
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory) / "docker-config"
            config_dir.mkdir()
            config = config_dir / "config.json"
            config.write_text(json.dumps({"auths": {
                "docker.io": {"auth": "dTpz"},
                "ghcr.io": {"username": "octo", "password": "gh-secret"},
            }}), encoding="utf-8")
            api = FakeDockerApi()
            staging = Path(tempfile.mkdtemp())
            auth_dir = Path(tempfile.mkdtemp())
            captured: dict = {}

            def on_start(container_id):
                if container_id != "copy-1":
                    return
                written = auth_dir / "config.json"
                captured["auth"] = (
                    json.loads(written.read_text()) if written.exists() else None
                )
                staging.joinpath("image.tar").write_bytes(b"tar")

            api.on_start = on_start
            client = Mock()
            client.api = api
            events = []
            with (
                patch.object(
                    tip.tempfile, "mkdtemp",
                    side_effect=[str(staging), str(auth_dir),
                                 tempfile.mkdtemp(prefix="sparkdeck-test-cas-")],
                ),
                patch.object(
                    tip.shutil, "disk_usage",
                    return_value=_USAGE(10 ** 15, 0, 10 ** 15),
                ),
                patch.dict(os.environ, {"DOCKER_CONFIG": str(config_dir)}),
            ):
                async for event in throttled_image_pull_events(
                    client, "nginx", 1000.0,
                ):
                    events.append(event)
            self.assertEqual(events[-1], {"status": "Successfully pulled nginx"})
            self.assertEqual(
                captured["auth"], {"auths": {"docker.io": {"auth": "dTpz"}}},
            )
            self.assertNotIn("ghcr.io", captured["auth"]["auths"])


def inspect_kwargs(api):
    """The size-preflight inspect container kwargs."""
    return next(
        c for c in api.created
        if c.get("image") == SKOPEO_IMAGE
        and list(c.get("command") or [])[:1] == ["inspect"]
    )


class ImageTotalBytesTests(unittest.IsolatedAsyncioTestCase):
    async def total(self, payload, copy_binds=None):
        api = FakeDockerApi()
        base_logs = api.container_logs

        def logs(container_id, stderr=True, stdout=False, tail=12):
            if stdout and not stderr:
                return json.dumps(payload).encode()
            return base_logs(container_id, stderr=stderr, stdout=stdout, tail=tail)

        api.container_logs = logs
        client = Mock()
        client.api = api
        self.api = api
        return await tip._image_total_bytes(
            client, "net-1", "http://sparkdeck-proxy:8899",
            "docker.io/library/nginx:latest", "amd64", copy_binds or [],
        )

    async def test_layers_data_sizes_are_aggregated(self):
        self.assertEqual(
            await self.total({"LayersData": [
                {"Size": 60, "Digest": "sha256:x"}, {"Size": 40}, {"Digest": "y"},
            ]}),
            100,
        )

    async def test_top_level_size_is_the_fallback(self):
        self.assertEqual(await self.total({"Size": 77}), 77)

    async def test_missing_sizes_read_as_unknown(self):
        self.assertEqual(await self.total({"LayersData": [{"Digest": "x"}]}), 0)
        self.assertEqual(await self.total({}), 0)

    async def test_preflight_dials_through_the_limiter_proxy(self):
        # The preflight shares the copy container's proxy environment: on
        # proxy-only hosts a direct connect cannot reach the registry, and
        # a direct one would also bypass the token bucket.
        await self.total({"Size": 10})
        created = inspect_kwargs(self.api)
        environment = created["environment"]
        self.assertEqual(environment["HTTPS_PROXY"], "http://sparkdeck-proxy:8899")
        self.assertEqual(environment["HTTP_PROXY"], "http://sparkdeck-proxy:8899")
        self.assertEqual(environment["NO_PROXY"], "localhost,127.0.0.1")

    async def test_preflight_keeps_credentials_and_ca_binds_only(self):
        # The staging archive must stay out of the preflight, but the
        # registry CA bind must come along for --tls-verify=true.
        binds = [
            "/staging:/archive:z",
            "/host/config.json:/auth/config.json:ro,z",
            "/sparkdeck-pull-cas-xyz:/etc/containers/certs.d:ro,z",
        ]
        await self.total({"Size": 10}, copy_binds=binds)
        created = inspect_kwargs(self.api)
        self.assertEqual(created["host_config"]["Binds"], [binds[1], binds[2]])


class RegistryCaBindsTests(unittest.TestCase):
    """Daemon certs.d copied into staging so binds stay colon-free."""

    @staticmethod
    def _write_ca(base, key):
        directory = os.path.join(base, key)
        os.mkdir(directory)
        path = os.path.join(directory, "ca.crt")
        with open(path, "w") as handle:
            handle.write("-----CERT-----")
        return path

    def test_ca_copies_into_staging_and_mounts_the_parent(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as staging:
            self._write_ca(source, "reg.corp")
            binds = registry_ca_binds("reg.corp", staging, bases=[source])
            self.assertEqual(
                binds, [f"{staging}:/etc/containers/certs.d:ro,z"],
            )
            self.assertTrue(os.path.isfile(
                os.path.join(staging, "reg.corp", "ca.crt"),
            ))

    @unittest.skipIf(os.name == "nt", "directory names cannot carry colons")
    def test_port_qualified_key_never_reaches_the_bind_spec(self):
        # A bind spec splits on colons: reg.corp:5443 in the source or the
        # target would be parsed as extra fields and fail container creation,
        # so the key must exist only INSIDE the mount.
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as staging:
            self._write_ca(source, "reg.corp:5443")
            binds = registry_ca_binds("reg.corp:5443", staging, bases=[source])
            self.assertEqual(
                binds, [f"{staging}:/etc/containers/certs.d:ro,z"],
            )
            self.assertNotIn("5443", binds[0])
            self.assertTrue(os.path.isfile(
                os.path.join(staging, "reg.corp:5443", "ca.crt"),
            ))

    @unittest.skipIf(os.name == "nt", "directory names cannot carry colons")
    def test_portless_operator_directory_lands_at_the_registry_key(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as staging:
            self._write_ca(source, "reg.corp")
            binds = registry_ca_binds("reg.corp:5443", staging, bases=[source])
            self.assertEqual(
                binds, [f"{staging}:/etc/containers/certs.d:ro,z"],
            )
            self.assertTrue(os.path.isfile(
                os.path.join(staging, "reg.corp:5443", "ca.crt"),
            ))

    def test_directory_without_certificates_yields_no_binds(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as staging:
            os.mkdir(os.path.join(source, "other.corp"))  # no .crt inside
            self.assertEqual(
                registry_ca_binds("other.corp", staging, bases=[source]), [],
            )

    def test_mtls_client_identity_is_copied_with_the_ca(self):
        # A TLS-authenticated registry keeps the client identity under the
        # exact names containers/image reads: client.cert/client.key. A
        # directory holding only those (or CA + identity) must stage them
        # all, or the helper's handshake presents no client certificate.
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as staging:
            directory = os.path.join(source, "reg.corp")
            os.mkdir(directory)
            for name in ("ca.crt", "client.cert", "client.key"):
                with open(os.path.join(directory, name), "w") as handle:
                    handle.write("x")
            binds = registry_ca_binds("reg.corp", staging, bases=[source])
            self.assertEqual(
                binds, [f"{staging}:/etc/containers/certs.d:ro,z"],
            )
            staged = sorted(os.listdir(os.path.join(staging, "reg.corp")))
            self.assertEqual(staged, ["ca.crt", "client.cert", "client.key"])

    def test_identity_only_directory_still_qualifies(self):
        # No CA at all (public root-signed registry) but mTLS required:
        # the identity files alone must produce the bind.
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as staging:
            directory = os.path.join(source, "reg.corp")
            os.mkdir(directory)
            for name in ("client.cert", "client.key"):
                with open(os.path.join(directory, name), "w") as handle:
                    handle.write("x")
            binds = registry_ca_binds("reg.corp", staging, bases=[source])
            self.assertEqual(
                binds, [f"{staging}:/etc/containers/certs.d:ro,z"],
            )
            self.assertTrue(os.path.isfile(
                os.path.join(staging, "reg.corp", "client.key"),
            ))

    def test_unrelated_files_do_not_qualify_a_directory(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as staging:
            directory = os.path.join(source, "reg.corp")
            os.mkdir(directory)
            for name in ("README.md", "client.pem"):
                with open(os.path.join(directory, name), "w") as handle:
                    handle.write("x")
            self.assertEqual(
                registry_ca_binds("reg.corp", staging, bases=[source]), [],
            )

    def test_missing_directories_yield_no_binds(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as staging:
            self.assertEqual(
                registry_ca_binds("nowhere.corp", staging, bases=[source]), [],
            )
            self.assertEqual(
                registry_ca_binds("", staging, bases=[source]), [],
            )

    def test_system_base_lookup_is_safe_when_absent(self):
        # Windows dev boxes have no /etc/docker/certs.d; the real-base call
        # must return nothing rather than raising.
        with tempfile.TemporaryDirectory() as staging:
            self.assertEqual(registry_ca_binds("reg.corp", staging), [])


class DaemonProxyConfigTests(unittest.TestCase):
    """docker info proxy settings mapped to host-env spellings (both APIs)."""

    def test_registry_config_shape_prefers_https_proxy(self):
        proxy, no_proxy = daemon_proxy_config({"RegistryConfig": {
            "HTTPProxy": "http://a:1", "HTTPSProxy": "http://b:2",
            "NoProxy": ["localhost", " .intra "],
        }})
        self.assertEqual(proxy, "http://b:2")
        self.assertEqual(no_proxy, "localhost,.intra")

    def test_top_level_proxies_shape(self):
        proxy, no_proxy = daemon_proxy_config({
            "Proxies": {"HTTPProxy": "http://a:1", "NoProxy": "x,y"},
        })
        self.assertEqual(proxy, "http://a:1")
        # NoProxy as a comma string is accepted too.
        self.assertEqual(no_proxy, "x,y")

    def test_httpsonly_spelling_variants(self):
        proxy, _ = daemon_proxy_config({
            "RegistryConfig": {"HttpProxy": "http://legacy:80"},
        })
        self.assertEqual(proxy, "http://legacy:80")

    def test_empty_daemon_reports_no_proxy(self):
        self.assertEqual(daemon_proxy_config({}), ("", ""))

    def test_top_level_daemon_fields_are_a_source(self):
        # The Engine API reports the daemon proxy as top-level /info fields
        # (HttpProxy/HttpsProxy/NoProxy) as well. A host configured only
        # through dockerd's daemon configuration or the systemd environment
        # leaves the nested objects empty, and a limiter without this source
        # would dial registries the network only reaches through the proxy.
        proxy, no_proxy = daemon_proxy_config({
            "HttpProxy": "http://proxy.corp:8080",
            "HttpsProxy": "https://proxy.corp:8443",
            "NoProxy": "localhost,.corp",
        })
        self.assertEqual(proxy, "https://proxy.corp:8443")
        self.assertEqual(no_proxy, "localhost,.corp")
        self.assertEqual(daemon_proxy_config(None), ("", ""))
        self.assertEqual(
            daemon_proxy_config({"RegistryConfig": {"NoProxy": [""]}}),
            ("", ""),
        )


class StagingParentTests(unittest.TestCase):
    def test_env_var_selects_the_parent_directory(self):
        with patch.dict(os.environ, {"SPARKDECK_IMAGE_STAGING_DIR": "/srv/large"}):
            self.assertEqual(staging_parent(), "/srv/large")

    def test_unset_env_var_uses_the_default_temp_parent(self):
        env = {k: v for k, v in os.environ.items()
               if k != "SPARKDECK_IMAGE_STAGING_DIR"}
        with patch.dict(os.environ, env, clear=True):
            self.assertIsNone(staging_parent())


class LoopbackRegistryTests(unittest.TestCase):
    def test_host_loopback_registries_are_detected(self):
        for image in (
            "localhost:5000/sparkdeck/sglang:latest",
            "localhost/sparkdeck/sglang",
            "127.0.0.1:5000/image:tag",
            "[::1]:5000/image:tag",
            "registry.localhost/sparkdeck/vllm:latest",
            "docker.io/library/nginx:latest".replace("docker.io", "LOCALHOST"),
        ):
            self.assertTrue(is_loopback_registry_image(image), image)

    def test_remote_and_unqualified_references_are_not_loopback(self):
        for image in (
            "nginx", "nginx:latest", "docker.io/library/nginx:latest",
            "ghcr.io/org/model:1.2", "10.0.0.5:5000/image:tag",
            "100.64.1.2:5000/image:tag",
            "docker.io/library/nginx@sha256:" + "a" * 64,
        ):
            self.assertFalse(is_loopback_registry_image(image), image)


if __name__ == "__main__":
    unittest.main()
