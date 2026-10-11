import json
import os
import subprocess
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import Mock, patch

import sparkdeck.throttled_image_pull as tip
from sparkdeck.throttled_image_pull import (
    PROGRESS_NAME,
    PROXY_IMAGE,
    SKOPEO_IMAGE,
    ThrottledPullError,
    _auth_entry,
    _read_counter,
    helper_credentials,
    is_loopback_registry_image,
    normalize_image_reference,
    select_docker_config,
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
                 load_events=None, verify_fails=False, inspect_size=100):
        self.copy_exit = copy_exit
        self.proxy_dies = proxy_dies
        self.stderr_tail = stderr_tail
        self.load_events = load_events if load_events is not None else [
            {"stream": "Loaded image: docker.io/library/nginx:latest"},
        ]
        self.verify_fails = verify_fails
        self.inspect_size = inspect_size
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

    def inspect_image(self, ref):
        # Helper images always exist (no pull); the requested image resolves
        # only after a successful load unless verify_fails is set.
        if ref in (PROXY_IMAGE, SKOPEO_IMAGE):
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
    async def collect(self, api, image="nginx", bps=1_000_000.0,
                      staging=None, authfile=None, auth_dir=None,
                      disk_free=10 ** 15):
        client = Mock()
        client.api = api
        staging = staging or Path(tempfile.mkdtemp())
        mkdtemp_values = [str(staging)] + ([str(auth_dir)] if auth_dir else [])
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
            async for event in throttled_image_pull_events(client, image, bps):
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
        self.assertTrue(any(bind.endswith(":/auth/config.json:ro")
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


class ImageTotalBytesTests(unittest.IsolatedAsyncioTestCase):
    async def total(self, payload):
        api = FakeDockerApi()
        base_logs = api.container_logs

        def logs(container_id, stderr=True, stdout=False, tail=12):
            if stdout and not stderr:
                return json.dumps(payload).encode()
            return base_logs(container_id, stderr=stderr, stdout=stdout, tail=tail)

        api.container_logs = logs
        client = Mock()
        client.api = api
        return await tip._image_total_bytes(
            client, "net-1", "docker.io/library/nginx:latest", "amd64", [],
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
