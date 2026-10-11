import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import sparkdeck.throttled_image_pull as tip
from sparkdeck.throttled_image_pull import (
    PROXY_IMAGE,
    SKOPEO_IMAGE,
    ThrottledPullError,
    _progress_event,
    _read_event_lines,
    normalize_image_reference,
    select_docker_config,
    skopeo_copy_args,
    throttled_image_pull_events,
)


class NormalizeImageReferenceTests(unittest.TestCase):
    def test_docker_hub_expansions_and_default_tag(self):
        self.assertEqual(normalize_image_reference("nginx"), (
            "docker://docker.io/library/nginx:latest",
            "docker-archive:/archive/image.tar:docker.io/library/nginx:latest",
        ))
        self.assertEqual(normalize_image_reference("org/app"), (
            "docker://docker.io/org/app:latest",
            "docker-archive:/archive/image.tar:docker.io/org/app:latest",
        ))
        self.assertEqual(normalize_image_reference("org/app:1.2"), (
            "docker://docker.io/org/app:1.2",
            "docker-archive:/archive/image.tar:docker.io/org/app:1.2",
        ))

    def test_registry_hosted_references_keep_their_host(self):
        self.assertEqual(normalize_image_reference("ghcr.io/sparkdeck/x"), (
            "docker://ghcr.io/sparkdeck/x:latest",
            "docker-archive:/archive/image.tar:ghcr.io/sparkdeck/x:latest",
        ))
        source, archive = normalize_image_reference("localhost:5000/x")
        self.assertEqual(source, "docker://localhost:5000/x:latest")
        self.assertTrue(archive.endswith(":latest"))

    def test_digest_reference_carries_no_tag_into_the_archive(self):
        source, archive = normalize_image_reference("nginx@sha256:" + "a" * 64)
        self.assertEqual(source, f"docker://docker.io/library/nginx@sha256:{'a' * 64}")
        self.assertEqual(archive, "docker-archive:/archive/image.tar")

    def test_invalid_references_are_rejected(self):
        for bad in ("", "  ", "two words", "docker://nginx", "nginx@"):
            with self.assertRaises(ThrottledPullError, msg=bad):
                normalize_image_reference(bad)


class SkopeoCopyArgsTests(unittest.TestCase):
    def test_base_copy_invocation(self):
        args = skopeo_copy_args(
            "docker://docker.io/library/nginx:latest",
            "docker-archive:/archive/image.tar:docker.io/library/nginx:latest",
            "amd64", has_authfile=False,
        )
        self.assertEqual(args[0], "copy")
        self.assertIn("--src-tls-verify=true", args)
        self.assertIn("--format-event-data", args)
        self.assertEqual(args[args.index("--override-arch") + 1], "amd64")
        self.assertNotIn("--authfile", args)
        self.assertEqual(args[-2:], [
            "docker://docker.io/library/nginx:latest",
            "docker-archive:/archive/image.tar:docker.io/library/nginx:latest",
        ])

    def test_authfile_and_missing_arch(self):
        args = skopeo_copy_args(
            "docker://a", "docker-archive:/archive/image.tar", "", has_authfile=True,
        )
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


class ReadEventLinesTests(unittest.TestCase):
    def test_parses_complete_lines_only_and_advances_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_bytes(b'{"type": "full"}\n{"type": "progress"}\n{"par')
            events, offset = _read_event_lines(path, 0)
            self.assertEqual([event["type"] for event in events], ["full", "progress"])
            self.assertEqual(offset, len(b'{"type": "full"}\n{"type": "progress"}\n'))
            # The partial trailing line is not consumed...
            events, offset = _read_event_lines(path, offset)
            self.assertEqual(events, [])
            # ...and becomes readable once the writer finishes it.
            with open(path, "ab") as handle:
                handle.write(b'tial": 1}\n')
            events, offset = _read_event_lines(path, offset)
            self.assertEqual(events, [{"partial": 1}])

    def test_missing_file_is_quiet(self):
        events, offset = _read_event_lines(
            Path(tempfile.gettempdir()) / "sparkdeck-absent-events.jsonl", 5,
        )
        self.assertEqual(events, [])
        self.assertEqual(offset, 5)


class ProgressEventMappingTests(unittest.TestCase):
    def test_full_event_provides_total(self):
        mapped = _progress_event({
            "type": "full",
            "data": {"id": "layer-" + "x" * 40, "bytes": 1000},
        })
        self.assertEqual(mapped["progressDetail"], {"current": 0, "total": 1000})
        self.assertEqual(len(mapped["id"]), 19)  # docker-style truncated id

    def test_progress_event_provides_offset(self):
        mapped = _progress_event({
            "type": "progress", "data": {"id": "layer", "offset": 42},
        })
        self.assertEqual(mapped["progressDetail"], {"current": 42})

    def test_unmappable_events_are_skipped(self):
        for event in (
            {"type": "full", "data": {}},
            {"type": "full", "data": {"bytes": 0}},
            {"type": "progress", "data": {"offset": 0}},
            {"type": "done", "data": {"id": "layer"}},
            {"type": "x"},
            {"no-data": True},
        ):
            self.assertIsNone(_progress_event(event), event)


class FakeDockerApi:
    """Minimal docker-py APIClient stand-in for the helper orchestration."""

    def __init__(self, *, copy_exit=0, proxy_dies=False, stderr_tail=b"fatal: copy failed"):
        self.copy_exit = copy_exit
        self.proxy_dies = proxy_dies
        self.stderr_tail = stderr_tail
        self.loaded = None
        self.created: list[dict] = []
        self.removed: list[str] = []
        self.removed_networks: list[str] = []
        self.pulls: list[str] = []
        self.networks: list[dict] = []
        self.on_start = None
        self._containers: dict[str, str] = {}
        self._copy_polls = 0

    # -- API surface used by the module (dispatched via asyncio.to_thread) --
    def version(self):
        return {"Arch": "amd64"}

    def inspect_image(self, ref):
        return {"Id": f"sha256:{ref}"}

    def pull(self, ref):
        self.pulls.append(ref)

    def create_network(self, name, driver):
        self.networks.append({"name": name, "driver": driver})
        return {"Id": "net-1"}

    def remove_network(self, net_id):
        self.removed_networks.append(net_id)

    def create_container(self, **kwargs):
        self.created.append(kwargs)
        role = "proxy" if kwargs.get("image") == PROXY_IMAGE else "copy"
        self._containers[f"{role}-1"] = role
        return {"Id": f"{role}-1"}

    def start_container(self, container_id):
        if self.on_start is not None:
            self.on_start(container_id)

    def exec_create(self, container, cmd):
        return {"Id": "exec-1"}

    def exec_start(self, exec_id, demux=False):
        return b""

    def exec_inspect(self, exec_id):
        return {"ExitCode": 0}

    def inspect_container(self, container_id):
        if self._containers.get(container_id) == "proxy":
            running = not self.proxy_dies
            return {"State": {"Running": running, "ExitCode": 0 if running else 137}}
        self._copy_polls += 1
        if self._copy_polls < 2:
            return {"State": {"Running": True, "ExitCode": 0}}
        return {"State": {"Running": False, "ExitCode": self.copy_exit}}

    def container_logs(self, container_id, stderr=True, stdout=False, tail=12):
        return self.stderr_tail

    def load_image(self, stream):
        self.loaded = stream.read()

    def remove_container(self, container_id, v=False, force=False):
        self.removed.append(container_id)


class ThrottledPullFlowTests(unittest.IsolatedAsyncioTestCase):
    async def collect(self, api, image="nginx", bps=1_000_000.0,
                      staging=None, authfile=None):
        client = Mock()
        client.api = api
        staging = staging or Path(tempfile.mkdtemp())
        events = []
        with (
            # The flow tests own registry auth explicitly; the developer
            # machine's own ~/.docker/config.json must not leak in.
            patch.object(tip, "select_docker_config", return_value=authfile),
            patch.object(tip.tempfile, "mkdtemp", return_value=str(staging)),
        ):
            async for event in throttled_image_pull_events(client, image, bps):
                events.append(event)
        return events, staging

    async def test_success_stream_maps_progress_and_loads_archive(self):
        layer = "sha256:deadbeefdeadbeefdeadbeef"
        api = FakeDockerApi()
        staging = Path(tempfile.mkdtemp())

        def on_start(container_id):
            if container_id != "copy-1":
                return
            # Stand in for the skopeo container writing into its /archive bind.
            staging.joinpath("events.jsonl").write_bytes(
                (json.dumps({"type": "full", "data": {"id": layer, "bytes": 50}}) + "\n"
                 + json.dumps({"type": "progress",
                               "data": {"id": layer, "offset": 50}}) + "\n").encode()
            )
            staging.joinpath("image.tar").write_bytes(b"tarbytes")

        api.on_start = on_start
        events, _ = await self.collect(api, staging=staging)

        self.assertEqual(events[0]["status"],
                         "Preparing bandwidth-limited pull of nginx")
        self.assertIn({"id": layer[:19], "status": "Downloading",
                       "progressDetail": {"current": 0, "total": 50}}, events)
        self.assertIn({"id": layer[:19], "status": "Downloading",
                       "progressDetail": {"current": 50}}, events)
        self.assertEqual(events[-1], {"status": "Successfully pulled nginx"})
        self.assertNotIn("error", json.dumps(events))
        self.assertEqual(api.loaded, b"tarbytes")
        self.assertEqual(sorted(api.removed), ["copy-1", "proxy-1"])
        self.assertEqual(api.removed_networks, ["net-1"])

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

    async def test_credential_helper_config_is_a_clear_error(self):
        api = FakeDockerApi()
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"credsStore": "desktop"}), encoding="utf-8")
            events, _ = await self.collect(
                api, image="docker.io/private/app:1", bps=1000.0, authfile=str(config),
            )
        self.assertIsNone(api.loaded)
        self.assertIn("credential helper", events[-1]["error"])
        # The network created before the auth check must still be cleaned up.
        self.assertEqual(api.removed_networks, ["net-1"])

    async def test_helper_containers_are_wired_to_the_proxy(self):
        api = FakeDockerApi()
        await self.collect(api, bps=1_000_000.0)
        proxy = next(c for c in api.created if c.get("image") == PROXY_IMAGE)
        copy = next(c for c in api.created if c.get("image") == SKOPEO_IMAGE)
        self.assertEqual(proxy["environment"]["SPARKDECK_DOWNLOAD_BPS"], "1000000.0")
        self.assertEqual(proxy["host_config"]["NetworkMode"], "net-1")
        self.assertEqual(
            copy["environment"]["HTTPS_PROXY"],
            f"http://{proxy['name']}:8899",
        )
        self.assertEqual(copy["host_config"]["NetworkMode"], "net-1")
        self.assertTrue(any("image.tar" in " ".join(c["command"])
                            for c in api.created if c.get("image") == SKOPEO_IMAGE))


if __name__ == "__main__":
    unittest.main()
