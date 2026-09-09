import io
import copy
import json
import tarfile
import unittest
from unittest.mock import AsyncMock, Mock

from docker.errors import APIError, ImageNotFound
from requests.exceptions import ConnectionError

from sparkdeck.image_patches import base_image_identity, build_patched_image, normalize_patch_request


BASE_ID = "sha256:" + "a" * 64
BUILT_ID = "sha256:" + "b" * 64


def request():
    return {"base_image": "local/runtime:base", "image": "local/runtime:patched",
            "files": [{"target": "/opt/model/patch.py", "content": "raise RuntimeError('never execute')\n"}]}


def archive_bytes(content):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        entry = tarfile.TarInfo("patch.py")
        entry.size = len(content)
        archive.addfile(entry, io.BytesIO(content))
    return buffer.getvalue()


class ImagePatchTests(unittest.TestCase):
    def fake_client(self, payload=None):
        payload = payload or request()
        client = Mock()
        base = Mock(id=BASE_ID, attrs={"Config": {"Entrypoint": ["vllm"]},
                                     "RootFS": {"Type": "layers", "Layers": ["sha256:" + "c" * 64]},
                                     "Os": "linux", "Architecture": "arm64", "Variant": "v8"})
        built = Mock(id=BUILT_ID)
        built.tag.return_value = True
        def get(reference):
            if reference == payload["base_image"]:
                return base
            if reference.startswith("sparkdeck-patch-build:"):
                return built
            raise ImageNotFound(reference)
        client.images.get.side_effect = get
        client.images.pull.return_value = base
        client.api.build.return_value = iter([{"stream": "Built image\n"}])
        client.containers.create.return_value.get_archive.return_value = (
            iter([archive_bytes(payload["files"][0]["content"].encode())]), {})
        return client, base, built

    def test_build_pins_local_base_copies_only_and_verifies_without_execution(self):
        payload = request()
        # Quotes and spaces remain one JSON destination, never Dockerfile syntax.
        payload["files"][0]["target"] = '/opt/patch "special".py'
        client, _, built = self.fake_client(payload)
        captured = {}
        def build(**kwargs):
            with tarfile.open(fileobj=kwargs["fileobj"], mode="r") as archive:
                captured["dockerfile"] = archive.extractfile("Dockerfile").read().decode()
                captured["patch"] = archive.extractfile("patch-0").read()
                captured["mode"] = archive.getmember("patch-0").mode
            return iter([{"stream": "COPY patch-0\n"}])
        client.api.build.side_effect = build
        result = build_patched_image(client, payload)
        self.assertEqual(result["base_id"], BASE_ID)
        self.assertEqual(result["image_id"], BUILT_ID)
        self.assertEqual(len(result["files"][0]["sha256"]), 64)
        lines = captured["dockerfile"].splitlines()
        self.assertEqual(lines[0], "FROM " + BASE_ID)
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[1][5:]), ["patch-0", payload["files"][0]["target"]])
        self.assertEqual(captured["mode"], 0o644)
        self.assertEqual(captured["patch"], payload["files"][0]["content"].encode())
        client.images.pull.assert_not_called()
        container = client.containers.create.return_value
        container.start.assert_not_called()
        container.exec_run.assert_not_called()
        container.remove.assert_called_once_with(v=True, force=True)
        built.tag.assert_called_once_with("local/runtime", tag="patched", force=False)

    def test_pulls_missing_base(self):
        client, _, _ = self.fake_client()
        original = client.images.get.side_effect
        client.images.get.side_effect = lambda ref: (_ for _ in ()).throw(ImageNotFound(ref)) if ref == request()["base_image"] else original(ref)
        build_patched_image(client, request())
        client.images.pull.assert_called_once_with(request()["base_image"])

    def test_rejects_existing_tag_before_build(self):
        client, _, _ = self.fake_client()
        client.images.get.side_effect = None
        with self.assertRaisesRegex(ValueError, "already exists"):
            build_patched_image(client, request())
        client.api.build.assert_not_called()

    def test_never_runs_base_onbuild_commands(self):
        client, base, _ = self.fake_client()
        base.attrs["Config"]["OnBuild"] = ["RUN python /patch.py"]
        with self.assertRaisesRegex(ValueError, "ONBUILD"):
            build_patched_image(client, request())
        client.api.build.assert_not_called()

    def test_cluster_base_mismatch_fails_before_build(self):
        client, _, _ = self.fake_client()
        with self.assertRaisesRegex(ValueError, "differs"):
            build_patched_image(client, {**request(), "expected_base_id": BUILT_ID})
        client.api.build.assert_not_called()

    def test_cross_store_ids_with_same_runtime_contents_build_using_local_id(self):
        client, base, _ = self.fake_client()
        first = copy.deepcopy(base.attrs)
        first["Config"].update(User="", WorkingDir="", AttachStdin=False, Labels=None)
        first.update(Id="sha256:9db7b59979c38555a39def84a31fb98b5296952f9e3afd4f6f11f05b07adfab0",
                     Created="2026-01-01", RepoTags=["busybox:1.37.0"], Size=4000000)
        identity = base_image_identity(Mock(attrs=first))
        base.id = "sha256:6df9636795d37473994366014c25264edeb6c00d7a57188ff62d5a94276b4297"
        captured = []
        def build(**kwargs):
            with tarfile.open(fileobj=kwargs["fileobj"], mode="r") as archive:
                captured.append(archive.extractfile("Dockerfile").read().decode())
            return iter([])
        client.api.build.side_effect = build
        result = build_patched_image(client, {**request(), "expected_base_identity": identity})
        self.assertEqual(result["base_identity"], identity)
        self.assertEqual(result["base_id"], base.id)
        self.assertTrue(captured[0].startswith("FROM " + base.id + "\n"))
        # A legacy coordinator's exact ID constraint remains authoritative.
        with self.assertRaisesRegex(ValueError, "differs"):
            build_patched_image(client, {**request(), "expected_base_id": BASE_ID,
                                         "expected_base_identity": identity})

    def test_changed_layers_runtime_config_or_platform_reject_before_build(self):
        for change in ("layers", "layer_order", "entrypoint", "env", "architecture", "variant", "volume", "port", "healthcheck", "user"):
            with self.subTest(change=change):
                client, base, _ = self.fake_client()
                base.attrs["RootFS"]["Layers"].append("sha256:" + "d" * 64)
                identity = base_image_identity(base)
                if change == "layers":
                    base.attrs["RootFS"]["Layers"][0] = "sha256:" + "e" * 64
                elif change == "layer_order":
                    base.attrs["RootFS"]["Layers"].reverse()
                elif change == "architecture":
                    base.attrs["Architecture"] = "amd64"
                elif change == "variant":
                    base.attrs["Variant"] = "v9"
                else:
                    field, value = {"entrypoint": ("Entrypoint", ["other"]), "env": ("Env", ["FOO=bar"]),
                                    "volume": ("Volumes", {"/opt/patch": {}}), "port": ("ExposedPorts", {"8000/tcp": {}}),
                                    "healthcheck": ("Healthcheck", {"Test": ["CMD", "true"]}), "user": ("User", "1000")}[change]
                    base.attrs["Config"][field] = value
                with self.assertRaisesRegex(ValueError, "differs"):
                    build_patched_image(client, {**request(), "expected_base_identity": identity})
                client.api.build.assert_not_called()

    def test_missing_or_malformed_identity_metadata_fails_closed(self):
        for missing in ("Config", "RootFS", "Os", "Architecture"):
            with self.subTest(missing=missing):
                client, base, _ = self.fake_client()
                base.attrs.pop(missing)
                with self.assertRaises(ValueError):
                    build_patched_image(client, request())
                client.api.build.assert_not_called()
        for layers in (None, ["not-a-digest"], "sha256:" + "c" * 64):
            client, base, _ = self.fake_client()
            base.attrs["RootFS"]["Layers"] = layers
            with self.assertRaisesRegex(ValueError, "layer identities"):
                build_patched_image(client, request())
            client.api.build.assert_not_called()

    def test_runtime_identity_constraint_requires_known_version_and_digest(self):
        for invalid in (None, BASE_ID, "runtime-v2:sha256:" + "a" * 64, "runtime-v1:sha256:short"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "runtime fingerprint"):
                normalize_patch_request({**request(), "expected_base_identity": invalid})

    def test_wrong_content_is_not_published_and_container_is_removed(self):
        client, _, built = self.fake_client()
        container = client.containers.create.return_value
        container.get_archive.return_value = (iter([archive_bytes(b"wrong")]), {})
        with self.assertRaisesRegex(ValueError, "did not match"):
            build_patched_image(client, request())
        built.tag.assert_not_called()
        container.remove.assert_called_once()
        client.images.remove.assert_called_once()

    def test_build_errors_are_not_published(self):
        client, _, built = self.fake_client()
        client.api.build.return_value = iter([{"errorDetail": {"message": "COPY failed"}}])
        with self.assertRaisesRegex(ValueError, "COPY failed"):
            build_patched_image(client, request())
        built.tag.assert_not_called()
        client.containers.create.assert_not_called()

    def test_cleanup_failures_do_not_turn_published_image_into_failed_build(self):
        client, _, built = self.fake_client()
        client.images.remove.side_effect = APIError("cleanup failed")
        client.containers.create.return_value.remove.side_effect = APIError("cleanup failed")
        logs = []
        result = build_patched_image(client, request(), logs.append)
        self.assertEqual(result["image_id"], BUILT_ID)
        built.tag.assert_called_once()
        self.assertEqual(len([line for line in logs if line.startswith("Warning:")]), 2)

    def test_tag_cleanup_failure_preserves_original_build_error(self):
        client, _, built = self.fake_client()
        client.api.build.return_value = iter([{"error": "original build failure"}])
        client.images.remove.side_effect = APIError("cleanup failed")
        logs = []
        with self.assertRaisesRegex(ValueError, "original build failure"):
            build_patched_image(client, request(), logs.append)
        built.tag.assert_not_called()
        self.assertTrue(any(line.startswith("Warning:") for line in logs))

    def test_container_cleanup_failure_preserves_original_verification_error(self):
        client, _, built = self.fake_client()
        container = client.containers.create.return_value
        container.get_archive.return_value = (iter([archive_bytes(b"wrong")]), {})
        container.remove.side_effect = APIError("cleanup failed")
        logs = []
        with self.assertRaisesRegex(ValueError, "did not match"):
            build_patched_image(client, request(), logs.append)
        built.tag.assert_not_called()
        self.assertTrue(any(line.startswith("Warning:") for line in logs))

    def test_transport_cleanup_failures_preserve_success_and_original_errors(self):
        for failure in (None, "build", "verification"):
            with self.subTest(failure=failure):
                client, _, built = self.fake_client()
                client.images.remove.side_effect = ConnectionError("Docker disconnected")
                container = client.containers.create.return_value
                container.remove.side_effect = ConnectionError("Docker disconnected")
                logs = []
                if failure == "build":
                    client.api.build.return_value = iter([{"error": "original build failure"}])
                    with self.assertRaisesRegex(ValueError, "original build failure"):
                        build_patched_image(client, request(), logs.append)
                    built.tag.assert_not_called()
                elif failure == "verification":
                    container.get_archive.return_value = (iter([archive_bytes(b"wrong")]), {})
                    with self.assertRaisesRegex(ValueError, "did not match"):
                        build_patched_image(client, request(), logs.append)
                    built.tag.assert_not_called()
                else:
                    result = build_patched_image(client, request(), logs.append)
                    self.assertEqual(result["image_id"], BUILT_ID)
                    built.tag.assert_called_once()
                warnings = [line for line in logs if line.startswith("Warning:")]
                self.assertEqual(len(warnings), 1 if failure == "build" else 2)

    def test_another_writer_claiming_tag_during_build_is_rejected(self):
        client, _, built = self.fake_client()
        original = client.images.get.side_effect
        count = 0
        def get(ref):
            nonlocal count
            if ref == request()["image"]:
                count += 1
                if count == 2:
                    return Mock()
            return original(ref)
        client.images.get.side_effect = get
        with self.assertRaisesRegex(ValueError, "already exists"):
            build_patched_image(client, request())
        built.tag.assert_not_called()

    def test_validation_rejects_injection_traversal_overlap_and_binary(self):
        for target in ["relative.py", "/", "/a/../b", "/a//b", "/a/./b", "/a/", "//a", "/a\nRUN evil", "/dev/a", "/sys/a", "/proc/a", "/a\\b"]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                normalize_patch_request({**request(), "files": [{"target": target, "content": "x"}]})
        for updates in [{"base_image": "a\nRUN evil"}, {"image": "a"},
                        {"image": request()["base_image"]}, {"files": []},
                        {"expected_base_id": "invalid"}, {"run": "python foo"},
                        {"files": [{"target": "/a", "content": "\x00"}]},
                        {"files": [{"target": "/a", "content": "\ud800"}]},
                        {"files": [{"target": "/a", "content": "x"}, {"target": "/a/b", "content": "x"}]}]:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                normalize_patch_request({**request(), **updates})

    def test_limits_are_utf8_bytes_per_file_and_total(self):
        with self.assertRaisesRegex(ValueError, "1 MiB"):
            normalize_patch_request({**request(), "files": [{"target": "/a", "content": "é" * (512 * 1024 + 1)}]})
        with self.assertRaisesRegex(ValueError, "4 MiB"):
            normalize_patch_request({**request(), "files": [{"target": f"/a{i}", "content": "x" * (1024 * 1024)} for i in range(5)]})


class PatchedImageLaunchTests(unittest.IsolatedAsyncioTestCase):
    async def test_locally_built_image_launches_single_and_tp_worker_without_pull(self):
        from manager import Manager

        for cluster_member in (None, {"deployment_id": "patched", "node_id": "node-3",
                                     "rank": 1, "nnodes": 2, "mode": "sharded"}):
            with self.subTest(cluster_member=cluster_member):
                manager = Manager.__new__(Manager)
                manager.settings = {"vllm_image": "unused/default:latest", "hf_cache": "/cache",
                                    "shm_size": "1g", "default_gpu_memory_utilization": 0.65}
                manager.client = Mock()
                manager._gpu_total_gb = AsyncMock(return_value=122.0)
                manager._try_fit_new_model = Mock()
                manager._read_gpu_memory_gb = Mock(return_value=(0.0, 0.0))
                manager._cluster_launch_update = Mock()
                manager._build_volumes = Mock(return_value={})
                manager._container_hf_environment = Mock(return_value={})
                manager._created_container_model_source = Mock(return_value="public_repository")
                manager._run_managed_container = Mock()
                manager._container_summary = Mock(return_value={"name": "patched"})
                await manager.create_container("org/model", port=8009, image=request()["image"],
                                               name="patched", cluster_member=cluster_member,
                                               infiniband_device=False)
                manager.client.images.get.assert_called_with(request()["image"])
                manager.client.images.pull.assert_not_called()
                options = manager._run_managed_container.call_args.args[0]
                self.assertEqual(options["image"], request()["image"])
                if cluster_member:
                    self.assertEqual(options["network_mode"], "host")
                else:
                    self.assertEqual(options["ports"], {"8000/tcp": 8009})


if __name__ == "__main__":
    unittest.main()
