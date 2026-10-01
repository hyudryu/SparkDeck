"""Regression coverage for Hugging Face's cache-wide Xet blob store."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sparkdeck.virtual_nas import VirtualNAS, _normalize_shared_hub_blobs


REVISION = "a" * 40


def shared_model(hub: Path):
    repository = hub / "models--org--model"
    local_blobs = repository / "blobs"
    local_blobs.mkdir(parents=True)
    snapshot = repository / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    shared = hub / "blobs"
    shared.mkdir(exist_ok=True)
    (shared / ".huggingface-shared-blobs").write_bytes(b"1\n")
    payloads = {}
    for index, (name, content) in enumerate([
        ("config.json", b"{}"), ("tokenizer.json", b"{}"),
        ("model.safetensors", b"actual-model-weights"),
    ], 1):
        digest = f"{index:064x}"
        target = shared / digest[:2] / digest
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(content)
        blob = local_blobs / digest
        blob.symlink_to(Path("..") / ".." / "blobs" / digest[:2] / digest)
        (snapshot / name).symlink_to(Path("..") / ".." / "blobs" / digest)
        payloads[name] = (blob, target, content)
    return repository, payloads


class SharedHubCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.hub = self.root / "hub"
        try:
            self.repository, self.payloads = shared_model(self.hub)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        self.nas = VirtualNAS(self.root / "state", lambda: self.hub, Mock(), lambda: True)

    def test_inventory_recognizes_existing_shared_weights_without_payload_copy(self):
        model = self.nas.inventory()[0]
        self.assertFalse(model["partial"])
        self.assertIn(REVISION, model["revisions"])
        self.assertGreaterEqual(model["size_bytes"], len(b"actual-model-weights"))
        for blob, target, content in self.payloads.values():
            self.assertFalse(blob.is_symlink())
            self.assertTrue(os.path.samefile(blob, target))
            self.assertEqual(target.read_bytes(), content)
        before = {blob: blob.stat().st_ino for blob, _, _ in self.payloads.values()}
        self.nas.inventory()
        self.assertEqual(before, {blob: blob.stat().st_ino for blob in before})

    def test_completed_download_does_not_redownload_shared_weights(self):
        with patch.dict("sys.modules", {"huggingface_hub": Mock()}) as modules:
            result = self.nas.download_model("org/model", REVISION)
            modules["huggingface_hub"].snapshot_download.assert_not_called()
        self.assertTrue(result["ok"])

    def test_fresh_download_normalizes_shared_blobs_before_completion(self):
        fresh_hub = self.root / "fresh-hub"
        nas = VirtualNAS(self.root / "fresh-state", lambda: fresh_hub, Mock(), lambda: True)
        hub_api = Mock()
        hub_api.snapshot_download.side_effect = lambda **kwargs: shared_model(Path(kwargs["cache_dir"]))
        with patch.dict("sys.modules", {"huggingface_hub": hub_api}):
            result = nas.download_model("org/model", REVISION)
        self.assertTrue(result["ok"])
        hub_api.snapshot_download.assert_called_once()
        self.assertFalse(nas.inventory()[0]["partial"])

    def test_inventory_normalizes_once_per_repository_not_once_per_file(self):
        with patch("sparkdeck.virtual_nas._normalize_shared_hub_blobs", wraps=_normalize_shared_hub_blobs) as normalize:
            self.nas.inventory()
            self.assertEqual(normalize.call_count, 1)
            normalize.reset_mock()
            self.nas.inventory()
            self.assertEqual(normalize.call_count, 1)

    def test_selected_file_check_repairs_existing_shared_cache(self):
        result = self.nas.has_model_files("org/model", REVISION, ["model.safetensors"])
        self.assertTrue(result["complete"])

    @unittest.skipIf(os.name == "nt", "POSIX symlink stream roundtrip requires Linux")
    async def test_whole_model_export_is_portable_without_shared_store(self):
        target_hub = self.root / "target-hub"
        target = VirtualNAS(self.root / "target-state", lambda: target_hub, Mock(), lambda: True)
        result = await target.import_model("org/model", self.nas.export_model("org/model"))
        self.assertTrue(result["ok"])
        self.assertFalse(target.inventory()[0]["partial"])
        snapshot = target_hub / self.repository.name / "snapshots" / REVISION
        self.assertEqual((snapshot / "model.safetensors").read_bytes(), b"actual-model-weights")
        self.assertFalse((target_hub / "blobs").exists())

    def test_partial_shared_cache_keeps_resume_revision_and_bytes(self):
        (self.repository / "snapshots" / REVISION / "tokenizer.json").unlink()
        model = self.nas.inventory()[0]
        self.assertTrue(model["partial"])
        self.assertIn(REVISION, model["partial_revisions"])
        self.assertGreaterEqual(model["partial_revision_size_bytes"][REVISION], len(b"actual-model-weights"))

    def test_unmarked_shared_store_stays_untrusted(self):
        (self.hub / "blobs" / ".huggingface-shared-blobs").unlink()
        _normalize_shared_hub_blobs(self.repository)
        self.assertTrue(all(blob.is_symlink() for blob, _, _ in self.payloads.values()))
        self.assertTrue(self.nas.inventory()[0]["partial"])

    def test_shared_payload_symlink_cannot_import_outside_file(self):
        blob, target, _ = self.payloads["model.safetensors"]
        outside = self.root / "outside"
        outside.write_bytes(b"private-content")
        target.unlink()
        target.symlink_to(outside)
        _normalize_shared_hub_blobs(self.repository)
        self.assertTrue(blob.is_symlink())
        self.assertTrue(self.nas.inventory()[0]["partial"])

    def test_symlinked_shared_prefix_cannot_import_outside_file(self):
        # All fixture hashes share a prefix; redirect the entire directory.
        prefix = self.hub / "blobs" / "00"
        outside = self.root / "outside-prefix"
        prefix.rename(outside)
        prefix.symlink_to(outside, target_is_directory=True)
        _normalize_shared_hub_blobs(self.repository)
        self.assertTrue(all(blob.is_symlink() for blob, _, _ in self.payloads.values()))

    def test_arbitrary_blob_pointer_is_not_normalized(self):
        blob, _, _ = self.payloads["model.safetensors"]
        outside = self.root / "outside"
        outside.write_bytes(b"private-content")
        blob.unlink()
        blob.symlink_to(outside)
        _normalize_shared_hub_blobs(self.repository)
        self.assertTrue(blob.is_symlink())
        self.assertTrue(self.nas.inventory()[0]["partial"])

    def test_link_failure_leaves_original_cache_intact_and_cleans_staging(self):
        with patch("sparkdeck.virtual_nas.os.link", side_effect=OSError("read-only")):
            _normalize_shared_hub_blobs(self.repository)
        self.assertTrue(all(blob.is_symlink() for blob, _, _ in self.payloads.values()))
        self.assertEqual(len(list((self.repository / "blobs").iterdir())), 3)
