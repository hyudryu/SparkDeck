import os
import tempfile
import time
import unittest
from pathlib import Path

from disk_manager import DiskScanJobs, browse_directories, delete_entries, scan_directory


class DiskManagerTests(unittest.TestCase):
    def test_scan_reports_files_and_recursive_directory_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "models" / "quantized"
            nested.mkdir(parents=True)
            (root / "notes.txt").write_bytes(b"abc")
            (nested / "model.gguf").write_bytes(b"1234567")

            result = scan_directory(str(root))
            by_path = {item["path"]: item for item in result["entries"]}

            self.assertEqual(result["total_size"], 10)
            self.assertEqual(result["file_count"], 2)
            self.assertEqual(result["directory_count"], 2)
            self.assertEqual(by_path["models"]["size"], 7)
            self.assertEqual(by_path[os.path.join("models", "quantized")]["size"], 7)
            self.assertEqual(by_path["notes.txt"]["type"], "file")

    def test_scan_emits_files_and_growing_folder_sizes_while_running(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "models"
            folder.mkdir()
            (folder / "one.bin").write_bytes(b"123")
            (folder / "two.bin").write_bytes(b"4567")
            changes = []

            scan_directory(str(root), lambda entry, _: changes.append(entry) if entry else None)

            model_sizes = [item["size"] for item in changes if item["path"] == "models"]
            discovered = {item["path"] for item in changes}
            self.assertEqual(model_sizes[0], 0)
            self.assertEqual(model_sizes[-1], 7)
            self.assertIn("models/one.bin", discovered)
            self.assertIn("models/two.bin", discovered)

    def test_background_scan_poll_returns_incremental_changes_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cache").mkdir()
            for index in range(20):
                (root / "cache" / f"{index}.bin").write_bytes(b"x" * (index + 1))
            jobs = DiskScanJobs()
            started = jobs.start(str(root))
            cursor = 0
            entries = {}
            deadline = time.time() + 2

            while time.time() < deadline:
                update = jobs.poll(started["scan_id"], cursor, limit=5)
                cursor = update["next_cursor"]
                entries.update({item["path"]: item for item in update["changes"]})
                if update["complete"]:
                    break
                time.sleep(0.005)

            self.assertTrue(update["complete"])
            self.assertEqual(update["status"], "complete")
            self.assertEqual(update["progress_pct"], 100.0)
            self.assertEqual(entries["cache"]["size"], 210)
            self.assertEqual(update["file_count"], 20)

    def test_scan_does_not_follow_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            outside_file = Path(outside) / "outside.bin"
            outside_file.write_bytes(b"outside contents")
            (root / "link").symlink_to(outside_file)

            result = scan_directory(str(root))

            self.assertEqual(result["file_count"], 1)
            self.assertEqual(result["entries"][0]["type"], "symlink")
            self.assertNotEqual(result["total_size"], outside_file.stat().st_size)

    def test_browse_lists_only_real_child_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "beta").mkdir()
            (root / "Alpha").mkdir()
            (root / "file.txt").touch()
            (root / "linked").symlink_to(root / "beta", target_is_directory=True)

            result = browse_directories(str(root))

            self.assertEqual([item["name"] for item in result["directories"]], ["Alpha", "beta"])

    def test_delete_permanently_removes_files_and_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "old" / "nested").mkdir(parents=True)
            (root / "old" / "nested" / "weights.bin").write_bytes(b"123")
            (root / "loose.log").write_text("log", encoding="utf-8")

            result = delete_entries(str(root), ["old", "old/nested/weights.bin", "loose.log"])

            self.assertCountEqual(result["deleted"], ["old", "loose.log"])
            self.assertEqual(result["errors"], [])
            self.assertFalse((root / "old").exists())
            self.assertFalse((root / "loose.log").exists())

    def test_delete_rejects_root_escape_and_symlink_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            outside_file = Path(outside) / "keep.txt"
            outside_file.write_text("keep", encoding="utf-8")
            (root / "outside").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "outside the scanned directory"):
                delete_entries(str(root), ["../keep.txt"])
            with self.assertRaisesRegex(ValueError, "traverses outside"):
                delete_entries(str(root), ["outside/keep.txt"])

            self.assertTrue(outside_file.exists())

    def test_delete_can_unlink_a_symlink_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            target = Path(outside) / "keep.txt"
            target.write_text("keep", encoding="utf-8")
            link = root / "link"
            link.symlink_to(target)

            result = delete_entries(str(root), ["link"])

            self.assertEqual(result["deleted"], ["link"])
            self.assertFalse(link.exists())
            self.assertTrue(target.exists())


if __name__ == "__main__":
    unittest.main()
