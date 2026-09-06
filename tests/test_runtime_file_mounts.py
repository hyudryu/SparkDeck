import unittest
from unittest.mock import AsyncMock, Mock, patch

from manager import Manager
from sparkdeck.runtime_file_mounts import normalize_runtime_file_mounts, runtime_file_volumes

MOUNTS = [{"source": "/opt/patches/qwen3_dflash.py", "target": "/opt/vllm/qwen3_dflash.py"}]


class RuntimeFileMountTests(unittest.TestCase):
    def test_rejects_invalid_shapes_paths_and_overlap(self):
        invalid = [
            {}, ["/a:/b"], [{**MOUNTS[0], "readonly": False}],
            [{"source": "relative", "target": "/b"}],
            [{"source": "/a", "target": "/../b"}],
            [{"source": "/a", "target": "/."}],
            [{"source": "/a", "target": "/b/"}],
            [{"source": "/a", "target": "/b\n"}],
            MOUNTS * 17,
            MOUNTS + [{"source": "/other", "target": "/opt/vllm"}],
            MOUNTS + [{"source": MOUNTS[0]["source"], "target": "/other"}],
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_runtime_file_mounts(value)
        with self.assertRaisesRegex(ValueError, "only supported for vLLM"):
            normalize_runtime_file_mounts(MOUNTS, "sglang")

    def test_save_accepts_remote_path_without_touching_local_files(self):
        with patch("sparkdeck.runtime_file_mounts.Path") as local:
            self.assertEqual(normalize_runtime_file_mounts(MOUNTS), MOUNTS)
        local.assert_not_called()

    def test_rejects_managed_target_and_source_collisions(self):
        managed = {"/cache": {"bind": "/root/.cache/huggingface", "mode": "rw"}}
        with self.assertRaisesRegex(ValueError, "target conflicts"):
            runtime_file_volumes([{**MOUNTS[0], "target": "/root/.cache"}], managed)
        with patch("sparkdeck.runtime_file_mounts.Path.is_file", return_value=True):
            with self.assertRaisesRegex(ValueError, "source conflicts"):
                runtime_file_volumes([{**MOUNTS[0], "source": "/cache"}], managed)

    def test_missing_directory_or_special_file_is_rejected(self):
        with patch("sparkdeck.runtime_file_mounts.Path.is_file", return_value=False):
            with self.assertRaisesRegex(ValueError, "existing regular file on this node"):
                runtime_file_volumes(MOUNTS, {})


class RuntimeFileLaunchTests(unittest.IsolatedAsyncioTestCase):
    def manager(self):
        instance = Manager.__new__(Manager)
        instance.settings = {"vllm_image": "vllm/test", "hf_cache": "/cache", "shm_size": "1g", "default_gpu_memory_utilization": 0.65}
        instance.client = Mock()
        instance._gpu_total_gb = AsyncMock(return_value=122.0)
        instance._try_fit_new_model = Mock()
        instance._read_gpu_memory_gb = Mock(return_value=(0.0, 0.0))
        instance._cluster_launch_update = Mock()
        instance._build_volumes = Mock(return_value={"/cache": {"bind": "/root/.cache/huggingface", "mode": "rw"}})
        instance._container_hf_environment = Mock(return_value={})
        instance._created_container_model_source = Mock(return_value="public_repository")
        instance._run_managed_container = Mock()
        instance._container_summary = Mock(return_value={"name": "test"})
        return instance

    async def test_worker_rejects_missing_source_before_docker_or_eviction(self):
        instance = self.manager()
        with patch("sparkdeck.runtime_file_mounts.Path.is_file", return_value=False):
            with self.assertRaisesRegex(ValueError, "existing regular file"):
                await instance.create_container("org/model", port=8009, runtime_file_mounts=MOUNTS)
        instance.client.images.get.assert_not_called()
        instance._run_managed_container.assert_not_called()
        instance._try_fit_new_model.assert_not_called()

    async def test_patch_uses_read_only_bind_mount_api_and_preserves_cache_and_env(self):
        instance = self.manager()
        with patch("sparkdeck.runtime_file_mounts.Path.is_file", return_value=True):
            await instance.create_container("org/model", port=8009, name="test", runtime_file_mounts=MOUNTS, environment={"NCCL_DEBUG": "WARN"})
        options = instance._run_managed_container.call_args.args[0]
        self.assertEqual(options["mounts"], [{"Target": MOUNTS[0]["target"], "Source": MOUNTS[0]["source"], "Type": "bind", "ReadOnly": True}])
        self.assertEqual(options["volumes"], instance._build_volumes.return_value)
        self.assertEqual(options["environment"], {"NCCL_DEBUG": "WARN"})
        self.assertEqual(options["ports"], {"8000/tcp": 8009})
