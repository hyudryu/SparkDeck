import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from cluster import AgentCredentials, NodeRegistry, normalize_agent_url
from manager import Manager


class AgentCredentialsTests(unittest.TestCase):
    def test_pairing_code_is_one_time_and_token_is_durable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            credentials = AgentCredentials(Path(directory))
            code = credentials.data["pairing_code"]

            paired = credentials.pair(code)

            self.assertTrue(credentials.accepts_token(paired["agent_token"]))
            self.assertNotEqual(credentials.data["pairing_code"], code)
            with self.assertRaisesRegex(ValueError, "invalid pairing code"):
                credentials.pair(code)
            self.assertEqual(
                oct((Path(directory) / "agent.json").stat().st_mode & 0o777),
                "0o600",
            )

    def test_agent_url_rejects_embedded_credentials(self) -> None:
        self.assertEqual(normalize_agent_url("http://spark-2:7878/"), "http://spark-2:7878")
        with self.assertRaises(ValueError):
            normalize_agent_url("http://user:secret@spark-2:7878")


class SparkRunReferenceTests(unittest.TestCase):
    UUID = "13321ed7-516e-412a-ba13-bf00c4d805c3"

    def test_normalizes_common_portal_inputs(self) -> None:
        expected = f"@spark-arena/{self.UUID}"
        for value in (
            self.UUID,
            f"https://spark-arena.com/api/recipes/{self.UUID}/raw",
            f"https://spark-arena.com/recipes/{self.UUID}",
            f"sparkrun run @spark-arena/{self.UUID} --solo",
        ):
            with self.subTest(value=value):
                self.assertEqual(Manager._normalize_spark_reference(value), expected)

    def test_preserves_local_recipe_reference(self) -> None:
        self.assertEqual(
            Manager._normalize_spark_reference("recipes/custom.yaml"),
            "recipes/custom.yaml",
        )

    def test_cli_error_does_not_expose_traceback(self) -> None:
        stderr = b"Traceback (most recent call last):\nValueError: recipe not found\n"
        self.assertEqual(
            Manager._sparkrun_error(stderr, "missing"),
            "recipe not found",
        )


class NodeRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_pairing_persists_secret_but_returns_public_config(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/api/agent/pair")
            return httpx.Response(200, json={
                "node_id": "remote-1",
                "agent_token": "durable-secret",
                "name": "Spark 2",
                "protocol_version": 1,
            })

        with tempfile.TemporaryDirectory() as directory:
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            try:
                registry = NodeRegistry(Path(directory), client)
                public = await registry.pair_remote(
                    "http://spark-2:7878", "123456", fabric_ip="169.254.10.2"
                )
            finally:
                await client.aclose()
            self.assertNotIn("agent_token", public)
            saved = json.loads((Path(directory) / "nodes.json").read_text())
            self.assertEqual(saved[0]["agent_token"], "durable-secret")
            self.assertEqual(saved[0]["fabric_ip"], "169.254.10.2")

    async def test_probe_distinguishes_online_and_degraded_nodes(self) -> None:
        docker_ready = True

        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["authorization"], "Bearer secret")
            return httpx.Response(200, json={
                "node_id": "remote-1",
                "protocol_version": 1,
                "docker_ready": docker_ready,
                "fabric_ready": True,
                "interfaces": [],
                "stats": {},
                "containers": [],
            })

        with tempfile.TemporaryDirectory() as directory:
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            try:
                registry = NodeRegistry(Path(directory), client)
                node = {
                    "id": "remote-1", "name": "Spark 2",
                    "agent_url": "http://spark-2:7878", "agent_token": "secret",
                    "enabled": True,
                }
                registry.nodes = [node]
                online = await registry.probe(node, force=True)
                self.assertEqual(online["status"], "online")

                docker_ready = False
                degraded = await registry.probe(node, force=True)
                self.assertEqual(degraded["status"], "degraded")
                self.assertTrue(degraded["online"])
                self.assertIn("Docker", degraded["status_message"])
            finally:
                await client.aclose()


class LlamaRpcClusterTests(unittest.IsolatedAsyncioTestCase):
    def test_gguf_variant_size_includes_every_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = (
                root / "hub" / "models--example--Huge-GGUF" /
                "snapshots" / "snapshot-1" / "Q8_0"
            )
            snapshot.mkdir(parents=True)
            for shard, size in ((1, 11), (2, 13), (3, 17)):
                path = snapshot / f"Huge-Q8_0-{shard:05d}-of-00003.gguf"
                path.write_bytes(b"x" * size)

            instance = Manager.__new__(Manager)
            instance.settings = {"hf_cache": str(root)}
            models = instance._scan_gguf_models()

            variant = models[0]["variants"][0]
            self.assertEqual(variant["size_bytes"], 41)
            self.assertEqual(len(variant["files"]), 3)
            self.assertIn("-00001-of-00003.gguf", variant["path"])

    def test_gguf_scan_omits_variant_without_first_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = (
                root / "hub" / "models--example--Incomplete-GGUF" /
                "snapshots" / "snapshot-1" / "Q8_0"
            )
            snapshot.mkdir(parents=True)
            for shard in (2, 3):
                path = snapshot / f"Incomplete-Q8_0-{shard:05d}-of-00003.gguf"
                path.write_bytes(b"incomplete")

            instance = Manager.__new__(Manager)
            instance.settings = {"hf_cache": str(root)}

            self.assertEqual(instance._scan_gguf_models(), [])

    def test_gguf_scan_separates_dspark_drafts_from_target_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = (
                root / "hub" / "models--unsloth--DeepSeek-V4-GGUF" /
                "snapshots" / "snapshot-1"
            )
            target = snapshot / "UD-Q8_K_XL" / "DeepSeek-V4-UD-Q8_K_XL.gguf"
            q8_draft = snapshot / "dspark-DeepSeek-V4-Q8_0.gguf"
            bf16_draft = snapshot / "dspark" / "dspark-DeepSeek-V4-BF16.gguf"
            for path, size in ((target, 17), (q8_draft, 11), (bf16_draft, 13)):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x" * size)

            instance = Manager.__new__(Manager)
            instance.settings = {"hf_cache": str(root)}
            model = instance._scan_gguf_models()[0]

            self.assertEqual([v["quant"] for v in model["variants"]], ["UD-Q8_K_XL"])
            self.assertEqual(
                [draft["quant"] for draft in model["dspark_drafts"]],
                ["Q8_0", "BF16"],
            )

    def test_dspark_arguments_use_separate_draft_and_shared_token_count(self) -> None:
        argv, draft = Manager._llama_speculative_args(
            ["llama-server", "-m", "target.gguf"],
            {
                "mtp_enabled": False,
                "dspark_enabled": True,
                "mtp_predict_tokens": 3,
            },
            {
                "id": "unsloth/DeepSeek-V4-GGUF",
                "dspark_drafts": [
                    {"quant": "BF16", "path": "/draft-bf16.gguf"},
                    {"quant": "Q8_0", "path": "/draft-q8.gguf"},
                ],
            },
        )

        self.assertEqual(argv[argv.index("--spec-type") + 1], "draft-dspark")
        self.assertEqual(argv[argv.index("--spec-draft-model") + 1], "/draft-q8.gguf")
        self.assertEqual(argv[argv.index("--spec-draft-n-max") + 1], "3")
        self.assertEqual(argv[argv.index("--spec-draft-ngl") + 1], "99")
        self.assertEqual(draft["quant"], "Q8_0")

    def test_dspark_requires_a_downloaded_draft(self) -> None:
        with self.assertRaisesRegex(LookupError, "no DSPARK draft GGUF"):
            Manager._llama_speculative_args(
                ["llama-server"],
                {
                    "mtp_enabled": False,
                    "dspark_enabled": True,
                    "mtp_predict_tokens": 3,
                },
                {"id": "unsloth/DeepSeek-V4-GGUF", "dspark_drafts": []},
            )

    def test_mtp_and_dspark_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be enabled"):
            Manager._llama_speculative_args(
                ["llama-server"],
                {
                    "mtp_enabled": True,
                    "dspark_enabled": True,
                    "mtp_predict_tokens": 3,
                },
                {},
            )

    def test_rpc_progress_uses_fabric_transfer_counter(self) -> None:
        instance = Manager.__new__(Manager)
        instance._llama_pid = 123
        instance._interface_tx_counter = lambda interfaces: 1050
        instance._rpc_tx_counter = lambda endpoints: self.fail(
            "route lookup should not run during progress polling"
        )
        info = {
            "size_bytes": 200,
            "tensor_parallel_size": 2,
            "rpc_endpoints": ["169.254.1.2:50052"],
            "rpc_tx_start": 1000,
            "rpc_interfaces": ["cx7-test"],
        }

        percent, detail = instance._llama_transfer_progress(info)

        self.assertEqual(percent, 48)
        self.assertIn("RPC transfer over cx7-test", detail)
        self.assertIn("50.0 B / ~100.0 B", detail)

    def test_cached_local_load_stays_indeterminate_without_physical_io(self) -> None:
        instance = Manager.__new__(Manager)
        instance._llama_pid = 123
        instance._proc_read_bytes = lambda pid: 1000
        info = {
            "size_bytes": 200,
            "rpc_endpoints": [],
            "read_bytes_start": 1000,
        }

        percent, detail = instance._llama_transfer_progress(info)

        self.assertIsNone(percent)
        self.assertIn("filesystem cache", detail)

    def test_llama_logs_are_read_incrementally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance._llama_log_dir = Path(directory)
            instance._llama_current_log = instance._llama_log_dir / "llama-server-1.log"
            instance._llama_current_log.write_text("first\nsecond\n")
            instance._llama_launching = None
            instance._llama_model = "example/model"
            instance._llama_pid = None
            instance._llama_adopt_tried = True

            first = instance.get_llama_server_logs(since=0, limit_bytes=4096)
            second = instance.get_llama_server_logs(
                since=first["next_offset"], limit_bytes=4096
            )

            self.assertEqual(first["text"], "first\nsecond\n")
            self.assertTrue(first["complete"])
            self.assertEqual(second["text"], "")
            self.assertEqual(second["log_id"], "llama-server-1.log")

    def test_llama_log_chunks_preserve_split_utf8_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance._llama_log_dir = Path(directory)
            instance._llama_current_log = instance._llama_log_dir / "llama-server-utf8.log"
            expected = "a" * 4095 + "€" + "z"
            instance._llama_current_log.write_bytes(expected.encode())
            instance._llama_launching = None
            instance._llama_model = "example/model"
            instance._llama_pid = None
            instance._llama_adopt_tried = True

            first = instance.get_llama_server_logs(since=0, limit_bytes=4096)
            second = instance.get_llama_server_logs(
                since=first["next_offset"], limit_bytes=4096
            )

            self.assertNotIn("�", first["text"] + second["text"])
            self.assertEqual(first["text"] + second["text"], expected)

    def test_llama_logs_resolve_selected_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance._llama_log_dir = Path(directory)
            first_path = instance._llama_log_dir / "llama-server-a.log"
            second_path = instance._llama_log_dir / "llama-server-b.log"
            first_path.write_text("model A log")
            second_path.write_text("model B log")
            instance._llama_model_logs = {
                "example/a": str(first_path),
                "example/b": str(second_path),
            }
            instance._llama_current_log = second_path
            instance._llama_current_log_model = "example/b"
            instance._llama_launching = None
            instance._llama_model = "example/b"
            instance._llama_pid = None
            instance._llama_adopt_tried = True

            result = instance.get_llama_server_logs(model_path="example/a")

            self.assertEqual(result["text"], "model A log")
            self.assertEqual(result["model"], "example/a")

    def test_llama_log_names_are_collision_resistant(self) -> None:
        log_dir = Path("/tmp/logs")

        first = Manager._new_llama_log_path(log_dir)
        second = Manager._new_llama_log_path(log_dir)

        self.assertNotEqual(first, second)
        self.assertRegex(first.name, r"^llama-server-\d+-[0-9a-f]{8}\.log$")

    def test_tp2_arguments_use_rpc_tensor_split(self) -> None:
        argv = Manager._llama_tensor_parallel_args(
            ["llama-server", "-m", "model.gguf"],
            ["10.100.144.2:50052"],
            2,
        )

        self.assertEqual(argv[argv.index("--rpc") + 1], "10.100.144.2:50052")
        self.assertEqual(argv[argv.index("--split-mode") + 1], "tensor")
        self.assertEqual(argv[argv.index("--tensor-split") + 1], "1,1")
        self.assertEqual(argv[argv.index("--fit") + 1], "off")

    def test_layer_split_arguments_use_rpc_workers(self) -> None:
        argv = Manager._llama_tensor_parallel_args(
            ["llama-server", "-m", "model.gguf"],
            ["10.100.144.2:50052"],
            2,
            "layer",
        )

        self.assertEqual(argv[argv.index("--rpc") + 1], "10.100.144.2:50052")
        self.assertEqual(argv[argv.index("--split-mode") + 1], "layer")
        self.assertEqual(argv[argv.index("--tensor-split") + 1], "1,1")

    def test_rejects_unknown_llama_split_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported llama.cpp split mode"):
            Manager._llama_tensor_parallel_args(
                ["llama-server", "-m", "model.gguf"],
                ["10.100.144.2:50052"],
                2,
                "unsupported",
            )

    async def test_starts_remote_rpc_worker_on_connectx_fabric(self) -> None:
        instance = Manager.__new__(Manager)
        instance.settings = {"llama_rpc_port": 50052}
        instance._llama_rpc_nodes = []
        instance._llama_rpc_endpoints = []
        requests = []

        async def cluster_nodes(local_stats=None):
            return [
                {
                    "id": "local", "local": True, "name": "Spark 1",
                    "online": True, "fabric_ip": "10.100.144.1",
                    "fabric_interface": "enp1s0f0np0", "interfaces": [],
                },
                {
                    "id": "remote-1", "local": False, "name": "Spark 2",
                    "online": True, "fabric_ip": None,
                    "fabric_interface": None,
                    "interfaces": [{
                        "name": "enp1s0f0np0", "up": True, "rdma": True,
                        "ipv4": ["10.100.144.2"],
                    }],
                },
            ]

        class Registry:
            async def request(self, node_id, method, path, **kwargs):
                requests.append((node_id, method, path, kwargs))
                return {"running": True}

        instance.cluster_nodes = cluster_nodes
        instance.node_registry = Registry()

        endpoints = await instance._start_llama_rpc_cluster(2)

        self.assertEqual(endpoints, ["10.100.144.2:50052"])
        self.assertEqual(instance._llama_rpc_nodes, ["remote-1"])
        self.assertEqual(requests[0][0:3], (
            "remote-1", "POST", "/api/agent/llama-rpc"
        ))
        self.assertEqual(requests[0][3]["json_body"], {
            "host": "10.100.144.2", "port": 50052,
        })

    async def test_saved_short_quant_matches_ggml_model_filename(self) -> None:
        instance = Manager.__new__(Manager)
        instance._scan_gguf_models = lambda: [{
            "id": "example/model-GGUF",
            "variants": [{"quant": "ggml-model-Q8_0", "path": "/m.gguf"}],
        }]

        _, variant = await instance._find_gguf("example/model-GGUF", "Q8_0")

        self.assertEqual(variant["path"], "/m.gguf")


class DistributedLaunchTests(unittest.IsolatedAsyncioTestCase):
    def test_usage_alias_is_persisted_and_can_be_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.token_stats = {"model/a [Q8]": {"input": 1}}
            instance.usage_aliases = {}
            instance.usage_aliases_path = Path(directory) / "usage_aliases.json"

            saved = instance.update_usage_alias("model/a [Q8]", "Fast local")
            self.assertEqual(saved["alias"], "Fast local")
            self.assertEqual(
                json.loads(instance.usage_aliases_path.read_text()),
                {"model/a [Q8]": "Fast local"},
            )

            cleared = instance.update_usage_alias("model/a [Q8]", None)
            self.assertIsNone(cleared["alias"])
            self.assertEqual(json.loads(instance.usage_aliases_path.read_text()), {})

    async def test_recipe_clone_controls_and_update_are_durable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.lock = asyncio.Lock()
            instance.recipes = []
            instance.recipes_path = Path(directory) / "recipes.json"

            base = await instance.add_recipe(
                "model/a", name="Base", extra_args=[], force_new=True
            )
            clone = await instance.add_recipe(
                "model/a",
                name="Variant",
                extra_args=[],
                launch_controls={"max_concurrency": 8},
                force_new=True,
            )
            updated = await instance.update_recipe(
                clone["id"],
                {"launch_controls": {"context_window": 32768}},
            )

            self.assertNotEqual(base["id"], clone["id"])
            self.assertEqual(
                instance._cli_option(updated["extra_args"], {"--max-num-seqs"}, int),
                8,
            )
            self.assertEqual(
                instance._cli_option(updated["extra_args"], {"--max-model-len"}, int),
                32768,
            )
            saved = json.loads(instance.recipes_path.read_text())
            self.assertEqual(len(saved), 2)

    def test_hf_cache_mount_uses_image_hf_home(self) -> None:
        instance = Manager.__new__(Manager)
        image = type("Image", (), {
            "attrs": {"Config": {"Env": ["HOME=/tmp", "HF_HOME=/cache/huggingface"]}}
        })()
        instance.client = type("Client", (), {
            "images": type("Images", (), {"get": lambda self, name: image})()
        })()

        volumes = instance._build_volumes(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            "/host/huggingface",
            "custom-vllm:latest",
        )

        self.assertEqual(
            volumes["/host/huggingface"]["bind"], "/cache/huggingface"
        )

    def test_hf_cache_mount_falls_back_for_standard_images(self) -> None:
        instance = Manager.__new__(Manager)
        image = type("Image", (), {"attrs": {"Config": {"Env": []}}})()
        instance.client = type("Client", (), {
            "images": type("Images", (), {"get": lambda self, name: image})()
        })()

        volumes = instance._build_volumes(
            "example/Model", "/host/huggingface", "standard-vllm:latest"
        )

        self.assertEqual(
            volumes["/host/huggingface"]["bind"], "/root/.cache/huggingface"
        )

    def test_hf_token_is_masked_publicly_and_exported_to_containers(self) -> None:
        instance = Manager.__new__(Manager)
        instance.settings = {"hf_token": "hf_test_secret", "hf_cache": ""}

        public = instance.public_settings()
        environment = instance._container_hf_environment()

        self.assertEqual(public["hf_token"], "")
        self.assertTrue(public["hf_token_configured"])
        self.assertNotIn("hf_test_secret", json.dumps(public))
        self.assertEqual(environment["HF_TOKEN"], "hf_test_secret")
        self.assertEqual(environment["HUGGING_FACE_HUB_TOKEN"], "hf_test_secret")

    async def test_members_are_persisted_before_slow_node_launches_finish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.settings = {
                "cluster_fabric_ip": "169.254.10.1",
                "cluster_fabric_interface": "cx7-local",
            }
            instance.deployments_path = Path(directory) / "deployments.json"
            instance.deployments = []
            entered = 0
            both_entered = asyncio.Event()
            release = asyncio.Event()

            async def cluster_nodes(local_stats=None):
                return [
                    {
                        "id": "local", "name": "Spark 1", "online": True,
                        "docker_ready": True, "fabric_ip": "169.254.10.1",
                        "fabric_interface": "cx7-local", "interfaces": [],
                    },
                    {
                        "id": "remote-1", "name": "Spark 2", "online": True,
                        "docker_ready": True, "fabric_ip": "169.254.10.2",
                        "fabric_interface": "cx7-remote", "interfaces": [],
                    },
                ]

            async def allocate_port():
                return 8008

            async def create_member(node_id, payload):
                nonlocal entered
                entered += 1
                if entered == 2:
                    both_entered.set()
                await release.wait()
                return {"id": f"container-{node_id}", "status": "running"}

            instance.cluster_nodes = cluster_nodes
            instance._allocate_port = allocate_port
            instance._create_member = create_member

            task = asyncio.create_task(instance.create_deployment({
                "model": "deepseek-ai/DeepSeek-V4-Flash",
                "engine": "vllm",
                "deployment_mode": "sharded",
                "node_ids": ["local", "remote-1"],
            }))
            await asyncio.wait_for(both_entered.wait(), 1)

            self.assertEqual(len(instance.deployments[0]["members"]), 2)
            self.assertTrue(all(
                member["status"] == "queued"
                for member in instance.deployments[0]["members"]
            ))
            persisted = json.loads(instance.deployments_path.read_text())
            self.assertEqual(len(persisted[0]["members"]), 2)
            self.assertEqual(persisted[0]["api_port"], 8008)

            release.set()
            await task

    async def test_cluster_logs_show_progress_before_container_exists(self) -> None:
        instance = Manager.__new__(Manager)
        instance.cluster_member_launches = {}
        instance._cluster_launch_update(
            "cluster-test-r0", "pulling_image",
            "Downloading Docker image example/vllm:latest; this can take several minutes",
            model="example/model",
            cluster_member={"deployment_id": "test", "node_id": "local", "rank": 0},
        )

        async def is_managed_container(name):
            return False

        instance.is_managed_container = is_managed_container
        logs = await instance.get_cluster_member_logs("cluster-test-r0")

        self.assertIn("Controller launch progress", logs)
        self.assertIn("Downloading Docker image", logs)
        self.assertIn("Container has not been created yet", logs)

    async def test_combined_logs_fall_back_to_coordinator_status(self) -> None:
        instance = Manager.__new__(Manager)
        instance.deployments = [{
            "id": "test",
            "members": [{
                "node_id": "remote-1", "node_name": "Spark 2", "rank": 1,
                "container_name": "cluster-test-r1", "status": "queued",
                "phase": {"phase": "queued", "message": "Waiting for image pull"},
            }],
        }]

        async def member_action(member, action):
            raise RuntimeError("older agent has not created the container")

        instance._member_action = member_action
        result = await instance.deployment_logs("test")

        self.assertIn("Coordinator launch status", result["members"][0]["logs"])
        self.assertIn("Waiting for image pull", result["members"][0]["logs"])
        self.assertIn("older agent", result["members"][0]["logs"])
        self.assertIsNone(result["members"][0]["error"])

    async def test_remove_cluster_member_is_idempotent_when_already_absent(self) -> None:
        instance = Manager.__new__(Manager)
        instance.cluster_member_launches = {}

        async def is_managed_container(name):
            return False

        instance.is_managed_container = is_managed_container

        self.assertEqual(
            await instance.remove_cluster_member("missing-member"),
            {"ok": True, "already_absent": True},
        )

    async def test_remove_deployment_accepts_missing_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.deployments_path = Path(directory) / "deployments.json"
            instance.deployments = [{
                "id": "stale-deployment",
                "status": "error",
                "members": [
                    {"node_id": "local", "container_name": "missing-r0"},
                    {"node_id": "remote-1", "container_name": "missing-r1"},
                ],
            }]

            async def member_action(member, action):
                if member["node_id"] == "local":
                    raise ValueError("cluster member not found")
                raise RuntimeError(
                    'Spark 2 agent error: {"detail":"cluster member not found"}'
                )

            instance._member_action = member_action

            result = await instance.deployment_action("stale-deployment", "remove")

            self.assertEqual(result, {"ok": True, "errors": []})
            self.assertEqual(instance.deployments, [])
            self.assertEqual(json.loads(instance.deployments_path.read_text()), [])

    async def test_vllm_sharded_launch_generates_coordinated_rank_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.settings = {
                "cluster_fabric_ip": "169.254.10.1",
                "cluster_fabric_interface": "cx7-local",
            }
            instance.deployments_path = Path(directory) / "deployments.json"
            instance.deployments = []
            captured = []

            async def cluster_nodes(local_stats=None):
                return [
                    {
                        "id": "local", "name": "Spark 1", "online": True,
                        "docker_ready": True,
                        "fabric_ip": "169.254.10.1", "fabric_interface": "cx7-local",
                        "interfaces": [],
                    },
                    {
                        "id": "remote-1", "name": "Spark 2", "online": True,
                        "docker_ready": True,
                        "fabric_ip": "169.254.10.2", "fabric_interface": "cx7-remote",
                        "interfaces": [],
                    },
                ]

            async def allocate_port():
                return 8007

            async def create_member(node_id, payload):
                captured.append((node_id, payload))
                return {"id": f"container-{node_id}", "status": "running"}

            instance.cluster_nodes = cluster_nodes
            instance._allocate_port = allocate_port
            instance._create_member = create_member

            deployment = await instance.create_deployment({
                "model": "deepseek-ai/DeepSeek-V4-Flash",
                "engine": "vllm",
                "deployment_mode": "sharded",
                "node_ids": ["local", "remote-1"],
                "extra_args": [
                    "--max-model-len", "65536", "--tensor-parallel-size", "2",
                    "--pipeline-parallel-size", "1", "--headless",
                ],
            })

            self.assertEqual(deployment["status"], "starting")
            self.assertEqual(len(captured), 2)
            for rank, (node_id, payload) in enumerate(captured):
                args = payload["extra_args"]
                self.assertEqual(args[args.index("--node-rank") + 1], str(rank))
                self.assertEqual(args[args.index("--nnodes") + 1], "2")
                self.assertEqual(args[args.index("--tensor-parallel-size") + 1], "2")
                self.assertEqual(args[args.index("--pipeline-parallel-size") + 1], "1")
                self.assertEqual(args[args.index("--master-addr") + 1], "169.254.10.1")
                self.assertEqual("--headless" in args, rank > 0)
                self.assertEqual(args.count("--tensor-parallel-size"), 1)
                self.assertEqual(args.count("--pipeline-parallel-size"), 1)
                self.assertEqual(payload["cluster_member"]["fabric_interface"],
                                 "cx7-local" if node_id == "local" else "cx7-remote")

    async def test_vllm_sharded_launch_rejects_invalid_explicit_parallelism(self) -> None:
        instance = Manager.__new__(Manager)
        instance.settings = {
            "cluster_fabric_ip": "169.254.10.1",
            "cluster_fabric_interface": "cx7-local",
        }
        instance.deployments = []

        async def cluster_nodes(local_stats=None):
            return [
                {
                    "id": "local", "name": "Spark 1", "online": True,
                    "docker_ready": True, "fabric_ip": "169.254.10.1",
                    "fabric_interface": "cx7-local", "interfaces": [],
                },
                {
                    "id": "remote-1", "name": "Spark 2", "online": True,
                    "docker_ready": True, "fabric_ip": "169.254.10.2",
                    "fabric_interface": "cx7-remote", "interfaces": [],
                },
            ]

        instance.cluster_nodes = cluster_nodes

        with self.assertRaisesRegex(ValueError, "multiply to the 2 selected nodes"):
            await instance.create_deployment({
                "model": "example/Model",
                "engine": "vllm",
                "deployment_mode": "sharded",
                "node_ids": ["local", "remote-1"],
                "extra_args": [
                    "--tensor-parallel-size", "2",
                    "--pipeline-parallel-size", "2",
                ],
            })

        self.assertEqual(instance.deployments, [])

    def test_stopped_deployment_launch_settings_are_saved_and_marked_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.deployments_path = Path(directory) / "deployments.json"
            instance.deployments = [{
                "id": "deployment-1",
                "name": "Old name",
                "model": "example/Old",
                "engine": "vllm",
                "mode": "sharded",
                "node_ids": ["local", "remote-1"],
                "status": "stopped",
                "api_port": 8000,
                "members": [],
            }]

            updated = instance.update_deployment_settings("deployment-1", {
                "deployment_name": "Faster cluster",
                "model": "example/New",
                "engine": "vllm",
                "deployment_mode": "sharded",
                "node_ids": ["local", "remote-1"],
                "extra_args": [
                    "--max-model-len", "131072",
                    "--tensor-parallel-size", "2",
                    "--pipeline-parallel-size", "1",
                ],
                "image": "example/vllm:test",
            })

            self.assertTrue(updated["settings_dirty"])
            self.assertEqual(updated["name"], "Faster cluster")
            self.assertEqual(updated["launch_settings"]["port"], 8000)
            self.assertEqual(
                updated["launch_settings"]["extra_args"][1], "131072"
            )
            persisted = json.loads(instance.deployments_path.read_text())
            self.assertEqual(persisted[0]["launch_settings"]["image"], "example/vllm:test")

    def test_cluster_launch_controls_parse_and_round_trip_dspark_flags(self) -> None:
        instance = Manager.__new__(Manager)
        args = [
            "--max-model-len", "500000",
            "--max-num-seqs", "12",
            "--kv-cache-dtype", "nvfp4_ds_mla",
            "--max-cudagraph-capture-size", "72",
            "--max-num-batched-tokens", "8192",
            "--default-chat-template-kwargs", '{"thinking":true,"tools":true}',
            "--speculative-config",
            '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}',
            "--image-specific-flag", "kept",
        ]

        parsed = instance._deployment_launch_controls({
            "engine": "vllm", "extra_args": args,
        })

        self.assertEqual(parsed["context_window"], 500000)
        self.assertEqual(parsed["max_concurrency"], 12)
        self.assertEqual(parsed["kv_cache_dtype"], "nvfp4_ds_mla")
        self.assertEqual(parsed["thinking_mode"], "enabled")
        self.assertEqual(parsed["dspark_num_speculative_tokens"], 5)
        self.assertEqual(parsed["max_cudagraph_capture_size"], 72)
        self.assertEqual(parsed["max_num_batched_tokens"], 8192)

        updated = instance._apply_deployment_launch_controls(
            args,
            "vllm",
            {
                "context_window": 262144,
                "max_concurrency": 8,
                "kv_cache_dtype": "fp8_per_token_head",
                "thinking_mode": "disabled",
                "dspark_num_speculative_tokens": 3,
                "max_cudagraph_capture_size": 48,
                "max_num_batched_tokens": 4096,
            },
        )

        self.assertEqual(
            instance._cli_option(updated, {"--max-model-len"}, int), 262144
        )
        self.assertEqual(instance._cli_option(updated, {"--max-num-seqs"}, int), 8)
        self.assertEqual(
            instance._cli_option(updated, {"--kv-cache-dtype"}),
            "fp8_per_token_head",
        )
        self.assertEqual(
            instance._cli_option(updated, {"--max-cudagraph-capture-size"}, int), 48
        )
        self.assertEqual(
            instance._cli_option(updated, {"--max-num-batched-tokens"}, int), 4096
        )
        speculative = json.loads(
            instance._cli_option(updated, {"--speculative-config"})
        )
        self.assertEqual(speculative["num_speculative_tokens"], 3)
        self.assertEqual(speculative["method"], "dspark")
        self.assertEqual(speculative["draft_sample_method"], "probabilistic")
        template = json.loads(
            instance._cli_option(updated, {"--default-chat-template-kwargs"})
        )
        self.assertFalse(template["thinking"])
        self.assertTrue(template["tools"])
        self.assertEqual(
            instance._cli_option(updated, {"--image-specific-flag"}), "kept"
        )

    def test_running_deployment_settings_cannot_be_changed(self) -> None:
        instance = Manager.__new__(Manager)
        instance.deployments = [{
            "id": "deployment-1", "status": "ready", "members": [],
        }]

        with self.assertRaisesRegex(ValueError, "stop the cluster"):
            instance.update_deployment_settings("deployment-1", {"model": "example/New"})

    def test_running_deployment_can_be_renamed_without_dirtying_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.deployments_path = Path(directory) / "deployments.json"
            instance.deployments = [{
                "id": "deployment-1",
                "name": "example/Model",
                "model": "example/Model",
                "status": "ready",
                "settings_dirty": False,
                "launch_settings": {
                    "deployment_name": "example/Model",
                    "model": "example/Model",
                },
                "members": [],
            }]

            updated = instance.update_deployment_alias(
                "deployment-1", "Production reasoning"
            )

            self.assertEqual(updated["name"], "Production reasoning")
            self.assertEqual(
                updated["launch_settings"]["deployment_name"],
                "Production reasoning",
            )
            self.assertFalse(updated["settings_dirty"])
            persisted = json.loads(instance.deployments_path.read_text())
            self.assertEqual(persisted[0]["name"], "Production reasoning")

    async def test_starting_dirty_deployment_rebuilds_every_rank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.deployments_path = Path(directory) / "deployments.json"
            old = {
                "id": "deployment-old",
                "name": "Editable cluster",
                "model": "example/Model",
                "engine": "vllm",
                "mode": "sharded",
                "node_ids": ["local", "remote-1"],
                "status": "stopped",
                "settings_dirty": True,
                "members": [
                    {"node_id": "local", "container_name": "old-r0"},
                    {"node_id": "remote-1", "container_name": "old-r1"},
                ],
                "launch_settings": {
                    "deployment_name": "Editable cluster",
                    "model": "example/Model",
                    "engine": "vllm",
                    "deployment_mode": "sharded",
                    "node_ids": ["local", "remote-1"],
                    "extra_args": ["--tensor-parallel-size", "2"],
                    "port": 8000,
                },
            }
            instance.deployments = [old]
            removed = []
            launched = []

            async def member_action(member, action):
                removed.append((member["container_name"], action))
                return {"ok": True}

            async def create_deployment(body):
                launched.append(body)
                replacement = {
                    "id": "deployment-new", "status": "starting", "members": [],
                }
                instance.deployments.append(replacement)
                return replacement

            instance._member_action = member_action
            instance.create_deployment = create_deployment

            result = await instance.deployment_action("deployment-old", "start")

            self.assertTrue(result["ok"])
            self.assertEqual(removed, [("old-r0", "remove"), ("old-r1", "remove")])
            self.assertEqual(launched[0]["port"], 8000)
            self.assertEqual([d["id"] for d in instance.deployments], ["deployment-new"])

    async def test_idle_monitor_never_stops_one_cluster_member(self) -> None:
        instance = Manager.__new__(Manager)
        instance.settings = {"idle_timeout_seconds": 1}
        instance._activity = {
            "cluster-worker": {"counter": 0, "last_active": 0},
        }
        stopped = []

        async def list_containers():
            return [{
                "name": "cluster-worker",
                "managed": True,
                "status": "running",
                "deployment_id": "deployment-1",
                "phase": {"phase": "ready"},
                "port": 8000,
            }]

        async def read_activity_counter(port):
            self.fail("cluster members should not be polled independently")

        async def stop_container(name):
            stopped.append(name)

        instance.list_containers = list_containers
        instance._read_activity_counter = read_activity_counter
        instance.stop_container = stop_container

        await instance._idle_tick()

        self.assertEqual(stopped, [])

    @staticmethod
    def _health_deployment(status="ready"):
        return {
            "id": "deployment-1",
            "status": status,
            "members": [
                {"node_id": "local", "rank": 0, "container_name": "rank-0"},
                {"node_id": "remote-1", "rank": 1, "container_name": "rank-1"},
            ],
        }

    @staticmethod
    def _health_nodes(rank_0, rank_1):
        return [
            {
                "id": "local", "online": True, "docker_ready": True,
                "containers": [{"name": "rank-0", **rank_0}],
            },
            {
                "id": "remote-1", "online": True, "docker_ready": True,
                "containers": [{"name": "rank-1", **rank_1}],
            },
        ]

    def test_cluster_health_detects_running_and_stopped_split(self) -> None:
        instance = Manager.__new__(Manager)
        issue = instance._cluster_health_issue(
            self._health_deployment(),
            self._health_nodes(
                {"status": "running"},
                {"status": "exited"},
            ),
        )

        self.assertIn("ranks are split", issue)
        self.assertIn("rank 1: exited", issue)

    def test_cluster_health_detects_asymmetric_running_generation(self) -> None:
        instance = Manager.__new__(Manager)
        issue = instance._cluster_health_issue(
            self._health_deployment(),
            self._health_nodes(
                {
                    "status": "running", "restart_count": 1,
                    "started_at": "2026-08-08T18:52:28Z",
                },
                {
                    "status": "running", "restart_count": 0,
                    "started_at": "2026-08-08T16:09:00Z",
                },
            ),
        )

        self.assertIn("different Docker restart generations", issue)

    def test_cluster_health_accepts_aligned_start_after_recovery(self) -> None:
        instance = Manager.__new__(Manager)
        issue = instance._cluster_health_issue(
            self._health_deployment(status="starting"),
            self._health_nodes(
                {
                    "status": "running", "restart_count": 1,
                    "started_at": "2026-08-08T19:00:00Z",
                },
                {
                    "status": "running", "restart_count": 0,
                    "started_at": "2026-08-08T19:00:02Z",
                },
            ),
        )

        self.assertIsNone(issue)

    def test_cluster_health_detects_restart_after_baseline(self) -> None:
        instance = Manager.__new__(Manager)
        deployment = self._health_deployment()
        deployment["health_restart_counts"] = {"rank-0": 1, "rank-1": 0}
        issue = instance._cluster_health_issue(
            deployment,
            self._health_nodes(
                {
                    "status": "running", "restart_count": 2,
                    "started_at": "2026-08-08T19:05:00Z",
                },
                {
                    "status": "running", "restart_count": 0,
                    "started_at": "2026-08-08T19:00:00Z",
                },
            ),
        )

        self.assertEqual(issue, "Docker restarted only part of the cluster")

    async def test_cluster_recovery_stops_all_ranks_before_starting_any(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.deployments_path = Path(directory) / "deployments.json"
            instance.deployments = [self._health_deployment()]
            events = []
            both_stopping = asyncio.Event()
            release_stops = asyncio.Event()

            async def member_action(member, action):
                events.append((action, member["rank"]))
                if action == "stop":
                    if sum(1 for event in events if event[0] == "stop") == 2:
                        both_stopping.set()
                    await release_stops.wait()
                return {"ok": True}

            instance._member_action = member_action
            recovery = asyncio.create_task(instance._recover_cluster_deployment(
                "deployment-1", "cluster ranks are split"
            ))
            await asyncio.wait_for(both_stopping.wait(), 1)
            self.assertFalse(any(action == "start" for action, _ in events))
            release_stops.set()
            await recovery

            first_start = next(i for i, event in enumerate(events) if event[0] == "start")
            self.assertTrue(all(action == "stop" for action, _ in events[:first_start]))
            self.assertEqual(instance.deployments[0]["status"], "starting")
            self.assertIsNone(instance.deployments[0]["error"])

    async def test_cluster_recovery_does_not_start_if_a_rank_cannot_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.deployments_path = Path(directory) / "deployments.json"
            instance.deployments = [self._health_deployment()]
            starts = []

            async def member_action(member, action):
                if action == "stop" and member["rank"] == 1:
                    raise RuntimeError("GPU process did not stop")
                if action == "start":
                    starts.append(member["rank"])
                return {"ok": True}

            instance._member_action = member_action
            await instance._recover_cluster_deployment(
                "deployment-1", "cluster ranks are split"
            )

            self.assertEqual(starts, [])
            self.assertEqual(instance.deployments[0]["status"], "degraded")
            self.assertIn("no rank was restarted", instance.deployments[0]["error"])

    async def test_partial_manual_start_remains_eligible_for_health_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.deployments_path = Path(directory) / "deployments.json"
            instance.deployments = [self._health_deployment(status="stopped")]

            async def member_action(member, action):
                if member["rank"] == 1:
                    raise RuntimeError("rank 1 did not start")
                return {"ok": True}

            instance._member_action = member_action
            result = await instance.deployment_action("deployment-1", "start")

            self.assertFalse(result["ok"])
            self.assertEqual(instance.deployments[0]["status"], "degraded")

    def test_node_in_use_by_deployment_cannot_be_removed(self) -> None:
        instance = Manager.__new__(Manager)
        instance.deployments = [{
            "id": "deployment-1",
            "name": "DeepSeek cluster",
            "node_ids": ["local", "remote-1"],
            "members": [],
        }]

        with self.assertRaisesRegex(ValueError, "remove that deployment first"):
            instance.remove_cluster_node("remote-1")


if __name__ == "__main__":
    unittest.main()
