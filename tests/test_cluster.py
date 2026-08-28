import asyncio
import json
import socket
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import docker
import httpx
import requests

from cluster import (
    AGENT_PROTOCOL_VERSION,
    COORDINATOR_ID_HEADER,
    AgentCredentials,
    NodeRegistry,
    normalize_agent_url,
)
from manager import Manager, PERSISTED_DEPLOYMENT_ARGS_ERROR
from sparkdeck.onboarding import resolve_agent_connection


class AgentCredentialsTests(unittest.TestCase):
    def test_pairing_code_and_previous_controller_token_are_rotated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            credentials = AgentCredentials(Path(directory))
            code = credentials.data["pairing_code"]
            previous_token = credentials.data["agent_token"]

            paired = credentials.pair(code, "controller-a")

            self.assertTrue(credentials.accepts_token(paired["agent_token"]))
            self.assertFalse(credentials.accepts_token(previous_token))
            self.assertNotEqual(credentials.data["pairing_code"], code)
            with self.assertRaisesRegex(ValueError, "invalid pairing code"):
                credentials.pair(code, "controller-a")
            self.assertEqual(
                oct((Path(directory) / "agent.json").stat().st_mode & 0o777),
                "0o600",
            )

    def test_pairing_does_not_activate_credentials_before_atomic_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            credentials = AgentCredentials(Path(directory))
            original = dict(credentials.data)

            with mock.patch(
                "cluster._atomic_json_write", side_effect=OSError("disk full")
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    credentials.pair(original["pairing_code"], "controller-a")

            self.assertEqual(credentials.data, original)
            self.assertTrue(credentials.accepts_token(original["agent_token"]))

    def test_legacy_token_is_claimed_by_only_one_controller(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            credentials = AgentCredentials(Path(directory))
            token = credentials.data["agent_token"]

            self.assertTrue(credentials.authorize_controller(token, "controller-a"))
            self.assertTrue(credentials.authorize_controller(token, "controller-a"))
            self.assertFalse(credentials.authorize_controller(token, "controller-b"))
            reloaded = AgentCredentials(Path(directory))
            self.assertFalse(reloaded.authorize_controller(token, "controller-b"))

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
    async def test_authenticated_consent_can_reach_a_disabled_joined_worker(self) -> None:
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200, json={"applied": True, "enabled": False}, request=request,
            )

        with tempfile.TemporaryDirectory() as directory:
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            registry = NodeRegistry(Path(directory), client, "controller")
            registry.nodes = [{
                "id": "remote-1",
                "name": "Spark Disabled",
                "agent_url": "https://worker.tail.example:7878",
                "agent_token": "agent-secret",
                "enabled": False,
            }]
            registry._connection_targets = mock.AsyncMock(return_value=[(
                httpx.URL("https://100.100.20.30:7878/api/agent/community-consent"),
                {"Host": "worker.tail.example:7878"},
                {"sni_hostname": "worker.tail.example"},
            )])
            try:
                with self.assertRaisesRegex(ValueError, "is disabled"):
                    await registry.request(
                        "remote-1", "PUT", "/api/agent/community-consent",
                        json_body={"enabled": False},
                    )

                result = await registry.request(
                    "remote-1", "PUT", "/api/agent/community-consent",
                    json_body={"enabled": False}, allow_disabled=True,
                )
            finally:
                await client.aclose()

        self.assertEqual(result, {"applied": True, "enabled": False})
        self.assertEqual(len(requests), 1)
        self.assertEqual(
            requests[0].headers["authorization"], "Bearer agent-secret",
        )
        self.assertEqual(
            json.loads(requests[0].content), {"enabled": False},
        )

    async def test_authenticated_request_pins_validated_agent_address(self) -> None:
        requests = []
        resolutions = []

        def resolve(host, port, **kwargs):
            resolutions.append((host, port))
            address = "100.100.20.30" if len(resolutions) == 1 else "203.0.113.10"
            return [(
                socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                (address, port),
            )]

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"ok": True}, request=request)

        with tempfile.TemporaryDirectory() as directory:
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            registry = NodeRegistry(
                Path(directory), client, "controller",
                connection_resolver=resolve_agent_connection,
            )
            registry.nodes = [{
                "id": "remote-1",
                "name": "Spark 2",
                "agent_url": "https://worker.tail.example:7878/sparkdeck",
                "agent_token": "agent-secret",
                "enabled": True,
            }]
            try:
                with mock.patch(
                    "sparkdeck.onboarding.socket.getaddrinfo", side_effect=resolve,
                ):
                    result = await registry.request(
                        "remote-1", "POST", "/api/agent/containers",
                        json_body={"hf_token": "hf-secret"},
                    )
            finally:
                await client.aclose()

        self.assertEqual(result, {"ok": True})
        self.assertEqual(resolutions, [("worker.tail.example", 7878)])
        self.assertEqual(
            str(requests[0].url),
            "https://100.100.20.30:7878/sparkdeck/api/agent/containers",
        )
        self.assertEqual(requests[0].headers["host"], "worker.tail.example:7878")
        self.assertEqual(requests[0].extensions["sni_hostname"], "worker.tail.example")
        self.assertEqual(requests[0].headers["authorization"], "Bearer agent-secret")
        self.assertEqual(json.loads(requests[0].content)["hf_token"], "hf-secret")

    async def test_authenticated_request_tries_each_pinned_safe_address(self) -> None:
        requests = []
        resolution_count = 0

        def resolve(host, port, **kwargs):
            nonlocal resolution_count
            resolution_count += 1
            return [
                (
                    socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                    ("fd7a:115c:a1e0::10", port, 0, 0),
                ),
                (
                    socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                    ("100.100.20.30", port),
                ),
            ]

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if len(requests) == 1:
                raise httpx.ConnectError("IPv6 unavailable", request=request)
            return httpx.Response(200, json={"ok": True}, request=request)

        with tempfile.TemporaryDirectory() as directory:
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            registry = NodeRegistry(
                Path(directory), client, "controller",
                connection_resolver=resolve_agent_connection,
            )
            registry.nodes = [{
                "id": "remote-1", "name": "Spark 2", "enabled": True,
                "agent_url": "https://worker.tail.example:7878",
                "agent_token": "agent-secret",
            }]
            try:
                with mock.patch(
                    "sparkdeck.onboarding.socket.getaddrinfo", side_effect=resolve,
                ):
                    result = await registry.request(
                        "remote-1", "POST", "/api/agent/containers",
                        json_body={"hf_token": "hf-secret"},
                    )
            finally:
                await client.aclose()

        self.assertEqual(result, {"ok": True})
        self.assertEqual(resolution_count, 1)
        self.assertEqual([str(request.url) for request in requests], [
            "https://[fd7a:115c:a1e0::10]:7878/api/agent/containers",
            "https://100.100.20.30:7878/api/agent/containers",
        ])
        self.assertTrue(all(
            request.headers["authorization"] == "Bearer agent-secret"
            and request.headers["host"] == "worker.tail.example:7878"
            and request.extensions["sni_hostname"] == "worker.tail.example"
            for request in requests
        ))

    async def test_pairing_tries_each_pinned_safe_address(self) -> None:
        requests = []
        resolution_count = 0

        def resolve(host, port, **kwargs):
            nonlocal resolution_count
            resolution_count += 1
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("100.64.0.10", port)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("100.64.0.11", port)),
            ]

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if len(requests) == 1:
                raise httpx.ConnectTimeout("first address timed out", request=request)
            return httpx.Response(200, json={
                "node_id": "remote-1", "agent_token": "agent-secret",
                "protocol_version": AGENT_PROTOCOL_VERSION,
            }, request=request)

        with tempfile.TemporaryDirectory() as directory:
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            registry = NodeRegistry(
                Path(directory), client, "controller",
                connection_resolver=resolve_agent_connection,
            )
            try:
                with mock.patch(
                    "sparkdeck.onboarding.socket.getaddrinfo", side_effect=resolve,
                ):
                    paired = await registry.pair_remote(
                        "http://worker.tail.example:7878/sparkdeck/", "123456",
                    )
            finally:
                await client.aclose()

        self.assertEqual(paired["id"], "remote-1")
        self.assertEqual(
            paired["agent_url"], "http://worker.tail.example:7878/sparkdeck"
        )
        self.assertEqual(resolution_count, 1)
        self.assertEqual([str(request.url) for request in requests], [
            "http://100.64.0.10:7878/sparkdeck/api/agent/pair",
            "http://100.64.0.11:7878/sparkdeck/api/agent/pair",
        ])
        self.assertTrue(all(
            json.loads(request.content)["pairing_code"] == "123456"
            and request.headers["host"] == "worker.tail.example:7878"
            for request in requests
        ))

    async def test_open_stream_tries_each_pinned_safe_address(self) -> None:
        requests = []
        resolution_count = 0

        def resolve(host, port, **kwargs):
            nonlocal resolution_count
            resolution_count += 1
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("100.64.0.20", port)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("100.64.0.21", port)),
            ]

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if len(requests) == 1:
                raise httpx.ConnectError("first address refused", request=request)
            return httpx.Response(200, content=b"stream", request=request)

        with tempfile.TemporaryDirectory() as directory:
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            registry = NodeRegistry(
                Path(directory), client, "controller",
                connection_resolver=resolve_agent_connection,
            )
            registry.nodes = [{
                "id": "remote-1", "name": "Spark 2", "enabled": True,
                "agent_url": "http://worker.tail.example:7878/sparkdeck",
                "agent_token": "agent-secret",
            }]
            try:
                with mock.patch(
                    "sparkdeck.onboarding.socket.getaddrinfo", side_effect=resolve,
                ):
                    response = await registry.open_stream(
                        "remote-1", "GET", "/api/agent/files",
                    )
                    body = await response.aread()
                    await response.aclose()
            finally:
                await client.aclose()

        self.assertEqual(body, b"stream")
        self.assertEqual(resolution_count, 1)
        self.assertEqual([str(request.url) for request in requests], [
            "http://100.64.0.20:7878/sparkdeck/api/agent/files",
            "http://100.64.0.21:7878/sparkdeck/api/agent/files",
        ])
        self.assertTrue(all(
            request.headers["authorization"] == "Bearer agent-secret"
            and request.headers["host"] == "worker.tail.example:7878"
            for request in requests
        ))

    async def test_agent_connection_rejects_public_dns_with_base_path(self) -> None:
        def resolve(host, port, **kwargs):
            return [(
                socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                ("203.0.113.10", port),
            )]

        with mock.patch(
            "sparkdeck.onboarding.socket.getaddrinfo", side_effect=resolve,
        ):
            with self.assertRaisesRegex(ValueError, "Tailscale or loopback"):
                await resolve_agent_connection(
                    "https://worker.example:7878/sparkdeck"
                )

    async def test_pairing_persists_secret_but_returns_public_config(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/api/agent/pair")
            self.assertEqual(json.loads(request.content), {
                "pairing_code": "123456",
                "controller_id": "controller",
            })
            return httpx.Response(200, json={
                "node_id": "remote-1",
                "agent_token": "durable-secret",
                "name": "Spark 2",
                "protocol_version": 1,
            })

        with tempfile.TemporaryDirectory() as directory:
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            try:
                registry = NodeRegistry(Path(directory), client, "controller")
                public = await registry.pair_remote(
                    "http://spark-2:7878", "123456", fabric_ip="169.254.10.2",
                )
            finally:
                await client.aclose()
            self.assertNotIn("agent_token", public)
            saved = json.loads((Path(directory) / "nodes.json").read_text())
            self.assertEqual(saved[0]["agent_token"], "durable-secret")
            self.assertEqual(saved[0]["fabric_ip"], "169.254.10.2")
            self.assertFalse(saved[0]["usage_reconciled"])

    async def test_probe_distinguishes_online_and_degraded_nodes(self) -> None:
        docker_ready = True

        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["authorization"], "Bearer secret")
            self.assertEqual(request.headers[COORDINATOR_ID_HEADER], "controller")
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
                registry = NodeRegistry(Path(directory), client, "controller")
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
    @staticmethod
    def _usage_sync_manager(
        root: Path, node_id: str, token_stats: dict, hourly: dict,
    ) -> Manager:
        instance = Manager.__new__(Manager)
        instance.settings = {}
        instance.agent_credentials = mock.Mock(node_id=node_id)
        instance.token_stats = json.loads(json.dumps(token_stats))
        instance.hourly_token_stats = json.loads(json.dumps(hourly))
        instance.token_stats_path = root / f"{node_id}-tokens.json"
        instance.hourly_token_stats_path = root / f"{node_id}-hourly.json"
        instance.token_usage_sync_path = root / f"{node_id}-sync.json"
        instance.token_usage_sync = instance._load_token_usage_sync()
        instance._rebuild_synced_token_usage()
        return instance

    def test_distributed_network_environment_matches_selected_rdma_port(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            infiniband = (
                root / "enp1s0f1np1" / "device" / "infiniband"
            )
            (infiniband / "rocep1s0f1").mkdir(parents=True)

            environment = Manager._distributed_network_environment(
                "enp1s0f1np1", root,
            )

        self.assertEqual(environment, {
            "NCCL_SOCKET_IFNAME": "enp1s0f1np1",
            "GLOO_SOCKET_IFNAME": "enp1s0f1np1",
            "NCCL_IB_GID_INDEX": None,
            "NCCL_NET": "IB",
            "NCCL_IB_DISABLE": "0",
            "NCCL_IB_HCA": "rocep1s0f1",
            "UCX_NET_DEVICES": "rocep1s0f1:1",
        })

    def test_distributed_network_environment_overrides_stale_rdma_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = Manager._distributed_network_environment(
                "eth0", Path(directory),
            )

        self.assertEqual(environment["NCCL_NET"], "Socket")
        self.assertEqual(environment["NCCL_IB_DISABLE"], "1")
        self.assertEqual(environment["NCCL_IB_HCA"], "")
        self.assertEqual(environment["UCX_NET_DEVICES"], "")

    def test_usage_alias_is_persisted_and_can_be_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.token_stats = {"model/a [Q8]": {"input": 1}}
            instance.usage_aliases = {}
            instance.usage_aliases_path = Path(directory) / "usage_aliases.json"
            instance.usage_merge_groups = {}
            instance.usage_merge_groups_path = (
                Path(directory) / "usage_merge_groups.json"
            )

            saved = instance.update_usage_alias(
                "model/a [Q8]", "Fast local",
                merge_group="DeepSeek", update_merge_group=True,
            )
            self.assertEqual(saved["alias"], "Fast local")
            self.assertEqual(saved["merge_group"], "DeepSeek")
            self.assertEqual(
                json.loads(instance.usage_aliases_path.read_text()),
                {"model/a [Q8]": "Fast local"},
            )
            self.assertEqual(
                json.loads(instance.usage_merge_groups_path.read_text()),
                {"model/a [Q8]": "DeepSeek"},
            )

            cleared = instance.update_usage_alias(
                "model/a [Q8]", None,
                merge_group=None, update_merge_group=True,
            )
            self.assertIsNone(cleared["alias"])
            self.assertIsNone(cleared["merge_group"])
            self.assertEqual(json.loads(instance.usage_aliases_path.read_text()), {})
            self.assertEqual(
                json.loads(instance.usage_merge_groups_path.read_text()), {}
            )

    def test_usage_alias_update_is_atomic_when_merge_group_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.token_stats = {"model/a": {"input": 1}}
            instance.usage_aliases = {"model/a": "Original"}
            instance.usage_merge_groups = {}
            instance.usage_aliases_path = Path(directory) / "usage_aliases.json"
            instance.usage_merge_groups_path = Path(directory) / "usage_merge_groups.json"
            instance._save_usage_aliases()

            with self.assertRaisesRegex(ValueError, "120 characters"):
                instance.update_usage_alias(
                    "model/a", "Changed", merge_group="x" * 121,
                    update_merge_group=True,
                )

            self.assertEqual(instance.usage_aliases, {"model/a": "Original"})
            self.assertEqual(
                json.loads(instance.usage_aliases_path.read_text()),
                {"model/a": "Original"},
            )

    def test_one_usage_model_can_be_erased_without_touching_others(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instance = Manager.__new__(Manager)
            instance.token_stats_path = root / "token_stats.json"
            instance.hourly_token_stats_path = root / "hourly.json"
            instance.speed_samples_path = root / "speed.json"
            instance.usage_aliases_path = root / "aliases.json"
            instance.usage_merge_groups_path = root / "groups.json"
            instance.token_stats = {
                "model/a": {"input": 10}, "model/b": {"input": 20},
            }
            instance.session_token_stats = {
                "model/a": {"input": 1}, "model/b": {"input": 2},
            }
            instance.speed_samples = {
                "model/a": [{"tokens": 1}], "model/b": [{"tokens": 2}],
            }
            instance.hourly_token_stats = {
                "2026-08-14T01": {
                    "model/a": {"input": 3}, "model/b": {"input": 4},
                },
            }
            instance.usage_aliases = {
                "model/a": "A", "model/b": "B",
            }
            instance.usage_merge_groups = {
                "model/a": "group", "model/b": "group",
            }

            result = instance.erase_usage_model("model/a")

            self.assertEqual(result, {"ok": True, "model": "model/a"})
            for mapping in (
                instance.token_stats, instance.session_token_stats,
                instance.speed_samples, instance.usage_aliases,
                instance.usage_merge_groups,
            ):
                self.assertNotIn("model/a", mapping)
                self.assertIn("model/b", mapping)
            self.assertNotIn(
                "model/a", instance.hourly_token_stats["2026-08-14T01"]
            )
            self.assertIn(
                "model/b", instance.hourly_token_stats["2026-08-14T01"]
            )
            self.assertNotIn(
                "model/a", json.loads(instance.token_stats_path.read_text())
            )

    def test_token_usage_sync_fans_out_three_node_totals_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            node1 = self._usage_sync_manager(
                root, "node-1",
                {"model/a": {"input": 10, "output": 2, "requests": 1}},
                {"2026-08-15T07": {"model/a": {"input": 10, "output": 2,
                                                  "requests": 1}}},
            )
            node2 = self._usage_sync_manager(
                root, "node-2",
                {"model/b": {"input": 20, "output": 4, "requests": 1}},
                {"2026-08-15T07": {"model/b": {"input": 20, "output": 4,
                                                  "requests": 1}}},
            )
            node3 = self._usage_sync_manager(
                root, "node-3",
                {"model/a": {"input": 5, "output": 3, "requests": 1},
                 "model/c": {"input": 30, "output": 6, "requests": 1}},
                {"2026-08-15T07": {
                    "model/a": {"input": 5, "output": 3, "requests": 1},
                    "model/c": {"input": 30, "output": 6, "requests": 1},
                }},
            )

            # Node 1 acts as the paired coordinator: pull both peers, then
            # push the union back to them.
            self.assertTrue(node1.merge_token_usage_sync(
                node2.token_usage_sync_snapshot()
            ))
            self.assertTrue(node1.merge_token_usage_sync(
                node3.token_usage_sync_snapshot()
            ))
            union = node1.token_usage_sync_snapshot()
            self.assertTrue(node2.merge_token_usage_sync(union))
            self.assertTrue(node3.merge_token_usage_sync(union))

            for node in (node1, node2, node3):
                self.assertEqual(node.token_stats["model/a"]["input"], 15)
                self.assertEqual(node.token_stats["model/a"]["output"], 5)
                self.assertEqual(node.token_stats["model/b"]["input"], 20)
                self.assertEqual(node.token_stats["model/c"]["input"], 30)
                self.assertEqual(
                    node.hourly_token_stats["2026-08-15T07"]["model/a"]["input"],
                    15,
                )

            # Replaying the same full-mesh snapshots must not add them again.
            before = json.loads(json.dumps(node1.token_stats))
            self.assertFalse(node1.merge_token_usage_sync(union))
            self.assertEqual(node1.token_stats, before)

    async def test_token_usage_sync_always_pulls_every_paired_node(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self._usage_sync_manager(
                root, "controller",
                {"model/a": {"input": 10, "output": 2, "requests": 1}},
                {"2026-08-26T18": {"model/a": {"input": 10, "output": 2}}},
            )
            worker = self._usage_sync_manager(
                root, "worker",
                {"model/b": {"input": 20, "output": 4, "requests": 1}},
                {"2026-08-26T18": {"model/b": {"input": 20, "output": 4}}},
            )
            worker._reset_synced_token_usage()
            worker._record_local_synced_tokens(
                "model/b", 20, 4, 0, None, None, "2026-08-26T18"
            )
            worker._rebuild_synced_token_usage()
            requests = []

            class Registry:
                nodes = [{"id": "worker", "name": "Worker", "enabled": True}]

                async def request(self, node_id, method, path, **kwargs):
                    requests.append((node_id, method, path, kwargs))
                    if method == "GET":
                        # Older workers reported this hidden opt-in as false,
                        # but their ledger is still valid and must be counted.
                        return {
                            **worker.token_usage_sync_snapshot(),
                            "enabled": False,
                        }
                    worker.merge_token_usage_sync(kwargs["json_body"])
                    return {"enabled": True, "changed": True}

                def mark_usage_reconciled(self, node_id):
                    self.nodes[0]["usage_reconciled"] = True

            controller.node_registry = Registry()
            controller._token_usage_sync_status = {
                "enabled": False, "last_sync_at": None,
                "peers": 0, "error": None,
            }

            status = await controller.sync_token_usage_once()

            self.assertTrue(status["enabled"])
            self.assertEqual(status["peers"], 1)
            self.assertIsNone(status["error"])
            self.assertEqual(controller.token_stats["model/a"]["input"], 10)
            self.assertEqual(controller.token_stats["model/b"]["input"], 20)
            self.assertEqual(worker.token_stats["model/a"]["input"], 10)
            self.assertTrue(controller.node_registry.nodes[0]["usage_reconciled"])
            self.assertEqual(
                [(method, path) for _, method, path, _ in requests],
                [("GET", "/api/agent/token-usage"),
                 ("POST", "/api/agent/token-usage")],
            )

    def test_legacy_usage_sync_opt_out_is_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.settings_path = Path(directory) / "settings.json"
            instance.settings_path.write_text(json.dumps({
                "sync_token_usage": False,
                "max_retries": 7,
            }))

            settings = instance._load_settings()

        self.assertNotIn("sync_token_usage", settings)
        self.assertEqual(settings["max_retries"], 7)

    def test_legacy_linux_hf_cache_default_is_migrated_to_user_home(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.settings_path = Path(directory) / "settings.json"
            instance.settings_path.write_text(json.dumps({
                "hf_cache": "/home/hyudryu/.cache/huggingface",
                "max_retries": 7,
            }))

            settings = instance._load_settings()

            self.assertEqual(
                settings["hf_cache"],
                str(Path.home() / ".cache" / "huggingface"),
            )
            self.assertEqual(
                json.loads(instance.settings_path.read_text())["hf_cache"],
                str(Path.home() / ".cache" / "huggingface"),
            )
            self.assertEqual(settings["max_retries"], 7)

    def test_legacy_hf_cache_migration_keeps_parsed_settings_when_write_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.settings_path = Path(directory) / "settings.json"
            instance.settings_path.write_text(json.dumps({
                "hf_cache": "/home/hyudryu/.cache/huggingface",
                "max_retries": 7,
            }))

            with mock.patch(
                "manager._atomic_private_json_write",
                side_effect=OSError("read only"),
            ):
                settings = instance._load_settings()

        self.assertEqual(settings["max_retries"], 7)
        self.assertEqual(settings["hf_cache"], str(Path.home() / ".cache" / "huggingface"))

    def test_token_usage_sync_reset_epoch_rejects_stale_totals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            node1 = self._usage_sync_manager(
                root, "node-1", {"model/a": {"input": 10}},
                {"2026-08-15T07": {"model/a": {"input": 10}}},
            )
            node2 = self._usage_sync_manager(
                root, "node-2", {"model/b": {"input": 20}},
                {"2026-08-15T07": {"model/b": {"input": 20}}},
            )
            stale = node2.token_usage_sync_snapshot()
            node1.merge_token_usage_sync(stale)
            node1._reset_synced_token_usage()
            node1._rebuild_synced_token_usage()

            self.assertEqual(node1.token_stats, {})
            self.assertFalse(node1.merge_token_usage_sync(stale))
            self.assertEqual(node1.token_stats, {})
            self.assertTrue(node2.merge_token_usage_sync(
                node1.token_usage_sync_snapshot()
            ))
            self.assertEqual(node2.token_stats, {})
            # Lifetime reset preserves historical hourly analysis counters.
            self.assertIn("model/a", node2.hourly_token_stats["2026-08-15T07"])
            self.assertIn("model/b", node2.hourly_token_stats["2026-08-15T07"])

    def test_first_sync_reconciles_reset_without_erasing_either_node(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self._usage_sync_manager(
                root, "controller", {"model/a": {"input": 10}},
                {"2026-08-15T07": {"model/a": {"input": 10}}},
            )
            worker = self._usage_sync_manager(
                root, "worker", {"model/b": {"input": 20}},
                {"2026-08-15T07": {"model/b": {"input": 20}}},
            )
            worker._reset_synced_token_usage()
            worker._record_local_synced_tokens(
                "model/b", 20, 0, 0, None, None, "2026-08-15T07"
            )
            worker._rebuild_synced_token_usage()

            controller.reconcile_token_usage_sync(
                worker.token_usage_sync_snapshot(), "worker"
            )

            self.assertEqual(controller.token_stats["model/a"]["input"], 10)
            self.assertEqual(controller.token_stats["model/b"]["input"], 20)

    def test_reconciliation_preserves_usage_recorded_during_first_sync(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self._usage_sync_manager(
                root, "controller", {"model/a": {"input": 10}}, {}
            )
            worker = self._usage_sync_manager(
                root, "worker", {"model/b": {"input": 20}}, {}
            )
            controller.reconcile_token_usage_sync(
                worker.token_usage_sync_snapshot(), "worker"
            )
            union = {
                **controller.token_usage_sync_snapshot(),
                "reconcile_origin": "worker",
            }
            worker._record_local_synced_tokens(
                "model/b", 5, 0, 0, None, None, "2026-08-26T19"
            )
            worker._rebuild_synced_token_usage()

            worker.merge_token_usage_sync(union)

            self.assertEqual(worker.token_stats["model/a"]["input"], 10)
            self.assertEqual(worker.token_stats["model/b"]["input"], 25)

    def test_reconciliation_preserves_coordinator_model_tombstones(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self._usage_sync_manager(
                root, "controller", {"model/a": {"input": 10}}, {}
            )
            worker = self._usage_sync_manager(
                root, "worker", {"model/b": {"input": 20}}, {}
            )
            controller.token_usage_sync["model_epochs"]["future/model"] = [
                4, "controller",
            ]
            controller.reconcile_token_usage_sync(
                worker.token_usage_sync_snapshot(), "worker"
            )
            union = {
                **controller.token_usage_sync_snapshot(),
                "reconcile_origin": "worker",
            }

            worker._record_local_synced_tokens(
                "future/model", 7, 0, 0, None, None, "2026-08-26T19"
            )
            worker._rebuild_synced_token_usage()
            worker.merge_token_usage_sync(union)

            self.assertEqual(
                worker.token_usage_sync["model_epochs"]["future/model"],
                [4, "controller"],
            )
            self.assertEqual(
                worker.token_usage_sync["origins"]["worker"]["model_epochs"][
                    "future/model"
                ],
                [4, "controller"],
            )
            self.assertEqual(worker.token_stats["future/model"]["input"], 7)

    def test_reconciliation_drops_usage_learned_from_a_previous_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_controller = self._usage_sync_manager(
                root, "old-controller", {"private/model": {"input": 99}}, {}
            )
            worker = self._usage_sync_manager(
                root, "worker", {"local/model": {"input": 20}}, {}
            )
            worker.merge_token_usage_sync(
                old_controller.token_usage_sync_snapshot()
            )
            new_controller = self._usage_sync_manager(
                root, "new-controller", {"new/model": {"input": 5}}, {}
            )

            new_controller.reconcile_token_usage_sync(
                worker.token_usage_sync_snapshot(), "worker"
            )

            self.assertEqual(
                set(new_controller.token_usage_sync["origins"]),
                {"new-controller", "worker"},
            )
            self.assertEqual(
                set(new_controller.token_stats), {"new/model", "local/model"}
            )

    def test_rolling_speed_combines_samples_from_every_cluster_origin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self._usage_sync_manager(
                root, "controller", {"model": {"output": 100}}, {}
            )
            worker = self._usage_sync_manager(
                root, "worker", {"model": {"output": 200}}, {}
            )
            controller.token_usage_sync["origins"]["controller"][
                "speed_samples"
            ] = {"model": [{
                "tokens": 100, "started_at": 1000, "ended_at": 1010,
            }]}
            worker.token_usage_sync["origins"]["worker"]["speed_samples"] = {
                "model": [{
                    "tokens": 200, "started_at": 1000, "ended_at": 1010,
                }]
            }
            controller.merge_token_usage_sync(worker.token_usage_sync_snapshot())

            speed = controller.rolling_generation_speed(["model"])

            self.assertEqual(speed["tokens"], 300)
            self.assertAlmostEqual(speed["active_time_s"], 10.0)
            self.assertAlmostEqual(speed["tok_s"], 30.0)

    def test_cluster_speed_is_independent_of_remote_wall_clock_skew(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self._usage_sync_manager(
                root, "controller", {"model": {"output": 100}}, {}
            )
            worker = self._usage_sync_manager(
                root, "worker", {"model": {"output": 200}}, {}
            )
            controller.token_usage_sync["origins"]["controller"][
                "speed_samples"
            ] = {"model": [{
                "tokens": 100, "started_at": 1000, "ended_at": 1010,
            }]}
            worker.token_usage_sync["origins"]["worker"]["speed_samples"] = {
                "model": [{
                    "tokens": 200, "started_at": 1060, "ended_at": 1070,
                }]
            }
            controller.merge_token_usage_sync(worker.token_usage_sync_snapshot())

            speed = controller.rolling_generation_speed(["model"])

            self.assertEqual(speed["tokens"], 300)
            self.assertAlmostEqual(speed["tok_s"], 30.0)

    def test_hourly_usage_is_utc_and_bounded_to_activity_window(self) -> None:
        with mock.patch("manager.datetime") as mocked_datetime:
            mocked_datetime.now.return_value = datetime(2026, 8, 26, 18)
            self.assertEqual(Manager._usage_hour_key(), "2026-08-26T18")
            mocked_datetime.now.assert_called_once_with(timezone.utc)

        hourly = {
            "2025-08-20T00": {"model": {"input": 1}},
            "2025-08-22T00": {"model": {"input": 2}},
            "2026-08-26T18": {"model": {"input": 3}},
        }
        changed = Manager._prune_hourly_usage(
            hourly, datetime(2026, 8, 26, 18, tzinfo=timezone.utc)
        )

        self.assertTrue(changed)
        self.assertNotIn("2025-08-20T00", hourly)
        self.assertIn("2025-08-22T00", hourly)
        self.assertIn("2026-08-26T18", hourly)

    def test_hourly_ledger_pruning_scans_at_most_once_per_utc_hour(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = self._usage_sync_manager(
                Path(directory), "worker", {"model": {"input": 1}},
                {"2026-08-26T18": {"model": {"input": 1}}},
            )
            instance.token_usage_sync["hourly_retention_cutoff"] = "stale"

            with mock.patch.object(
                Manager,
                "_prune_hourly_usage",
                wraps=Manager._prune_hourly_usage,
            ) as prune:
                instance._record_local_synced_tokens(
                    "model", 1, 0, 0, None, None, "2026-08-26T19"
                )
                first_count = prune.call_count
                instance._record_local_synced_tokens(
                    "model", 1, 0, 0, None, None, "2026-08-26T19"
                )
                instance.token_usage_sync_snapshot()

            self.assertEqual(first_count, 1)
            self.assertEqual(prune.call_count, 1)

    def test_usage_merge_group_combines_counters_cost_and_concurrent_speed(self) -> None:
        instance = Manager.__new__(Manager)
        instance.token_stats = {
            "deepseek-thinking": {
                "input": 100, "cached": 40, "output": 100,
                "requests": 1, "gen_tokens": 100, "gen_time_s": 10,
            },
            "deepseek-chat": {
                "input": 200, "cached": 80, "output": 200,
                "requests": 1, "gen_tokens": 200, "gen_time_s": 10,
            },
        }
        instance.usage_aliases = {"deepseek-chat": "Non-thinking"}
        instance.usage_merge_groups = {
            "deepseek-thinking": "DeepSeek",
            "deepseek-chat": "DeepSeek",
        }
        instance.speed_samples = {
            "deepseek-thinking": [{
                "tokens": 100, "started_at": 1000, "ended_at": 1010,
            }],
            "deepseek-chat": [{
                "tokens": 200, "started_at": 1000, "ended_at": 1010,
            }],
        }
        instance.deployments = []
        instance.unsloth_settings = {}

        rows = instance.usage_rows()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "DeepSeek")
        self.assertEqual(rows[0]["stats"]["input"], 300)
        self.assertEqual(rows[0]["stats"]["cached"], 120)
        self.assertEqual(rows[0]["stats"]["output"], 300)
        self.assertEqual(rows[0]["stats"]["requests"], 2)
        self.assertAlmostEqual(rows[0]["speed"]["tok_s"], 30.0)

    def test_usage_routing_rule_rolls_source_into_destination_row(self) -> None:
        instance = Manager.__new__(Manager)
        instance.token_stats = {
            "xxx/abcdefg": {
                "input": 100, "cached": 25, "output": 40,
                "requests": 2, "gen_tokens": 40, "gen_time_s": 4,
            },
            "xxx/1234567": {
                "input": 300, "cached": 75, "output": 60,
                "requests": 3, "gen_tokens": 60, "gen_time_s": 6,
            },
        }
        instance.usage_aliases = {}
        instance.usage_merge_groups = {}
        instance.usage_routing_rules = {"xxx/abcdefg": "xxx/1234567"}
        instance.speed_samples = {}
        instance.deployments = []
        instance.unsloth_settings = {}

        rows = instance.usage_rows()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["key"], "model:xxx/1234567")
        self.assertEqual(rows[0]["label"], "xxx/1234567")
        self.assertEqual(rows[0]["stats"]["input"], 400)
        self.assertEqual(rows[0]["stats"]["cached"], 100)
        self.assertEqual(rows[0]["stats"]["output"], 100)
        self.assertEqual(rows[0]["stats"]["requests"], 5)
        self.assertEqual(rows[0]["routed_sources"], ["xxx/abcdefg"])
        self.assertEqual(
            [member["model"] for member in rows[0]["members"]],
            ["xxx/1234567", "xxx/abcdefg"],
        )

    def test_usage_routing_rule_prices_source_at_final_master_rates(self) -> None:
        instance = Manager.__new__(Manager)
        instance.token_stats = {
            "gemini-2.5-flash": {
                "input": 1_000_000, "cached": 0, "output": 1_000_000,
            },
            "claude-opus-4": {
                "input": 1_000_000, "cached": 0, "output": 1_000_000,
            },
        }
        instance.usage_aliases = {}
        instance.usage_merge_groups = {}
        instance.usage_routing_rules = {
            "gemini-2.5-flash": "master-alias",
            "master-alias": "claude-opus-4",
        }
        instance.speed_samples = {}
        instance.deployments = []
        instance.unsloth_settings = {}

        rows = instance.usage_rows()

        # Claude Opus costs $15/M input + $75/M output. Both the direct
        # master usage and routed Gemini usage must use those rates.
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "claude-opus-4")
        self.assertEqual(rows[0]["total_cost"], 180.0)

    def test_usage_routing_falls_back_to_priced_member_when_master_unpriced(self) -> None:
        instance = Manager.__new__(Manager)
        instance.token_stats = {
            "legacy-master": {
                "input": 1_000_000, "cached": 0, "output": 1_000_000,
            },
            "claude-opus-4": {
                "input": 1_000_000, "cached": 0, "output": 1_000_000,
            },
        }
        instance.usage_aliases = {}
        instance.usage_merge_groups = {}
        instance.usage_routing_rules = {"claude-opus-4": "legacy-master"}
        instance.speed_samples = {}
        instance.deployments = []
        instance.unsloth_settings = {}

        row = instance.usage_rows()[0]

        self.assertEqual(row["route_target"], "legacy-master")
        self.assertEqual(row["pricing_model"], "claude-opus-4")
        self.assertEqual(row["total_cost"], 180.0)

    def test_usage_cache_estimate_splits_hits_misses_and_adjusts_cost(self) -> None:
        instance = Manager.__new__(Manager)
        instance.token_stats = {
            "claude-opus-4": {
                "input": 1_000_000, "cached": 100_000, "output": 0,
            },
        }
        instance.session_token_stats = {}
        instance.usage_aliases = {}
        instance.usage_merge_groups = {}
        instance.usage_routing_rules = {}
        instance.usage_cache_estimates = {
            "model:claude-opus-4": {
                "rate_pct": 80.0,
                "legacy_input": 1_000_000,
                "measured_cached": 100_000,
                "estimated_cached": 700_000,
            },
        }
        instance.speed_samples = {}
        instance.deployments = []
        instance.unsloth_settings = {}

        row = instance.usage_rows()[0]

        self.assertEqual(row["stats"]["measured_cached"], 100_000)
        self.assertEqual(row["stats"]["estimated_cached"], 700_000)
        self.assertEqual(row["stats"]["cached"], 800_000)
        self.assertEqual(row["stats"]["input_miss"], 200_000)
        self.assertTrue(row["cost_estimated"])
        self.assertEqual(row["total_cost"], 4.2)

    def test_usage_routing_carries_source_cache_estimate_to_destination(self) -> None:
        instance = Manager.__new__(Manager)
        instance.token_stats = {
            "legacy-source": {
                "input": 1_000_000, "cached": 100_000, "output": 0,
            },
        }
        instance.usage_aliases = {}
        instance.usage_merge_groups = {}
        instance.usage_routing_rules = {"legacy-source": "claude-opus-4"}
        instance.usage_cache_estimates = {
            "model:legacy-source": {"estimated_cached": 700_000},
        }
        instance.speed_samples = {}
        instance.deployments = []
        instance.unsloth_settings = {}

        row = instance.usage_rows()[0]

        self.assertEqual(row["key"], "model:claude-opus-4")
        self.assertEqual(row["stats"]["measured_cached"], 100_000)
        self.assertEqual(row["stats"]["estimated_cached"], 700_000)
        self.assertEqual(row["stats"]["cached"], 800_000)
        self.assertEqual(row["total_cost"], 4.2)

    def test_usage_merge_group_reprices_estimated_cache_hits(self) -> None:
        instance = Manager.__new__(Manager)
        instance.token_stats = {
            "claude-opus-4": {
                "input": 1_000_000, "cached": 100_000, "output": 0,
            },
        }
        instance.usage_aliases = {}
        instance.usage_merge_groups = {"claude-opus-4": "Combined"}
        instance.usage_routing_rules = {}
        instance.usage_cache_estimates = {
            "group:Combined": {"estimated_cached": 700_000},
        }
        instance.speed_samples = {}
        instance.deployments = []
        instance.unsloth_settings = {}

        row = instance.usage_rows()[0]

        self.assertEqual(row["stats"]["cached"], 800_000)
        self.assertTrue(row["cost_estimated"])
        self.assertEqual(row["total_cost"], 4.2)

    def test_usage_routing_rules_persist_update_delete_and_reject_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.usage_routing_rules = {}
            instance.usage_routing_rules_path = (
                Path(directory) / "usage_routing_rules.json"
            )

            instance.update_usage_routing_rule("model/a", "model/b")
            instance.update_usage_routing_rule("model/b", "model/c")
            self.assertEqual(
                json.loads(instance.usage_routing_rules_path.read_text()),
                {"model/a": "model/b", "model/b": "model/c"},
            )
            with self.assertRaisesRegex(ValueError, "cycle"):
                instance.update_usage_routing_rule("model/c", "model/a")
            self.assertNotIn("model/c", instance.usage_routing_rules)

            instance.update_usage_routing_rule("model/a", "model/c")
            self.assertEqual(instance.usage_routing_rules["model/a"], "model/c")
            instance.delete_usage_routing_rule("model/a")
            self.assertNotIn("model/a", instance.usage_routing_rules)
            self.assertEqual(
                json.loads(instance.usage_routing_rules_path.read_text()),
                {"model/b": "model/c"},
            )

    def test_get_token_stats_exposes_routing_rules(self) -> None:
        instance = Manager.__new__(Manager)
        instance.token_stats = {
            "xxx/abcdefg": {
                "input": 100, "cached": 25, "output": 40,
                "requests": 2, "gen_tokens": 40, "gen_time_s": 4,
            },
            "xxx/1234567": {
                "input": 300, "cached": 75, "output": 60,
                "requests": 3, "gen_tokens": 60, "gen_time_s": 6,
            },
        }
        instance.usage_aliases = {}
        instance.usage_merge_groups = {"xxx/abcdefg": "Fleet"}
        instance.usage_routing_rules = {"xxx/abcdefg": "xxx/1234567"}
        instance.speed_samples = {}
        instance.deployments = []
        instance.unsloth_settings = {}

        stats = instance.get_token_stats()

        self.assertEqual(stats["routing_rules"], {"xxx/abcdefg": "xxx/1234567"})
        self.assertEqual(stats["merge_groups"], {"xxx/abcdefg": "Fleet"})
        self.assertEqual(stats["groups"][0]["key"], "model:xxx/1234567")

    def test_rolling_speed_uses_only_newest_one_million_output_tokens(self) -> None:
        instance = Manager.__new__(Manager)
        instance.speed_samples = {
            "model": [
                {"tokens": 9_000_000, "started_at": 0, "ended_at": 90},
                {"tokens": 2_000_000, "started_at": 100, "ended_at": 110},
            ],
        }

        speed = instance.rolling_generation_speed(["model"])

        self.assertEqual(speed["tokens"], 1_000_000)
        self.assertAlmostEqual(speed["active_time_s"], 5.0)

    def test_rolling_speed_sums_five_overlapping_streams(self) -> None:
        instance = Manager.__new__(Manager)
        instance.speed_samples = {
            "model": [
                {"tokens": 100, "started_at": 1000, "ended_at": 1010}
                for _ in range(5)
            ],
        }

        speed = instance.rolling_generation_speed(["model"])

        self.assertEqual(speed["tokens"], 500)
        self.assertAlmostEqual(speed["active_time_s"], 10.0)
        self.assertAlmostEqual(speed["tok_s"], 50.0)

    def test_rolling_speed_cuts_boundary_across_concurrent_streams(self) -> None:
        instance = Manager.__new__(Manager)
        instance.speed_samples = {
            "model": [
                {"tokens": 300_000, "started_at": 1000, "ended_at": 1010}
                for _ in range(5)
            ],
        }

        speed = instance.rolling_generation_speed(["model"])

        self.assertEqual(speed["tokens"], 1_000_000)
        self.assertAlmostEqual(speed["active_time_s"], 20 / 3)
        self.assertAlmostEqual(speed["tok_s"], 150_000.0)

    def test_cluster_pricing_applies_cache_miss_hit_and_output_rates(self) -> None:
        instance = Manager.__new__(Manager)
        instance.deployments = [{
            "model": "deepseek/model",
            "pricing_model_key": "deepseek/model [nvfp4]",
            "launch_settings": {
                "input_cost_per_1m": 2.0,
                "cache_cost_per_1m": 0.5,
                "output_cost_per_1m": 4.0,
            },
        }]
        instance.unsloth_settings = {}

        cost = instance.calculate_cost("deepseek/model [nvfp4]", {
            "input": 1_000_000,
            "cached": 250_000,
            "output": 500_000,
        })

        self.assertEqual(cost["input_cost_per_1m"], 2.0)
        self.assertEqual(cost["cache_cost_per_1m"], 0.5)
        self.assertEqual(cost["output_cost_per_1m"], 4.0)
        self.assertEqual(cost["total_cost"], 3.62)

    def test_cluster_pricing_matches_exact_usage_identity(self) -> None:
        instance = Manager.__new__(Manager)
        instance.deployments = [
            {
                "model": "org/model",
                "pricing_model_key": "org/model",
                "launch_settings": {"output_cost_per_1m": 1.0},
            },
            {
                "model": "org/model-large",
                "pricing_model_key": "org/model-large",
                "launch_settings": {"output_cost_per_1m": 9.0},
            },
        ]
        instance.unsloth_settings = {}

        cost = instance.calculate_cost(
            "org/model-large", {"input": 0, "cached": 0, "output": 1_000_000}
        )

        self.assertEqual(cost["output_cost_per_1m"], 9.0)

    def test_pricing_rejects_non_finite_numbers(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "non-negative number"):
                    Manager._pricing_value(value, "output_cost_per_1m")

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

    async def test_recipe_delete_is_durable_and_clears_transient_launch_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.lock = asyncio.Lock()
            instance.recipes = [{"id": "delete-me"}, {"id": "keep-me"}]
            instance.recipe_launches = {
                "delete-me": {"phase": "ready"},
                "keep-me": {"phase": "starting"},
            }
            instance.recipes_path = Path(directory) / "recipes.json"

            self.assertTrue(await instance.delete_recipe("delete-me"))
            self.assertFalse(await instance.delete_recipe("missing"))

            self.assertEqual(instance.recipes, [{"id": "keep-me"}])
            self.assertEqual(instance.recipe_launches, {"keep-me": {"phase": "starting"}})
            self.assertEqual(
                json.loads(instance.recipes_path.read_text()),
                [{"id": "keep-me"}],
            )

    async def test_sglang_launch_survives_recipe_deletion_after_progress_starts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.lock = asyncio.Lock()
            instance.recipes = [{"id": "delete-me"}]
            instance.recipe_launches = {}
            instance.recipes_path = Path(directory) / "recipes.json"
            instance.settings = {
                "hf_cache": str(Path(directory) / "huggingface"),
                "shm_size": "1g",
            }
            instance.client = mock.Mock()
            instance.client.images.get.return_value = mock.Mock()
            container = mock.Mock()
            container.reload.return_value = None
            instance._run_managed_container = mock.Mock(return_value=container)
            instance._container_summary = mock.Mock(return_value={"name": "sglang-test"})
            instance._created_container_model_source = mock.Mock(return_value="org/model")
            instance._cluster_launch_update = mock.Mock()
            instance._build_volumes = mock.Mock(return_value={})
            instance._container_hf_environment = mock.Mock(return_value={})
            instance._allocate_port = mock.AsyncMock(return_value=8001)

            async def delete_during_evict(*, protect: str) -> None:
                self.assertEqual(protect, "sglang")
                self.assertTrue(await instance.delete_recipe("delete-me"))

            instance.evict_other_backends = mock.AsyncMock(side_effect=delete_during_evict)

            result = await instance.create_container(
                "org/model", engine="sglang", recipe_id="delete-me",
            )

            self.assertEqual(result, {"name": "sglang-test", "model_source": "org/model"})
            self.assertEqual(instance.recipes, [])
            self.assertEqual(instance.recipe_launches, {})
            instance._run_managed_container.assert_called_once()

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

    def test_created_container_model_source_uses_node_runtime_provenance(self) -> None:
        instance = Manager.__new__(Manager)
        instance._resolve_local_path = mock.Mock(return_value=None)
        container = mock.Mock()

        container.exec_run.return_value = (0, b"")
        self.assertEqual(
            instance._created_container_model_source(container, "models/customer"),
            "local",
        )
        container.exec_run.assert_called_once_with([
            "test", "-e", "--", "models/customer",
        ])

        container.exec_run.reset_mock()
        container.exec_run.return_value = (1, b"")
        self.assertEqual(
            instance._created_container_model_source(container, "org/model"),
            "public_repository",
        )

        instance._resolve_local_path.return_value = "/models/mounted"
        container.exec_run.reset_mock()
        self.assertEqual(
            instance._created_container_model_source(container, "/models/mounted"),
            "local",
        )
        container.exec_run.assert_not_called()

        instance._resolve_local_path.return_value = None
        container.exec_run.side_effect = RuntimeError("container unavailable")
        self.assertEqual(
            instance._created_container_model_source(container, "org/model"),
            "unknown",
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

    async def test_cluster_logs_preserve_launch_progress_when_docker_is_offline(self) -> None:
        instance = Manager.__new__(Manager)
        instance.cluster_member_launches = {}
        instance._cluster_launch_update(
            "cluster-test-r0", "pulling_image", "Downloading Docker image",
            model="example/model",
            cluster_member={"deployment_id": "test", "node_id": "local", "rank": 0},
        )

        async def is_managed_container(name):
            raise docker.errors.DockerException("daemon offline")

        instance.is_managed_container = is_managed_container
        logs = await instance.get_cluster_member_logs("cluster-test-r0")

        self.assertIn("Controller launch progress", logs)
        self.assertIn("Downloading Docker image", logs)
        self.assertIn("Docker is unavailable: daemon offline", logs)

    async def test_cluster_logs_preserve_progress_on_docker_transport_failure(self) -> None:
        instance = Manager.__new__(Manager)
        instance.cluster_member_launches = {}
        instance._cluster_launch_update(
            "cluster-test-r0", "creating", "Creating the model container",
            model="example/model",
            cluster_member={"deployment_id": "test", "node_id": "local", "rank": 0},
        )

        async def is_managed_container(name):
            raise requests.exceptions.ConnectionError("socket disappeared")

        instance.is_managed_container = is_managed_container
        logs = await instance.get_cluster_member_logs("cluster-test-r0")

        self.assertIn("Controller launch progress", logs)
        self.assertIn("Creating the model container", logs)
        self.assertIn("Docker is unavailable: socket disappeared", logs)

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
                return {
                    "id": f"container-{node_id}", "status": "running",
                    "model_source": "public_repository",
                }

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
            self.assertEqual(deployment["model_source"], "public_repository")
            self.assertEqual(
                deployment["launch_settings"]["model_source"],
                "public_repository",
            )
            self.assertTrue(all(
                member["model_source"] == "public_repository"
                for member in deployment["members"]
            ))
            self.assertEqual(len(captured), 2)
            for rank, (node_id, payload) in enumerate(captured):
                args = payload["extra_args"]
                self.assertEqual(args[args.index("--node-rank") + 1], str(rank))
                self.assertEqual(args[args.index("--nnodes") + 1], "2")
                self.assertEqual(args[args.index("--tensor-parallel-size") + 1], "2")
                self.assertEqual(args[args.index("--pipeline-parallel-size") + 1], "1")
                self.assertEqual(args[args.index("--master-addr") + 1], "169.254.10.1")
                self.assertEqual("--headless" in args, rank > 0)
                self.assertEqual(args.count("--enable-prompt-tokens-details"), 1)
                self.assertEqual(args.count("--tensor-parallel-size"), 1)
                self.assertEqual(args.count("--pipeline-parallel-size"), 1)
                self.assertEqual(payload["cluster_member"]["fabric_interface"],
                                 "cx7-local" if node_id == "local" else "cx7-remote")

    def test_vllm_prompt_token_details_flag_is_idempotent_and_overridable(self) -> None:
        self.assertEqual(
            Manager._with_vllm_prompt_token_details(["--max-model-len", "1024"]),
            ["--max-model-len", "1024", "--enable-prompt-tokens-details"],
        )
        self.assertEqual(
            Manager._with_vllm_prompt_token_details([
                "--enable-prompt-tokens-details",
            ]),
            ["--enable-prompt-tokens-details"],
        )
        self.assertEqual(
            Manager._with_vllm_prompt_token_details([
                "--no-enable-prompt-tokens-details",
            ]),
            ["--no-enable-prompt-tokens-details"],
        )

    def test_saved_vllm_deployments_are_migrated_for_cache_reporting(self) -> None:
        instance = Manager.__new__(Manager)
        instance.deployments = [
            {
                "engine": "vllm",
                "settings_dirty": False,
                "launch_settings": {
                    "engine": "vllm",
                    "extra_args": ["--max-model-len", "1024"],
                },
            },
            {
                "engine": "sglang",
                "settings_dirty": False,
                "launch_settings": {"engine": "sglang", "extra_args": []},
            },
        ]

        self.assertTrue(instance._migrate_vllm_prompt_token_details())
        self.assertIn(
            "--enable-prompt-tokens-details",
            instance.deployments[0]["launch_settings"]["extra_args"],
        )
        self.assertTrue(instance.deployments[0]["settings_dirty"])
        self.assertEqual(
            instance.deployments[1]["launch_settings"]["extra_args"], []
        )
        self.assertFalse(instance._migrate_vllm_prompt_token_details())

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
                    "--max-cudagraph-capture-size", "72",
                    "--tensor-parallel-size", "2",
                    "--pipeline-parallel-size", "1",
                ],
                "image": "example/vllm:test",
                "launch_controls": {
                    "context_window": 131072,
                    "max_cudagraph_capture_size": 12,
                },
            })

            self.assertTrue(updated["settings_dirty"])
            self.assertEqual(updated["name"], "Faster cluster")
            self.assertEqual(updated["launch_settings"]["port"], 8000)
            self.assertEqual(
                instance._cli_option(
                    updated["launch_settings"]["extra_args"],
                    {"--max-model-len", "--max-model-length"}, int,
                ),
                131072,
            )
            self.assertEqual(
                instance._cli_option(
                    updated["launch_settings"]["extra_args"],
                    {"--max-cudagraph-capture-size"}, int,
                ),
                12,
            )
            persisted = json.loads(instance.deployments_path.read_text())
            self.assertEqual(persisted[0]["launch_settings"]["image"], "example/vllm:test")
            self.assertEqual(
                instance._cli_option(
                    persisted[0]["launch_settings"]["extra_args"],
                    {"--max-cudagraph-capture-size"}, int,
                ),
                12,
            )

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

    def test_vllm_capacity_log_parser_extracts_safe_full_context_limit(self) -> None:
        reports = Manager._parse_vllm_capacity_log("""
            GPU KV cache size: 2,288,000 tokens
            Maximum concurrency for 400,000 tokens per request: 5.72x
        """)

        self.assertEqual(reports, [{
            "context_tokens": 400_000,
            "maximum_concurrency": 5.72,
            "safe_concurrency": 5,
            "gpu_kv_cache_tokens": 2_288_000,
        }])

    async def test_capacity_tick_records_safe_configuration_without_redeploy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.settings = {"vllm_auto_adjust_concurrency": True}
            instance.deployments_path = Path(directory) / "deployments.json"
            deployment = {
                "id": "deployment-1",
                "model": "example/Model",
                "engine": "vllm",
                "status": "starting",
                "members": [{"rank": 0}],
                "launch_settings": {
                    "engine": "vllm",
                    "extra_args": [
                        "--max-model-len", "400000", "--max-num-seqs", "5",
                    ],
                },
            }
            instance.deployments = [deployment]
            instance._deployment_capacity_reports = mock.AsyncMock(return_value=[
                {"rank": 0, "context_tokens": 400_000,
                 "maximum_concurrency": 5.72, "safe_concurrency": 5,
                 "gpu_kv_cache_tokens": 2_288_000},
            ])
            instance._auto_reduce_vllm_concurrency = mock.AsyncMock()

            await instance._deployment_capacity_tick()

            instance._auto_reduce_vllm_concurrency.assert_not_awaited()
            self.assertEqual(deployment["kv_capacity"]["status"], "within_limit")
            self.assertEqual(
                deployment["kv_capacity"]["reports"][0]["gpu_kv_cache_tokens"],
                2_288_000,
            )

    async def test_capacity_tick_ignores_report_for_another_context_length(self) -> None:
        instance = Manager.__new__(Manager)
        instance.settings = {"vllm_auto_adjust_concurrency": True}
        deployment = {
            "id": "deployment-1",
            "engine": "vllm",
            "status": "starting",
            "members": [{"rank": 0}],
            "launch_settings": {
                "engine": "vllm",
                "extra_args": [
                    "--max-model-len", "400000", "--max-num-seqs", "6",
                ],
            },
        }
        instance.deployments = [deployment]
        instance._deployment_capacity_reports = mock.AsyncMock(return_value=[
            {"rank": 0, "context_tokens": 131_072,
             "maximum_concurrency": 2.5, "safe_concurrency": 2,
             "gpu_kv_cache_tokens": 327_680},
        ])
        instance._auto_reduce_vllm_concurrency = mock.AsyncMock()

        await instance._deployment_capacity_tick()

        instance._auto_reduce_vllm_concurrency.assert_not_awaited()
        self.assertNotIn("kv_capacity", deployment)

    async def test_capacity_tick_uses_lowest_rank_and_requests_redeploy(self) -> None:
        instance = Manager.__new__(Manager)
        instance.settings = {"vllm_auto_adjust_concurrency": True}
        deployment = {
            "id": "deployment-1",
            "engine": "vllm",
            "status": "starting",
            "members": [{"rank": 0}, {"rank": 1}],
            "launch_settings": {
                "engine": "vllm",
                "extra_args": [
                    "--max-model-len", "400000", "--max-num-seqs", "6",
                ],
            },
        }
        instance.deployments = [deployment]
        instance._deployment_capacity_reports = mock.AsyncMock(return_value=[
            {"rank": 0, "context_tokens": 400_000,
             "maximum_concurrency": 5.72, "safe_concurrency": 5},
            {"rank": 1, "context_tokens": 400_000,
             "maximum_concurrency": 4.98, "safe_concurrency": 4},
        ])
        instance._auto_reduce_vllm_concurrency = mock.AsyncMock()

        await instance._deployment_capacity_tick()

        args = instance._auto_reduce_vllm_concurrency.await_args.args
        self.assertEqual(args[0], "deployment-1")
        self.assertEqual(args[1], 4)
        self.assertEqual(args[2]["maximum_concurrency"], 4.98)

    async def test_capacity_reports_deep_scans_old_deployment_only_once(self) -> None:
        instance = Manager.__new__(Manager)
        deployment = {
            "id": "deployment-1",
            "members": [{"rank": 0, "node_id": "local"}],
        }
        calls = []

        async def member_action(member, action, **kwargs):
            tail = kwargs["log_tail"]
            calls.append(tail)
            if tail == 100_000:
                return {"logs": """
                    GPU KV cache size: 1,229,206 tokens
                    Maximum concurrency for 420,000 tokens per request: 2.93x
                """}
            return {"logs": "recent runtime output"}

        instance._member_action = member_action

        first = await instance._deployment_capacity_reports(deployment)
        second = await instance._deployment_capacity_reports(deployment)

        self.assertEqual(calls, [3000, 100_000, 3000])
        self.assertEqual(first[0]["safe_concurrency"], 2)
        self.assertEqual(first[0]["gpu_kv_cache_tokens"], 1_229_206)
        self.assertEqual(second, [])

    async def test_capacity_redeploy_guard_delays_model_resolution(self) -> None:
        instance = Manager.__new__(Manager)
        instance._capacity_redeploying_models = {"served-model"}
        container = {
            "name": "rank-0",
            "model": "example/Model",
            "served_models": ["served-model"],
            "status": "running",
        }
        instance.list_containers = mock.AsyncMock(return_value=[container])
        instance._check_ready = mock.AsyncMock(return_value=True)
        instance._mark_active = mock.Mock()
        instance._sparkrun_targets = mock.AsyncMock(return_value={})

        resolving = asyncio.create_task(
            instance._resolve_vllm_target("served-model")
        )
        await asyncio.sleep(0.01)

        self.assertFalse(resolving.done())
        instance.list_containers.assert_not_awaited()
        instance._capacity_redeploying_models.clear()
        resolved = await asyncio.wait_for(resolving, timeout=1)

        self.assertIs(resolved, container)
        instance.list_containers.assert_awaited_once()

    async def test_capacity_redeploy_stops_all_ranks_and_saves_lower_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance._deployment_action_lock = asyncio.Lock()
            instance.deployments_path = Path(directory) / "deployments.json"
            deployment = {
                "id": "deployment-1",
                "model": "example/Model",
                "engine": "vllm",
                "status": "starting",
                "settings_dirty": False,
                "members": [
                    {"rank": 0, "node_id": "local", "container_name": "rank-0"},
                    {"rank": 1, "node_id": "remote", "container_name": "rank-1"},
                ],
                "launch_settings": {
                    "engine": "vllm",
                    "extra_args": [
                        "--max-model-len", "400000", "--max-num-seqs", "6",
                    ],
                },
            }
            instance.deployments = [deployment]
            actions = []

            async def member_action(member, action, **kwargs):
                self.assertIn(
                    "example/Model", instance._capacity_redeploying_models
                )
                self.assertIn(
                    "served-model", instance._capacity_redeploying_models
                )
                actions.append((action, member["rank"]))
                return {"ok": True}

            replacement = {"id": "deployment-2"}
            instance._member_action = member_action
            instance.list_containers = mock.AsyncMock(return_value=[{
                "name": "rank-0",
                "model": "example/Model",
                "served_models": ["served-model"],
            }])
            instance._deployment_action_locked = mock.AsyncMock(return_value={
                "ok": True, "deployment": replacement,
            })
            observation = {
                "maximum_concurrency": 4.98,
                "safe_concurrency": 4,
            }

            await instance._auto_reduce_vllm_concurrency(
                "deployment-1", 4, observation,
            )

            controls = instance._deployment_launch_controls(
                deployment["launch_settings"]
            )
            self.assertEqual(controls["max_concurrency"], 4)
            self.assertEqual(actions, [("stop", 0), ("stop", 1)])
            instance._deployment_action_locked.assert_awaited_once_with(
                "deployment-1", "start"
            )
            self.assertEqual(
                replacement["auto_concurrency_adjustment"]["from"], 6
            )
            self.assertEqual(
                replacement["auto_concurrency_adjustment"]["to"], 4
            )
            self.assertEqual(
                replacement["kv_capacity"]["status"], "awaiting_recheck"
            )
            self.assertEqual(instance._capacity_redeploying_models, set())

    async def test_capacity_redeploy_does_not_restart_after_partial_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance._deployment_action_lock = asyncio.Lock()
            instance.deployments_path = Path(directory) / "deployments.json"
            deployment = {
                "id": "deployment-1",
                "engine": "vllm",
                "status": "starting",
                "members": [
                    {"rank": 0, "node_id": "local", "container_name": "rank-0"},
                    {"rank": 1, "node_id": "remote", "container_name": "rank-1"},
                ],
                "launch_settings": {
                    "engine": "vllm",
                    "extra_args": [
                        "--max-model-len", "400000", "--max-num-seqs", "6",
                    ],
                },
            }
            instance.deployments = [deployment]

            async def member_action(member, action, **kwargs):
                if member["rank"] == 1:
                    raise RuntimeError("node unavailable")
                return {"ok": True}

            instance._member_action = member_action
            instance._deployment_action_locked = mock.AsyncMock()

            await instance._auto_reduce_vllm_concurrency(
                "deployment-1", 4, {
                    "maximum_concurrency": 4.98,
                    "safe_concurrency": 4,
                },
            )

            instance._deployment_action_locked.assert_not_awaited()
            self.assertEqual(deployment["status"], "error")
            self.assertIn("no rank was restarted", deployment["error"])

    def test_running_deployment_settings_cannot_be_changed(self) -> None:
        instance = Manager.__new__(Manager)
        instance.deployments = [{
            "id": "deployment-1", "status": "ready", "members": [],
        }]

        with self.assertRaisesRegex(ValueError, "stop the cluster"):
            instance.update_deployment_settings("deployment-1", {"model": "example/New"})

    async def test_running_deployment_pricing_can_be_changed_without_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.deployments_path = Path(directory) / "deployments.json"
            instance.deployments = [{
                "id": "deployment-1",
                "model": "example/Model",
                "status": "ready",
                "settings_dirty": False,
                "launch_settings": {"model": "example/Model"},
            }]

            result = await instance.update_deployment_pricing("deployment-1", {
                "input_cost_per_1m": 1.25,
                "cache_cost_per_1m": 0.1,
                "output_cost_per_1m": 3.5,
            })

            self.assertTrue(result["ok"])
            self.assertFalse(instance.deployments[0]["settings_dirty"])
            saved = json.loads(instance.deployments_path.read_text())[0]
            self.assertEqual(saved["launch_settings"]["input_cost_per_1m"], 1.25)
            self.assertEqual(saved["launch_settings"]["cache_cost_per_1m"], 0.1)
            self.assertEqual(saved["launch_settings"]["output_cost_per_1m"], 3.5)

    async def test_pricing_recovers_full_legacy_launch_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.deployments_path = Path(directory) / "deployments.json"
            instance.deployments = [{
                "id": "deployment-1", "name": "Legacy", "model": "org/model",
                "engine": "vllm", "mode": "single", "node_ids": ["local"],
                "api_port": 8123, "launch_settings": None,
            }]
            instance.list_containers = mock.AsyncMock(return_value=[{
                "deployment_id": "deployment-1", "rank": 0,
                "image": "vllm/image:tag", "stats_key": "org/model [awq]",
                "load_settings": {
                    "command_flags": "--max-model-len 400000 --quantization awq",
                    "gpu_memory_utilization": 0.8,
                },
            }])

            await instance.update_deployment_pricing(
                "deployment-1", {"output_cost_per_1m": 2.5}
            )

            saved = instance.deployments[0]
            self.assertEqual(saved["launch_settings"]["image"], "vllm/image:tag")
            self.assertIn("--max-model-len", saved["launch_settings"]["extra_args"])
            self.assertEqual(saved["pricing_model_key"], "org/model [awq]")

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

    async def test_sanitized_malformed_deployment_cannot_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.deployments_path = Path(directory) / "deployments.json"
            deployment = {
                "id": "deployment-bad", "name": "Unsafe", "model": "example/Model",
                "engine": "vllm", "mode": "single", "node_ids": ["local"],
                "status": "error", "settings_dirty": True,
                "error": PERSISTED_DEPLOYMENT_ARGS_ERROR,
                "launch_settings_error": PERSISTED_DEPLOYMENT_ARGS_ERROR,
                "members": [{"node_id": "local", "container_name": "old-r0"}],
                "launch_settings": {
                    "model": "example/Model", "engine": "vllm",
                    "deployment_mode": "single", "node_ids": ["local"],
                    "extra_args": [],
                },
            }
            instance.deployments = [deployment]
            instance._member_action = mock.AsyncMock(return_value={"ok": True})
            instance.create_deployment = mock.AsyncMock()

            stopped = await instance.deployment_action("deployment-bad", "stop")
            with self.assertRaisesRegex(ValueError, "edit extra_args"):
                await instance.deployment_action("deployment-bad", "start")

            self.assertTrue(stopped["ok"])
            instance._member_action.assert_awaited_once_with(
                deployment["members"][0], "stop",
            )
            instance.create_deployment.assert_not_awaited()
            self.assertEqual(deployment["status"], "stopped")
            self.assertEqual(
                deployment["launch_settings_error"],
                PERSISTED_DEPLOYMENT_ARGS_ERROR,
            )

    async def test_explicit_extra_args_repair_allows_dirty_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.deployments_path = Path(directory) / "deployments.json"
            instance.deployments = [{
                "id": "deployment-bad", "name": "Unsafe", "model": "example/Model",
                "engine": "vllm", "mode": "single", "node_ids": ["local"],
                "status": "error", "settings_dirty": True,
                "error": PERSISTED_DEPLOYMENT_ARGS_ERROR,
                "launch_settings_error": PERSISTED_DEPLOYMENT_ARGS_ERROR,
                "members": [],
                "launch_settings": {
                    "model": "example/Model", "engine": "vllm",
                    "deployment_mode": "single", "node_ids": ["local"],
                    "extra_args": [],
                },
            }]

            with self.assertRaisesRegex(ValueError, "explicitly repaired"):
                instance.update_deployment_settings(
                    "deployment-bad", {"deployment_name": "Still unsafe"},
                )
            with self.assertRaisesRegex(ValueError, "explicitly repaired"):
                instance.update_deployment_settings(
                    "deployment-bad", {"extra_args": None},
                )
            updated = instance.update_deployment_settings(
                "deployment-bad", {"extra_args": ["--max-model-len", "8192"]},
            )

            self.assertEqual(updated["status"], "stopped")
            self.assertNotIn("launch_settings_error", updated)
            self.assertIsNone(updated["error"])
            self.assertIn("--max-model-len", updated["launch_settings"]["extra_args"])
            persisted = json.loads(instance.deployments_path.read_text())
            self.assertNotIn("launch_settings_error", persisted[0])

            instance._member_action = mock.AsyncMock(return_value={"ok": True})

            async def create_deployment(body):
                replacement = {
                    "id": "deployment-new", "status": "starting", "members": [],
                }
                instance.deployments.append(replacement)
                return replacement

            instance.create_deployment = mock.AsyncMock(side_effect=create_deployment)
            result = await instance.deployment_action("deployment-bad", "start")

            self.assertTrue(result["ok"])
            instance.create_deployment.assert_awaited_once()
            launch_body = instance.create_deployment.await_args.args[0]
            self.assertIn("--max-model-len", launch_body["extra_args"])
            self.assertEqual(
                [deployment["id"] for deployment in instance.deployments],
                ["deployment-new"],
            )

    def test_explicit_args_repair_preserves_unrelated_deployment_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.deployments_path = Path(directory) / "deployments.json"
            instance.deployments = [{
                "id": "deployment-bad", "name": "Unsafe", "model": "example/Model",
                "engine": "vllm", "mode": "single", "node_ids": ["local"],
                "status": "error", "settings_dirty": True,
                "error": (
                    "worker unavailable; " + PERSISTED_DEPLOYMENT_ARGS_ERROR
                ),
                "launch_settings_error": PERSISTED_DEPLOYMENT_ARGS_ERROR,
                "members": [],
                "launch_settings": {
                    "model": "example/Model", "engine": "vllm",
                    "deployment_mode": "single", "node_ids": ["local"],
                    "extra_args": [],
                },
            }]

            updated = instance.update_deployment_settings(
                "deployment-bad", {"extra_args": ["--max-model-len", "8192"]},
            )

            self.assertEqual(updated["status"], "error")
            self.assertEqual(updated["error"], "worker unavailable")
            self.assertNotIn("launch_settings_error", updated)
            persisted = json.loads(instance.deployments_path.read_text())
            self.assertEqual(persisted[0]["error"], "worker unavailable")

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

    async def test_controller_detaches_worker_before_removing_registry_record(self) -> None:
        instance = Manager.__new__(Manager)
        instance.deployments = []
        instance.node_registry = mock.Mock()
        instance.node_registry.get.return_value = {"id": "remote-1", "name": "Worker"}
        instance.node_registry.request = mock.AsyncMock(return_value={"ok": True})
        instance.node_registry.remove.return_value = True

        removed = await instance.detach_cluster_node("remote-1")

        self.assertTrue(removed)
        instance.node_registry.request.assert_awaited_once_with(
            "remote-1", "POST", "/api/agent/onboarding/detach", timeout=10,
        )
        instance.node_registry.remove.assert_called_once_with("remote-1")

    async def test_force_forget_skips_unreachable_worker_detach(self) -> None:
        instance = Manager.__new__(Manager)
        instance.deployments = []
        instance.node_registry = mock.Mock()
        instance.node_registry.get.return_value = {"id": "remote-1", "name": "Worker"}
        instance.node_registry.request = mock.AsyncMock()
        instance.node_registry.remove.return_value = True

        removed = await instance.detach_cluster_node("remote-1", force=True)

        self.assertTrue(removed)
        instance.node_registry.request.assert_not_awaited()
        instance.node_registry.remove.assert_called_once_with("remote-1")

    async def test_protocol_v1_worker_without_detach_endpoint_uses_legacy_removal(self) -> None:
        request = httpx.Request("POST", "http://worker/api/agent/onboarding/detach")
        response = httpx.Response(404, request=request)
        missing_endpoint = httpx.HTTPStatusError(
            "not found", request=request, response=response,
        )
        agent_error = RuntimeError("Worker agent error: not found")
        agent_error.__cause__ = missing_endpoint
        instance = Manager.__new__(Manager)
        instance.deployments = []
        instance.node_registry = mock.Mock()
        instance.node_registry.get.return_value = {
            "id": "remote-1",
            "name": "Worker",
            "protocol_version": AGENT_PROTOCOL_VERSION,
        }
        instance.node_registry.request = mock.AsyncMock(side_effect=agent_error)
        instance.node_registry.remove.return_value = True

        removed = await instance.detach_cluster_node("remote-1")

        self.assertTrue(removed)
        instance.node_registry.remove.assert_called_once_with("remote-1")


if __name__ == "__main__":
    unittest.main()
