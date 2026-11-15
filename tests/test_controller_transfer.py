"""Controller-role transfer: bundle building, re-binding, and promotion."""

import asyncio
import base64
import hashlib
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx

from cluster import (
    AGENT_PROTOCOL_VERSION,
    AgentCredentials,
    CONTROLLER_TRANSFER_CAPABILITY,
    LOCAL_NODE_ID,
)
from sparkdeck.controller_transfer import TRANSFER_CONFIRMATION
from sparkdeck import controller_transfer as transfer_module
from sparkdeck.controller_transfer import (
    TransferService,
    remap_node_refs,
    snapshot_store,
)
from sparkdeck.onboarding import ControllerAssignment, OnboardingService
from sparkdeck import transfer_helper


class FakeRegistry:
    def __init__(self, nodes):
        self.nodes = nodes
        self.request = AsyncMock()

    def get(self, node_id):
        if node_id == LOCAL_NODE_ID:
            return {"id": LOCAL_NODE_ID, "name": "This node", "local": True}
        return next((n for n in self.nodes if n.get("id") == node_id), None)


class FakeManager:
    def __init__(self, data_dir: Path, http: httpx.AsyncClient | None = None):
        self.data_dir = data_dir
        self.http = http or httpx.AsyncClient()
        self.agent_credentials = AgentCredentials(data_dir)
        self.settings = {
            "cluster_node_name": "Controller One",
            "hf_token": "hf-secret",
            "hf_cache": "/keep/me",
            "cluster_fabric_ip": "10.0.0.9",
            "cluster_node_hidden_from_dashboard": False,
            "flagship_pricing": {"Fable 5": {"input": 1.0, "output": 2.0}},
            "routeros_gateway_node_id": LOCAL_NODE_ID,
        }
        self.deployments = []
        self.is_joined_worker = Mock(return_value=False)
        self.adopt_worker_role = AsyncMock()

    @staticmethod
    def _network_interfaces():
        return [
            {"name": "eth0", "ipv4": ["192.168.1.5"]},
            {"name": "tailscale0", "ipv4": ["100.100.20.30"]},
        ]


def make_worker(node_id: str, name: str, url: str) -> dict:
    return {
        "id": node_id,
        "name": name,
        "agent_url": url,
        "agent_token": f"{node_id}-token",
        "fabric_ip": None,
        "fabric_interface": None,
        "enabled": True,
        "hidden_from_dashboard": False,
        "protocol_version": AGENT_PROTOCOL_VERSION,
        "paired_at": time.time(),
        "usage_reconciled": False,
        "forward_token_hash": hashlib.sha256(f"{node_id}-fwd".encode()).hexdigest(),
    }


def write_store(path: Path, rows: list[tuple[str, str]]) -> None:
    path.unlink(missing_ok=True)
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            "CREATE TABLE deployments ("
            " id TEXT PRIMARY KEY, settings_json TEXT)"
        )
        for deployment_id, settings_json in rows:
            connection.execute(
                "INSERT INTO deployments (id, settings_json) VALUES (?, ?)",
                (deployment_id, settings_json),
            )
        connection.commit()
    finally:
        connection.close()


class RemapNodeRefsTests(unittest.TestCase):
    def test_remaps_known_keys_and_leaves_other_values_alone(self):
        value = {
            "node_ids": [LOCAL_NODE_ID, "worker-b", "target-id"],
            "routeros_gateway_node_id": LOCAL_NODE_ID,
            "members": [{"node_id": LOCAL_NODE_ID, "rank": 0}],
            "selected_nodes": [
                {"id": LOCAL_NODE_ID, "name": "This device"},
                {"id": "target-id", "name": "Target"},
                {"id": "worker-b", "name": "B"},
            ],
            "instances": [{"node_ids": ["target-id"]}],
            "alias": LOCAL_NODE_ID,
            "count": 2,
        }
        result = remap_node_refs(value, {
            LOCAL_NODE_ID: "old-controller",
            "target-id": LOCAL_NODE_ID,
        })
        self.assertEqual(result["node_ids"], ["old-controller", "worker-b", LOCAL_NODE_ID])
        self.assertEqual(result["routeros_gateway_node_id"], "old-controller")
        self.assertEqual(result["members"][0]["node_id"], "old-controller")
        self.assertEqual(result["selected_nodes"][0]["id"], "old-controller")
        self.assertEqual(result["selected_nodes"][1]["id"], LOCAL_NODE_ID)
        self.assertEqual(result["selected_nodes"][2]["id"], "worker-b")
        self.assertEqual(result["instances"][0]["node_ids"], [LOCAL_NODE_ID])
        # Unknown keys never move, even when they hold the sentinel text.
        self.assertEqual(result["alias"], LOCAL_NODE_ID)
        self.assertEqual(result["count"], 2)
        # The input is deep-copied.
        self.assertEqual(value["node_ids"], [LOCAL_NODE_ID, "worker-b", "target-id"])


class SnapshotStoreTests(unittest.TestCase):
    def test_snapshot_remaps_node_references_into_a_standalone_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sparkdeck.sqlite3"
            write_store(source, [
                ("dep-1", json.dumps({"node_ids": [LOCAL_NODE_ID, "worker-b"]})),
                ("dep-2", json.dumps({"node_ids": ["target-id"]})),
                ("dep-3", json.dumps({"tensor_parallel_size": 2})),
            ])
            destination = root / "copy.sqlite3"
            snapshot_store(source, destination, {
                LOCAL_NODE_ID: "old-controller",
                "target-id": LOCAL_NODE_ID,
            })
            connection = sqlite3.connect(str(destination))
            try:
                rows = dict(
                    connection.execute("SELECT id, settings_json FROM deployments")
                )
            finally:
                connection.close()
            self.assertEqual(
                json.loads(rows["dep-1"])["node_ids"],
                ["old-controller", "worker-b"],
            )
            self.assertEqual(json.loads(rows["dep-2"])["node_ids"], [LOCAL_NODE_ID])
            self.assertEqual(json.loads(rows["dep-3"]), {"tensor_parallel_size": 2})
            # The live store is untouched.
            original = sqlite3.connect(str(source))
            try:
                rows = dict(
                    original.execute("SELECT id, settings_json FROM deployments")
                )
            finally:
                original.close()
            self.assertEqual(
                json.loads(rows["dep-1"])["node_ids"],
                [LOCAL_NODE_ID, "worker-b"],
            )


class RebindCredentialTests(unittest.TestCase):
    def test_rebind_keeps_identity_and_bearer_but_moves_the_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            credentials = AgentCredentials(Path(temporary))
            pairing_code = credentials.data["pairing_code"]
            paired = credentials.pair(pairing_code, "old-controller")
            token = paired["agent_token"]
            node_id = credentials.node_id

            credentials.rebind_controller("new-controller")

            self.assertEqual(credentials.node_id, node_id)
            self.assertEqual(credentials.data["agent_token"], token)
            self.assertEqual(
                credentials.data["paired_controller_id"], "new-controller"
            )
            self.assertFalse(credentials.authorize_controller(token, "old-controller"))
            self.assertTrue(credentials.authorize_controller(token, "new-controller"))

    def test_rebind_requires_a_controller_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            credentials = AgentCredentials(Path(temporary))
            with self.assertRaises(ValueError):
                credentials.rebind_controller("  ")


class OnboardingRebindTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    async def test_rebind_verifies_the_promoted_controller_and_persists(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "role": "controller",
                "node": {
                    "id": "new-controller",
                    "protocol_version": AGENT_PROTOCOL_VERSION,
                },
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        manager = FakeManager(self.root, http)
        service = OnboardingService(manager, self.root)
        ControllerAssignment(self.root).save({
            "controller_url": "http://127.0.0.1:9000",
            "forward_token": "forward-secret",
            "node_id": manager.agent_credentials.node_id,
            "controller_node_id": "old-controller",
            "protocol_version": AGENT_PROTOCOL_VERSION,
            "joined_at": time.time(),
        })

        result = await service.rebind({
            "controller_url": "http://127.0.0.1:9100",
            "controller_node_id": "new-controller",
        })

        self.assertTrue(result["ok"])
        assignment = ControllerAssignment(self.root).load()
        self.assertEqual(assignment["controller_url"], "http://127.0.0.1:9100")
        self.assertEqual(assignment["controller_node_id"], "new-controller")
        # The forward token survives: its hash already moved with the registry.
        self.assertEqual(assignment["forward_token"], "forward-secret")
        self.assertEqual(
            manager.agent_credentials.data["paired_controller_id"],
            "new-controller",
        )
        await http.aclose()

    async def test_rebind_rejects_unconfirmed_identity(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "role": "controller",
                "node": {"id": "someone-else", "protocol_version": AGENT_PROTOCOL_VERSION},
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        manager = FakeManager(self.root, http)
        service = OnboardingService(manager, self.root)
        ControllerAssignment(self.root).save({
            "controller_url": "http://127.0.0.1:9000",
            "forward_token": "forward-secret",
            "node_id": manager.agent_credentials.node_id,
            "controller_node_id": "old-controller",
            "protocol_version": AGENT_PROTOCOL_VERSION,
            "joined_at": time.time(),
        })

        with self.assertRaisesRegex(ValueError, "did not confirm"):
            await service.rebind({
                "controller_url": "http://127.0.0.1:9100",
                "controller_node_id": "new-controller",
            })
        assignment = ControllerAssignment(self.root).load()
        self.assertEqual(assignment["controller_node_id"], "old-controller")
        await http.aclose()

    async def test_rebind_rejects_non_workers_and_duplicates(self):
        manager = FakeManager(self.root)
        service = OnboardingService(manager, self.root)
        with self.assertRaisesRegex(ValueError, "joined worker"):
            await service.rebind({
                "controller_url": "http://127.0.0.1:9100",
                "controller_node_id": "new-controller",
            })
        ControllerAssignment(self.root).save({
            "controller_url": "http://127.0.0.1:9000",
            "forward_token": "forward-secret",
            "node_id": manager.agent_credentials.node_id,
            "controller_node_id": "old-controller",
            "protocol_version": AGENT_PROTOCOL_VERSION,
            "joined_at": time.time(),
        })
        # Claim the credential the way a completed rebind would; an unclaimed
        # agent with a matching assignment is the partial-write recovery case
        # instead (covered by its own test).
        manager.agent_credentials.rebind_controller("old-controller")
        with self.assertRaisesRegex(ValueError, "already bound"):
            await service.rebind({
                "controller_url": "http://127.0.0.1:9100",
                "controller_node_id": "old-controller",
            })
        await manager.http.aclose()

    async def test_rebind_adopts_an_unassigned_but_paired_node(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json={
                    "role": "controller",
                    "join_code": "JOIN123",
                    "node": {
                        "id": "new-controller",
                        "protocol_version": AGENT_PROTOCOL_VERSION,
                    },
                })
            return httpx.Response(200, json={
                "ok": True,
                "role": "controller",
                "protocol_version": AGENT_PROTOCOL_VERSION,
                "forward_token": "adopted-forward",
                "node": {"id": "new-controller", "name": "New"},
                "cluster": {"nodes": []},
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        manager = FakeManager(self.root, http)
        # The lost-assignment state: the credential claim still names the old
        # controller, but controller.json is gone, so this node runs as a
        # standalone controller.
        manager.agent_credentials.rebind_controller("old-controller")
        service = OnboardingService(manager, self.root)

        result = await service.rebind(
            {
                "controller_url": "http://127.0.0.1:9100",
                "controller_node_id": "new-controller",
            },
            current_controller_id="old-controller",
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["adopted"])
        assignment = ControllerAssignment(self.root).load()
        self.assertEqual(assignment["controller_url"], "http://127.0.0.1:9100")
        self.assertEqual(assignment["controller_node_id"], "new-controller")
        self.assertEqual(assignment["forward_token"], "adopted-forward")
        self.assertIsNotNone(assignment.get("adopted_at"))
        manager.adopt_worker_role.assert_awaited_once()
        await http.aclose()

    async def test_rebind_adopts_requires_the_calling_controller_claim(self):
        manager = FakeManager(self.root)
        manager.agent_credentials.rebind_controller("old-controller")
        service = OnboardingService(manager, self.root)
        with self.assertRaisesRegex(ValueError, "only a joined worker"):
            await service.rebind(
                {
                    "controller_url": "http://127.0.0.1:9100",
                    "controller_node_id": "new-controller",
                },
                current_controller_id="someone-else",
            )
        with self.assertRaisesRegex(ValueError, "only a joined worker"):
            await service.rebind({
                "controller_url": "http://127.0.0.1:9100",
                "controller_node_id": "new-controller",
            })
        await manager.http.aclose()

    async def test_rebind_finishes_a_partial_credential_write(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "role": "controller",
                "node": {
                    "id": "new-controller",
                    "protocol_version": AGENT_PROTOCOL_VERSION,
                },
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        manager = FakeManager(self.root, http)
        service = OnboardingService(manager, self.root)
        ControllerAssignment(self.root).save({
            "controller_url": "http://127.0.0.1:9100",
            "forward_token": "forward-secret",
            "node_id": manager.agent_credentials.node_id,
            # The previous attempt saved the assignment but the credential
            # write failed, so the agent still answers for the old owner.
            "controller_node_id": "new-controller",
            "protocol_version": AGENT_PROTOCOL_VERSION,
            "joined_at": time.time(),
        })

        result = await service.rebind({
            "controller_url": "http://127.0.0.1:9100",
            "controller_node_id": "new-controller",
        })

        self.assertTrue(result["ok"])
        self.assertEqual(
            manager.agent_credentials.data["paired_controller_id"],
            "new-controller",
        )
        # A further identical rebind is now genuinely a duplicate.
        with self.assertRaisesRegex(ValueError, "already bound"):
            await service.rebind({
                "controller_url": "http://127.0.0.1:9100",
                "controller_node_id": "new-controller",
            })
        await http.aclose()


class BuildBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _service(self) -> tuple[TransferService, FakeManager]:
        manager = FakeManager(self.root)
        manager.node_registry = FakeRegistry([
            make_worker("target-id", "Target", "http://127.0.0.1:7000"),
            make_worker("worker-b", "Bravo", "http://127.0.0.1:7001"),
        ])
        onboarding = Mock()
        onboarding.port = 7878
        onboarding.assignment = ControllerAssignment(self.root)
        write_store(self.root / "sparkdeck.sqlite3", [
            ("dep-1", json.dumps({"node_ids": [LOCAL_NODE_ID]})),
            ("dep-2", json.dumps({"node_ids": ["target-id"]})),
        ])
        (self.root / "recipes.json").write_text(json.dumps({
            "recipes": [{"id": "r1", "node_ids": [LOCAL_NODE_ID]}],
            "routeros_gateway_node_id": LOCAL_NODE_ID,
        }), encoding="utf-8")
        return TransferService(manager, onboarding, self.root, self.root), manager

    def test_bundle_moves_registry_state_and_remaps_references(self):
        service, manager = self._service()
        state = {
            "id": "job-1",
            "target_node_id": "target-id",
            "target_name": "Target",
            "target_url": "http://127.0.0.1:7000",
            "warnings": ["w1"],
            "started_at": time.time(),
        }
        bundle = service._build_bundle(state)

        self.assertEqual(
            {node["id"] for node in bundle["nodes"]},
            {"worker-b", manager.agent_credentials.node_id},
        )
        old_entry = next(
            node for node in bundle["nodes"]
            if node["id"] == manager.agent_credentials.node_id
        )
        self.assertEqual(
            old_entry["agent_url"], "http://100.100.20.30:7878"
        )
        self.assertEqual(
            old_entry["agent_token"],
            manager.agent_credentials.data["agent_token"],
        )
        self.assertEqual(
            old_entry["forward_token_hash"],
            hashlib.sha256(
                bundle["old_controller"]["forward_token"].encode()
            ).hexdigest(),
        )
        self.assertEqual(
            bundle["json_files"]["recipes.json"]["recipes"][0]["node_ids"],
            [manager.agent_credentials.node_id],
        )
        self.assertEqual(
            bundle["json_files"]["recipes.json"]["routeros_gateway_node_id"],
            manager.agent_credentials.node_id,
        )
        overlay = bundle["settings_overlay"]
        self.assertEqual(overlay["hf_token"], "hf-secret")
        self.assertNotIn("hf_cache", overlay)
        self.assertNotIn("cluster_node_name", overlay)
        self.assertNotIn("cluster_fabric_ip", overlay)
        # Cluster-wide settings that pin nodes move with the remap too.
        self.assertEqual(
            overlay["routeros_gateway_node_id"],
            manager.agent_credentials.node_id,
        )

        store_bytes = base64.b64decode(bundle["store_b64"])
        self.assertEqual(
            hashlib.sha256(store_bytes).hexdigest(), bundle["store_sha256"]
        )
        probe = sqlite3.connect(":memory:")
        probe.deserialize(store_bytes)
        settings_json = probe.execute(
            "SELECT settings_json FROM deployments WHERE id = 'dep-1'"
        ).fetchone()[0]
        target_json = probe.execute(
            "SELECT settings_json FROM deployments WHERE id = 'dep-2'"
        ).fetchone()[0]
        probe.close()
        self.assertEqual(
            json.loads(settings_json)["node_ids"],
            [manager.agent_credentials.node_id],
        )
        # References to the promoted worker become the local sentinel on the
        # new controller.
        self.assertEqual(
            json.loads(target_json)["node_ids"],
            [LOCAL_NODE_ID],
        )

        self.assertTrue(bundle["transfer_state"]["incoming"])
        self.assertEqual(bundle["transfer_state"]["phase"], "promoted")
        self.assertEqual(bundle["new_controller_node_id"], "target-id")

    def test_status_redacts_the_private_forward_token(self):
        service, _manager = self._service()
        service._write({
            "active": True,
            "phase": "rebinding",
            "old_controller_forward_token": "secret-forward",
        })
        status = service.status()
        self.assertNotIn("old_controller_forward_token", status)
        self.assertEqual(status["phase"], "rebinding")
        self.assertEqual(status["confirmation"], TRANSFER_CONFIRMATION)

    def test_defers_cluster_actions_while_completing(self):
        service, _manager = self._service()
        self.assertFalse(service.defers_cluster_actions())
        service._write({
            "active": True,
            "incoming": True,
            "phase": "completing",
        })
        self.assertTrue(service.defers_cluster_actions())
        service._write({
            "active": False,
            "incoming": True,
            "phase": "completed",
        })
        self.assertFalse(service.defers_cluster_actions())


class BootReconciliationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _service(self) -> tuple[TransferService, FakeManager]:
        manager = FakeManager(self.root)
        manager.node_registry = FakeRegistry([
            make_worker("worker-b", "Bravo", "http://127.0.0.1:7001"),
        ])
        manager.node_registry.probe = AsyncMock(
            return_value={"online": True, "name": "Bravo"}
        )
        onboarding = Mock()
        onboarding.assignment = ControllerAssignment(self.root)
        return TransferService(manager, onboarding, self.root, self.root), manager

    async def test_promoted_controller_boots_into_completing_and_finishes(self):
        service, _manager = self._service()
        service._write({
            "id": "job-1",
            "active": True,
            "incoming": True,
            "committed": True,
            "phase": "promoted",
            "nodes": [{"id": "worker-b", "name": "Bravo", "status": "pending"}],
        })
        with patch.object(transfer_module, "COMPLETION_POLL_SECONDS", 0.01):
            service.reconcile()
            for _ in range(200):
                state = service._read()
                if not state.get("active"):
                    break
                await asyncio.sleep(0.01)
        state = service._read()
        self.assertEqual(state["phase"], "completed")
        self.assertFalse(state["active"])
        self.assertEqual(
            state["nodes"][0]["status"], "online"
        )
        await service.close()

    async def test_interrupted_pre_commit_transfer_fails_at_boot(self):
        service, _manager = self._service()
        service._write({
            "id": "job-1",
            "active": True,
            "incoming": False,
            "committed": False,
            "phase": "staging",
        })
        with patch.object(transfer_module, "COMPLETION_POLL_SECONDS", 0.01):
            service.reconcile()
        state = service._read()
        self.assertEqual(state["phase"], "failed")
        self.assertFalse(state["active"])
        await service.close()


class StartTransferFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _service(self) -> tuple[TransferService, FakeManager]:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "role": "controller",
                "node": {
                    "id": "target-id",
                    "protocol_version": AGENT_PROTOCOL_VERSION,
                },
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        manager = FakeManager(self.root, http)
        manager.node_registry = FakeRegistry([
            make_worker("target-id", "Target", "http://127.0.0.1:7000"),
            make_worker("worker-b", "Bravo", "http://127.0.0.1:7001"),
        ])

        async def registry_request(node_id, method, path, json_body=None, timeout=30, allow_disabled=False):
            if path == "/api/agent/controller-transfer/receive":
                return {"ok": True, "job_id": (json_body or {}).get("job_id")}
            if path == "/api/agent/controller-transfer/promote":
                return {"ok": True, "helper_pid": 99}
            if path == "/api/agent/onboarding/rebind":
                return {"ok": True}
            if path == "/api/v1/onboarding":
                return {"role": "worker", "node": {"id": "target-id"}}
            if path == "/api/agent/info":
                return {"node_id": node_id, "capabilities": [CONTROLLER_TRANSFER_CAPABILITY]}
            return {"ok": True}

        manager.node_registry.request = AsyncMock(side_effect=registry_request)
        manager.node_registry.probe = AsyncMock(return_value={"online": True})
        onboarding = Mock()
        onboarding.port = 7878
        onboarding.assignment = ControllerAssignment(self.root)
        write_store(self.root / "sparkdeck.sqlite3", [])
        (self.root / "recipes.json").write_text("{}", encoding="utf-8")
        (self.root / "token_stats.json").write_text("{}", encoding="utf-8")
        return TransferService(manager, onboarding, self.root, self.root), manager

    async def test_start_runs_through_staging_promotion_rebind_and_demotion(self):
        service, manager = self._service()
        service.set_launch_drain_hook(lambda: True)
        with patch.object(service, "_spawn_helper", return_value=7) as spawn:
            status = await service.start("target-id", TRANSFER_CONFIRMATION)
            self.assertTrue(status["active"])
            task = service._task
            await task

        spawn.assert_called_once_with("demote")
        self.assertNotIn("Transfer failed", (task.exception() or "") if task.exception() else "")
        # The old controller rejoined as a worker under the new controller.
        assignment = ControllerAssignment(self.root).load()
        self.assertEqual(assignment["controller_node_id"], "target-id")
        self.assertEqual(
            manager.agent_credentials.data["paired_controller_id"], "target-id"
        )
        self.assertTrue(assignment["forward_token"])
        # The controller state is archived; the machine's own usage counters
        # stay live as worker state.
        archives = list(self.root.glob("controller-transfer-archive-*"))
        self.assertEqual(len(archives), 1)
        self.assertTrue((archives[0] / "recipes.json").exists())
        self.assertFalse((self.root / "recipes.json").exists())
        self.assertFalse((self.root / "sparkdeck.sqlite3").exists())
        self.assertTrue((self.root / "token_stats.json").exists())
        await manager.http.aclose()

    async def test_cancel_is_locked_once_promotion_may_have_committed(self):
        service, _manager = self._service()
        service._write({
            "id": "job-1",
            "active": True,
            "committed": False,
            "promote_requested": True,
            "phase": "promoting",
            "target_node_id": "target-id",
        })
        with self.assertRaisesRegex(RuntimeError, "can only be resumed"):
            await service.cancel()
        self.assertTrue(service.blocks_management_writes())

    async def test_resume_releases_the_job_when_the_target_stayed_a_worker(self):
        service, _manager = self._service()
        service._write({
            "id": "job-1",
            "active": True,
            "committed": False,
            "promote_requested": True,
            "phase": "promoting",
            "target_node_id": "target-id",
            "target_name": "Target",
            "target_url": "http://127.0.0.1:7000",
            "nodes": [],
            "old_controller_forward_token": "raw",
        })
        with patch.object(
            service, "_wait_for_new_controller",
            AsyncMock(side_effect=RuntimeError("controller never came up")),
        ):
            await service.resume()
            await service._task
        state = service._read()
        # The target still answered as this controller's worker, so the flow
        # released the job for a clean retry or cancel.
        self.assertFalse(state["active"])
        self.assertFalse(state["promote_requested"])
        self.assertFalse(service.blocks_management_writes())
        await service.close()
        await service.manager.http.aclose()

    async def test_onboarding_writes_open_tracks_incoming_role(self):
        service, _manager = self._service()
        self.assertTrue(service.onboarding_writes_open())
        service._write({"id": "job-1", "active": True, "incoming": False})
        self.assertFalse(service.onboarding_writes_open())
        service._write({"id": "job-1", "active": True, "incoming": True})
        self.assertTrue(service.onboarding_writes_open())
        await service.close()
        await service.manager.http.aclose()


class RebindReconciliationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _service(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/nodes":
                return httpx.Response(200, json={"items": [
                    {"id": "worker-b", "online": True},
                    {"id": "worker-c", "online": False},
                ]})
            return httpx.Response(200, json={})

        manager = FakeManager(self.root, httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ))
        manager.node_registry = FakeRegistry([
            make_worker("worker-b", "Bravo", "http://127.0.0.1:7001"),
            make_worker("worker-c", "Charlie", "http://127.0.0.1:7002"),
        ])
        onboarding = Mock()
        onboarding.assignment = ControllerAssignment(self.root)
        return TransferService(manager, onboarding, self.root, self.root), manager

    async def test_lost_rebind_reply_is_reconciled_through_the_new_controller(self):
        from cluster import NodeAgentResponseError

        service, _manager = self._service()
        # The worker already re-bound; the old controller's credentials now
        # draw a 401 on every retry.
        async def reject(node_id, method, path, json_body=None, timeout=30, allow_disabled=False):
            raise NodeAgentResponseError("Bravo", 401, "invalid node-agent token")

        service.manager.node_registry.request = AsyncMock(side_effect=reject)
        state = {
            "id": "job-1",
            "target_node_id": "target-id",
            "target_url": "http://127.0.0.1:7000",
            "nodes": [{"id": "worker-b", "name": "Bravo", "status": "pending"}],
        }
        await service._rebind_nodes(state)
        self.assertEqual(state["nodes"][0]["status"], "rebound")
        self.assertNotIn("error", state["nodes"][0])
        await service.manager.http.aclose()

    async def test_offline_under_new_controller_stays_failed(self):
        from cluster import NodeAgentResponseError

        service, _manager = self._service()
        async def reject(node_id, method, path, json_body=None, timeout=30, allow_disabled=False):
            raise NodeAgentResponseError("Bravo", 401, "invalid node-agent token")

        service.manager.node_registry.request = AsyncMock(side_effect=reject)
        state = {
            "id": "job-1",
            "target_node_id": "target-id",
            "target_url": "http://127.0.0.1:7000",
            "nodes": [{"id": "worker-c", "name": "Charlie", "status": "pending"}],
        }
        await service._rebind_nodes(state)
        self.assertEqual(state["nodes"][0]["status"], "failed")
        await service.manager.http.aclose()


class AmbiguityReleaseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _service(self, responses: list[dict]) -> tuple[TransferService, FakeManager]:
        manager = FakeManager(self.root)
        manager.node_registry = FakeRegistry([])
        seen: list[str] = []

        async def registry_request(node_id, method, path, json_body=None, timeout=30, allow_disabled=False):
            seen.append(path)
            return responses[min(len(seen) - 1, len(responses) - 1)]

        manager.node_registry.request = AsyncMock(side_effect=registry_request)
        onboarding = Mock()
        onboarding.assignment = ControllerAssignment(self.root)
        return (
            TransferService(manager, onboarding, self.root, self.root),
            manager,
        )

    async def test_target_still_worker_requires_a_worker_role(self):
        service, _manager = self._service([
            {"role": "worker", "node": {"id": "target-id"}},
        ])
        state = {"id": "job-1", "target_node_id": "target-id"}
        self.assertTrue(await service._target_still_worker(state))
        await service.manager.http.aclose()

    async def test_promoted_target_is_not_reversible(self):
        service, _manager = self._service([
            {"role": "controller", "node": {"id": "target-id"}},
        ])
        state = {"id": "job-1", "target_node_id": "target-id"}
        self.assertFalse(await service._target_still_worker(state))
        await service.manager.http.aclose()

    async def test_unreachable_target_is_not_reversible(self):
        service, _manager = self._service([])
        async def unreachable(node_id, method, path, json_body=None, timeout=30, allow_disabled=False):
            raise RuntimeError("target is offline")

        service.manager.node_registry.request = AsyncMock(side_effect=unreachable)
        state = {"id": "job-1", "target_node_id": "target-id"}
        self.assertFalse(await service._target_still_worker(state))
        await service.manager.http.aclose()


class AgentReceiveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manager = FakeManager(self.root)
        self.onboarding = Mock()
        self.onboarding.assignment = ControllerAssignment(self.root)
        self.onboarding.assignment.save({
            "controller_url": "http://127.0.0.1:9000",
            "forward_token": "forward-secret",
            "node_id": self.manager.agent_credentials.node_id,
            "controller_node_id": "old-controller",
            "protocol_version": AGENT_PROTOCOL_VERSION,
            "joined_at": time.time(),
        })
        self.service = TransferService(self.manager, self.onboarding, self.root, self.root)

    def tearDown(self):
        self.temp.cleanup()

    def _store_bytes(self) -> bytes:
        path = self.root / "seed.sqlite3"
        write_store(path, [("dep-1", json.dumps({"node_ids": ["x"]}))])
        return path.read_bytes()

    def _bundle(self, **overrides) -> dict:
        store_bytes = self._store_bytes()
        bundle = {
            "bundle_version": 1,
            "job_id": "job-1",
            "created_at": time.time(),
            "old_controller": {
                "node_id": "old-controller",
                "name": "Controller One",
                "agent_url": "http://127.0.0.1:7002",
                "agent_token": "old-token",
                "forward_token": "old-raw-forward",
            },
            "target": {"node_id": self.manager.agent_credentials.node_id, "name": "Target"},
            "controller_url": "http://127.0.0.1:7000",
            "new_controller_node_id": self.manager.agent_credentials.node_id,
            "nodes": [],
            "settings_overlay": {"hf_token": "t"},
            "json_files": {},
            "store_sha256": hashlib.sha256(store_bytes).hexdigest(),
            "store_b64": base64.b64encode(store_bytes).decode("ascii"),
            "transfer_state": {"id": "job-1", "phase": "promoted"},
        }
        bundle.update(overrides)
        return bundle

    async def test_receive_stages_the_bundle_and_discard_removes_it(self):
        result = await self.service.agent_receive(self._bundle())
        self.assertTrue(result["ok"])
        self.assertEqual(result["job_id"], "job-1")
        staging = self.root / "controller-transfer-incoming"
        self.assertTrue((staging / "bundle.json").is_file())
        self.assertTrue((staging / "store.sqlite3").is_file())
        status = self.service.agent_status()
        self.assertEqual(status["staged"]["job_id"], "job-1")
        self.assertEqual(status["capability"], CONTROLLER_TRANSFER_CAPABILITY)
        discard = self.service.agent_discard()
        self.assertTrue(discard["discarded"])
        self.assertIsNone(self.service.agent_status()["staged"])

    async def test_receive_rejects_wrong_target_and_bad_checksums(self):
        other_target = self._bundle(
            target={"node_id": "someone-else", "name": "Other"},
        )
        with self.assertRaisesRegex(ValueError, "different node"):
            await self.service.agent_receive(other_target)
        bad_hash = self._bundle()
        bad_hash["store_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum"):
            await self.service.agent_receive(bad_hash)
        bad_version = self._bundle(bundle_version=2)
        with self.assertRaisesRegex(ValueError, "bundle version"):
            await self.service.agent_receive(bad_version)

    async def test_promote_requires_matching_staged_job(self):
        await self.service.agent_receive(self._bundle())
        with patch.object(self.service, "_spawn_helper", return_value=4242) as spawn:
            result = await self.service.agent_promote({"job_id": "job-1"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["helper_pid"], 4242)
        spawn.assert_called_once_with("promote")
        state = self.service._read()
        self.assertEqual(state["phase"], "promoting")
        with self.assertRaisesRegex(RuntimeError, "no staged controller state"):
            await self.service.agent_promote({"job_id": "other-job"})


class TransferHelperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir(parents=True)
        # A joined worker with its own local state.
        (self.data / "controller.json").write_text(json.dumps({
            "controller_url": "http://127.0.0.1:9000",
            "forward_token": "fwd",
            "node_id": "target-node",
            "controller_node_id": "old-controller",
        }), encoding="utf-8")
        (self.data / "settings.json").write_text(json.dumps({
            "cluster_node_name": "Target",
            "hf_cache": "/keep/me",
        }), encoding="utf-8")
        (self.data / "sparkdeck.sqlite3").write_bytes(b"old store")

    def tearDown(self):
        self.temp.cleanup()

    def _stage(self) -> None:
        store_path = self.root / "seed.sqlite3"
        write_store(store_path, [("dep-1", json.dumps({"node_ids": ["old"]}))])
        store_bytes = store_path.read_bytes()
        bundle = {
            "bundle_version": 1,
            "job_id": "job-1",
            "new_controller_node_id": "target-node",
            "nodes": [{"id": "old-controller", "name": "Controller One"}],
            "settings_overlay": {"hf_token": "moved"},
            "json_files": {"recipes.json": {"recipes": []}},
            "transfer_state": {
                "id": "job-1",
                "active": True,
                "incoming": True,
                "committed": True,
                "phase": "promoted",
            },
        }
        staging = self.data / "controller-transfer-incoming"
        staging.mkdir(parents=True)
        bundle_bytes = json.dumps(bundle).encode("utf-8")
        (staging / "bundle.json").write_bytes(bundle_bytes)
        (staging / "store.sqlite3").write_bytes(store_bytes)
        (staging / "manifest.json").write_text(json.dumps({
            "job_id": "job-1",
            "bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
            "store_sha256": hashlib.sha256(store_bytes).hexdigest(),
        }), encoding="utf-8")

    def test_promote_installs_state_and_drops_the_assignment(self):
        self._stage()
        state_path = self.data / "controller-transfer.json"
        with patch.object(transfer_helper, "restart_service") as restart, patch.object(
            transfer_helper, "_wait_for_controller_role", return_value=True
        ):
            transfer_helper.promote(self.root, state_path)
        restart.assert_called_once_with(self.root)

        self.assertFalse((self.data / "controller.json").exists())
        settings = json.loads((self.data / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(settings["hf_token"], "moved")
        self.assertEqual(settings["cluster_node_name"], "Target")
        self.assertEqual(settings["hf_cache"], "/keep/me")
        nodes = json.loads((self.data / "nodes.json").read_text(encoding="utf-8"))
        self.assertEqual(nodes[0]["id"], "old-controller")
        recipes = json.loads((self.data / "recipes.json").read_text(encoding="utf-8"))
        self.assertEqual(recipes, {"recipes": []})
        # The transferred store replaced the worker's own file.
        connection = sqlite3.connect(str(self.data / "sparkdeck.sqlite3"))
        try:
            rows = connection.execute("SELECT id FROM deployments").fetchall()
        finally:
            connection.close()
        self.assertEqual(rows, [("dep-1",)])
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["phase"], "promoted")
        backups = list(self.data.glob("pre-transfer-backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(
            (backups[0] / "controller.json").read_text(encoding="utf-8"),
            json.dumps({
                "controller_url": "http://127.0.0.1:9000",
                "forward_token": "fwd",
                "node_id": "target-node",
                "controller_node_id": "old-controller",
            }),
        )

    def test_promote_rolls_back_when_the_controller_role_never_boots(self):
        self._stage()
        state_path = self.data / "controller-transfer.json"
        with patch.object(transfer_helper, "restart_service") as restart, patch.object(
            transfer_helper, "_wait_for_controller_role", return_value=False
        ):
            with self.assertRaises(RuntimeError):
                transfer_helper.promote(self.root, state_path)
        restart.assert_called_with(self.root)

        self.assertTrue((self.data / "controller.json").exists())
        self.assertEqual((self.data / "sparkdeck.sqlite3").read_bytes(), b"old store")
        settings = json.loads((self.data / "settings.json").read_text(encoding="utf-8"))
        self.assertNotIn("hf_token", settings)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["phase"], "recovery_required")

    def test_promote_restores_everything_on_an_early_install_failure(self):
        self._stage()
        (self.data / "recipes.json").write_text(
            json.dumps({"recipes": ["original"]}), encoding="utf-8"
        )
        state_path = self.data / "controller-transfer.json"
        with patch.object(
            transfer_helper, "_install_private_bytes", side_effect=OSError("disk full")
        ):
            with self.assertRaises(OSError):
                transfer_helper.promote(self.root, state_path)

        # Nothing about the worker changed, including machine-local settings
        # that were never backed up when the failure happened mid-install.
        self.assertTrue((self.data / "controller.json").exists())
        self.assertEqual((self.data / "sparkdeck.sqlite3").read_bytes(), b"old store")
        self.assertEqual(
            json.loads((self.data / "recipes.json").read_text(encoding="utf-8")),
            {"recipes": ["original"]},
        )
        settings = json.loads((self.data / "settings.json").read_text(encoding="utf-8"))
        self.assertNotIn("hf_token", settings)
        self.assertEqual(settings["cluster_node_name"], "Target")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["phase"], "failed")


class TransferWriteGateTests(unittest.IsolatedAsyncioTestCase):
    """The transfer freeze must not block the transfer's own recovery flow.

    A promoted controller waiting on a lost worker adopts it through the
    standard join flow, which writes via /api/v1/onboarding/register. If the
    freeze covered onboarding, the controller would wait forever for a node
    it refuses to admit.
    """

    async def _gated(self, method: str, path: str, onboarding_open: bool = True):
        from types import SimpleNamespace

        import server
        from server import _SecurityHeadersMiddleware

        request = Mock()
        request.method = method
        request.url = SimpleNamespace(path=path)
        request.headers = {}
        with patch.object(
            server.controller_transfer, "blocks_management_writes", return_value=True
        ), patch.object(
            server.controller_transfer,
            "onboarding_writes_open",
            return_value=onboarding_open,
        ):
            return await _SecurityHeadersMiddleware._management_response(request)

    async def _gated_incoming(self, method: str, path: str):
        return await self._gated(method, path, onboarding_open=True)

    async def _gated_outgoing(self, method: str, path: str):
        return await self._gated(method, path, onboarding_open=False)

    async def test_frozen_management_writes_return_503(self):
        for gated in (self._gated_incoming, self._gated_outgoing):
            response = await gated("POST", "/api/v1/deployments")
            self.assertEqual(response.status_code, 503)

    async def test_reads_agent_and_transfer_traffic_stay_open(self):
        for gated in (self._gated_incoming, self._gated_outgoing):
            for method, path in (
                ("GET", "/api/v1/deployments"),
                ("POST", "/api/agent/command"),
                ("POST", "/api/v1/controller-transfer/resume"),
            ):
                self.assertIsNone(await gated(method, path), f"{method} {path}")

    async def test_onboarding_writes_open_only_on_incoming_controller(self):
        for method, path in (
            ("POST", "/api/v1/onboarding/register"),
            ("POST", "/api/v1/onboarding/unregister"),
        ):
            self.assertIsNone(
                await self._gated_incoming(method, path), f"{method} {path}"
            )
            response = await self._gated_outgoing(method, path)
            self.assertEqual(response.status_code, 503)
