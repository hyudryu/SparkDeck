import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from sparkdeck.onboarding import (
    FORWARD_HOP_HEADER,
    FORWARD_NODE_HEADER,
    FORWARD_TOKEN_HEADER,
)


with patch("docker.from_env", return_value=Mock()):
    import server


class NodeRenameApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.assignment = patch.object(
            server.onboarding.assignment, "load", return_value=None,
        )
        self.assignment.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        self.assignment.stop()

    async def test_versioned_patch_delegates_and_returns_public_node(self):
        rename = AsyncMock(return_value={
            "id": "remote-1", "name": "Compute", "online": True,
            "name_sync": "synchronized",
        })
        with patch.object(server.manager, "rename_cluster_node", rename):
            response = await self.client.patch(
                "/api/v1/nodes/remote-1", json={"name": " Compute "},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["name"], "Compute")
        rename.assert_awaited_once_with("remote-1", " Compute ")

    async def test_versioned_patch_maps_validation_and_missing_node(self):
        rename = AsyncMock(side_effect=ValueError("name must not be empty"))
        with patch.object(server.manager, "rename_cluster_node", rename):
            invalid = await self.client.patch("/api/v1/nodes/local", json={"name": ""})
        rename.side_effect = LookupError("node not found")
        with patch.object(server.manager, "rename_cluster_node", rename):
            missing = await self.client.patch(
                "/api/v1/nodes/missing", json={"name": "Worker"},
            )

        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(missing.status_code, 404)

    async def test_versioned_patch_updates_dashboard_visibility(self):
        visibility = AsyncMock(return_value={
            "id": "remote-1", "hidden_from_dashboard": True,
        })
        with patch.object(
            server.manager, "set_cluster_node_dashboard_hidden", visibility,
        ):
            response = await self.client.patch(
                "/api/v1/nodes/remote-1",
                json={"hidden_from_dashboard": True},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["hidden_from_dashboard"])
        visibility.assert_awaited_once_with("remote-1", True)

    async def test_versioned_patch_rejects_mixed_node_updates(self):
        response = await self.client.patch(
            "/api/v1/nodes/remote-1",
            json={"name": "Worker", "hidden_from_dashboard": True},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("only name or hidden_from_dashboard", response.text)

    async def test_agent_patch_requires_auth_and_updates_only_local_node(self):
        rename = AsyncMock(return_value={"id": "local", "name": "Worker"})
        with (
            patch.object(server.manager.agent_credentials, "authorize_controller", return_value=False),
            patch.object(server.manager, "rename_cluster_node", rename),
        ):
            unauthorized = await self.client.patch(
                "/api/agent/node", json={"name": "Worker"},
            )
        with (
            patch.object(server.manager.agent_credentials, "authorize_controller", return_value=True),
            patch.object(server.manager, "rename_cluster_node", rename),
        ):
            accepted = await self.client.patch(
                "/api/agent/node",
                headers={"authorization": "Bearer paired"},
                json={"name": "Worker"},
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(accepted.status_code, 200)
        rename.assert_awaited_once_with("local", "Worker")

    async def test_versioned_delete_detaches_or_force_forgets_remote_node(self):
        detach = AsyncMock(return_value=True)
        with patch.object(server.manager, "detach_cluster_node", detach):
            removed = await self.client.delete("/api/v1/nodes/remote-1")
            forgotten = await self.client.delete("/api/v1/nodes/offline-1?force=true")

        self.assertEqual(removed.status_code, 200)
        self.assertEqual(removed.json(), {
            "ok": True, "node_id": "remote-1", "forced": False,
        })
        self.assertEqual(forgotten.status_code, 200)
        self.assertTrue(forgotten.json()["forced"])
        self.assertEqual(detach.await_args_list[0].args, ("remote-1",))
        self.assertEqual(detach.await_args_list[0].kwargs, {"force": False})
        self.assertEqual(detach.await_args_list[1].args, ("offline-1",))
        self.assertEqual(detach.await_args_list[1].kwargs, {"force": True})

    async def test_versioned_delete_maps_in_use_unreachable_and_missing_nodes(self):
        detach = AsyncMock(side_effect=ValueError("node is still used by model"))
        with patch.object(server.manager, "detach_cluster_node", detach):
            in_use = await self.client.delete("/api/v1/nodes/remote-1")
        detach.side_effect = RuntimeError("could not contact Worker")
        with patch.object(server.manager, "detach_cluster_node", detach):
            unreachable = await self.client.delete("/api/v1/nodes/remote-1")
        detach.side_effect = None
        detach.return_value = False
        with patch.object(server.manager, "detach_cluster_node", detach):
            missing = await self.client.delete("/api/v1/nodes/missing")

        self.assertEqual(in_use.status_code, 409)
        self.assertEqual(unreachable.status_code, 409)
        self.assertEqual(missing.status_code, 404)

    async def test_legacy_delete_also_detaches_before_removing(self):
        detach = AsyncMock(return_value=True)
        with patch.object(server.manager, "detach_cluster_node", detach):
            removed = await self.client.delete("/api/nodes/remote-1")

        self.assertEqual(removed.status_code, 200)
        detach.assert_awaited_once_with("remote-1")

    async def test_agent_detach_requires_auth_and_revokes_local_assignment(self):
        detach = AsyncMock(return_value={"ok": True, "role": "controller", "revoked": True})
        with (
            patch.object(server.manager.agent_credentials, "authorize_controller", return_value=False),
            patch.object(server.onboarding, "detach", detach),
        ):
            unauthorized = await self.client.post("/api/agent/onboarding/detach")
        with (
            patch.object(server.manager.agent_credentials, "authorize_controller", return_value=True),
            patch.object(server.onboarding, "detach", detach),
        ):
            accepted = await self.client.post(
                "/api/agent/onboarding/detach",
                headers={"authorization": "Bearer paired"},
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(accepted.status_code, 200)
        detach.assert_awaited_once()

    async def test_force_forgotten_unregister_bypasses_only_its_auth_gate(self):
        headers = {
            FORWARD_HOP_HEADER: "1",
            FORWARD_NODE_HEADER: "forgotten-worker",
            FORWARD_TOKEN_HEADER: "stale-token",
        }
        with (
            patch.object(server.onboarding.assignment, "load", return_value=None),
            patch.object(server.manager.node_registry, "get", return_value=None),
            patch.object(
                server.manager.node_registry,
                "accepts_forward_token",
                return_value=False,
            ),
        ):
            unregister = await self.client.post(
                "/api/v1/onboarding/unregister", headers=headers,
            )
            rejected = await self.client.get("/api/state", headers=headers)

        self.assertEqual(unregister.status_code, 404)
        self.assertEqual(unregister.json(), {"detail": "worker is not registered"})
        self.assertEqual(rejected.status_code, 401)
        self.assertEqual(
            rejected.json(), {"detail": "invalid worker forwarding credential"},
        )

    async def test_legacy_patch_preserves_general_registry_updates(self):
        update = Mock(return_value={"id": "remote-1", "enabled": False})
        with patch.object(server.manager.node_registry, "update", update):
            response = await self.client.patch(
                "/api/nodes/remote-1", json={"enabled": False},
            )

        self.assertEqual(response.status_code, 200)
        update.assert_called_once_with("remote-1", {"enabled": False})


if __name__ == "__main__":
    unittest.main()
