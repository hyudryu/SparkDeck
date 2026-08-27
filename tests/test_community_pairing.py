import threading
import time
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

import httpx
import jwt as pyjwt
from cryptography.hazmat.primitives.asymmetric import rsa

from manager import Manager


with patch("docker.from_env", return_value=Mock()):
    import server


def _rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


SIGNING_KEY = _rsa_key()
OTHER_KEY = _rsa_key()


def _id_token(signing_key=SIGNING_KEY, **claim_overrides):
    claims = {
        "sub": "user-sub-123",
        "email": "user@example.com",
        "iss": server.COGNITO_ISSUER,
        "aud": server.COGNITO_CLIENT_ID,
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
        "token_use": "id",
    }
    claims.update(claim_overrides)
    return pyjwt.encode(claims, signing_key, algorithm="RS256",
                        headers={"kid": "test-key"})


def _jwks_stub(public_key):
    client = Mock()
    client.get_signing_key_from_jwt.return_value = Mock(key=public_key)
    return client


def _bearer(token=None):
    return {"Authorization": f"Bearer {token or _id_token()}"}


class CommunityPairingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app),
            base_url="http://test",
        )
        self.jwks_patch = patch.object(
            server, "_cognito_jwks",
            return_value=_jwks_stub(SIGNING_KEY.public_key()),
        )
        self.jwks_patch.start()
        self.set_setting_patch = patch.object(
            server.sparkdeck.store, "set_setting",
        )
        self.set_setting = self.set_setting_patch.start()
        self.get_setting_patch = patch.object(
            server.sparkdeck.store, "get_setting",
            return_value={"status": "not_paired"},
        )
        self.get_setting = self.get_setting_patch.start()
        self.promote_patch = patch.object(
            server.sparkdeck.store, "promote_outbox_for_pairing",
            Mock(return_value=0),
        )
        self.promote = self.promote_patch.start()
        self.push_pair_patch = patch.object(
            server.manager, "push_community_pairing",
            AsyncMock(return_value={"applied": [], "conflicts": [], "errors": []}),
        )
        self.push_pair = self.push_pair_patch.start()
        self.push_unpair_patch = patch.object(
            server.manager, "push_community_unpair",
            AsyncMock(return_value={"applied": [], "conflicts": [], "errors": []}),
        )
        self.push_unpair = self.push_unpair_patch.start()

    async def asyncTearDown(self):
        await self.client.aclose()
        self.push_unpair_patch.stop()
        self.push_pair_patch.stop()
        self.promote_patch.stop()
        self.get_setting_patch.stop()
        self.set_setting_patch.stop()
        self.jwks_patch.stop()

    async def test_csp_permits_the_cognito_idp_origin(self):
        response = await self.client.get("/api/v1/community/sync")

        self.assertEqual(response.status_code, 200)
        csp = response.headers["Content-Security-Policy"]
        self.assertIn(
            "connect-src 'self' https://cognito-idp.us-east-2.amazonaws.com", csp)

    async def test_auth_config_uses_the_backend_runtime_values(self):
        with (
            patch.object(server, "COGNITO_IDP_ORIGIN", "https://idp.example"),
            patch.object(server, "COGNITO_CLIENT_ID", "custom-client"),
        ):
            response = await self.client.get("/api/v1/community/auth-config")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "idp_endpoint": "https://idp.example/",
            "client_id": "custom-client",
        })

    async def test_valid_id_token_pairs_the_device(self):
        response = await self.client.post(
            "/api/v1/community/pair", json={"id_token": _id_token()})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "pairing": {
                "status": "paired",
                "sub": "user-sub-123",
                "email": "user@example.com",
            },
            "cluster": {"applied": [], "conflicts": [], "errors": []},
        })
        self.set_setting.assert_called_once_with("device_pairing", {
            "status": "paired",
            "sub": "user-sub-123",
            "email": "user@example.com",
        })
        self.promote.assert_called_once_with()
        self.push_pair.assert_awaited_once_with("user-sub-123", "user@example.com", None)

    async def test_pairing_refuses_to_overwrite_a_different_account(self):
        self.get_setting.return_value = {
            "status": "paired", "sub": "other-sub", "email": "other@example.com",
        }

        response = await self.client.post(
            "/api/v1/community/pair", json={"id_token": _id_token()})

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json(), {
            "error": "already_paired",
            "existing": {"email": "other@example.com"},
        })
        self.set_setting.assert_not_called()
        self.promote.assert_not_called()
        self.push_pair.assert_not_awaited()

    async def test_repairing_the_same_account_is_a_noop_success(self):
        self.get_setting.return_value = {
            "status": "paired", "sub": "user-sub-123",
            "email": "user@example.com",
        }

        response = await self.client.post(
            "/api/v1/community/pair", json={"id_token": _id_token()})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["pairing"]["sub"], "user-sub-123")
        self.push_pair.assert_awaited_once_with("user-sub-123", "user@example.com", None)

    async def test_pairing_stores_the_refresh_token_without_echoing_it(self):
        response = await self.client.post(
            "/api/v1/community/pair",
            json={"id_token": _id_token(), "refresh_token": "refresh-secret-1"})

        self.assertEqual(response.status_code, 200)
        self.set_setting.assert_called_once_with("device_pairing", {
            "status": "paired",
            "sub": "user-sub-123",
            "email": "user@example.com",
            "refresh_token": "refresh-secret-1",
        })
        self.assertNotIn("refresh-secret-1", response.text)
        self.assertEqual(response.json()["pairing"], {
            "status": "paired",
            "sub": "user-sub-123",
            "email": "user@example.com",
        })
        self.push_pair.assert_awaited_once_with(
            "user-sub-123", "user@example.com", "refresh-secret-1")

    async def test_pairing_rejects_a_non_string_refresh_token(self):
        response = await self.client.post(
            "/api/v1/community/pair",
            json={"id_token": _id_token(), "refresh_token": 42})

        self.assertEqual(response.status_code, 400)
        self.set_setting.assert_not_called()

    async def test_jwks_verification_runs_off_the_event_loop(self):
        verify_threads = []

        def recording_verify(token):
            verify_threads.append(threading.get_ident())
            return {"sub": "user-sub-123", "email": "user@example.com"}

        with patch.object(
            server, "_verify_cognito_id_token", side_effect=recording_verify,
        ):
            response = await self.client.post(
                "/api/v1/community/pair", json={"id_token": _id_token()})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(verify_threads), 1)
        self.assertNotEqual(verify_threads[0], threading.get_ident())

    async def test_pairing_reports_cluster_conflicts_without_failing(self):
        self.push_pair.return_value = {
            "applied": ["Spark Two"],
            "conflicts": [{"node": "Spark Three", "email": "other@example.com"}],
            "errors": ["Spark Four: unreachable"],
        }

        response = await self.client.post(
            "/api/v1/community/pair", json={"id_token": _id_token()})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["cluster"], {
            "applied": ["Spark Two"],
            "conflicts": [{"node": "Spark Three", "email": "other@example.com"}],
            "errors": ["Spark Four: unreachable"],
        })

    async def test_wrong_signing_key_is_rejected(self):
        response = await self.client.post(
            "/api/v1/community/pair",
            json={"id_token": _id_token(signing_key=OTHER_KEY)})

        self.assertEqual(response.status_code, 401)
        self.set_setting.assert_not_called()
        self.push_pair.assert_not_awaited()

    async def test_wrong_audience_is_rejected(self):
        response = await self.client.post(
            "/api/v1/community/pair",
            json={"id_token": _id_token(aud="some-other-client")})

        self.assertEqual(response.status_code, 401)
        self.set_setting.assert_not_called()

    async def test_expired_token_is_rejected(self):
        response = await self.client.post(
            "/api/v1/community/pair",
            json={"id_token": _id_token(exp=int(time.time()) - 60)})

        self.assertEqual(response.status_code, 401)
        self.set_setting.assert_not_called()

    async def test_wrong_issuer_is_rejected(self):
        response = await self.client.post(
            "/api/v1/community/pair",
            json={"id_token": _id_token(iss="https://issuer.example")})

        self.assertEqual(response.status_code, 401)
        self.set_setting.assert_not_called()

    async def test_missing_id_token_is_a_client_error(self):
        response = await self.client.post("/api/v1/community/pair", json={})

        self.assertEqual(response.status_code, 400)
        self.set_setting.assert_not_called()

    async def test_non_id_token_is_rejected(self):
        response = await self.client.post(
            "/api/v1/community/pair",
            json={"id_token": _id_token(token_use="access")},
        )

        self.assertEqual(response.status_code, 401)
        self.set_setting.assert_not_called()

    async def test_unpair_marks_the_device_not_paired(self):
        self.get_setting.return_value = {
            "status": "paired", "sub": "user-sub-123",
            "email": "user@example.com",
        }

        response = await self.client.delete(
            "/api/v1/community/pair", headers=_bearer())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "pairing": {"status": "not_paired"},
            "cluster": {"applied": [], "conflicts": [], "errors": []},
        })
        self.set_setting.assert_called_once_with(
            "device_pairing", {"status": "not_paired"})
        self.push_unpair.assert_awaited_once_with("user-sub-123")

    async def test_unpair_without_local_pairing_skips_the_fanout(self):
        response = await self.client.delete(
            "/api/v1/community/pair", headers=_bearer())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "pairing": {"status": "not_paired"},
            "cluster": {"applied": [], "conflicts": [], "errors": []},
        })
        self.push_unpair.assert_not_awaited()

    async def test_unpair_requires_the_matching_account(self):
        self.get_setting.return_value = {
            "status": "paired", "sub": "other-sub", "email": "other@example.com",
        }

        response = await self.client.delete(
            "/api/v1/community/pair", headers=_bearer())

        self.assertEqual(response.status_code, 403)
        self.set_setting.assert_not_called()
        self.push_unpair.assert_not_awaited()

    async def test_unpair_requires_a_valid_bearer_when_paired(self):
        self.get_setting.return_value = {
            "status": "paired", "sub": "user-sub-123", "email": "user@example.com",
        }

        response = await self.client.delete("/api/v1/community/pair")

        self.assertEqual(response.status_code, 401)
        self.set_setting.assert_not_called()
        self.push_unpair.assert_not_awaited()

    async def test_aggregates_require_matching_pairing_and_consent(self):
        self.get_setting.side_effect = lambda key, default=None: {
            "device_pairing": {
                "status": "paired", "sub": "user-sub-123",
                "email": "user@example.com",
            },
            "community_consent": True,
        }.get(key, default)
        with patch.object(
            server.sparkdeck, "community_aggregates",
            AsyncMock(return_value={"items": []}),
        ) as aggregate:
            response = await self.client.get(
                "/api/v1/community/aggregates", headers=_bearer())

        self.assertEqual(response.status_code, 200)
        aggregate.assert_awaited_once_with()

    async def test_aggregates_are_denied_before_service_access(self):
        self.get_setting.side_effect = lambda key, default=None: {
            "device_pairing": {
                "status": "paired", "sub": "user-sub-123",
                "email": "user@example.com",
            },
            "community_consent": False,
        }.get(key, default)
        with patch.object(
            server.sparkdeck, "community_aggregates", AsyncMock(),
        ) as aggregate:
            missing = await self.client.get("/api/v1/community/aggregates")
            no_consent = await self.client.get(
                "/api/v1/community/aggregates", headers=_bearer())

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(no_consent.status_code, 403)
        aggregate.assert_not_awaited()

    async def test_consent_changes_propagate_and_surface_cluster_failures(self):
        cluster_result = {
            "applied": ["Spark Two"],
            "conflicts": [],
            "errors": ["Spark Three: unreachable"],
        }
        with (
            patch.object(
                server.sparkdeck, "set_community_consent", AsyncMock(),
            ) as set_consent,
            patch.object(
                server.manager, "push_community_consent",
                AsyncMock(return_value=cluster_result),
            ) as push_consent,
            patch.object(
                server, "_community_sync_status",
                return_value={"consent": False, "pairing": {"status": "paired"}},
            ),
        ):
            enabled = await self.client.put(
                "/api/v1/community/consent", json={"enabled": True},
            )
            disabled = await self.client.put(
                "/api/v1/community/consent", json={"enabled": False},
            )

        self.assertEqual(enabled.status_code, 200)
        self.assertEqual(disabled.status_code, 200)
        self.assertEqual(disabled.json(), {
            "consent": False,
            "pairing": {"status": "paired"},
            "cluster": cluster_result,
        })
        self.assertEqual(
            set_consent.await_args_list, [call(True), call(False)],
        )
        self.assertEqual(
            push_consent.await_args_list, [call(True), call(False)],
        )


class AgentCommunityPairingTests(unittest.IsolatedAsyncioTestCase):
    """The agent endpoint applies controller pushes without overriding."""

    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app),
            base_url="http://test",
        )
        self.agent_patch = patch.object(server, "_require_agent")
        self.agent_patch.start()
        self.get_setting_patch = patch.object(
            server.sparkdeck.store, "get_setting",
        )
        self.get_setting = self.get_setting_patch.start()
        self.set_setting_patch = patch.object(
            server.sparkdeck.store, "set_setting",
        )
        self.set_setting = self.set_setting_patch.start()
        self.promote_patch = patch.object(
            server.sparkdeck.store, "promote_outbox_for_pairing",
            Mock(return_value=0),
        )
        self.promote = self.promote_patch.start()

    async def asyncTearDown(self):
        await self.client.aclose()
        self.promote_patch.stop()
        self.set_setting_patch.stop()
        self.get_setting_patch.stop()
        self.agent_patch.stop()

    def pair_locally(self, sub="user-sub-123", email="user@example.com"):
        self.get_setting.return_value = {
            "status": "paired", "sub": sub, "email": email,
        }

    async def test_applies_pairing_when_not_paired(self):
        self.get_setting.return_value = {"status": "not_paired"}

        response = await self.client.put(
            "/api/agent/community-pairing",
            json={"sub": "user-sub-123", "email": "user@example.com"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"applied": True})
        self.set_setting.assert_called_once_with("device_pairing", {
            "status": "paired", "sub": "user-sub-123",
            "email": "user@example.com",
        })
        self.promote.assert_called_once_with()

    async def test_applies_pairing_with_a_refresh_token(self):
        self.get_setting.return_value = {"status": "not_paired"}

        response = await self.client.put(
            "/api/agent/community-pairing",
            json={"sub": "user-sub-123", "email": "user@example.com",
                  "refresh_token": "refresh-secret-1"})

        self.assertEqual(response.status_code, 200)
        self.set_setting.assert_called_once_with("device_pairing", {
            "status": "paired", "sub": "user-sub-123",
            "email": "user@example.com", "refresh_token": "refresh-secret-1",
        })
        self.assertNotIn("refresh-secret-1", response.text)

    async def test_same_account_pairing_is_a_noop(self):
        self.pair_locally()

        response = await self.client.put(
            "/api/agent/community-pairing",
            json={"sub": "user-sub-123", "email": "user@example.com"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"applied": True, "already": True})
        self.set_setting.assert_not_called()
        self.promote.assert_called_once_with()

    async def test_different_account_pairing_is_refused(self):
        self.pair_locally()

        response = await self.client.put(
            "/api/agent/community-pairing",
            json={"sub": "other-sub", "email": "other@example.com"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "applied": False, "existing": {"email": "user@example.com"},
        })
        self.set_setting.assert_not_called()

    async def test_pairing_requires_a_sub(self):
        self.get_setting.return_value = {"status": "not_paired"}

        response = await self.client.put(
            "/api/agent/community-pairing", json={"email": "user@example.com"})

        self.assertEqual(response.status_code, 400)
        self.set_setting.assert_not_called()

    async def test_unpair_when_not_paired_is_a_noop(self):
        self.get_setting.return_value = {"status": "not_paired"}

        response = await self.client.request(
            "DELETE", "/api/agent/community-pairing",
            json={"sub": "user-sub-123"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"applied": True, "already": True})
        self.set_setting.assert_not_called()

    async def test_unpair_applies_for_the_same_account(self):
        self.pair_locally()

        response = await self.client.request(
            "DELETE", "/api/agent/community-pairing",
            json={"sub": "user-sub-123"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"applied": True})
        self.set_setting.assert_called_once_with(
            "device_pairing", {"status": "not_paired"})

    async def test_unpair_is_refused_for_a_different_account(self):
        self.pair_locally()

        response = await self.client.request(
            "DELETE", "/api/agent/community-pairing", json={"sub": "other-sub"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "applied": False, "existing": {"email": "user@example.com"},
        })
        self.set_setting.assert_not_called()

    async def test_unpair_requires_a_sub(self):
        self.get_setting.return_value = {"status": "not_paired"}

        response = await self.client.request(
            "DELETE", "/api/agent/community-pairing", json={})

        self.assertEqual(response.status_code, 400)
        self.set_setting.assert_not_called()

    async def test_consent_enable_and_withdrawal_use_the_shared_service_setter(self):
        with patch.object(
            server.sparkdeck, "set_community_consent", AsyncMock(),
        ) as set_consent:
            enabled = await self.client.put(
                "/api/agent/community-consent", json={"enabled": True},
            )
            disabled = await self.client.put(
                "/api/agent/community-consent", json={"enabled": False},
            )

        self.assertEqual(
            enabled.json(), {"applied": True, "enabled": True},
        )
        self.assertEqual(
            disabled.json(), {"applied": True, "enabled": False},
        )
        self.assertEqual(
            set_consent.await_args_list, [call(True), call(False)],
        )

    async def test_agent_consent_requires_a_boolean(self):
        with patch.object(
            server.sparkdeck, "set_community_consent", AsyncMock(),
        ) as set_consent:
            response = await self.client.put(
                "/api/agent/community-consent", json={"enabled": "false"},
            )

        self.assertEqual(response.status_code, 400)
        set_consent.assert_not_awaited()


class CommunityPairingFanoutTests(unittest.IsolatedAsyncioTestCase):
    """The controller merges per-node results without failing the request."""

    def manager_with_nodes(self, nodes, request):
        instance = Manager.__new__(Manager)
        instance.node_registry = Mock(nodes=nodes, request=request)
        return instance

    def nodes(self, *names):
        return [
            {"id": f"node-{index}", "name": name, "enabled": True}
            for index, name in enumerate(names, start=1)
        ]

    async def test_pairing_fanout_merges_applied_conflicts_and_errors(self):
        request = AsyncMock(side_effect=[
            {"applied": True},
            {"applied": False, "existing": {"email": "other@example.com"}},
            RuntimeError("Spark Four agent error: unreachable"),
        ])
        instance = self.manager_with_nodes(
            self.nodes("Spark Two", "Spark Three", "Spark Four"), request)

        result = await instance.push_community_pairing(
            "user-sub-123", "user@example.com", "refresh-1")

        self.assertEqual(result, {
            "applied": ["Spark Two"],
            "conflicts": [{"node": "Spark Three", "email": "other@example.com"}],
            "errors": ["Spark Four: Spark Four agent error: unreachable"],
        })
        self.assertEqual(request.await_count, 3)
        first_call = request.await_args_list[0]
        self.assertEqual(first_call.args[:3], (
            "node-1", "PUT", "/api/agent/community-pairing"))
        self.assertEqual(first_call.kwargs["json_body"], {
            "sub": "user-sub-123", "email": "user@example.com",
            "refresh_token": "refresh-1",
        })

    async def test_unpair_fanout_skips_disabled_nodes(self):
        request = AsyncMock(return_value={"applied": True})
        nodes = self.nodes("Spark Two") + [
            {"id": "node-off", "name": "Spark Off", "enabled": False},
        ]
        instance = self.manager_with_nodes(nodes, request)

        result = await instance.push_community_unpair("user-sub-123")

        self.assertEqual(result, {
            "applied": ["Spark Two"], "conflicts": [], "errors": [],
        })
        request.assert_awaited_once()
        self.assertEqual(request.await_args.args[:3], (
            "node-1", "DELETE", "/api/agent/community-pairing"))
        self.assertEqual(
            request.await_args.kwargs["json_body"], {"sub": "user-sub-123"})

    async def test_no_peers_reports_empty_result(self):
        instance = self.manager_with_nodes([], AsyncMock())

        result = await instance.push_community_pairing("user-sub-123", None)

        self.assertEqual(result, {"applied": [], "conflicts": [], "errors": []})

    async def test_consent_withdrawal_reaches_all_joined_nodes_and_reports_failures(self):
        request = AsyncMock(side_effect=[
            {"applied": True, "enabled": False},
            RuntimeError("unreachable"),
        ])
        nodes = self.nodes("Spark Two", "Spark Disabled")
        nodes[1]["enabled"] = False
        instance = self.manager_with_nodes(nodes, request)

        result = await instance.push_community_consent(False)

        self.assertEqual(result, {
            "applied": ["Spark Two"],
            "conflicts": [],
            "errors": ["Spark Disabled: unreachable"],
        })
        self.assertEqual(request.await_count, 2)
        for node_id, item in zip(("node-1", "node-2"), request.await_args_list):
            self.assertEqual(item.args[:3], (
                node_id, "PUT", "/api/agent/community-consent",
            ))
            self.assertEqual(item.kwargs["json_body"], {"enabled": False})
            self.assertEqual(item.kwargs["timeout"], 20)
            self.assertTrue(item.kwargs["allow_disabled"])


if __name__ == "__main__":
    unittest.main()
