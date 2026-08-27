import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

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
        self.set_setting_patch.stop()
        self.jwks_patch.stop()

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
        self.push_pair.assert_awaited_once_with("user-sub-123", "user@example.com")

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

    async def test_unpair_marks_the_device_not_paired(self):
        with patch.object(
            server.sparkdeck.store, "get_setting",
            return_value={"status": "paired", "sub": "user-sub-123",
                          "email": "user@example.com"},
        ):
            response = await self.client.delete("/api/v1/community/pair")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "pairing": {"status": "not_paired"},
            "cluster": {"applied": [], "conflicts": [], "errors": []},
        })
        self.set_setting.assert_called_once_with(
            "device_pairing", {"status": "not_paired"})
        self.push_unpair.assert_awaited_once_with("user-sub-123")


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

    async def asyncTearDown(self):
        await self.client.aclose()
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

    async def test_same_account_pairing_is_a_noop(self):
        self.pair_locally()

        response = await self.client.put(
            "/api/agent/community-pairing",
            json={"sub": "user-sub-123", "email": "user@example.com"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"applied": True, "already": True})
        self.set_setting.assert_not_called()

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
            "user-sub-123", "user@example.com")

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


if __name__ == "__main__":
    unittest.main()
