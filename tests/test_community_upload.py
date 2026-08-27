import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import jwt as pyjwt
from cryptography.hazmat.primitives.asymmetric import rsa

from sparkdeck.models import BenchmarkSample, ModelIdentity, RuntimeKind
from sparkdeck.storage import SparkDeckStore


with patch("docker.from_env", return_value=Mock()):
    import server


SIGNING_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _id_token():
    return pyjwt.encode({
        "sub": "user-sub-123",
        "email": "user@example.com",
        "iss": server.COGNITO_ISSUER,
        "aud": server.COGNITO_CLIENT_ID,
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
        "token_use": "id",
    }, SIGNING_KEY, algorithm="RS256", headers={"kid": "test-key"})


def _jwks_stub():
    client = Mock()
    client.get_signing_key_from_jwt.return_value = Mock(
        key=SIGNING_KEY.public_key())
    return client


def _bearer():
    return {"Authorization": f"Bearer {_id_token()}"}


def _sample(sample_id="sample-1"):
    return BenchmarkSample(
        id=sample_id, created_at="2026-08-25T00:00:00+00:00",
        deployment_id=None, model=ModelIdentity("org/model"),
        runtime=RuntimeKind.VLLM, runtime_version=None,
        hardware={"architecture": "x86_64"},
        configuration={"context_length": 4096},
        input_tokens=20, output_tokens=30, latency_ms=100,
        ttft_ms=10, generation_tokens_per_second=300,
        prompt_tokens_per_second=200, cold_start=False,
        eligible_for_community=True,
    )


def _stub_http(test_case, handler):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5)
    patcher = patch.object(server, "_community_http", http)
    patcher.start()
    test_case.addCleanup(patcher.stop)
    return http


class CommunityAggregatesProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app),
            base_url="http://test",
        )
        self.get_setting_patch = patch.object(
            server.sparkdeck.store, "get_setting",
        )
        self.get_setting = self.get_setting_patch.start()
        self.get_setting.side_effect = lambda key, default=None: {
            "community_consent": True,
            "device_pairing": {
                "status": "paired", "sub": "user-sub-123",
                "email": "user@example.com", "refresh_token": "refresh-1",
            },
        }.get(key, default)
        api_url_patch = patch.object(
            server, "COMMUNITY_API_URL", "https://community.example",
        )
        api_url_patch.start()
        self.addCleanup(api_url_patch.stop)
        self.jwks_patch = patch.object(
            server, "_cognito_jwks", return_value=_jwks_stub(),
        )
        self.jwks_patch.start()

    async def asyncTearDown(self):
        await self.client.aclose()
        self.jwks_patch.stop()
        self.get_setting_patch.stop()

    async def test_passthrough_forwards_the_authorization_header(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={
                "items": [{
                    "model_id": "org/model",
                    "context_window_size": 4096,
                    "inference_tokens_per_second": 42.5,
                    "sample_count": 12,
                }],
                "availability": "ok",
                "evidence_policy": {"minimum_samples": 10},
            })

        http = _stub_http(self, handler)
        self.addAsyncCleanup(http.aclose)

        authorization = _bearer()["Authorization"]
        response = await self.client.get(
            "/api/v1/community/aggregates",
            headers={"Authorization": authorization},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["availability"], "ok")
        self.assertEqual(response.json()["items"][0]["model_id"], "org/model")
        self.assertEqual(str(seen[0].url), "https://community.example/v1/aggregates")
        self.assertEqual(seen[0].headers["authorization"], authorization)

    async def test_upstream_unauthorized_maps_to_401(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"message": "Unauthorized"})

        http = _stub_http(self, handler)
        self.addAsyncCleanup(http.aclose)

        response = await self.client.get(
            "/api/v1/community/aggregates", headers=_bearer())

        self.assertEqual(response.status_code, 401)

    async def test_unconfigured_falls_back_to_the_local_stub(self):
        with patch.object(server, "COMMUNITY_API_URL", ""):
            response = await self._request_unconfigured()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["availability"], "not_configured")

    async def _request_unconfigured(self):
        http = _stub_http(self, Mock(side_effect=AssertionError("no HTTP")))
        self.addAsyncCleanup(http.aclose)
        with patch.object(
            server.sparkdeck, "community_aggregates",
            AsyncMock(return_value={
                "items": [], "availability": "not_configured",
                "evidence_policy": {},
            }),
        ):
            return await self.client.get(
                "/api/v1/community/aggregates", headers=_bearer())

    async def test_upstream_outage_reports_unavailable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        http = _stub_http(self, handler)
        self.addAsyncCleanup(http.aclose)

        response = await self.client.get(
            "/api/v1/community/aggregates", headers=_bearer())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["availability"], "unavailable")
        self.assertEqual(response.json()["items"], [])


class CommunityUploadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SparkDeckStore(Path(self.temp.name) / "sparkdeck.sqlite3")
        self.store_patch = patch.object(server.sparkdeck, "store", self.store)
        self.store_patch.start()
        self.api_url_patch = patch.object(
            server, "COMMUNITY_API_URL", "https://community.example",
        )
        self.api_url_patch.start()
        self.requests: list[httpx.Request] = []
        server._community_token_cache.update({
            "refresh_token": None, "id_token": None, "expires_at": 0.0,
        })

    async def asyncTearDown(self):
        self.api_url_patch.stop()
        self.store_patch.stop()
        self.store.close()
        self.temp.cleanup()
        server._community_token_cache.update({
            "refresh_token": None, "id_token": None, "expires_at": 0.0,
        })

    def configure(self, consent=True, paired=True):
        self.store.set_setting("community_consent", consent)
        if paired:
            self.store.set_setting("device_pairing", {
                "status": "paired", "sub": "user-sub-123",
                "email": "user@example.com", "refresh_token": "refresh-1",
            })

    def handler(self, sample_response=None, id_token="id-1"):
        def route(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if "cognito-idp" in str(request.url):
                return httpx.Response(200, json={
                    "AuthenticationResult": {
                        "IdToken": id_token, "AccessToken": "a", "ExpiresIn": 3600,
                    },
                })
            if sample_response is not None:
                return sample_response(request)
            return httpx.Response(201, json={"accepted": True})
        return route

    def sample_requests(self):
        return [r for r in self.requests if "cognito-idp" not in str(r.url)]

    def idp_requests(self):
        return [r for r in self.requests if "cognito-idp" in str(r.url)]

    async def test_pending_samples_are_uploaded_and_marked_synced(self):
        self.configure()
        self.store.add_benchmark(_sample(), queue=True)
        http = _stub_http(self, self.handler())
        self.addAsyncCleanup(http.aclose)

        result = await server.community_upload_once()

        self.assertEqual(result, {"uploaded": 1, "failed": 0})
        self.assertEqual(self.store.sync_status()["outbox"]["synced"], 1)
        uploads = self.sample_requests()
        self.assertEqual(len(uploads), 1)
        self.assertEqual(
            str(uploads[0].url), "https://community.example/v1/samples")
        self.assertEqual(
            uploads[0].headers["authorization"], "Bearer id-1")
        self.assertEqual(json.loads(uploads[0].content), {
            "model_id": "org/model",
            "context_window_size": 4096,
            "inference_tokens_per_second": 300.0,
        })

    async def test_transport_errors_leave_samples_pending(self):
        self.configure()
        self.store.add_benchmark(_sample(), queue=True)

        def fail(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        http = _stub_http(self, self.handler(sample_response=fail))
        self.addAsyncCleanup(http.aclose)

        result = await server.community_upload_once()

        self.assertEqual(result, {"uploaded": 0, "failed": 0})
        outbox = self.store.sync_status()["outbox"]
        self.assertEqual(outbox["pending"], 1)
        self.assertEqual(outbox["synced"], 0)
        self.assertEqual(outbox["failed"], 0)

    async def test_schema_rejection_marks_samples_failed(self):
        self.configure()
        self.store.add_benchmark(_sample(), queue=True)
        http = _stub_http(self, self.handler(
            sample_response=lambda request: httpx.Response(
                400, json={"message": "strict validation"}),
        ))
        self.addAsyncCleanup(http.aclose)

        result = await server.community_upload_once()

        self.assertEqual(result, {"uploaded": 0, "failed": 1})
        outbox = self.store.sync_status()["outbox"]
        self.assertEqual(outbox["failed"], 1)
        self.assertEqual(outbox["pending"], 0)

    async def test_deleted_queued_sample_is_rechecked_before_its_turn(self):
        self.configure()
        self.store.add_benchmark(_sample("sample-1"), queue=True)
        self.store.add_benchmark(_sample("sample-2"), queue=True)

        def route(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if "cognito-idp" in str(request.url):
                return httpx.Response(200, json={
                    "AuthenticationResult": {
                        "IdToken": "id-1", "ExpiresIn": 3600,
                    },
                })
            # Deleting after the batch snapshot must stop the second upload.
            self.store.delete_benchmark("sample-2")
            return httpx.Response(201, json={"accepted": True})

        http = _stub_http(self, route)
        self.addAsyncCleanup(http.aclose)

        result = await server.community_upload_once()

        self.assertEqual(result, {"uploaded": 1, "failed": 0})
        self.assertEqual(len(self.sample_requests()), 1)
        self.assertEqual(self.store.sync_status()["outbox"]["synced"], 1)

    async def test_consent_off_is_a_noop(self):
        self.configure(consent=False)
        self.store.add_benchmark(_sample(), queue=True)
        http = _stub_http(self, self.handler())
        self.addAsyncCleanup(http.aclose)

        result = await server.community_upload_once()

        self.assertEqual(result["reason"], "consent_off")
        self.assertEqual(self.requests, [])

    async def test_expired_id_token_triggers_one_refresh_and_retries(self):
        self.configure()
        self.store.add_benchmark(_sample(), queue=True)
        idp_calls = []
        sample_attempts = []

        def route(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if "cognito-idp" in str(request.url):
                idp_calls.append(request)
                return httpx.Response(200, json={
                    "AuthenticationResult": {
                        "IdToken": f"id-{len(idp_calls)}", "ExpiresIn": 3600,
                    },
                })
            sample_attempts.append(request.headers["authorization"])
            if len(sample_attempts) == 1:
                return httpx.Response(401, json={"message": "expired"})
            return httpx.Response(201, json={"accepted": True})

        http = _stub_http(self, route)
        self.addAsyncCleanup(http.aclose)

        result = await server.community_upload_once()

        self.assertEqual(result, {"uploaded": 1, "failed": 0})
        self.assertEqual(len(idp_calls), 2)
        self.assertEqual(sample_attempts, ["Bearer id-1", "Bearer id-2"])
        body = json.loads(idp_calls[0].content)
        self.assertEqual(body["AuthFlow"], "REFRESH_TOKEN_AUTH")
        self.assertEqual(
            body["AuthParameters"], {"REFRESH_TOKEN": "refresh-1"})

    async def test_id_tokens_are_cached_until_near_expiry(self):
        self.configure()
        http = _stub_http(self, self.handler())
        self.addAsyncCleanup(http.aclose)

        first = await server._community_id_token("refresh-1")
        second = await server._community_id_token("refresh-1")

        self.assertEqual(first, "id-1")
        self.assertEqual(second, "id-1")
        self.assertEqual(len(self.idp_requests()), 1)


if __name__ == "__main__":
    unittest.main()
