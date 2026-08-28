import mimetypes
import tempfile
import unittest
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from sparkdeck.web import (
    SPA_PATHS,
    configure_static_asset_mime_types,
    register_spa_routes,
)


class SpaRoutingTests(unittest.IsolatedAsyncioTestCase):
    def test_cluster_route_is_in_direct_refresh_allowlist(self):
        self.assertIn("/cluster", SPA_PATHS)
        self.assertIn("/fan-control", SPA_PATHS)
        self.assertIn("/storage", SPA_PATHS)
        self.assertIn("/usage", SPA_PATHS)
        self.assertIn("/switch", SPA_PATHS)

    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        frontend_dist = root / "frontend"
        static_dir = root / "static"
        frontend_dist.mkdir()
        static_dir.mkdir()
        (frontend_dist / "index.html").write_text(
            "<!doctype html><title>SparkDeck SPA</title><main>sparkdeck-spa</main>",
            encoding="utf-8",
        )
        (frontend_dist / "bundle.js").write_text("window.sparkdeck = true", encoding="utf-8")
        (static_dir / "legacy.css").write_text("body {}", encoding="utf-8")

        app = FastAPI()

        @app.get("/api/health")
        async def api_health():
            return PlainTextResponse("api-health")

        @app.get("/v1/models")
        async def openai_models():
            return PlainTextResponse("openai-models")

        app.mount("/static/app", StaticFiles(directory=frontend_dist), name="app-static")
        app.mount("/static", StaticFiles(directory=static_dir), name="legacy-static")
        register_spa_routes(app, frontend_dist)

        @app.get("/disk-manager")
        async def disk_manager():
            return PlainTextResponse("disk-manager")

        mcp = FastAPI()

        @mcp.get("/{path:path}")
        async def mcp_fallback(path: str):
            return PlainTextResponse(f"mcp:{path}")

        app.mount("", mcp, name="mcp")
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        self.temp_dir.cleanup()

    async def test_every_primary_browser_route_serves_the_spa_entry(self) -> None:
        for path in SPA_PATHS:
            with self.subTest(path=path):
                response = await self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("sparkdeck-spa", response.text)

    async def test_spa_routes_do_not_shadow_reserved_surfaces(self) -> None:
        expectations = {
            "/api/health": "api-health",
            "/v1/models": "openai-models",
            "/static/app/bundle.js": "window.sparkdeck = true",
            "/static/legacy.css": "body {}",
            "/disk-manager": "disk-manager",
            "/mcp": "mcp:mcp",
            "/unknown-browser-path": "mcp:unknown-browser-path",
        }
        for path, expected in expectations.items():
            with self.subTest(path=path):
                response = await self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.text, expected)

    async def test_javascript_assets_override_a_plain_text_host_mapping(self) -> None:
        mimetypes.add_type("text/plain", ".js", strict=True)
        self.addCleanup(configure_static_asset_mime_types)
        configure_static_asset_mime_types()

        response = await self.client.get("/static/app/bundle.js")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/javascript; charset=utf-8")

    async def test_missing_frontend_build_returns_actionable_503(self) -> None:
        app = FastAPI()
        register_spa_routes(app, Path(self.temp_dir.name) / "missing")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/settings")
        self.assertEqual(response.status_code, 503)
        self.assertIn("npm --prefix frontend run build", response.text)


if __name__ == "__main__":
    unittest.main()
