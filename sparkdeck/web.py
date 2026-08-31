"""Browser entry routing for the SparkDeck single-page application."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse


SPA_PATHS = (
    "/",
    "/dashboard",
    "/explore",
    "/models",
    "/chat",
    "/compare",
    "/cluster",
    "/fan-control",
    "/switch",
    "/storage",
    "/benchmarks",
    "/usage",
    "/images",
    "/settings",
    "/logs",
)

SPA_ROUTE_PATTERNS = (
    "/models/{deployment_id}",
)


def configure_static_asset_mime_types() -> None:
    """Keep browser module assets executable across host MIME registries.

    Windows can register ``.js`` as ``text/plain``. Starlette delegates static
    response types to :mod:`mimetypes`, and browsers reject ES modules served
    with that type. Register the web-standard types after Python has loaded the
    host registry so SparkDeck behaves consistently on every platform.
    """
    mimetypes.add_type("text/javascript", ".js", strict=True)
    mimetypes.add_type("text/javascript", ".mjs", strict=True)
    mimetypes.add_type("application/wasm", ".wasm", strict=True)


def register_spa_routes(app: FastAPI, frontend_dist: Path) -> None:
    """Serve the SPA entry only for known browser routes.

    An explicit allowlist is intentional: a catch-all route here would consume
    unknown API, static, disk-manager, or MCP requests before their mounted
    applications get a chance to handle them.
    """

    async def spa_entry() -> Response:
        index_file = frontend_dist / "index.html"
        if index_file.exists():
            # The entry points at content-hashed assets and must be revalidated
            # on every navigation. Caching it can strand a normal refresh on an
            # old bundle until the user forces a hard reload.
            return FileResponse(
                index_file,
                headers={"Cache-Control": "no-store, max-age=0"},
            )
        return Response(
            "SparkDeck's web app has not been built. "
            "Run ./run.sh or npm --prefix frontend run build.",
            status_code=503,
            media_type="text/plain",
        )

    for path in (*SPA_PATHS, *SPA_ROUTE_PATTERNS):
        slug = (
            path.strip("/")
            .replace("/", "_")
            .replace("{", "")
            .replace("}", "")
            or "root"
        )
        app.add_api_route(
            path,
            spa_entry,
            methods=["GET"],
            include_in_schema=False,
            name=f"sparkdeck_spa_{slug}",
        )
