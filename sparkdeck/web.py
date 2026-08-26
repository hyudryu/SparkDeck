"""Browser entry routing for the SparkDeck single-page application."""

from __future__ import annotations

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
    "/benchmarks",
    "/images",
    "/settings",
    "/logs",
)


def register_spa_routes(app: FastAPI, frontend_dist: Path) -> None:
    """Serve the SPA entry only for known browser routes.

    An explicit allowlist is intentional: a catch-all route here would consume
    unknown API, static, disk-manager, or MCP requests before their mounted
    applications get a chance to handle them.
    """

    async def spa_entry() -> Response:
        index_file = frontend_dist / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return Response(
            "SparkDeck's web app has not been built. "
            "Run ./run.sh or npm --prefix frontend run build.",
            status_code=503,
            media_type="text/plain",
        )

    for path in SPA_PATHS:
        slug = path.strip("/").replace("/", "_") or "root"
        app.add_api_route(
            path,
            spa_entry,
            methods=["GET"],
            include_in_schema=False,
            name=f"sparkdeck_spa_{slug}",
        )
