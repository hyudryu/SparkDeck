"""Hugging Face-backed public model catalog with a bounded local cache."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx


class HuggingFaceCatalog:
    def __init__(self, http: httpx.AsyncClient, ttl_seconds: float = 300.0):
        self.http = http
        self.ttl_seconds = ttl_seconds
        self._cache: dict[tuple[str, int], tuple[float, list[dict[str, Any]]]] = {}
        self._lock = asyncio.Lock()

    async def search(self, query: str, limit: int = 24) -> list[dict[str, Any]]:
        query = query.strip()
        limit = min(100, max(1, int(limit)))
        key = (query.casefold(), limit)
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached and now - cached[0] < self.ttl_seconds:
            return cached[1]
        async with self._lock:
            cached = self._cache.get(key)
            if cached and now - cached[0] < self.ttl_seconds:
                return cached[1]
            response = await self.http.get(
                "https://huggingface.co/api/models",
                params={"search": query, "limit": limit, "sort": "downloads", "direction": -1},
                timeout=15,
            )
            response.raise_for_status()
            items = [self._public_item(item) for item in response.json() if isinstance(item, dict)]
            self._cache[key] = (time.monotonic(), items)
            return items

    @staticmethod
    def _public_item(item: dict[str, Any]) -> dict[str, Any]:
        tags = [str(tag) for tag in item.get("tags", [])]
        formats = []
        if any(tag.casefold() == "gguf" for tag in tags):
            formats.append("gguf")
        runtime_compatibility = {
            "vllm": "compatible" if any(tag in tags for tag in ("transformers", "safetensors")) else "unknown",
            "llama.cpp": "compatible" if "gguf" in formats else "unknown",
            "sglang": "compatible" if any(tag in tags for tag in ("transformers", "safetensors")) else "unknown",
        }
        return {
            "id": item.get("id") or item.get("modelId"),
            "author": item.get("author"),
            "downloads": item.get("downloads", 0),
            "likes": item.get("likes", 0),
            "pipeline_tag": item.get("pipeline_tag"),
            "last_modified": item.get("lastModified"),
            "private": bool(item.get("private", False)),
            "gated": item.get("gated", False),
            "formats": formats,
            "runtime_compatibility": runtime_compatibility,
            "community": None,
        }
