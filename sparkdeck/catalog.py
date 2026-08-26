"""Hugging Face-backed public model catalog with a bounded local cache."""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, Callable

import httpx


class HuggingFaceCatalog:
    def __init__(
        self,
        http: httpx.AsyncClient,
        ttl_seconds: float = 300.0,
        token_provider: Callable[[], str] | None = None,
    ):
        self.http = http
        self.ttl_seconds = ttl_seconds
        self.token_provider = token_provider
        self._cache: dict[tuple[str, int, str], tuple[float, list[dict[str, Any]]]] = {}
        self._lock = asyncio.Lock()

    async def search(self, query: str, limit: int = 24) -> list[dict[str, Any]]:
        query = query.strip()
        limit = min(100, max(1, int(limit)))
        token = str(self.token_provider() or "") if self.token_provider else ""
        token_key = hashlib.sha256(token.encode("utf-8")).hexdigest() if token else "public"
        key = (query.casefold(), limit, token_key)
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
                headers={"Authorization": f"Bearer {token}"} if token else {},
                timeout=15,
            )
            response.raise_for_status()
            raw_items = response.json()
            if not isinstance(raw_items, list):
                raise ValueError("Hugging Face returned an invalid model list")
            items = [
                public for item in raw_items
                if isinstance(item, dict) and not item.get("private")
                if (public := self._public_item(item)).get("id")
            ]
            self._cache[key] = (time.monotonic(), items)
            return items

    @staticmethod
    def _public_item(item: dict[str, Any]) -> dict[str, Any]:
        repository = str(item.get("id") or item.get("modelId") or "").strip()
        tags = [str(tag)[:100] for tag in item.get("tags", []) if isinstance(tag, str)][:100]
        folded_tags = {tag.casefold() for tag in tags}
        formats = []
        if "gguf" in folded_tags:
            formats.append("gguf")
        transformer_model = bool(folded_tags & {"transformers", "safetensors"})
        runtime_compatibility = [
            {"runtime": "vllm", "supported": transformer_model},
            {"runtime": "llama.cpp", "supported": "gguf" in formats},
            {"runtime": "sglang", "supported": transformer_model},
        ]
        return {
            "id": repository,
            "author": str(item.get("author") or repository.partition("/")[0] or "")[:200] or None,
            "name": repository.split("/")[-1],
            "downloads": _nonnegative_int(item.get("downloads")),
            "likes": _nonnegative_int(item.get("likes")),
            "pipeline_tag": str(item.get("pipeline_tag") or "")[:100] or None,
            "last_modified": str(item.get("lastModified") or "")[:100] or None,
            "private": False,
            "gated": bool(item.get("gated", False)),
            "tags": tags,
            "formats": formats,
            "runtime_compatibility": runtime_compatibility,
            "community": None,
        }


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0
