"""Hugging Face-backed public model catalog with a bounded local cache."""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from typing import Any, Callable

import httpx


_SAFETENSORS_BYTES_PER_VALUE = {
    "BOOL": 1,
    "U8": 1, "I8": 1,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E4M3FN": 1, "F8_E5M2FNUZ": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
    "U4": 0.5, "I4": 0.5,
}


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
                params=[
                    ("search", query), ("limit", limit),
                    ("sort", "downloads"), ("direction", -1),
                    *(("expand[]", field) for field in (
                        "author", "downloads", "likes", "tags", "safetensors",
                        "gguf", "pipeline_tag", "gated", "private", "lastModified",
                    )),
                ],
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
        parameter_count, weight_size_bytes, weight_size_source = _weight_metadata(item)
        return {
            "id": repository,
            "author": str(item.get("author") or repository.partition("/")[0] or "")[:200] or None,
            "name": repository.split("/")[-1],
            "downloads": _nonnegative_int(item.get("downloads")),
            "likes": _nonnegative_int(item.get("likes")),
            "parameter_count": parameter_count,
            "weight_size_bytes": weight_size_bytes,
            "weight_size_source": weight_size_source,
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


def _weight_metadata(item: dict[str, Any]) -> tuple[int | None, int | None, str | None]:
    """Return public parameter and weight-byte metadata reported by the Hub."""
    safetensors = item.get("safetensors")
    if isinstance(safetensors, dict):
        total = _positive_int(safetensors.get("total"))
        parameters = safetensors.get("parameters")
        if isinstance(parameters, dict) and parameters:
            weight_size = 0.0
            for dtype, count in parameters.items():
                width = _SAFETENSORS_BYTES_PER_VALUE.get(str(dtype).upper())
                values = _positive_int(count)
                if width is None or values is None:
                    weight_size = 0.0
                    break
                weight_size += values * width
            if weight_size > 0:
                return total or sum(
                    _positive_int(count) or 0 for count in parameters.values()
                ), math.ceil(weight_size), "safetensors"
        if total:
            return total, None, None

    gguf = item.get("gguf")
    if isinstance(gguf, dict):
        total = _positive_int(gguf.get("total"))
        weight_size = _positive_int(gguf.get("totalFileSize"))
        if total or weight_size:
            return total, weight_size, "gguf" if weight_size else None
    return None, None, None


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None
