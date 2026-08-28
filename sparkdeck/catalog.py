"""Hugging Face-backed public model catalog with a bounded local cache."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
from urllib.parse import quote
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

_QUANTIZATION_PATTERN = re.compile(
    r"(?:^|[/_.-])((?:UD-)?(?:IQ|Q)\d(?:_[A-Z0-9]+)*|NVFP4|FP8|BF16|FP16|F16|AWQ|GPTQ)(?=$|[/_.-])",
    re.IGNORECASE,
)


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
        self._detail_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
        # Deduplicate only identical requests. Different repositories must be
        # able to fetch concurrently during bulk community enrichment.
        self._search_locks: dict[tuple[str, int, str], asyncio.Lock] = {}
        self._detail_locks: dict[tuple[str, str], asyncio.Lock] = {}

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
        async with self._search_locks.setdefault(key, asyncio.Lock()):
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

    async def details(self, repository: str) -> dict[str, Any]:
        """Return public Hub metadata plus every downloadable GGUF quantization."""
        repository = repository.strip().strip("/")
        if (
            repository.count("/") != 1
            or any(part in {"", ".", ".."} for part in repository.split("/"))
        ):
            raise ValueError("model repository must be an owner/name Hugging Face ID")
        token = str(self.token_provider() or "") if self.token_provider else ""
        token_key = hashlib.sha256(token.encode("utf-8")).hexdigest() if token else "public"
        key = (repository.casefold(), token_key)
        now = time.monotonic()
        cached = self._detail_cache.get(key)
        if cached and now - cached[0] < self.ttl_seconds:
            return cached[1]
        async with self._detail_locks.setdefault(key, asyncio.Lock()):
            cached = self._detail_cache.get(key)
            if cached and now - cached[0] < self.ttl_seconds:
                return cached[1]
            detail_params = [
                *(('expand[]', field) for field in (
                    "author", "downloads", "likes", "tags", "safetensors",
                    "gguf", "pipeline_tag", "gated", "private", "lastModified",
                    "siblings",
                )),
            ]
            request_headers = {
                "Authorization": f"Bearer {token}"
            } if token else {}
            response = await self.http.get(
                f"https://huggingface.co/api/models/{quote(repository, safe='/')}",
                params=detail_params,
                headers=request_headers,
                timeout=15,
            )
            if response.status_code in {301, 302, 303, 307, 308}:
                redirected = response.headers.get("location", "")
                redirected_url = httpx.URL("https://huggingface.co").join(redirected)
                if (
                    redirected_url.scheme != "https"
                    or redirected_url.host != "huggingface.co"
                    or not redirected_url.path.startswith("/api/models/")
                ):
                    raise ValueError("Hugging Face returned an unsafe model redirect")
                response = await self.http.get(
                    redirected_url.copy_with(query=None),
                    params=detail_params,
                    headers=request_headers,
                    timeout=15,
                )
            response.raise_for_status()
            raw = response.json()
            if not isinstance(raw, dict) or raw.get("private"):
                raise ValueError("Hugging Face returned an invalid public model")
            raw.setdefault("id", repository)
            detail_repository = str(raw.get("id") or repository)
            siblings = raw.get("siblings")
            if _gguf_sizes_missing(siblings):
                tree_path = (
                    f"/api/models/{quote(detail_repository, safe='/')}/tree/main"
                )
                tree_url = httpx.URL(f"https://huggingface.co{tree_path}")
                tree_params: dict[str, Any] | None = {
                    "recursive": "true", "limit": 1000,
                }
                tree: list[dict[str, Any]] = []
                seen_pages: set[str] = set()
                while True:
                    tree_response = await self.http.get(
                    tree_url,
                    params=tree_params,
                    headers=request_headers,
                    timeout=15,
                    )
                    tree_response.raise_for_status()
                    current_page = str(tree_response.url)
                    if current_page in seen_pages:
                        raise ValueError(
                            "Hugging Face returned a repeated model tree page"
                        )
                    seen_pages.add(current_page)
                    page = tree_response.json()
                    if not isinstance(page, list):
                        raise ValueError("Hugging Face returned an invalid model tree")
                    tree.extend(entry for entry in page if isinstance(entry, dict))

                    next_href = tree_response.links.get("next", {}).get("url")
                    if not next_href:
                        break
                    next_url = tree_response.url.join(str(next_href))
                    if (
                        next_url.scheme != "https"
                        or next_url.host != "huggingface.co"
                        or next_url.path != tree_path
                    ):
                        raise ValueError(
                            "Hugging Face returned an unsafe model tree page"
                        )
                    tree_url = next_url
                    tree_params = None
                sizes = {
                    str(entry.get("path") or ""): _positive_int(entry.get("size"))
                    for entry in tree
                }
                if isinstance(siblings, list):
                    siblings = [
                        {
                            **sibling,
                            **(
                                {"size": sizes[str(sibling.get("rfilename") or "")]}
                                if sizes.get(str(sibling.get("rfilename") or ""))
                                else {}
                            ),
                        }
                        if isinstance(sibling, dict) else sibling
                        for sibling in siblings
                    ]
                    raw["siblings"] = siblings
            item = self._public_item(raw)
            item["quantizations"] = _gguf_quantizations(raw.get("siblings"))
            self._detail_cache[key] = (time.monotonic(), item)
            return item

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


def quantization_from_text(*values: Any) -> str | None:
    """Conservatively extract a recognized weight quantization marker."""
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        match = _QUANTIZATION_PATTERN.search(text)
        if match:
            value = match.group(1).upper()
            return "FP16" if value == "F16" else value
    return None


def _gguf_quantizations(raw_siblings: Any) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    if not isinstance(raw_siblings, list):
        return []
    for sibling in raw_siblings:
        if not isinstance(sibling, dict):
            continue
        filename = str(
            sibling.get("rfilename") or sibling.get("path") or sibling.get("name") or ""
        ).strip()
        folded_filename = filename.casefold()
        if (
            not folded_filename.endswith(".gguf")
            or "mmproj" in folded_filename
            or folded_filename.startswith("mtp/")
            or "/mtp-" in folded_filename
        ):
            continue
        quantization = quantization_from_text(filename)
        if not quantization:
            continue
        size = _positive_int(sibling.get("size"))
        lfs = sibling.get("lfs")
        if size is None and isinstance(lfs, dict):
            size = _positive_int(lfs.get("size"))
        group = grouped.setdefault(quantization, {
            "name": quantization,
            "files": [],
            "weight_size_bytes": 0,
        })
        file_item: dict[str, Any] = {"filename": filename}
        if size is not None:
            file_item["size_bytes"] = size
            group["weight_size_bytes"] += size
        group["files"].append(file_item)
    for group in grouped.values():
        if not group["weight_size_bytes"]:
            group["weight_size_bytes"] = None
        group["files"].sort(key=lambda item: item["filename"].casefold())
    return sorted(grouped.values(), key=lambda item: item["name"].casefold())


def _gguf_sizes_missing(raw_siblings: Any) -> bool:
    if not isinstance(raw_siblings, list):
        return False
    for sibling in raw_siblings:
        if not isinstance(sibling, dict):
            continue
        filename = str(sibling.get("rfilename") or "")
        if not filename.casefold().endswith(".gguf"):
            continue
        lfs = sibling.get("lfs")
        if _positive_int(sibling.get("size")) is None and not (
            isinstance(lfs, dict) and _positive_int(lfs.get("size")) is not None
        ):
            return True
    return False
