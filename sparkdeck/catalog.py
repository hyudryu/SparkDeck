"""Hugging Face-backed public model catalog with a bounded local cache."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
from contextlib import asynccontextmanager
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
_GGUF_SHARD_PATTERN = re.compile(
    r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$",
    re.IGNORECASE,
)
_REVISION_PATTERN = re.compile(r"[0-9a-f]{4,64}$", re.IGNORECASE)
_SNAPSHOT_DIRECTORY_PATTERN = re.compile(r"(?:^|/)checkpoint-\d+(?:/|$)", re.IGNORECASE)


@asynccontextmanager
async def _keyed_lock(pool: dict[Any, dict[str, Any]], key: Any):
    """Hold one keyed lock and evict it after its last waiter exits."""
    entry = pool.get(key)
    if entry is None:
        entry = {"lock": asyncio.Lock(), "users": 0}
        pool[key] = entry
    entry["users"] += 1
    try:
        async with entry["lock"]:
            yield
    finally:
        entry["users"] -= 1
        if entry["users"] == 0 and pool.get(key) is entry:
            pool.pop(key, None)


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
        self._enrich_timeout = 8.0
        self._cache: dict[tuple[str, int, str], tuple[float, list[dict[str, Any]]]] = {}
        self._detail_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
        # Deduplicate only identical requests. Different repositories must be
        # able to fetch concurrently during bulk community enrichment.
        self._search_locks: dict[tuple[str, int, str], dict[str, Any]] = {}
        self._detail_locks: dict[tuple[str, str], dict[str, Any]] = {}

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
        async with _keyed_lock(self._search_locks, key):
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
                        "siblings", "sha",
                    )),
                ],
                headers={"Authorization": f"Bearer {token}"} if token else {},
                timeout=15,
            )
            response.raise_for_status()
            raw_items = response.json()
            if not isinstance(raw_items, list):
                raise ValueError("Hugging Face returned an invalid model list")
            items = []
            for item in raw_items:
                if not isinstance(item, dict) or item.get("private"):
                    continue
                public = self._public_item(item)
                if not public.get("id"):
                    continue
                public["quantizations"] = _gguf_quantizations(item.get("siblings"))
                items.append(public)
            if await self._enrich_tree_weight_sizes(items):
                # A partially enriched (timed-out) result still carries
                # inflated estimates, so it must not be cached as complete.
                self._cache[key] = (time.monotonic(), items)
            return items

    async def _enrich_tree_weight_sizes(self, items: list[dict[str, Any]]) -> bool:
        """Replace inflated safetensors metadata estimates with tree sizes.

        Hub safetensors metadata double-counts tensors shared across shards,
        so sizes derived from element counts can be far larger than the real
        download. Each candidate costs one tree request, pinned to the search
        result's revision; failures keep the estimate so search never fails
        here. Returns True when every candidate finished, meaning the result
        is complete enough to cache.
        """
        candidates = [
            item for item in items
            if _is_safetensors_candidate(item)
        ]
        if not candidates:
            return True
        token = str(self.token_provider() or "") if self.token_provider else ""
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        semaphore = asyncio.Semaphore(8)
        failed = 0

        async def load(item: dict[str, Any]) -> None:
            nonlocal failed
            revision = str(item.get("revision") or "")
            if not _REVISION_PATTERN.fullmatch(revision):
                revision = "main"
            async with semaphore:
                try:
                    tree = await self._fetch_tree(str(item["id"]), headers, revision)
                except (httpx.HTTPError, ValueError):
                    # A transient Hub failure must not be cached as a
                    # completed enrichment; the estimate is still inflated.
                    failed += 1
                    return
            size = _tree_weight_size(tree)
            if size:
                item["weight_size_bytes"] = size
                item["weight_size_source"] = "tree"
            else:
                # A tree response without usable sizes leaves the inflated
                # estimate in place and must not be cached as complete.
                failed += 1

        tasks = [asyncio.create_task(load(item)) for item in candidates]
        _, pending = await asyncio.wait(tasks, timeout=self._enrich_timeout)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            return False
        return failed == 0

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
        async with _keyed_lock(self._detail_locks, key):
            cached = self._detail_cache.get(key)
            if cached and now - cached[0] < self.ttl_seconds:
                return cached[1]
            detail_params = [
                *(('expand[]', field) for field in (
                    "author", "downloads", "likes", "tags", "safetensors",
                    "gguf", "pipeline_tag", "gated", "private", "lastModified",
                    "siblings", "sha",
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
            # Pin the tree fetch to the metadata response's commit so the
            # weight size and the advertised revision stay consistent.
            revision = str(raw.get("sha") or "")
            if not _REVISION_PATTERN.fullmatch(revision):
                revision = "main"
            siblings = raw.get("siblings")
            item = self._public_item(raw)
            tree: list[dict[str, Any]] | None = None
            safetensors_lookup = (
                item["weight_size_source"] == "safetensors"
                or (
                    item["weight_size_source"] is None
                    and _has_safetensors(raw)
                )
            )
            tree_failed = False
            if _gguf_sizes_missing(siblings):
                tree = await self._fetch_tree(
                    detail_repository, request_headers, revision,
                )
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
            elif safetensors_lookup:
                # Hub safetensors metadata double-counts tensors shared across
                # shards, inflating any size derived from element counts.
                # Prefer real weight-file sizes; keep the estimate on failure.
                try:
                    tree = await self._fetch_tree(
                        detail_repository, request_headers, revision,
                    )
                except (httpx.HTTPError, ValueError):
                    tree = None
                    tree_failed = True
            if tree is not None and safetensors_lookup:
                tree_weight_size = _tree_weight_size(tree)
                if tree_weight_size:
                    item["weight_size_bytes"] = tree_weight_size
                    item["weight_size_source"] = "tree"
            item["quantizations"] = _gguf_quantizations(raw.get("siblings"))
            if not tree_failed:
                # A fallback produced after a transient tree error must be
                # retried after the Hub recovers, not cached as complete.
                self._detail_cache[key] = (time.monotonic(), item)
            return item

    async def _fetch_tree(
        self,
        repository: str,
        headers: dict[str, str],
        revision: str = "main",
    ) -> list[dict[str, Any]]:
        """Return the full recursive file listing for a model repository."""
        tree_path = (
            f"/api/models/{quote(repository, safe='/')}"
            f"/tree/{quote(revision, safe='')}"
        )
        tree_url = httpx.URL(f"https://huggingface.co{tree_path}")
        tree_params: dict[str, Any] | None = {"recursive": "true", "limit": 1000}
        tree: list[dict[str, Any]] = []
        seen_pages: set[str] = set()
        while True:
            tree_response = await self.http.get(
                tree_url,
                params=tree_params,
                headers=headers,
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
        return tree

    @staticmethod
    def _public_item(item: dict[str, Any]) -> dict[str, Any]:
        repository = str(item.get("id") or item.get("modelId") or "").strip()
        tags = [str(tag)[:100] for tag in item.get("tags", []) if isinstance(tag, str)][:100]
        folded_tags = {tag.casefold() for tag in tags}
        siblings = item.get("siblings")
        has_gguf_sibling = isinstance(siblings, list) and any(
            isinstance(sibling, dict)
            and str(sibling.get("rfilename") or sibling.get("path") or "")
            .casefold().endswith(".gguf")
            for sibling in siblings
        )
        formats = []
        gguf_metadata = item.get("gguf")
        if (
            "gguf" in folded_tags
            or (isinstance(gguf_metadata, dict) and bool(gguf_metadata))
            or has_gguf_sibling
        ):
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
            # The commit the file listing was resolved at, so clients can
            # compare locally cached snapshot revisions against current files.
            "revision": str(item.get("sha") or "")[:64] or None,
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
            return canonical_quantization(match.group(1))
    return None


def canonical_quantization(value: Any) -> str | None:
    """Canonicalize one explicit or inferred public quantization label."""
    normalized = str(value or "").strip().upper()
    if (
        not normalized or len(normalized) > 100
        or any(ord(character) < 32 for character in normalized)
    ):
        return None
    return {"F16": "FP16"}.get(normalized, normalized)


def _gguf_quantizations(raw_siblings: Any) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
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
        quantization = quantization_from_text(filename) or "unknown"
        size = _positive_int(sibling.get("size"))
        lfs = sibling.get("lfs")
        if size is None and isinstance(lfs, dict):
            size = _positive_int(lfs.get("size"))
        path = filename.replace("\\", "/")
        parent, _, basename = path.rpartition("/")
        shard = _GGUF_SHARD_PATTERN.match(basename)
        artifact_key = (
            f"{parent}/{shard.group('stem')}" if parent and shard
            else shard.group("stem") if shard
            else path
        )
        artifacts = grouped.setdefault(quantization, {})
        group = artifacts.setdefault(artifact_key, {
            "filename": filename,
            "files": [],
            "weight_size_bytes": 0,
            "shard_count": int(shard.group("count")) if shard else 1,
            "shard_indexes": set(),
        })
        if shard:
            group["shard_indexes"].add(int(shard.group("index")))
            if int(shard.group("count")) != group["shard_count"]:
                group["shard_count"] = 0
        file_item: dict[str, Any] = {"filename": filename}
        if size is not None:
            file_item["size_bytes"] = size
            group["weight_size_bytes"] += size
        group["files"].append(file_item)
    result = []
    for quantization, artifact_groups in grouped.items():
        artifacts = []
        for group in artifact_groups.values():
            group["files"].sort(key=lambda item: item["filename"].casefold())
            verified = (
                group["shard_count"] == 1
                or group["shard_indexes"]
                == set(range(1, group["shard_count"] + 1))
            )
            if not verified:
                continue
            artifacts.append({
                "filename": group["files"][0]["filename"],
                "files": group["files"],
                "weight_size_bytes": (
                    group["weight_size_bytes"]
                    if verified and group["weight_size_bytes"] else None
                ),
                "sharded": group["shard_count"] > 1,
            })
        if not artifacts:
            continue
        artifacts.sort(key=lambda item: (
            item["weight_size_bytes"] is None,
            item["weight_size_bytes"] or math.inf,
            item["filename"].casefold(),
        ))
        preferred = artifacts[0]
        result.append({
            "name": quantization,
            "files": preferred["files"],
            "weight_size_bytes": preferred["weight_size_bytes"],
            "artifacts": artifacts,
        })
    return sorted(result, key=lambda item: item["name"].casefold())


def _has_safetensors(item: dict[str, Any]) -> bool:
    """Whether a raw Hub payload carries safetensors weights evidence."""
    if isinstance(item.get("safetensors"), dict):
        return True
    siblings = item.get("siblings")
    return isinstance(siblings, list) and any(
        isinstance(sibling, dict)
        and str(sibling.get("rfilename") or "").casefold().endswith(".safetensors")
        for sibling in siblings
    )


def _is_safetensors_candidate(item: dict[str, Any]) -> bool:
    """Whether a public search item could hold inflated or missing sizes."""
    source = item.get("weight_size_source")
    if source == "safetensors":
        return True
    if source is not None:
        return False
    return "safetensors" in {
        str(tag).casefold() for tag in item.get("tags") or []
    }


def _tree_weight_size(tree: list[dict[str, Any]]) -> int | None:
    """Sum the primary safetensors checkpoint from a repository tree listing.

    Only safetensors files are summed: repositories that also publish
    alternative copies of the same checkpoint (``pytorch_model.bin``, GGUF
    variants) must not have every copy counted into one total. Resolution is
    per directory: files with an unmarked sibling are alternatives, a
    quantization-named directory (``awq/``, ``bf16/``) is always an
    alternative, and a component directory whose files exist in only one
    dtype (``unet/model.fp16.safetensors`` with no unmarked counterpart) is a
    required part of the pipeline. Training snapshots in ``checkpoint-N``
    directories are never loaded by a normal deployment and are skipped. When
    nothing qualifies as primary (quant-collection repos), the largest
    single variant, aggregated across component directories, is reported.
    """
    per_directory: dict[str, dict[str, Any]] = {}
    for entry in tree:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("type") or "file") != "file":
            continue
        path = str(entry.get("path") or "").replace("\\", "/")
        if not path.casefold().endswith(".safetensors"):
            continue
        size = _positive_int(entry.get("size"))
        if size is None:
            # An incomplete listing must not override the metadata estimate.
            return None
        if _SNAPSHOT_DIRECTORY_PATTERN.search(path):
            continue
        directory = path.rpartition("/")[0]
        info = per_directory.setdefault(directory, {"unmarked": 0, "marked": {}})
        marker = quantization_from_text(path)
        if marker is None:
            info["unmarked"] += size
        else:
            info["marked"][marker] = info["marked"].get(marker, 0) + size
    primary = 0
    variants: dict[str, int] = {}
    for directory, info in per_directory.items():
        marked: dict[str, int] = info["marked"]
        if info["unmarked"]:
            primary += info["unmarked"]
        elif directory and quantization_from_text(directory) is None and marked:
            # A component directory existing in only one dtype is required.
            best = max(marked, key=marked.get)
            primary += marked.pop(best)
        for marker, size in marked.items():
            variants[marker] = variants.get(marker, 0) + size
    if primary:
        return primary
    if variants:
        return max(variants.values())
    return None


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
