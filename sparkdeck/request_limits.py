"""Bound request buffering for inference payloads such as base64 vision inputs."""

import json
from typing import Any

MAX_INFERENCE_REQUEST_BYTES = 32 * 1024 * 1024
# Cluster routing adds a few trusted private fields after the public request has
# passed the limit. Give authenticated agent requests enough headroom for that
# envelope without increasing the client-facing payload allowance.
MAX_CLUSTER_ROUTING_ENVELOPE_BYTES = 64 * 1024


class RequestBodyTooLarge(ValueError):
    """Raised before an inference request can exceed the buffering limit."""


def is_inference_request_path(path: str) -> bool:
    return path in {"/v1/chat/completions", "/v1/completions"} or path.startswith(
        "/api/agent/inference/"
    )


async def read_limited_request_body(request: Any, max_bytes: int) -> bytes:
    raw_length = request.headers.get("content-length")
    if raw_length is not None:
        try:
            content_length = int(raw_length)
        except (TypeError, ValueError):
            content_length = None
        if content_length is not None and content_length > max_bytes:
            raise RequestBodyTooLarge

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_bytes:
            raise RequestBodyTooLarge
        body.extend(chunk)
    return bytes(body)


async def read_limited_json(request: Any, max_bytes: int) -> Any:
    return json.loads(await read_limited_request_body(request, max_bytes))
