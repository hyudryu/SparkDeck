"""Stateless Responses protocol adapter for OpenAI-compatible chat engines."""

from __future__ import annotations

import codecs
import json
import time
import uuid
from collections.abc import AsyncIterator


def _id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex


def _flat_tools(request: dict):
    flattened = []
    names = set()
    for tool in request.get("tools", []):
        namespace = tool.get("name") if tool.get("type") == "namespace" else None
        for child in tool.get("tools", []) if namespace else [tool]:
            name = f"{namespace}__{child['name']}" if namespace else child.get("name")
            if not name or name in names:
                raise ValueError("Tool names must be present and unique after namespace expansion")
            names.add(name)
            flattened.append(({**child, "name": name}, namespace, child["name"]))
    return flattened


def _custom_names(request: dict) -> set[str]:
    return {tool["name"] for tool, _, _ in _flat_tools(request) if tool.get("type") == "custom"}


def _restore_namespace(item: dict, request: dict):
    for tool, namespace, name in _flat_tools(request):
        if tool["name"] == item.get("name") and namespace:
            item.update(name=name, namespace=namespace)
    return item


def _content(content, *, tool_output=False):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Message content must be text or a list of content parts")
    parts = []
    for part in content:
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": part["text"]})
        elif kind == "input_image" and not tool_output:
            if not part.get("image_url"):
                raise ValueError("input_image requires image_url; uploaded files are not supported")
            parts.append({"type": "image_url", "image_url": {
                "url": part["image_url"], "detail": part.get("detail", "auto")}})
        else:
            raise ValueError(f"Unsupported Responses content type: {kind}")
    if tool_output:
        return "\n".join(part["text"] for part in parts)
    return parts


def to_chat_request(body: dict) -> dict:
    """Validate and translate a stateless Responses request before inference."""
    if not isinstance(body, dict):
        raise ValueError("Request body must be an object")
    if not isinstance(body.get("model"), str) or not body["model"].strip():
        raise ValueError("model must be a nonempty string")
    if "stream" in body and not isinstance(body["stream"], bool):
        raise ValueError("stream must be a boolean")
    try:
        return _to_chat_request(body)
    except (TypeError, KeyError, AttributeError) as exc:
        raise ValueError("Invalid Responses request structure") from exc


def _to_chat_request(body: dict) -> dict:
    for field in ("previous_response_id", "conversation", "background"):
        if body.get(field):
            raise ValueError(f"{field} is not supported; send the full input history")
    if body.get("store"):
        raise ValueError("Stored responses are not supported; use store: false")
    if any(value != "reasoning.encrypted_content" for value in body.get("include", [])):
        raise ValueError("Responses include expansions are not supported")
    messages = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})
    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [{"role": "user", "content": inputs}]
    if not isinstance(inputs, list):
        raise ValueError("input must be a string or an array")
    for item in inputs:
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role", "user")
            if role not in ("user", "assistant", "system", "developer"):
                raise ValueError(f"Unsupported message role: {role}")
            messages.append({"role": "system" if role == "developer" else role,
                             "content": _content(item.get("content", ""))})
        elif kind in ("function_call", "custom_tool_call"):
            arguments = item.get("arguments", "") if kind == "function_call" else json.dumps({"input": item.get("input", "")})
            call = {"id": item["call_id"], "type": "function", "function": {
                "name": f"{item['namespace']}__{item['name']}" if item.get("namespace") else item["name"], "arguments": arguments}}
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif kind in ("function_call_output", "custom_tool_call_output"):
            messages.append({"role": "tool", "tool_call_id": item["call_id"],
                             "content": _content(item.get("output", ""), tool_output=True)})
        else:
            raise ValueError(f"Unsupported Responses input item: {kind}")
    result = {"model": body.get("model"), "messages": messages, "stream": bool(body.get("stream", False))}
    for field in ("temperature", "top_p", "parallel_tool_calls", "user", "seed"):
        if field in body:
            result[field] = body[field]
    if "max_output_tokens" in body:
        result["max_tokens"] = body["max_output_tokens"]
    if body.get("reasoning", {}).get("effort"):
        result["reasoning_effort"] = body["reasoning"]["effort"]
    tools = []
    for tool, _, _ in _flat_tools(body):
        kind = tool.get("type")
        if kind == "function":
            function = {key: tool[key] for key in ("name", "description", "parameters", "strict") if key in tool}
        elif kind == "custom":
            function = {"name": tool["name"], "description": tool.get("description", ""),
                        "parameters": {"type": "object", "properties": {"input": {"type": "string"}},
                                       "required": ["input"], "additionalProperties": False}}
            if tool.get("format", {}).get("type") == "grammar":
                function["description"] += "\nThe input must follow this grammar:\n" + tool["format"].get("definition", "")
        else:
            raise ValueError(f"Unsupported hosted tool type: {kind}")
        tools.append({"type": "function", "function": function})
    if tools:
        result["tools"] = tools
    if "tool_choice" in body:
        choice = body["tool_choice"]
        if isinstance(choice, dict):
            if choice.get("type") not in ("function", "custom"):
                raise ValueError("Unsupported tool_choice")
            name = f"{choice['namespace']}__{choice['name']}" if choice.get("namespace") else choice["name"]
            choice = {"type": "function", "function": {"name": name}}
        result["tool_choice"] = choice
    fmt = body.get("text", {}).get("format")
    if fmt:
        if fmt.get("type") == "json_schema":
            result["response_format"] = {"type": "json_schema", "json_schema": {
                key: fmt[key] for key in ("name", "schema", "strict", "description") if key in fmt}}
        elif fmt.get("type") in ("text", "json_object"):
            result["response_format"] = fmt
        else:
            raise ValueError("Unsupported text.format")
    if result["stream"]:
        result["stream_options"] = {"include_usage": True}
    return result


def _usage(usage: dict | None):
    if usage is None:
        return None
    return {"input_tokens": usage.get("prompt_tokens", 0),
            "input_tokens_details": {"cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)},
            "output_tokens": usage.get("completion_tokens", 0),
            "output_tokens_details": {"reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)},
            "total_tokens": usage.get("total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))}


def _response(request: dict) -> dict:
    return {"id": _id("resp_"), "object": "response", "created_at": int(time.time()),
            "status": "in_progress", "error": None, "incomplete_details": None,
            "model": request.get("model"), "output": [], "usage": None,
            "parallel_tool_calls": request.get("parallel_tool_calls", True), "store": False}


def _call_item(call: dict, custom: set[str], status="completed") -> dict:
    function = call.get("function", {})
    name = function.get("name", "")
    arguments = function.get("arguments", "")
    item = {"id": _id("fc_"), "type": "function_call", "status": status,
            "call_id": call.get("id") or _id("call_"), "name": name, "arguments": arguments}
    if name in custom:
        try:
            value = json.loads(arguments)["input"]
            if not isinstance(value, str):
                raise ValueError("Custom tool input must be a string")
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid arguments from custom tool {name}") from exc
        item.update(type="custom_tool_call", input=value)
        del item["arguments"]
    return item


def from_chat_response(result: dict, original_request: dict) -> dict:
    """Convert a complete upstream response; malformed payloads are errors."""
    if not isinstance(result, dict):
        raise ValueError("Upstream chat response must be an object")
    try:
        return _from_chat_response(result, original_request)
    except (TypeError, KeyError, AttributeError, IndexError) as exc:
        raise ValueError("Malformed upstream chat response") from exc


def _from_chat_response(result: dict, original_request: dict) -> dict:
    response = _response(original_request)
    if result.get("error"):
        response.update(status="failed", error=result["error"])
        return response
    choices = result.get("choices", [])
    if not choices:
        raise ValueError("Upstream chat response has no choices")
    choice = choices[0]
    if not choice.get("finish_reason"):
        raise ValueError("Upstream chat response has no finish reason")
    message = choice.get("message", {})
    status = "incomplete" if choice.get("finish_reason") in ("length", "content_filter") else "completed"
    if message.get("content") is not None:
        response["output"].append({"id": _id("msg_"), "type": "message", "role": "assistant",
                                   "status": status, "content": [{"type": "output_text", "text": message["content"], "annotations": []}]})
    if message.get("refusal"):
        response["output"].append({"id": _id("msg_"), "type": "message", "role": "assistant",
                                   "status": status, "content": [{"type": "refusal", "refusal": message["refusal"]}]})
    for call in message.get("tool_calls", []):
        response["output"].append(_restore_namespace(_call_item(call, _custom_names(original_request), status), original_request))
    response.update(status=status, usage=_usage(result.get("usage")))
    if status == "incomplete":
        response["incomplete_details"] = {"reason": "max_output_tokens" if choice["finish_reason"] == "length" else "content_filter"}
    return response


async def _sse_payloads(upstream):
    buffer = ""
    decoder = codecs.getincrementaldecoder("utf-8")()
    async for chunk in upstream:
        buffer += decoder.decode(chunk) if isinstance(chunk, bytes) else chunk
        buffer = buffer.replace("\r\n", "\n")
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            data = "\n".join(line[5:].lstrip() for line in frame.split("\n") if line.startswith("data:"))
            if data:
                yield data
    buffer += decoder.decode(b"", final=True)
    if buffer.strip():
        data = "\n".join(line[5:].lstrip() for line in buffer.splitlines() if line.startswith("data:"))
        if data:
            yield data


async def stream_chat_response(upstream: AsyncIterator[str], original_request: dict) -> AsyncIterator[str]:
    """Translate chat SSE into Responses events, retaining upstream ownership."""
    response = _response(original_request)
    sequence = 0
    custom = _custom_names(original_request)
    message = None
    calls = {}
    finish = None

    def event(kind, **fields):
        nonlocal sequence
        payload = {"type": kind, "sequence_number": sequence, **fields}
        sequence += 1
        return f"event: {kind}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"

    try:
        yield event("response.created", response=response)
        yield event("response.in_progress", response=response)
        async for data in _sse_payloads(upstream):
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise ValueError(str(chunk["error"]))
            if chunk.get("usage"):
                response["usage"] = _usage(chunk["usage"])
            for choice in chunk.get("choices", []):
                if choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta", {})
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                text = delta.get("content") or delta.get("refusal")
                refusal = bool(delta.get("refusal"))
                if text:
                    if message is None:
                        message = {"id": _id("msg_"), "type": "message", "role": "assistant", "status": "in_progress", "content": []}
                        response["output"].append(message)
                        output_index = len(response["output"]) - 1
                        yield event("response.output_item.added", output_index=output_index, item=message)
                        message["content"].append({"type": "refusal", "refusal": ""} if refusal else {"type": "output_text", "text": "", "annotations": []})
                        yield event("response.content_part.added", item_id=message["id"], output_index=output_index,
                                    content_index=0, part=message["content"][0])
                    output_index = response["output"].index(message)
                    part = message["content"][0]
                    if refusal != (part["type"] == "refusal"):
                        raise ValueError("Upstream mixed text and refusal in the same message")
                    part["refusal" if refusal else "text"] += text
                    yield event("response.refusal.delta" if refusal else "response.output_text.delta", item_id=message["id"], output_index=output_index, content_index=0, delta=text)
                for call_delta in delta.get("tool_calls", []):
                    index = call_delta.get("index", 0)
                    state = calls.setdefault(index, {"id": "", "function": {"name": "", "arguments": ""}, "item": None})
                    state["id"] += call_delta.get("id", "")
                    function = call_delta.get("function", {})
                    state["function"]["name"] += function.get("name", "")
                    state["function"]["arguments"] += function.get("arguments", "")
        if finish is None:
            raise ValueError("Upstream stream ended before a finish reason")
        status = "incomplete" if finish in ("length", "content_filter") else "completed"
        # Tool metadata may itself be fragmented, so publish after assembly.
        # This also permits lossless conversion of custom-tool JSON arguments.
        for state in calls.values():
            item = _restore_namespace(_call_item(state, custom, "in_progress"), original_request)
            field = "input" if item["type"] == "custom_tool_call" else "arguments"
            value = item[field]
            item[field] = ""
            output_index = len(response["output"])
            response["output"].append(item)
            yield event("response.output_item.added", output_index=output_index, item=item)
            kind = "custom_tool_call_input" if field == "input" else "function_call_arguments"
            yield event(f"response.{kind}.delta", item_id=item["id"], output_index=output_index, delta=value)
            item[field] = value
            yield event(f"response.{kind}.done", item_id=item["id"], output_index=output_index, **{field: value})
        for index, item in enumerate(response["output"]):
            item["status"] = status
            if item["type"] == "message":
                part = item["content"][0]
                if part["type"] == "refusal":
                    yield event("response.refusal.done", item_id=item["id"], output_index=index, content_index=0, refusal=part["refusal"])
                else:
                    yield event("response.output_text.done", item_id=item["id"], output_index=index, content_index=0, text=part["text"], logprobs=[])
                yield event("response.content_part.done", item_id=item["id"], output_index=index, content_index=0, part=part)
            yield event("response.output_item.done", output_index=index, item=item)
        response["status"] = status
        if status == "incomplete":
            response["incomplete_details"] = {"reason": "max_output_tokens" if finish == "length" else "content_filter"}
        yield event(f"response.{status}", response=response)
    except Exception as exc:
        response.update(status="failed", error={"code": "upstream_error", "message": str(exc)})
        yield event("response.failed", response=response)
    finally:
        close = getattr(upstream, "aclose", None)
        if close is not None:
            await close()
