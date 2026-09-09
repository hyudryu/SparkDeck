import asyncio
import json

import pytest

from sparkdeck.responses import from_chat_response, stream_chat_response, to_chat_request


def test_request_roundtrips_tools_and_content():
    request = to_chat_request({"model": "local", "instructions": "help", "input": [
        {"role": "developer", "content": [{"type": "input_text", "text": "rules"}]},
        {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,abc"}]},
        {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "result"},
        {"type": "custom_tool_call", "call_id": "c2", "name": "apply_patch", "input": "patch"},
        {"type": "custom_tool_call_output", "call_id": "c2", "output": [{"type": "input_text", "text": "done"}]},
    ], "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}},
                  {"type": "custom", "name": "apply_patch"}],
       "max_output_tokens": 128, "tool_choice": {"type": "custom", "name": "apply_patch"},
       "text": {"format": {"type": "json_schema", "name": "answer", "schema": {"type": "object"}}}})
    assert request["messages"][1]["role"] == "system"
    assert request["messages"][2]["content"][0]["image_url"]["url"].startswith("data:")
    assert json.loads(request["messages"][5]["tool_calls"][0]["function"]["arguments"]) == {"input": "patch"}
    assert request["messages"][6]["content"] == "done"
    assert request["tool_choice"]["function"]["name"] == "apply_patch"
    assert request["max_tokens"] == 128
    assert request["response_format"]["json_schema"]["name"] == "answer"


@pytest.mark.parametrize("extra", [
    {"previous_response_id": "resp_1"}, {"store": True}, {"conversation": "conv_1"},
    {"background": True}, {"tools": [{"type": "web_search"}]},
    {"input": [{"type": "item_reference", "id": "x"}]},
    {"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "x"}]}]},
])
def test_rejects_unsupported_state_before_dispatch(extra):
    with pytest.raises(ValueError):
        to_chat_request({"model": "local", **extra})


def test_nonstream_preserves_tools_usage_and_incomplete():
    result = from_chat_response({"choices": [{"message": {"content": "partial", "tool_calls": [
        {"id": "c1", "function": {"name": "apply_patch", "arguments": '{"input":"patch"}'}}
    ]}, "finish_reason": "length"}], "usage": {"prompt_tokens": 8, "completion_tokens": 3}},
        {"model": "local", "tools": [{"type": "custom", "name": "apply_patch"}]})
    assert result["status"] == "incomplete"
    assert result["incomplete_details"] == {"reason": "max_output_tokens"}
    assert result["output"][1]["type"] == "custom_tool_call"
    assert result["output"][1]["input"] == "patch"
    assert result["usage"]["total_tokens"] == 11


def frame(delta=None, finish=None, **extra):
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}], **extra}) + "\r\n\r\n"


def events(frames, request=None):
    async def run():
        async def upstream():
            for value in frames:
                yield value
        return [json.loads(value.split("data: ", 1)[1]) async for value in stream_chat_response(upstream(), request or {"model": "local"})]
    return asyncio.run(run())


def test_stream_fragmentation_unicode_lifecycle_and_usage():
    raw = (frame({"content": "Hello 🌍"}) + frame(finish="stop") +
           'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3}}\r\n\r\n' + 'data: [DONE]\r\n\r\n').encode()
    result = events([raw[index:index + 1] for index in range(len(raw))])
    assert [event["sequence_number"] for event in result] == list(range(len(result)))
    assert result[0]["type"] == "response.created"
    assert result[-1]["type"] == "response.completed"
    assert result[-1]["response"]["output"][0]["content"][0]["text"] == "Hello 🌍"
    assert result[-1]["response"]["usage"]["total_tokens"] == 5
    assert next(event for event in result if event["type"] == "response.output_item.added")["item"]["content"] == []


def test_tool_metadata_and_arguments_fragments_are_assembled():
    result = events([
        frame({"tool_calls": [{"index": 0, "id": "call_", "function": {"name": "re", "arguments": '{"'}}]}),
        frame({"tool_calls": [{"index": 0, "id": "1", "function": {"name": "ad", "arguments": 'path":"a"}'}}]}),
        frame(finish="tool_calls"), "data: [DONE]\n\n",
    ])
    item = result[-1]["response"]["output"][0]
    assert item["name"] == "read"
    assert item["call_id"] == "call_1"
    assert json.loads(item["arguments"]) == {"path": "a"}
    assert any(event["type"] == "response.function_call_arguments.delta" for event in result)


def test_custom_stream_roundtrip():
    result = events([frame({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "apply_patch", "arguments": '{"input":"patch"}'}}]}),
                     frame(finish="tool_calls")], {"model": "local", "tools": [{"type": "custom", "name": "apply_patch"}]})
    assert result[-1]["response"]["output"][0]["input"] == "patch"
    assert any(event["type"] == "response.custom_tool_call_input.done" for event in result)


@pytest.mark.parametrize("frames", [[frame({"content": "partial"})], ['data: {"error":{"message":"bad"}}\n\n']])
def test_stream_errors_emit_failed(frames):
    assert events(frames)[-1]["type"] == "response.failed"


def test_stream_length_is_incomplete():
    assert events([frame({"content": "partial"}, finish="length")])[-1]["type"] == "response.incomplete"


def test_closing_response_closes_upstream():
    async def run():
        closed = []
        async def upstream():
            try:
                yield frame({"content": "a"})
                await asyncio.Event().wait()
            finally:
                closed.append(True)
        stream = stream_chat_response(upstream(), {})
        async for event in stream:
            if "response.output_text.delta" in event:
                break
        await stream.aclose()
        assert closed == [True]
    asyncio.run(run())


def test_namespaced_tools_restore_identity_and_full_history():
    original = {"model": "local", "include": ["reasoning.encrypted_content"],
                "tools": [{"type": "namespace", "name": "functions", "tools": [
                    {"type": "function", "name": "read", "parameters": {"type": "object"}}]}],
                "input": [{"type": "function_call", "namespace": "functions", "name": "read",
                           "call_id": "call_1", "arguments": "{}"}]}
    converted = to_chat_request(original)
    assert converted["tools"][0]["function"]["name"] == "functions__read"
    assert converted["messages"][0]["tool_calls"][0]["function"]["name"] == "functions__read"
    response = from_chat_response({"choices": [{"message": {"tool_calls": [{"id": "call_1",
        "function": {"name": "functions__read", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]}, original)
    assert response["output"][0]["name"] == "read"
    assert response["output"][0]["namespace"] == "functions"


@pytest.mark.parametrize("body", [[], {}, {"model": ""}, {"model": "local", "stream": "false"},
    {"model": "local", "input": [None]}, {"model": "local", "tools": [None]},
    {"model": "local", "input": [{"role": "user", "content": [{"type": "input_text"}]}]}])
def test_malformed_request_is_a_validation_error(body):
    with pytest.raises(ValueError):
        to_chat_request(body)


@pytest.mark.parametrize("payload", [None, [], {"choices": [None]}, {"choices": [{"message": None}]}])
def test_malformed_upstream_is_a_validation_error(payload):
    with pytest.raises(ValueError):
        from_chat_response(payload, {"model": "local"})


def test_namespace_wire_name_collision_is_rejected():
    with pytest.raises(ValueError, match="unique"):
        to_chat_request({"model": "local", "tools": [
            {"type": "function", "name": "functions__read"},
            {"type": "namespace", "name": "functions", "tools": [{"type": "function", "name": "read"}]}]})


def test_refusal_and_null_usage_details():
    response = from_chat_response({"choices": [{"finish_reason": "stop", "message": {"content": None, "refusal": "declined"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "prompt_tokens_details": None, "completion_tokens_details": None}}, {"model": "local"})
    assert response["output"][0]["content"] == [{"type": "refusal", "refusal": "declined"}]
    assert response["usage"]["input_tokens_details"]["cached_tokens"] == 0
    streamed = events([frame({"refusal": "declined"}), frame(finish="stop")])
    assert any(event["type"] == "response.refusal.done" for event in streamed)
    assert streamed[-1]["response"]["output"][0]["content"][0]["refusal"] == "declined"


def test_nonstream_missing_finish_is_not_reported_as_success():
    with pytest.raises(ValueError, match="finish reason"):
        from_chat_response({"choices": [{"message": {"content": "partial"}}]}, {"model": "local"})
