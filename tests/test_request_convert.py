"""Tests for Anthropic → OpenAI request conversion.

Covers:
  - basic message shape translation
  - system prompt (string + content-block array)
  - tool injection (tools → system prompt XML spec)
  - tool_choice hints
  - tool_use replay in assistant history
  - tool_result replay in user history
  - image blocks dropped
  - model name pass-through
  - temperature/top_p/stop passthrough
"""
from __future__ import annotations

import pytest

from unified_api.converters.request import convert_request
from unified_api.errors import ConversionError
from unified_api.models import AnthropicMessageRequest

UPSTREAM_MODEL = "DeepSeek-V4-Pro"


def _make_req(**kwargs) -> AnthropicMessageRequest:
    base = {
        "model": UPSTREAM_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
    }
    base.update(kwargs)
    return AnthropicMessageRequest(**base)


# --- basic translation ---


def test_basic_user_message():
    req = _make_req(messages=[{"role": "user", "content": "hello"}])
    oai = convert_request(req, UPSTREAM_MODEL)
    assert oai.model == UPSTREAM_MODEL
    assert len(oai.messages) == 1
    assert oai.messages[0]["role"] == "user"
    assert oai.messages[0]["content"] == "hello"
    assert oai.max_tokens == 612  # 100 + 512 reasoning buffer


def test_model_passthrough():
    """Model name passes through unchanged — no alias mapping."""
    req = _make_req(model="my-custom-model")
    oai = convert_request(req, "my-custom-model")
    assert oai.model == "my-custom-model"


def test_unknown_role_rejected():
    req = _make_req(messages=[{"role": "system", "content": "bad"}])
    with pytest.raises(ConversionError):
        convert_request(req, UPSTREAM_MODEL)


def test_temperature_top_p_passed_through():
    req = _make_req(temperature=0.7, top_p=0.9)
    oai = convert_request(req, UPSTREAM_MODEL)
    assert oai.temperature == 0.7
    assert oai.top_p == 0.9


def test_stop_sequences_passed_to_stop():
    req = _make_req(stop_sequences=["END", "STOP"])
    oai = convert_request(req, UPSTREAM_MODEL)
    assert oai.stop == ["END", "STOP"]


def test_top_k_dropped():
    req = _make_req(top_k=40)
    oai = convert_request(req, UPSTREAM_MODEL)
    # top_k should not appear in OpenAI request
    assert "top_k" not in oai.model_dump()


# --- system prompt ---


def test_system_as_string():
    req = _make_req(system="You are helpful.")
    oai = convert_request(req, UPSTREAM_MODEL)
    assert oai.messages[0]["role"] == "system"
    assert oai.messages[0]["content"] == "You are helpful."


def test_system_as_content_array():
    req = _make_req(system=[{"type": "text", "text": "rule 1"}, {"type": "text", "text": "rule 2"}])
    oai = convert_request(req, UPSTREAM_MODEL)
    assert oai.messages[0]["role"] == "system"
    assert "rule 1" in oai.messages[0]["content"]
    assert "rule 2" in oai.messages[0]["content"]


def test_no_system_no_system_message():
    req = _make_req()
    oai = convert_request(req, UPSTREAM_MODEL)
    assert all(m["role"] != "system" for m in oai.messages)


# --- tools injection ---


def test_tools_injected_into_system_prompt():
    tools = [{
        "name": "get_weather",
        "description": "Get current weather",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        },
    }]
    req = _make_req(system="You are a bot.", tools=tools)
    oai = convert_request(req, UPSTREAM_MODEL)
    sys_msg = oai.messages[0]["content"]
    assert "get_weather" in sys_msg
    assert "<function_calls>" in sys_msg
    assert "<invoke name=" in sys_msg
    assert "city" in sys_msg


def test_tools_passed_to_openai_request():
    """Tools should be passed to the upstream in OpenAI function-tool format."""
    tools = [{
        "name": "f",
        "description": "d",
        "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
    }]
    req = _make_req(tools=tools)
    oai = convert_request(req, UPSTREAM_MODEL)
    dumped = oai.model_dump(exclude_none=True)
    assert "tools" in dumped
    assert dumped["tools"][0]["type"] == "function"
    assert dumped["tools"][0]["function"]["name"] == "f"
    assert dumped["tools"][0]["function"]["parameters"]["properties"]["x"]["type"] == "string"


def test_tool_choice_required_adds_constraint():
    tools = [{
        "name": "f",
        "description": "d",
        "input_schema": {"type": "object", "properties": {}},
    }]
    req = _make_req(tools=tools, tool_choice={"type": "required"})
    oai = convert_request(req, UPSTREAM_MODEL)
    sys_msg = oai.messages[0]["content"]
    assert "MUST" in sys_msg or "must" in sys_msg


def test_tool_choice_specific_tool_named():
    tools = [{"name": "foo", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
    req = _make_req(tools=tools, tool_choice={"type": "tool", "name": "foo"})
    oai = convert_request(req, UPSTREAM_MODEL)
    sys_msg = oai.messages[0]["content"]
    assert "foo" in sys_msg


# --- conversation history replay ---


def test_assistant_tool_use_in_history_replay():
    """Prior assistant turn with tool_use → replayed as XML."""
    req = _make_req(messages=[
        {"role": "user", "content": "what's the weather?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
            ],
        },
    ])
    oai = convert_request(req, UPSTREAM_MODEL)
    # assistant message should be present
    asst_msgs = [m for m in oai.messages if m["role"] == "assistant"]
    assert len(asst_msgs) == 1
    body = asst_msgs[0]["content"]
    assert "Let me check" in body
    assert "<function_calls>" in body
    assert "get_weather" in body
    assert "Paris" in body


def test_user_tool_result_in_history_replay():
    """Prior user turn with tool_result → rendered as <tool_result>."""
    req = _make_req(messages=[
        {"role": "user", "content": "what's the weather?"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [{"type": "text", "text": "sunny, 25C"}],
                },
            ],
        },
    ])
    oai = convert_request(req, UPSTREAM_MODEL)
    user_msgs = [m for m in oai.messages if m["role"] == "user"]
    # The last user message should carry the tool_result rendering
    last_user = user_msgs[-1]["content"]
    assert "tool_result" in last_user
    assert "sunny, 25C" in last_user


def test_image_blocks_dropped():
    req = _make_req(messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image", "source": {"type": "base64", "data": "..."}},
        ],
    }])
    oai = convert_request(req, UPSTREAM_MODEL)
    # Image dropped; only text remains
    user_msg = oai.messages[0]["content"]
    assert "describe this" in user_msg


def test_string_content_passes_through():
    req = _make_req(messages=[{"role": "user", "content": "plain string"}])
    oai = convert_request(req, UPSTREAM_MODEL)
    assert oai.messages[0]["content"] == "plain string"


# --- Anthropic → OpenAI tool schema conversion ---


def test_tool_choice_auto_converts():
    tools = [{"name": "f", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
    req = _make_req(tools=tools, tool_choice={"type": "auto"})
    oai = convert_request(req, UPSTREAM_MODEL)
    dumped = oai.model_dump(exclude_none=True)
    assert dumped["tool_choice"] == "auto"


def test_tool_choice_any_converts_to_required():
    tools = [{"name": "f", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
    req = _make_req(tools=tools, tool_choice={"type": "any"})
    oai = convert_request(req, UPSTREAM_MODEL)
    dumped = oai.model_dump(exclude_none=True)
    assert dumped["tool_choice"] == "required"


def test_tool_choice_specific_tool_converts():
    tools = [{"name": "foo", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
    req = _make_req(tools=tools, tool_choice={"type": "tool", "name": "foo"})
    oai = convert_request(req, UPSTREAM_MODEL)
    dumped = oai.model_dump(exclude_none=True)
    assert dumped["tool_choice"]["type"] == "function"
    assert dumped["tool_choice"]["function"]["name"] == "foo"


def test_multiple_tools_converted():
    tools = [
        {"name": "a", "description": "tool a", "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}},
        {"name": "b", "description": "tool b", "input_schema": {"type": "object", "properties": {"y": {"type": "integer"}}}},
    ]
    req = _make_req(tools=tools)
    oai = convert_request(req, UPSTREAM_MODEL)
    dumped = oai.model_dump(exclude_none=True)
    assert len(dumped["tools"]) == 2
    names = [t["function"]["name"] for t in dumped["tools"]]
    assert names == ["a", "b"]
