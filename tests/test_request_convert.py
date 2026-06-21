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
    # V4-Pro profile: max_tokens=100 + buffer 8192 = 8292, below floor 16384
    # → floor kicks in → 16384.
    assert oai.max_tokens == 16384


def test_reasoning_buffer_scales_above_floor():
    """Above the 16384 floor, buffer scales as max_tokens + 8192."""
    req = _make_req(max_tokens=16384)
    oai = convert_request(req, UPSTREAM_MODEL)
    assert oai.max_tokens == 24576  # 16384 + 8192


def test_max_tokens_capped_by_upstream_context_window():
    """Claude Code sends 32000 max_tokens + ~30k tokens of input (large system
    prompt + many tools). Old formula (max_tokens * 1.5 = 48000) overflowed
    upstream's 65535-token context → empty SSE body. New formula must cap
    below what fits, never above 65535 - input_estimate.
    """
    big_system = "x" * 85_000          # ~28k tokens, simulating Claude Code's
    many_tools = [
        {"name": f"tool_{i}", "description": "d" * 200,
         "input_schema": {"type": "object", "properties": {"p": {"type": "string"}}}}
        for i in range(48)
    ]
    req = _make_req(
        max_tokens=32000,
        system=big_system,
        tools=many_tools,
    )
    oai = convert_request(req, UPSTREAM_MODEL)
    # Must be capped: well below the 40192 that the old formula would produce
    assert oai.max_tokens < 32_000 + 8192
    # Must still be above the 16384 floor (input is large but not so large that
    # even the floor doesn't fit)
    assert oai.max_tokens >= 16384
    # Must fit within upstream context window when combined with input estimate
    # (input_chars / 3 + max_tokens < 65535)
    assert oai.max_tokens < 65535


def test_max_tokens_floor_respected_for_tiny_input():
    """Small request still gets the 16384 floor."""
    req = _make_req(max_tokens=100, messages=[{"role": "user", "content": "hi"}])
    oai = convert_request(req, UPSTREAM_MODEL)
    assert oai.max_tokens == 16384


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


# --- per-model profile selection ---


FLASH_MODEL = "DeepSeek-V4-Flash"


def test_flash_profile_uses_higher_floor_than_pro():
    """Flash's probe-verified failure at max_tokens=8192 forces a higher
    floor (32768) than V4-Pro (16384). Same tiny request → different result."""
    req = _make_req(model=FLASH_MODEL, messages=[{"role": "user", "content": "hi"}])
    oai_pro = convert_request(_make_req(messages=[{"role": "user", "content": "hi"}]),
                              UPSTREAM_MODEL)
    oai_flash = convert_request(req, FLASH_MODEL)
    assert oai_pro.max_tokens == 16384     # Pro floor
    assert oai_flash.max_tokens == 32768   # Flash floor


def test_flash_profile_uses_higher_reasoning_buffer():
    """Flash eats budget more aggressively → larger buffer."""
    # max_tokens=16384 + buffer. Pro: 16384+8192=24576. Flash: 16384+16384=32768.
    req_pro = _make_req(max_tokens=16384, messages=[{"role": "user", "content": "hi"}])
    req_flash = _make_req(model=FLASH_MODEL, max_tokens=16384,
                          messages=[{"role": "user", "content": "hi"}])
    assert convert_request(req_pro, UPSTREAM_MODEL).max_tokens == 24576
    assert convert_request(req_flash, FLASH_MODEL).max_tokens == 32768


def test_unknown_model_falls_back_to_default_profile():
    """Unknown model names get the conservative default (same as V4-Pro)."""
    req = _make_req(model="Some-Future-Model",
                    messages=[{"role": "user", "content": "hi"}])
    oai = convert_request(req, "Some-Future-Model")
    # Default profile: floor=16384, buffer=8192 → max_tokens=100 yields 16384
    assert oai.max_tokens == 16384


def test_flash_context_overflow_still_capped():
    """Flash has no max_tokens_param_cap, but context_budget still applies.
    Large input → max_tokens still gets capped to fit upstream_context."""
    big_system = "x" * 85_000
    many_tools = [
        {"name": f"tool_{i}", "description": "d" * 200,
         "input_schema": {"type": "object", "properties": {"p": {"type": "string"}}}}
        for i in range(48)
    ]
    req = _make_req(model=FLASH_MODEL, max_tokens=32000,
                    system=big_system, tools=many_tools)
    oai = convert_request(req, FLASH_MODEL)
    # Must be capped well below what 32000 + 16384 buffer would give
    assert oai.max_tokens < 32000 + 16384
    # Must still be above Flash's higher floor (32k)
    assert oai.max_tokens >= 32768
    # Must fit in upstream_context
    assert oai.max_tokens < 65535


def test_pro_max_tokens_param_cap_respected():
    """V4-Pro's litellm rejects max_tokens > 65535. Even with tiny input and
    huge client request, output must stay under the cap."""
    # Huge client request + tiny input → desired would be 100000 + 8192 = 108192
    # But V4-Pro profile caps max_tokens param at 65535.
    req = _make_req(max_tokens=100000,
                    messages=[{"role": "user", "content": "hi"}])
    oai = convert_request(req, UPSTREAM_MODEL)
    assert oai.max_tokens <= 65535


def test_flash_no_max_tokens_param_cap():
    """Flash API doesn't validate max_tokens param. With tiny input and huge
    client request, output is bounded only by context_budget, not param cap."""
    req = _make_req(model=FLASH_MODEL, max_tokens=100000,
                    messages=[{"role": "user", "content": "hi"}])
    oai = convert_request(req, FLASH_MODEL)
    # Tiny input → context_budget ≈ 65535 - 0 - 2048 = 63487
    # desired = 100000 + 16384 = 116384, capped by context_budget
    # No param_cap on Flash, so result = max(32768, min(116384, 63487)) = 63487
    assert 60000 < oai.max_tokens <= 65535
    assert oai.max_tokens > 32768   # above floor
