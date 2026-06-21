"""Tests for OpenAI → Anthropic response conversion (non-streaming).

Covers:
  - text-only response
  - reasoning_content → thinking block (when enabled)
  - <think>...</think> stripped from content
  - <function_calls> XML → tool_use blocks
  - finish_reason → stop_reason mapping (incl. tool_use override)
  - usage field mapping
  - model alias returned (not upstream_model)
"""
from __future__ import annotations

from unified_api.converters.response import convert_response
from unified_api.models import OpenAIChatResponse


def _make_resp(content: str = "", reasoning: str | None = None, finish: str = "stop",
               prompt_tokens: int = 10, completion_tokens: int = 20) -> OpenAIChatResponse:
    message: dict = {"content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return OpenAIChatResponse(
        id="chatcmpl-1",
        model="DeepSeek-V4-Pro",
        choices=[{"index": 0, "message": message, "finish_reason": finish}],
        usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    )


# --- text-only ---


def test_basic_text_response():
    resp = _make_resp(content="Hello there.")
    anth = convert_response(resp, requested_model_alias="claude-test", return_thinking=False)
    assert anth.id.startswith("msg_")
    assert anth.model == "claude-test"  # alias, not upstream
    assert anth.role == "assistant"
    assert anth.type == "message"
    assert len(anth.content) == 1
    assert anth.content[0]["type"] == "text"
    assert anth.content[0]["text"] == "Hello there."
    assert anth.stop_reason == "end_turn"


def test_usage_mapping():
    resp = _make_resp(prompt_tokens=42, completion_tokens=99)
    anth = convert_response(resp, "claude-test", return_thinking=False)
    assert anth.usage.input_tokens == 42
    assert anth.usage.output_tokens == 99


# --- reasoning_content → thinking ---


def test_reasoning_emitted_when_enabled():
    resp = _make_resp(content="answer", reasoning="let me think")
    anth = convert_response(resp, "claude-test", return_thinking=True)
    # Should have a thinking block + a text block
    types = [b["type"] for b in anth.content]
    assert "thinking" in types
    assert "text" in types
    thinking = next(b for b in anth.content if b["type"] == "thinking")
    assert thinking["thinking"] == "let me think"


def test_reasoning_dropped_when_disabled():
    resp = _make_resp(content="answer", reasoning="hidden")
    anth = convert_response(resp, "claude-test", return_thinking=False)
    types = [b["type"] for b in anth.content]
    assert "thinking" not in types
    assert "text" in types


# --- <think> tag handling ---


def test_think_tag_stripped_from_content():
    resp = _make_resp(content="visible<think>hidden reasoning</think>more")
    anth = convert_response(resp, "claude-test", return_thinking=False)
    text_blocks = [b for b in anth.content if b["type"] == "text"]
    combined = "".join(b["text"] for b in text_blocks)
    assert "hidden reasoning" not in combined
    assert "visible" in combined
    assert "more" in combined


def test_think_tag_emitted_as_thinking_when_enabled():
    resp = _make_resp(content="visible<think>hidden reasoning</think>more")
    anth = convert_response(resp, "claude-test", return_thinking=True)
    thinking_blocks = [b for b in anth.content if b["type"] == "thinking"]
    assert any("hidden reasoning" in b["thinking"] for b in thinking_blocks)


# --- <function_calls> XML → tool_use blocks ---


def test_tool_use_block_extracted():
    content = (
        '<function_calls><invoke name="get_weather">'
        '<parameter name="city">Paris</parameter>'
        '</invoke></function_calls>'
    )
    resp = _make_resp(content=content)
    anth = convert_response(resp, "claude-test", return_thinking=False)
    tool_uses = [b for b in anth.content if b["type"] == "tool_use"]
    assert len(tool_uses) == 1
    tu = tool_uses[0]
    assert tu["name"] == "get_weather"
    assert tu["input"] == {"city": "Paris"}
    assert tu["id"].startswith("toolu_")
    assert anth.stop_reason == "tool_use"  # overridden from end_turn


def test_text_plus_tool_use():
    content = (
        "I will check the weather.\n"
        '<function_calls><invoke name="get_weather">'
        '<parameter name="city">Paris</parameter>'
        '</invoke></function_calls>'
    )
    resp = _make_resp(content=content)
    anth = convert_response(resp, "claude-test", return_thinking=False)
    types = [b["type"] for b in anth.content]
    assert "text" in types
    assert "tool_use" in types


def test_multiple_tool_uses_in_one_response():
    content = (
        '<function_calls>'
        '<invoke name="a"><parameter name="x">1</parameter></invoke>'
        '<invoke name="b"><parameter name="y">2</parameter></invoke>'
        '</function_calls>'
    )
    resp = _make_resp(content=content)
    anth = convert_response(resp, "claude-test", return_thinking=False)
    tool_uses = [b for b in anth.content if b["type"] == "tool_use"]
    assert len(tool_uses) == 2
    names = [tu["name"] for tu in tool_uses]
    assert names == ["a", "b"]


# --- finish_reason mapping ---


def test_finish_length_maps_to_max_tokens():
    resp = _make_resp(content="partial", finish="length")
    anth = convert_response(resp, "claude-test", return_thinking=False)
    assert anth.stop_reason == "max_tokens"


def test_finish_tool_calls_maps_to_tool_use():
    resp = _make_resp(content="x", finish="tool_calls")
    anth = convert_response(resp, "claude-test", return_thinking=False)
    assert anth.stop_reason == "tool_use"


def test_finish_content_filter_maps_to_end_turn():
    resp = _make_resp(content="x", finish="content_filter")
    anth = convert_response(resp, "claude-test", return_thinking=False)
    assert anth.stop_reason == "end_turn"


# --- edge cases ---


def test_empty_content_gets_empty_text_block():
    """If the upstream returns empty content, we still emit a text block."""
    resp = _make_resp(content="")
    anth = convert_response(resp, "claude-test", return_thinking=False)
    assert len(anth.content) >= 1
    assert anth.content[0]["type"] == "text"


def test_xml_entities_decoded_in_tool_input():
    content = (
        '<function_calls><invoke name="echo">'
        '<parameter name="msg">5 &lt; 10</parameter>'
        '</invoke></function_calls>'
    )
    resp = _make_resp(content=content)
    anth = convert_response(resp, "claude-test", return_thinking=False)
    tool_use = next(b for b in anth.content if b["type"] == "tool_use")
    assert tool_use["input"]["msg"] == "5 < 10"


def test_extra_fields_in_openai_response_ignored():
    """Upstream adds feeType/consume/object — these must be ignored."""
    resp = OpenAIChatResponse(
        id="x",
        object="chat.completion",
        created="2024-01-01T00:00:00",  # string timestamp (upstream quirk)
        model="DeepSeek-V4-Pro",
        choices=[{"index": 0, "message": {"content": "hi"}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 1, "completion_tokens": 2, "feeType": "token", "consume": 0.001},
    )
    anth = convert_response(resp, "claude-test", return_thinking=False)
    assert anth.usage.input_tokens == 1
    assert anth.usage.output_tokens == 2
    assert anth.content[0]["text"] == "hi"


# --- reasoning field name variant ---


def test_reasoning_field_name_also_works():
    """Some upstreams use 'reasoning' instead of 'reasoning_content'."""
    resp = OpenAIChatResponse(
        id="x",
        model="m",
        choices=[{"index": 0, "message": {"content": "answer", "reasoning": "deep thought"}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 1, "completion_tokens": 2},
    )
    anth = convert_response(resp, "claude-test", return_thinking=True)
    thinking_blocks = [b for b in anth.content if b["type"] == "thinking"]
    assert len(thinking_blocks) >= 1
    assert thinking_blocks[0]["thinking"] == "deep thought"


# --- HKUST non-standard error format handling ---


def test_looks_like_error_with_detail_string():
    """HKUST may return {"detail": "error message"}."""
    from unified_api.upstream.client import _looks_like_error
    assert _looks_like_error({"detail": "Model not found"}) is True


def test_looks_like_error_with_detail_dict():
    from unified_api.upstream.client import _looks_like_error
    assert _looks_like_error({"detail": {"message": "bad request", "code": 400}}) is True


def test_looks_like_error_with_string_error():
    """Some upstreams return {"error": "simple error string"}."""
    from unified_api.upstream.client import _looks_like_error
    assert _looks_like_error({"error": "something went wrong"}) is True


def test_looks_like_error_normal_response_is_false():
    from unified_api.upstream.client import _looks_like_error
    assert _looks_like_error({"choices": [], "id": "x"}) is False


def test_error_from_body_detail_string():
    from unified_api.upstream.client import _error_from_body
    err = _error_from_body({"detail": "Rate limit exceeded"})
    assert "Rate limit exceeded" in str(err)


def test_error_from_body_detail_dict():
    from unified_api.upstream.client import _error_from_body
    err = _error_from_body({"detail": {"message": "Invalid API key", "code": 401}})
    assert err.status_code == 401
    assert "Invalid API key" in str(err)


def test_error_from_body_string_error():
    from unified_api.upstream.client import _error_from_body
    err = _error_from_body({"error": "unauthorized access"})
    assert "unauthorized access" in str(err)


def test_error_from_status_detail_string():
    from unified_api.upstream.client import _error_from_status
    err = _error_from_status(403, {"detail": "Access denied to this model"})
    assert err.status_code == 403
    assert "Access denied" in str(err)


def test_error_from_status_string_error():
    from unified_api.upstream.client import _error_from_status
    err = _error_from_status(500, {"error": "internal server error"})
    assert err.status_code == 500
    assert "internal server error" in str(err)
