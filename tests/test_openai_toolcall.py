"""Tests for OpenAI route tool call processing."""
from __future__ import annotations

from unified_api.models import OpenAIChatRequest
from unified_api.routes.chat import _normalize_oai_messages, _postprocess_tool_calls


def _make_oai_req(messages, tools=None):
    payload = {"model": "test", "messages": messages, "max_tokens": 100}
    if tools:
        payload["tools"] = tools
    return OpenAIChatRequest(**payload)


def test_normalize_no_tool_messages():
    req = _make_oai_req([{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}])
    result = _normalize_oai_messages(req)
    assert len(result.messages) == 2
    assert result.messages[0]["role"] == "system"


def test_normalize_tool_message():
    req = _make_oai_req([
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "calc", "arguments": '{"a": 1}'}}
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "result is 42"},
    ])
    result = _normalize_oai_messages(req)
    tool_msg = result.messages[1]
    assert tool_msg["role"] == "user"
    assert "tool_result" in tool_msg["content"]
    assert "call_1" in tool_msg["content"]


def test_normalize_assistant_tool_calls():
    req = _make_oai_req([
        {"role": "assistant", "content": "Let me calculate.", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "calc", "arguments": '{"a": 1, "b": 2}'}}
        ]},
    ])
    result = _normalize_oai_messages(req)
    msg = result.messages[0]
    assert msg["role"] == "assistant"
    assert "function_calls" in msg["content"]
    assert "tool_calls" not in msg


def test_normalize_multi_turn_tool_exchange():
    req = _make_oai_req([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "1+2?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "calc", "arguments": '{"a":1,"b":2}'}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "3"},
        {"role": "user", "content": "3+4?"},
    ])
    result = _normalize_oai_messages(req)
    assert result.messages[2]["role"] == "assistant"
    assert "function_calls" in result.messages[2]["content"]
    assert "tool_calls" not in result.messages[2]
    assert result.messages[3]["role"] == "user"
    assert "tool_result" in result.messages[3]["content"]


def test_postprocess_no_xml():
    resp = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}]}
    result = _postprocess_tool_calls(resp)
    assert result["choices"][0]["message"]["content"] == "Hello!"
    assert "tool_calls" not in result["choices"][0]["message"]


def test_postprocess_xml_tool_calls():
    xml = """<function_calls>
<invoke name="calc">
<parameter name="a">1</parameter>
<parameter name="b">2</parameter>
</invoke>
</function_calls>"""
    resp = {"choices": [{"index": 0, "message": {"role": "assistant", "content": xml}, "finish_reason": "stop"}]}
    result = _postprocess_tool_calls(resp)
    msg = result["choices"][0]["message"]
    assert "tool_calls" in msg
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "calc"
    assert msg["tool_calls"][0]["type"] == "function"
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["content"] is None or msg["content"] == ""


def test_postprocess_mixed_text_and_tool():
    xml = "Here is the result:\n" + """<function_calls>
<invoke name="calc">
<parameter name="a">1</parameter>
<parameter name="b">2</parameter>
</invoke>
</function_calls>"""
    resp = {"choices": [{"index": 0, "message": {"role": "assistant", "content": xml}, "finish_reason": "stop"}]}
    result = _postprocess_tool_calls(resp)
    msg = result["choices"][0]["message"]
    assert "tool_calls" in msg
    assert msg["content"] == "Here is the result:"
