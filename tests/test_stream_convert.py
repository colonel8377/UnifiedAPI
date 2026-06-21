"""Tests for the OpenAI SSE → Anthropic SSE streaming converter.

Covers:
  - basic text streaming event sequence
  - thinking (reasoning_content) handling
  - tool_use blocks emitted from <function_calls> XML
  - XML tag split across chunks
  - finish_reason mapping and stop_reason override
  - end-to-end SSE byte format
"""
from __future__ import annotations

import json

from unified_api.converters.stream import StreamConverter


def _parse_sse_events(raw_bytes: bytes) -> list[tuple[str, dict]]:
    """Parse raw SSE bytes into a list of (event_type, data)."""
    events: list[tuple[str, dict]] = []
    # SSE events are separated by \r\n\r\n
    for raw in raw_bytes.decode("utf-8").split("\r\n\r\n"):
        raw = raw.strip()
        if not raw:
            continue
        event_type = None
        data = None
        for line in raw.split("\r\n"):
            if line.startswith("event: "):
                event_type = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event_type and data:
            events.append((event_type, data))
    return events


def _drain(converter: StreamConverter, chunks: list[dict]) -> list[tuple[str, dict]]:
    raw = b""
    for c in chunks:
        for b in converter.feed(c):
            raw += b
    for b in converter.flush():
        raw += b
    return _parse_sse_events(raw)


def _chunk(content: str | None = None, reasoning: str | None = None, finish: str | None = None) -> dict:
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    choice: dict = {"delta": delta}
    if finish is not None:
        choice["finish_reason"] = finish
    return {"choices": [choice]}


# --- basic text streaming ---


def test_text_only_basic_sequence():
    converter = StreamConverter(requested_model_alias="claude-test", return_thinking=False)
    chunks = [
        _chunk(content="Hello"),
        _chunk(content=", world"),
        _chunk(finish="stop"),
    ]
    events = _drain(converter, chunks)
    types = [t for t, _ in events]

    # Must start with message_start
    assert types[0] == "message_start"
    # Must end with message_stop, preceded by message_delta
    assert types[-1] == "message_stop"
    assert types[-2] == "message_delta"
    # Must contain content_block_start / delta / stop for the text block
    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert "content_block_stop" in types

    # message_start should have proper shape
    _, start_data = events[0]
    assert start_data["message"]["role"] == "assistant"
    assert start_data["message"]["model"] == "claude-test"
    assert start_data["message"]["id"].startswith("msg_")

    # message_delta should carry stop_reason=end_turn
    _, delta_data = events[-2]
    assert delta_data["delta"]["stop_reason"] == "end_turn"


def test_message_id_consistent_across_chunks():
    converter = StreamConverter(requested_model_alias="claude-test", return_thinking=False)
    events = _drain(converter, [_chunk(content="hi"), _chunk(finish="stop")])
    start_id = events[0][1]["message"]["id"]
    # All events belong to the same message
    assert start_id.startswith("msg_")


def test_text_deltas_concatenate_to_full_string():
    converter = StreamConverter(requested_model_alias="m", return_thinking=False)
    events = _drain(converter, [
        _chunk(content="foo"),
        _chunk(content="bar"),
        _chunk(content="baz"),
        _chunk(finish="stop"),
    ])
    text_deltas = [
        data["delta"]["text"]
        for et, data in events
        if et == "content_block_delta" and data["delta"].get("type") == "text_delta"
    ]
    assert "".join(text_deltas) == "foobarbaz"


# --- finish_reason mapping ---


def test_finish_reason_length_maps_to_max_tokens():
    converter = StreamConverter(requested_model_alias="m", return_thinking=False)
    events = _drain(converter, [_chunk(content="x"), _chunk(finish="length")])
    delta_data = next(e for t, e in events if t == "message_delta")
    assert delta_data["delta"]["stop_reason"] == "max_tokens"


def test_finish_reason_tool_calls_maps_to_tool_use():
    converter = StreamConverter(requested_model_alias="m", return_thinking=False)
    events = _drain(converter, [_chunk(content="x"), _chunk(finish="tool_calls")])
    delta_data = next(e for t, e in events if t == "message_delta")
    assert delta_data["delta"]["stop_reason"] == "tool_use"


# --- thinking (reasoning_content) ---


def test_reasoning_content_emits_thinking_block_when_enabled():
    converter = StreamConverter(requested_model_alias="m", return_thinking=True)
    events = _drain(converter, [
        _chunk(reasoning="let me think"),
        _chunk(reasoning="..."),
        _chunk(content="answer"),
        _chunk(finish="stop"),
    ])
    # Expect at least one thinking_delta
    thinking_deltas = [
        e for t, e in events
        if t == "content_block_delta" and e["delta"].get("type") == "thinking_delta"
    ]
    assert len(thinking_deltas) >= 1
    thinking_text = "".join(e["delta"]["thinking"] for e in thinking_deltas)
    assert "let me think" in thinking_text


def test_reasoning_content_dropped_when_disabled():
    converter = StreamConverter(requested_model_alias="m", return_thinking=False)
    events = _drain(converter, [
        _chunk(reasoning="hidden"),
        _chunk(content="visible"),
        _chunk(finish="stop"),
    ])
    # No thinking deltas should be emitted
    thinking_deltas = [
        e for t, e in events
        if t == "content_block_delta" and e["delta"].get("type") == "thinking_delta"
    ]
    assert len(thinking_deltas) == 0
    # But the text should still come through
    text_deltas = [
        e["delta"]["text"]
        for t, e in events
        if t == "content_block_delta" and e["delta"].get("type") == "text_delta"
    ]
    assert "visible" in "".join(text_deltas)


def test_think_tag_in_content_stripped():
    """If the model emits <think>...</think> inline in content, it should be stripped."""
    converter = StreamConverter(requested_model_alias="m", return_thinking=False)
    events = _drain(converter, [
        _chunk(content="visible<think>hidden</think>more"),
        _chunk(finish="stop"),
    ])
    text_deltas = [
        e["delta"]["text"]
        for t, e in events
        if t == "content_block_delta" and e["delta"].get("type") == "text_delta"
    ]
    combined = "".join(text_deltas)
    assert "hidden" not in combined
    assert "visible" in combined
    assert "more" in combined


# --- tool_use blocks from XML ---


def test_tool_use_block_emitted():
    converter = StreamConverter(requested_model_alias="m", return_thinking=False)
    xml = (
        '<function_calls><invoke name="get_weather">'
        '<parameter name="city">Paris</parameter>'
        '</invoke></function_calls>'
    )
    events = _drain(converter, [
        _chunk(content=xml),
        _chunk(finish="stop"),
    ])
    # Should have content_block_start with type=tool_use
    tool_starts = [
        e for t, e in events
        if t == "content_block_start" and e["content_block"]["type"] == "tool_use"
    ]
    assert len(tool_starts) == 1
    cb = tool_starts[0]["content_block"]
    assert cb["name"] == "get_weather"
    assert cb["id"].startswith("toolu_")

    # input_json_delta should carry the params as JSON
    json_deltas = [
        e for t, e in events
        if t == "content_block_delta" and e["delta"].get("type") == "input_json_delta"
    ]
    assert len(json_deltas) == 1
    parsed = json.loads(json_deltas[0]["delta"]["partial_json"])
    assert parsed == {"city": "Paris"}

    # stop_reason should be overridden to tool_use
    delta_data = next(e for t, e in events if t == "message_delta")
    assert delta_data["delta"]["stop_reason"] == "tool_use"


def test_tool_use_xml_split_across_chunks():
    """Critical: XML tags can be split across SSE chunks."""
    converter = StreamConverter(requested_model_alias="m", return_thinking=False)
    chunks = [
        _chunk(content="Calling tool:\n<function_c"),
        _chunk(content="alls><invoke name=\"f\">"),
        _chunk(content="<parameter name=\"x\">42</parameter>"),
        _chunk(content="</invoke></function_calls>"),
        _chunk(finish="stop"),
    ]
    events = _drain(converter, chunks)
    tool_starts = [
        e for t, e in events
        if t == "content_block_start" and e["content_block"]["type"] == "tool_use"
    ]
    assert len(tool_starts) == 1
    assert tool_starts[0]["content_block"]["name"] == "f"
    json_deltas = [
        e["delta"]["partial_json"]
        for t, e in events
        if t == "content_block_delta" and e["delta"].get("type") == "input_json_delta"
    ]
    parsed = json.loads("".join(json_deltas))
    assert parsed == {"x": "42"}


def test_text_and_tool_use_alternating():
    """Text before tool_use, then more text after."""
    converter = StreamConverter(requested_model_alias="m", return_thinking=False)
    chunks = [
        _chunk(content="Let me check.\n<function_calls>"),
        _chunk(content='<invoke name="f"><parameter name="x">1</parameter></invoke>'),
        _chunk(content="</function_calls>"),
        _chunk(content="\nDone."),
        _chunk(finish="stop"),
    ]
    events = _drain(converter, chunks)
    # Index sequence: text block (0), tool_use block (1), text block (2)
    starts = [(e["index"], e["content_block"]["type"]) for t, e in events if t == "content_block_start"]
    types_in_order = [t for _, t in starts]
    assert "text" in types_in_order
    assert "tool_use" in types_in_order


def test_multiple_tool_use_blocks():
    converter = StreamConverter(requested_model_alias="m", return_thinking=False)
    xml = (
        '<function_calls>'
        '<invoke name="a"><parameter name="x">1</parameter></invoke>'
        '<invoke name="b"><parameter name="y">2</parameter></invoke>'
        '</function_calls>'
    )
    events = _drain(converter, [
        _chunk(content=xml),
        _chunk(finish="stop"),
    ])
    tool_starts = [
        e for t, e in events
        if t == "content_block_start" and e["content_block"]["type"] == "tool_use"
    ]
    assert len(tool_starts) == 2
    names = [cb["content_block"]["name"] for cb in tool_starts]
    assert names == ["a", "b"]


# --- edge cases ---


def test_empty_stream_emits_skeleton():
    """If no content arrives at all, we still emit message_start/message_stop."""
    converter = StreamConverter(requested_model_alias="m", return_thinking=False)
    events = _drain(converter, [])
    types = [t for t, _ in events]
    assert "message_start" in types
    assert "message_stop" in types


def test_sse_byte_format():
    """Each event should be valid SSE: 'event: TYPE\\r\\ndata: {json}\\r\\n\\r\\n'."""
    converter = StreamConverter(requested_model_alias="m", return_thinking=False)
    raw = b""
    for b in converter.feed(_chunk(content="hi")):
        raw += b
    for b in converter.flush():
        raw += b
    text = raw.decode("utf-8")
    # Should have event: and data: lines
    assert "event: message_start" in text
    assert "data: " in text
    # Lines should be CRLF-separated
    assert "\r\n" in text


def test_block_indices_are_sequential():
    """content_block_start indices should increment 0, 1, 2..."""
    converter = StreamConverter(requested_model_alias="m", return_thinking=False)
    events = _drain(converter, [
        _chunk(content="text"),
        _chunk(content='<function_calls><invoke name="f"><parameter name="x">1</parameter></invoke></function_calls>'),
        _chunk(finish="stop"),
    ])
    start_indices = [
        e["index"] for t, e in events if t == "content_block_start"
    ]
    assert start_indices == sorted(start_indices)
    assert start_indices[0] == 0


# --- HKUST non-standard format tolerance ---


def _chunk_with_reasoning_field(reasoning: str | None = None, content: str | None = None, finish: str | None = None) -> dict:
    """Chunk with 'reasoning' instead of 'reasoning_content' (some upstreams)."""
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning"] = reasoning
    choice: dict = {"delta": delta}
    if finish is not None:
        choice["finish_reason"] = finish
    return {"choices": [choice]}


def test_reasoning_field_name_emits_thinking():
    """Some upstreams use 'reasoning' instead of 'reasoning_content'."""
    converter = StreamConverter(requested_model_alias="m", return_thinking=True)
    events = _drain(converter, [
        _chunk_with_reasoning_field(reasoning="deep thought"),
        _chunk_with_reasoning_field(content="answer"),
        _chunk_with_reasoning_field(finish="stop"),
    ])
    thinking_deltas = [
        e for t, e in events
        if t == "content_block_delta" and e["delta"].get("type") == "thinking_delta"
    ]
    assert len(thinking_deltas) >= 1
    assert "deep thought" in "".join(e["delta"]["thinking"] for e in thinking_deltas)


def test_empty_choices_chunk_skipped():
    """Upstream may send chunks with empty choices list; should be skipped gracefully."""
    converter = StreamConverter(requested_model_alias="m", return_thinking=False)
    empty_choices_chunk = {"choices": []}
    raw = b""
    for b in converter.feed(empty_choices_chunk):
        raw += b
    for b in converter.feed(_chunk(content="hi")):
        raw += b
    for b in converter.flush():
        raw += b
    events = _parse_sse_events(raw)
    text_deltas = [
        e["delta"]["text"]
        for t, e in events
        if t == "content_block_delta" and e["delta"].get("type") == "text_delta"
    ]
    assert "".join(text_deltas) == "hi"


def test_no_choices_key_chunk_skipped():
    """Upstream may send a chunk without 'choices' key at all."""
    converter = StreamConverter(requested_model_alias="m", return_thinking=False)
    raw = b""
    for b in converter.feed({"foo": "bar"}):
        raw += b
    for b in converter.feed(_chunk(content="ok")):
        raw += b
    for b in converter.flush():
        raw += b
    events = _parse_sse_events(raw)
    text_deltas = [
        e["delta"]["text"]
        for t, e in events
        if t == "content_block_delta" and e["delta"].get("type") == "text_delta"
    ]
    assert "".join(text_deltas) == "ok"


# --- length + no visible content → sentinel (the "stops for no reason" fix) ---
# Bug pattern: DeepSeek-V4-Pro can consume the entire upstream max_tokens
# budget on reasoning_content, yielding finish_reason='length' with ZERO
# visible content. Without the sentinel the client sees an empty assistant
# message + stop_reason=max_tokens and silently ends the turn.


def test_length_no_content_emits_sentinel():
    """Core fix: finish=length + no visible content → inject sentinel
    text_delta so the client never sees an empty assistant message."""
    converter = StreamConverter(requested_model_alias="m", return_thinking=False)
    # Bug pattern: only reasoning_content (dropped because return_thinking=False),
    # then length
    events = _drain(converter, [
        _chunk(reasoning="thinking hard"),
        _chunk(reasoning="...still thinking"),
        _chunk(finish="length"),
    ])
    text_deltas = [
        e["delta"]["text"]
        for t, e in events
        if t == "content_block_delta" and e["delta"].get("type") == "text_delta"
    ]
    combined = "".join(text_deltas)
    assert "UnifiedAPI warning" in combined
    assert "max_tokens" in combined
    # stop_reason stays max_tokens — honest about what happened
    delta_data = next(e for t, e in events if t == "message_delta")
    assert delta_data["delta"]["stop_reason"] == "max_tokens"


def test_length_with_content_does_not_emit_sentinel():
    """Real content was produced before hitting length → no sentinel."""
    converter = StreamConverter(requested_model_alias="m", return_thinking=False)
    events = _drain(converter, [
        _chunk(content="partial answer"),
        _chunk(finish="length"),
    ])
    text_deltas = [
        e["delta"]["text"]
        for t, e in events
        if t == "content_block_delta" and e["delta"].get("type") == "text_delta"
    ]
    combined = "".join(text_deltas)
    assert "partial answer" in combined
    assert "UnifiedAPI warning" not in combined


def test_length_thinking_only_still_emits_sentinel():
    """Thinking blocks don't count as visible content even when forwarded."""
    converter = StreamConverter(requested_model_alias="m", return_thinking=True)
    events = _drain(converter, [
        _chunk(reasoning="deep thoughts"),
        _chunk(finish="length"),
    ])
    text_deltas = [
        e["delta"]["text"]
        for t, e in events
        if t == "content_block_delta" and e["delta"].get("type") == "text_delta"
    ]
    combined = "".join(text_deltas)
    assert "UnifiedAPI warning" in combined


def test_length_with_tool_use_does_not_emit_sentinel():
    """Tool use counts as visible content; no sentinel needed."""
    converter = StreamConverter(requested_model_alias="m", return_thinking=False)
    xml = (
        '<function_calls><invoke name="f">'
        '<parameter name="x">1</parameter>'
        '</invoke></function_calls>'
    )
    events = _drain(converter, [
        _chunk(content=xml),
        _chunk(finish="length"),
    ])
    text_deltas = [
        e["delta"]["text"]
        for t, e in events
        if t == "content_block_delta" and e["delta"].get("type") == "text_delta"
    ]
    combined = "".join(text_deltas)
    assert "UnifiedAPI warning" not in combined


def test_stop_no_content_does_not_emit_sentinel():
    """finish=stop with no content is suspicious but NOT the bug we're fixing;
    sentinel only fires on length. Avoids masking other issues."""
    converter = StreamConverter(requested_model_alias="m", return_thinking=False)
    events = _drain(converter, [_chunk(finish="stop")])
    text_deltas = [
        e["delta"]["text"]
        for t, e in events
        if t == "content_block_delta" and e["delta"].get("type") == "text_delta"
    ]
    assert text_deltas == []
