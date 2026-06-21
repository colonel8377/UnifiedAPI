"""Tests for the incremental `<function_calls>` XML parser.

Covers:
  - static parse_complete (one-shot)
  - incremental scanner with chunk-boundary slicing
  - multi-invoke parallel blocks
  - XML entity decoding
  - unclosed / malformed blocks (don't lose model output)
"""
from __future__ import annotations

import pytest

from unified_api.tools.xml_parser import (
    IncrementalXmlScanner,
    TextSegment,
    ToolUseSegment,
    parse_complete,
)


# --- parse_complete (one-shot) ---


def test_parse_complete_empty():
    assert parse_complete("") == []


def test_parse_complete_text_only():
    segs = parse_complete("hello world")
    assert segs == [TextSegment(text="hello world")]


def test_parse_complete_single_tool_use():
    text = (
        '<function_calls>\n'
        '<invoke name="get_weather">\n'
        '<parameter name="city">Paris</parameter>\n'
        '</invoke>\n'
        '</function_calls>'
    )
    segs = parse_complete(text)
    assert len(segs) == 1
    assert isinstance(segs[0], ToolUseSegment)
    assert segs[0].name == "get_weather"
    assert segs[0].params == {"city": "Paris"}


def test_parse_complete_text_plus_tool_use():
    text = (
        'I will check the weather.\n'
        '<function_calls>\n'
        '<invoke name="get_weather">\n'
        '<parameter name="city">Paris</parameter>\n'
        '</invoke>\n'
        '</function_calls>'
    )
    segs = parse_complete(text)
    assert len(segs) == 2
    assert isinstance(segs[0], TextSegment)
    assert "I will check the weather" in segs[0].text
    assert isinstance(segs[1], ToolUseSegment)
    assert segs[1].name == "get_weather"


def test_parse_complete_multi_params():
    text = (
        '<function_calls>\n'
        '<invoke name="search">\n'
        '<parameter name="query">rust async</parameter>\n'
        '<parameter name="limit">5</parameter>\n'
        '<parameter name="safe">true</parameter>\n'
        '</invoke>\n'
        '</function_calls>'
    )
    segs = parse_complete(text)
    assert len(segs) == 1
    inv = segs[0]
    assert isinstance(inv, ToolUseSegment)
    assert inv.params == {"query": "rust async", "limit": "5", "safe": "true"}


def test_parse_complete_parallel_invokes():
    text = (
        '<function_calls>\n'
        '<invoke name="get_weather">\n'
        '<parameter name="city">Paris</parameter>\n'
        '</invoke>\n'
        '<invoke name="get_time">\n'
        '<parameter name="zone">CET</parameter>\n'
        '</invoke>\n'
        '</function_calls>'
    )
    segs = parse_complete(text)
    assert len(segs) == 2
    assert all(isinstance(s, ToolUseSegment) for s in segs)
    assert segs[0].name == "get_weather"
    assert segs[1].name == "get_time"


def test_parse_complete_xml_entities_decoded():
    text = (
        '<function_calls>\n'
        '<invoke name="echo">\n'
        '<parameter name="msg">a &lt; b &amp; c &gt; d &quot; e &apos; f</parameter>\n'
        '</invoke>\n'
        '</function_calls>'
    )
    segs = parse_complete(text)
    assert isinstance(segs[0], ToolUseSegment)
    assert segs[0].params["msg"] == 'a < b & c > d " e \' f'


def test_parse_complete_multiple_blocks_in_series():
    text = (
        "first answer\n"
        '<function_calls><invoke name="a"><parameter name="x">1</parameter></invoke></function_calls>'
        " in between "
        '<function_calls><invoke name="b"><parameter name="y">2</parameter></invoke></function_calls>'
        " last"
    )
    segs = parse_complete(text)
    # TextSegment, ToolUse, TextSegment, ToolUse, TextSegment
    assert len(segs) == 5
    assert isinstance(segs[0], TextSegment)
    assert isinstance(segs[1], ToolUseSegment) and segs[1].name == "a"
    assert isinstance(segs[2], TextSegment) and "in between" in segs[2].text
    assert isinstance(segs[3], ToolUseSegment) and segs[3].name == "b"
    assert isinstance(segs[4], TextSegment) and segs[4].text == " last"


def test_parse_complete_unclosed_block_preserved_as_text():
    """If the model never closes <function_calls>, we must not lose the output."""
    text = '<function_calls><invoke name="x"><parameter name="y">z'
    segs = parse_complete(text)
    assert len(segs) == 1
    assert isinstance(segs[0], TextSegment)
    # Should contain the original content so the user sees what happened
    assert "<function_calls>" in segs[0].text


def test_parse_complete_case_insensitive_tags():
    text = (
        '<FUNCTION_CALLS>'
        '<INVOKE name="foo">'
        '<PARAMETER name="bar">baz</PARAMETER>'
        '</INVOKE>'
        '</FUNCTION_CALLS>'
    )
    segs = parse_complete(text)
    assert len(segs) == 1
    assert isinstance(segs[0], ToolUseSegment)
    assert segs[0].name == "foo"
    assert segs[0].params == {"bar": "baz"}


def test_parse_complete_empty_param_value():
    text = (
        '<function_calls>'
        '<invoke name="do"><parameter name="x"></parameter></invoke>'
        '</function_calls>'
    )
    segs = parse_complete(text)
    assert isinstance(segs[0], ToolUseSegment)
    assert segs[0].params == {"x": ""}


# --- IncrementalXmlScanner (streaming) ---


def _drain_feed(scanner: IncrementalXmlScanner, chunks: list[str]) -> list:
    out: list = []
    for c in chunks:
        out.extend(scanner.feed(c))
    out.extend(scanner.flush())
    return out


def test_scanner_plain_text_no_xml():
    scanner = IncrementalXmlScanner()
    segs = _drain_feed(scanner, ["hello ", "world"])
    assert all(isinstance(s, TextSegment) for s in segs)
    assert "".join(s.text for s in segs) == "hello world"


def test_scanner_single_complete_block():
    scanner = IncrementalXmlScanner()
    text = (
        '<function_calls><invoke name="f"><parameter name="x">1</parameter>'
        '</invoke></function_calls>'
    )
    segs = _drain_feed(scanner, [text])
    assert len(segs) == 1
    assert isinstance(segs[0], ToolUseSegment)
    assert segs[0].name == "f"
    assert segs[0].params == {"x": "1"}


def test_scanner_text_then_block_then_text():
    scanner = IncrementalXmlScanner()
    chunks = [
        "Hello!\n",
        "<function_calls>",
        '<invoke name="get_weather"><parameter name="city">Tokyo</parameter></invoke>',
        "</function_calls>",
        " Done.",
    ]
    segs = _drain_feed(scanner, chunks)
    # Expect: TextSegment, ToolUse, TextSegment
    types = [type(s).__name__ for s in segs]
    assert "ToolUseSegment" in types
    tool_segs = [s for s in segs if isinstance(s, ToolUseSegment)]
    assert len(tool_segs) == 1
    assert tool_segs[0].name == "get_weather"
    assert tool_segs[0].params == {"city": "Tokyo"}
    text_segs = [s for s in segs if isinstance(s, TextSegment)]
    full_text = "".join(s.text for s in text_segs)
    assert "Hello!" in full_text
    assert "Done." in full_text


def test_scanner_open_tag_split_across_chunks():
    """Critical: `<function_calls>` tag split like `<function_c` + `alls>`."""
    scanner = IncrementalXmlScanner()
    chunks = [
        "Answer: ",
        "<function_c",
        "alls>",
        '<invoke name="x"><parameter name="y">z</parameter></invoke>',
        "</function_calls>",
    ]
    segs = _drain_feed(scanner, chunks)
    tool_segs = [s for s in segs if isinstance(s, ToolUseSegment)]
    assert len(tool_segs) == 1
    assert tool_segs[0].name == "x"
    assert tool_segs[0].params == {"y": "z"}
    text_segs = [s for s in segs if isinstance(s, TextSegment)]
    # Text may be emitted progressively across multiple TextSegments because
    # the scanner buffers near chunk boundaries. Concatenate then check.
    full_text = "".join(s.text for s in text_segs)
    assert "Answer:" in full_text


def test_scanner_close_tag_split_across_chunks():
    scanner = IncrementalXmlScanner()
    chunks = [
        "<function_calls>",
        '<invoke name="x"><parameter name="y">z</parameter></invoke>',
        "</function_c",
        "alls>",
    ]
    segs = _drain_feed(scanner, chunks)
    tool_segs = [s for s in segs if isinstance(s, ToolUseSegment)]
    assert len(tool_segs) == 1
    assert tool_segs[0].params == {"y": "z"}


def test_scanner_param_value_split_across_chunks():
    scanner = IncrementalXmlScanner()
    chunks = [
        "<function_calls>",
        '<invoke name="emit"><parameter name="msg">Hel',
        "lo World</parameter></invoke>",
        "</function_calls>",
    ]
    segs = _drain_feed(scanner, chunks)
    tool_segs = [s for s in segs if isinstance(s, ToolUseSegment)]
    assert len(tool_segs) == 1
    assert tool_segs[0].params == {"msg": "Hello World"}


def test_scanner_invoke_name_split_across_chunks():
    scanner = IncrementalXmlScanner()
    chunks = [
        "<function_calls>",
        '<invoke name="get_we',
        'ather"><parameter name="city">NYC</parameter></invoke>',
        "</function_calls>",
    ]
    segs = _drain_feed(scanner, chunks)
    tool_segs = [s for s in segs if isinstance(s, ToolUseSegment)]
    assert len(tool_segs) == 1
    assert tool_segs[0].name == "get_weather"


def test_scanner_multiple_invokes_one_block():
    scanner = IncrementalXmlScanner()
    chunks = [
        "<function_calls>",
        '<invoke name="a"><parameter name="x">1</parameter></invoke>',
        '<invoke name="b"><parameter name="y">2</parameter></invoke>',
        "</function_calls>",
    ]
    segs = _drain_feed(scanner, chunks)
    tool_segs = [s for s in segs if isinstance(s, ToolUseSegment)]
    assert len(tool_segs) == 2
    assert tool_segs[0].name == "a"
    assert tool_segs[1].name == "b"


def test_scanner_parallel_blocks_split():
    scanner = IncrementalXmlScanner()
    chunks = [
        "<function_calls><invoke ",
        'name="a"><parameter name="x">1</parameter></invoke>',
        '<invoke name="b"><parameter name="y">2</parameter></invoke></function_calls>',
    ]
    segs = _drain_feed(scanner, chunks)
    tool_segs = [s for s in segs if isinstance(s, ToolUseSegment)]
    assert len(tool_segs) == 2


def test_scanner_xml_entities():
    scanner = IncrementalXmlScanner()
    text = (
        '<function_calls><invoke name="echo">'
        '<parameter name="msg">5 &lt; 10 &amp;&amp; 10 &gt; 5</parameter>'
        '</invoke></function_calls>'
    )
    segs = _drain_feed(scanner, [text])
    assert len(segs) == 1
    assert isinstance(segs[0], ToolUseSegment)
    assert segs[0].params["msg"] == "5 < 10 && 10 > 5"


def test_scanner_unclosed_at_flush_preserved():
    """If stream ends mid-block, we either parse what we have or emit as text."""
    scanner = IncrementalXmlScanner()
    chunks = [
        "<function_calls>",
        '<invoke name="incomplete"><parameter name="x">partial',
    ]
    segs = _drain_feed(scanner, chunks)
    # Either parsed as a tool_use OR preserved as text — both acceptable.
    # The critical invariant: no data is silently lost.
    if any(isinstance(s, ToolUseSegment) for s in segs):
        tool = next(s for s in segs if isinstance(s, ToolUseSegment))
        assert tool.name == "incomplete"
    else:
        text = "".join(s.text for s in segs if isinstance(s, TextSegment))
        assert "partial" in text


def test_scanner_empty_chunks_ignored():
    scanner = IncrementalXmlScanner()
    segs = []
    segs.extend(scanner.feed(""))
    segs.extend(scanner.feed(""))
    segs.extend(scanner.flush())
    assert segs == []


def test_scanner_progressive_text_emit_before_block():
    """Text should stream out incrementally, not buffer until the XML appears."""
    scanner = IncrementalXmlScanner()
    # Long text before any XML — should emit in chunks (modulo small buffer)
    long_text = "a" * 100
    segs = list(scanner.feed(long_text))
    # At least some text should emit (not all held back)
    emitted = "".join(s.text for s in segs if isinstance(s, TextSegment))
    assert len(emitted) > 50
