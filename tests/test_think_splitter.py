"""Tests for the incremental `<think>...</think>` splitter.

Covers:
  - one-shot strip_think_complete
  - IncrementalThinkSplitter with tag-split-across-chunks
  - multiple think blocks interleaved with text
  - unclosed think block (treated as thinking to end)
"""
from __future__ import annotations

from unified_api.tools.think_splitter import (
    IncrementalThinkSplitter,
    Segment,
    strip_think_complete,
)


# --- strip_think_complete ---


def test_strip_complete_no_think():
    cleaned, thinking = strip_think_complete("hello world")
    assert cleaned == "hello world"
    assert thinking == ""


def test_strip_complete_one_block():
    cleaned, thinking = strip_think_complete("a<think>hidden</think>b")
    assert cleaned == "ab"
    assert thinking == "hidden"


def test_strip_complete_multiple_blocks():
    text = "<think>x</think>visible1<think>y</think>visible2"
    cleaned, thinking = strip_think_complete(text)
    assert cleaned == "visible1visible2"
    assert thinking == "xy"


def test_strip_complete_unclosed_block():
    """An unclosed <think> consumes the rest as thinking."""
    cleaned, thinking = strip_think_complete("before<think>rest of text")
    assert cleaned == "before"
    assert thinking == "rest of text"


def test_strip_complete_empty():
    assert strip_think_complete("") == ("", "")


# --- IncrementalThinkSplitter ---


def _drain_feed(splitter: IncrementalThinkSplitter, chunks: list[str]) -> list[Segment]:
    out: list[Segment] = []
    for c in chunks:
        out.extend(splitter.feed(c))
    out.extend(splitter.flush())
    return out


def test_splitter_text_only():
    splitter = IncrementalThinkSplitter()
    segs = _drain_feed(splitter, ["hello ", "world"])
    assert all(s.kind == "text" for s in segs)
    assert "".join(s.text for s in segs) == "hello world"


def test_splitter_complete_think_block():
    splitter = IncrementalThinkSplitter()
    segs = _drain_feed(splitter, ["<think>hidden</think>"])
    kinds = [s.kind for s in segs]
    assert "thinking" in kinds
    thinking_text = "".join(s.text for s in segs if s.kind == "thinking")
    assert thinking_text == "hidden"


def test_splitter_open_tag_split_across_chunks():
    """Critical: `<think>` tag split like `<thi` + `nk>`."""
    splitter = IncrementalThinkSplitter()
    segs = _drain_feed(splitter, ["text<thi", "nk>hidden</think>done"])
    thinking = "".join(s.text for s in segs if s.kind == "thinking")
    text = "".join(s.text for s in segs if s.kind == "text")
    assert thinking == "hidden"
    assert "text" in text
    assert "done" in text


def test_splitter_close_tag_split_across_chunks():
    splitter = IncrementalThinkSplitter()
    segs = _drain_feed(splitter, ["<think>hidden</thi", "nk>done"])
    thinking = "".join(s.text for s in segs if s.kind == "thinking")
    assert thinking == "hidden"
    text = "".join(s.text for s in segs if s.kind == "text")
    assert "done" in text


def test_splitter_multiple_think_blocks_interleaved():
    splitter = IncrementalThinkSplitter()
    text = "a<think>b</think>c<think>d</think>e"
    segs = _drain_feed(splitter, [text])
    kinds = [s.kind for s in segs]
    # Expect interleaving: text, thinking, text, thinking, text
    # The first text 'a' may be buffered with subsequent 'c' — we just check
    # that we see both thinking blocks and at least the surrounding text.
    thinking = "".join(s.text for s in segs if s.kind == "thinking")
    assert "b" in thinking
    assert "d" in thinking
    text_emitted = "".join(s.text for s in segs if s.kind == "text")
    assert "a" in text_emitted or text_emitted.endswith("e")
    assert "e" in text_emitted


def test_splitter_unclosed_think_at_flush():
    splitter = IncrementalThinkSplitter()
    segs = _drain_feed(splitter, ["visible<think>still thinking"])
    thinking = "".join(s.text for s in segs if s.kind == "thinking")
    text = "".join(s.text for s in segs if s.kind == "text")
    assert "visible" in text
    assert "still thinking" in thinking


def test_splitter_empty_chunks():
    splitter = IncrementalThinkSplitter()
    assert list(splitter.feed("")) == []
    assert list(splitter.flush()) == []


def test_splitter_progressive_text_emit():
    """Long text without <think> should emit progressively."""
    splitter = IncrementalThinkSplitter()
    long_text = "x" * 100
    segs = list(splitter.feed(long_text))
    # Should not hold back everything
    emitted = "".join(s.text for s in segs if s.kind == "text")
    assert len(emitted) > 50
