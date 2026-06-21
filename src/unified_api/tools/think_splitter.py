"""Incremental `<think>...</think>` extractor.

DeepSeek-V4-Pro occasionally wraps its spontaneous reasoning in
`<think>...</think>` tags within the visible `content` stream. We strip
those out so they don't leak as text into Anthropic content blocks.

Two interfaces:
  - `strip_think_complete(text)` — one-shot, for non-streaming responses
  - `IncrementalThinkSplitter` — stateful, for streaming
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

_OPEN = "<think>"
_CLOSE = "</think>"


@dataclass
class Segment:
    """A piece of classified content."""
    kind: str  # "thinking" or "text"
    text: str


def strip_think_complete(text: str) -> tuple[str, str]:
    """Remove all `<think>...</think>` blocks from text.

    Returns (cleaned_text, concatenated_thinking_text).
    Unclosed `<think>` (no matching close) is treated as thinking up to end.
    """
    if not text:
        return "", ""
    cleaned_parts: list[str] = []
    thinking_parts: list[str] = []
    pos = 0
    while True:
        open_idx = text.find(_OPEN, pos)
        if open_idx == -1:
            cleaned_parts.append(text[pos:])
            break
        cleaned_parts.append(text[pos:open_idx])
        close_idx = text.find(_CLOSE, open_idx + len(_OPEN))
        if close_idx == -1:
            # think block never closes — rest is thinking
            thinking_parts.append(text[open_idx + len(_OPEN):])
            break
        thinking_parts.append(text[open_idx + len(_OPEN):close_idx])
        pos = close_idx + len(_CLOSE)
    return "".join(cleaned_parts), "".join(thinking_parts)


class IncrementalThinkSplitter:
    """Stateful stream scanner that classifies content as text or thinking.

    Feed content chunks via `feed()`, get back `Segment` instances. The
    scanner is robust to `<think>` tags being split across chunks.

    Note: tags are emitted lower-cased by the model; we match exactly.
    """

    # We only need to buffer up to len("<think>")-1 characters at chunk
    # boundaries to detect partial opens.
    _MAX_TAG_LEN = len(_CLOSE)  # 8, "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False

    def feed(self, content: str) -> Iterator[Segment]:
        if not content:
            return
        self._buffer += content
        # Process as much as we can; keep at most _MAX_TAG_LEN-1 chars buffered
        # to allow partial-tag detection.
        while True:
            if self._in_think:
                close_idx = self._buffer.find(_CLOSE)
                if close_idx == -1:
                    # Might be mid-close-tag at boundary; emit up to safe point.
                    safe_emit = len(self._buffer) - (self._MAX_TAG_LEN - 1)
                    if safe_emit > 0:
                        yield Segment("thinking", self._buffer[:safe_emit])
                        self._buffer = self._buffer[safe_emit:]
                        return
                    return
                # Emit up to close
                if close_idx > 0:
                    yield Segment("thinking", self._buffer[:close_idx])
                self._buffer = self._buffer[close_idx + len(_CLOSE):]
                self._in_think = False
                continue
            # not in think: look for opening
            open_idx = self._buffer.find(_OPEN)
            if open_idx == -1:
                # No opening yet; emit text up to safe point
                safe_emit = len(self._buffer) - (self._MAX_TAG_LEN - 1)
                if safe_emit > 0:
                    yield Segment("text", self._buffer[:safe_emit])
                    self._buffer = self._buffer[safe_emit:]
                    return
                return
            # Emit text before opening
            if open_idx > 0:
                yield Segment("text", self._buffer[:open_idx])
            self._buffer = self._buffer[open_idx + len(_OPEN):]
            self._in_think = True
            continue

    def flush(self) -> Iterator[Segment]:
        """Drain any remaining buffer at end-of-stream."""
        if self._buffer:
            if self._in_think:
                yield Segment("thinking", self._buffer)
            else:
                yield Segment("text", self._buffer)
            self._buffer = ""
        return
