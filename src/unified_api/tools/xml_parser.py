"""Incremental `<function_calls>` XML parser.

Two interfaces:
  - `parse_complete(text)` — one-shot, splits content into text + tool invokes
  - `IncrementalXmlScanner` — stateful, for streaming

The upstream model emits tool calls in this shape (we inject the format
via the system prompt — see `prompt_builder.py`):

    <function_calls>
    <invoke name="get_weather">
    <parameter name="city">Paris</parameter>
    </invoke>
    </function_calls>

We parse it back into structured (name, params) pairs.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_OPEN_TAG = "<function_calls>"
_CLOSE_TAG = "</function_calls>"
_MAX_TAG_LEN = max(len(_OPEN_TAG), len(_CLOSE_TAG))


@dataclass
class ParsedInvoke:
    name: str
    params: dict[str, Any]


@dataclass
class TextSegment:
    text: str


@dataclass
class ToolUseSegment:
    """One complete invoke block."""
    name: str
    params: dict[str, Any]


# --- static parser (non-streaming) ---

_OPEN_RE = re.compile(_OPEN_TAG, re.IGNORECASE)
_CLOSE_RE = re.compile(_CLOSE_TAG, re.IGNORECASE)


def parse_complete(text: str) -> list[TextSegment | ToolUseSegment]:
    """Parse a complete content string into a sequence of text + tool segments.

    - Text outside `<function_calls>` blocks → TextSegment
    - Each `<invoke>` inside → ToolUseSegment
    - Multiple `<function_calls>` blocks in series are all parsed.
    - Adjacent TextSegments are NOT merged — the caller can decide.
    """
    if not text:
        return []
    segments: list[TextSegment | ToolUseSegment] = []
    pos = 0
    while True:
        open_m = _OPEN_RE.search(text, pos)
        if not open_m:
            tail = text[pos:]
            if tail:
                segments.append(TextSegment(text=tail))
            break
        # text before block
        before = text[pos:open_m.start()]
        if before:
            segments.append(TextSegment(text=before))
        close_m = _CLOSE_RE.search(text, open_m.end())
        if not close_m:
            # Block never closes — treat the partial block as text so the
            # client sees what the model produced rather than losing it.
            logger.debug("Unclosed <function_calls> block; emitting as text")
            segments.append(TextSegment(text=text[open_m.start():]))
            break
        xml_chunk = text[open_m.start():close_m.end()]
        for invoke in _parse_invokes_from_xml(xml_chunk):
            segments.append(ToolUseSegment(name=invoke.name, params=invoke.params))
        pos = close_m.end()
    return segments


def _parse_invokes_from_xml(xml_text: str) -> list[ParsedInvoke]:
    """Parse all <invoke> blocks inside one <function_calls> wrapper."""
    results: list[ParsedInvoke] = []
    try:
        # ElementTree is strict; the model usually emits clean XML so this works.
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.debug("ElementTree parse failed (%s); falling back to regex", e)
        return _parse_invokes_regex(xml_text)

    if root.tag.lower() != "function_calls":
        return results

    for invoke in root:
        if invoke.tag.lower() != "invoke":
            continue
        name = invoke.get("name")
        if not name:
            continue
        params: dict[str, Any] = {}
        for param in invoke:
            if param.tag.lower() != "parameter":
                continue
            pname = param.get("name")
            if not pname:
                continue
            params[pname] = _decode_param_value(param.text or "")
        results.append(ParsedInvoke(name=name, params=params))
    return results


_PARAM_RE = re.compile(
    r'<invoke\s+name="([^"]+)">(.*?)</invoke>',
    re.IGNORECASE | re.DOTALL,
)
_PARAM_INNER_RE = re.compile(
    r'<parameter\s+name="([^"]+)">(.*?)</parameter>',
    re.IGNORECASE | re.DOTALL,
)


def _parse_invokes_regex(xml_text: str) -> list[ParsedInvoke]:
    """Fallback when ElementTree rejects the XML (e.g., stray characters).

    Tolerant regex-based parser. Still handles XML entity escapes.
    """
    results: list[ParsedInvoke] = []
    for invoke_m in _PARAM_RE.finditer(xml_text):
        name = invoke_m.group(1).strip()
        inner = invoke_m.group(2)
        params: dict[str, Any] = {}
        for param_m in _PARAM_INNER_RE.finditer(inner):
            pname = param_m.group(1).strip()
            pvalue = _decode_param_value(param_m.group(2))
            params[pname] = pvalue
        results.append(ParsedInvoke(name=name, params=params))
    return results


def _decode_param_value(raw: str) -> Any:
    """Decode XML entities and try to coerce to native types."""
    decoded = (
        raw.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&apos;", "'")
            .replace("&amp;", "&")  # last to avoid double-decoding
    )
    return decoded


# --- incremental scanner (streaming) ---


class IncrementalXmlScanner:
    """Stateful scanner that yields text + tool events as content streams in.

    State machine:
      - IDLE: scanning for `<function_calls>` opening. Buffer last N chars at
        chunk boundaries to detect partial opening tags.
      - IN_XML: accumulating until `</function_calls>` found. On close, parse
        and yield one ToolUseSegment per invoke.

    Feed via `feed(chunk)`, then call `flush()` at end-of-stream.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._in_xml = False

    def feed(self, content: str) -> Iterator[TextSegment | ToolUseSegment]:
        if not content:
            return
        self._buffer += content
        while True:
            if self._in_xml:
                close_idx = self._buffer.lower().find(_CLOSE_TAG.lower())
                if close_idx == -1:
                    # Not closed yet; hold the whole buffer until we see the close.
                    return
                xml_chunk = self._buffer[:close_idx + len(_CLOSE_TAG)]
                self._buffer = self._buffer[close_idx + len(_CLOSE_TAG):]
                self._in_xml = False
                for invoke in _parse_invokes_from_xml(xml_chunk):
                    yield ToolUseSegment(name=invoke.name, params=invoke.params)
                continue
            # IDLE: look for opening tag (case-insensitive)
            lower_buf = self._buffer.lower()
            open_idx = lower_buf.find(_OPEN_TAG.lower())
            if open_idx == -1:
                # No opening yet; emit text up to a safe boundary
                # (keep last _MAX_TAG_LEN-1 chars in case the tag spans chunks).
                safe_emit = len(self._buffer) - (_MAX_TAG_LEN - 1)
                if safe_emit > 0:
                    yield TextSegment(text=self._buffer[:safe_emit])
                    self._buffer = self._buffer[safe_emit:]
                return
            # Emit text before opening
            if open_idx > 0:
                yield TextSegment(text=self._buffer[:open_idx])
            self._buffer = self._buffer[open_idx + len(_OPEN_TAG):]
            self._in_xml = True
            continue

    def flush(self) -> Iterator[TextSegment | ToolUseSegment]:
        """Drain buffer at end-of-stream.

        - If still in IDLE: emit remaining buffer as text.
        - If still in IN_XML (unclosed): try to parse what we have; if that
          fails, emit as text (don't lose the model's output).
        """
        if not self._buffer:
            return
        if self._in_xml:
            partial = f"{_OPEN_TAG}{self._buffer}"
            invokes = _parse_invokes_from_xml(partial)
            if invokes:
                for inv in invokes:
                    yield ToolUseSegment(name=inv.name, params=inv.params)
            else:
                # Unclosed and unparseable — preserve as text
                logger.debug("Unclosed <function_calls> at flush; emitting as text")
                yield TextSegment(text=partial)
        else:
            yield TextSegment(text=self._buffer)
        self._buffer = ""
