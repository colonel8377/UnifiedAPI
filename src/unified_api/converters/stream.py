"""OpenAI SSE → Anthropic SSE streaming converter.

Stateful state machine: feed OpenAI chunks via `feed()`, then call
`flush()` at end-of-stream. Each call yields SSE-formatted byte events
matching Anthropic's Messages streaming event contract.

Pipeline (applied to each `delta`):
  1. `delta.reasoning_content` → thinking block (if return_thinking)
  2. `delta.content` → think_splitter → classifies as thinking or text
       a. thinking → thinking block (if return_thinking)
       b. text → xml_scanner → splits into text and tool_use segments
            i.  text segment → text block
            ii. tool_use segment → emits a complete tool_use block
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Iterator

from ..tools.think_splitter import IncrementalThinkSplitter
from ..tools.xml_parser import IncrementalXmlScanner, TextSegment, ToolUseSegment

logger = logging.getLogger(__name__)


_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def sse(event_type: str, data: dict[str, Any]) -> bytes:
    """Format one Anthropic-style SSE event."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\r\ndata: {payload}\r\n\r\n".encode("utf-8")


class StreamConverter:
    """Convert an OpenAI streaming chat completion into Anthropic SSE events."""

    def __init__(self, *, requested_model_alias: str, return_thinking: bool) -> None:
        self._alias = requested_model_alias
        self._return_thinking = return_thinking
        self._message_id = f"msg_{uuid.uuid4().hex[:24]}"

        self._started = False
        self._next_index = 0
        self._current_block: str | None = None  # "thinking" | "text"
        self._finish_reason: str | None = None
        self._has_tool_use = False

        # Incremental scanners for content (the visible `delta.content` field)
        self._think_splitter = IncrementalThinkSplitter()
        self._xml_scanner = IncrementalXmlScanner()

        # Native OpenAI tool_calls accumulator (streaming)
        self._native_tool_calls: dict[int, dict[str, Any]] = {}  # index → {id, name, arguments}

        # Captured upstream usage from final streaming chunk
        self._usage: dict[str, Any] | None = None

    def feed(self, chunk: dict[str, Any]) -> Iterator[bytes]:
        # Capture usage from chunk (present in the final chunk when include_usage=true)
        usage = chunk.get("usage")
        if isinstance(usage, dict) and usage:
            self._usage = usage

        choices = chunk.get("choices") or []
        if not choices:
            return
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        finish_reason = choice.get("finish_reason")

        if not self._started:
            self._started = True
            yield self._emit_message_start()

        # 1. Structured reasoning field → thinking
        # Some upstreams use "reasoning_content", others use "reasoning"
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning and self._return_thinking:
            yield from self._ensure_block("thinking")
            yield sse("content_block_delta", {
                "type": "content_block_delta",
                "index": self._next_index - 1,
                "delta": {"type": "thinking_delta", "thinking": reasoning},
            })

        # 2. content → think_splitter → xml_scanner → events
        content = delta.get("content")
        if isinstance(content, str) and content:
            yield from self._process_content(content)

        # 3. Native OpenAI tool_calls (streaming)
        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            yield from self._accumulate_native_tool_calls(tool_calls)

        if isinstance(finish_reason, str) and finish_reason:
            self._finish_reason = finish_reason

    def flush(self) -> Iterator[bytes]:
        # Emit any accumulated native OpenAI tool_calls first
        yield from self._emit_accumulated_tool_calls()

        # Drain scanners (in case there's buffered text at EOF)
        for seg in self._think_splitter.flush():
            yield from self._emit_think_segment(seg.kind, seg.text)
        for seg in self._xml_scanner.flush():
            yield from self._emit_xml_segment(seg)

        if not self._started:
            yield self._emit_message_start()
        yield from self._close_current_block()
        yield self._emit_message_delta()
        yield self._emit_message_stop()

    # --- internal helpers ---

    def _process_content(self, content: str) -> Iterator[bytes]:
        """Route content through think_splitter → xml_scanner."""
        for seg in self._think_splitter.feed(content):
            yield from self._emit_think_segment(seg.kind, seg.text)

    def _emit_think_segment(self, kind: str, text: str) -> Iterator[bytes]:
        if kind == "thinking":
            if not self._return_thinking:
                return
            yield from self._ensure_block("thinking")
            yield sse("content_block_delta", {
                "type": "content_block_delta",
                "index": self._next_index - 1,
                "delta": {"type": "thinking_delta", "thinking": text},
            })
        elif kind == "text":
            # Feed into the XML scanner
            for xml_seg in self._xml_scanner.feed(text):
                yield from self._emit_xml_segment(xml_seg)

    def _emit_xml_segment(self, seg: Any) -> Iterator[bytes]:
        if isinstance(seg, TextSegment):
            text = seg.text
            if text:
                yield from self._ensure_block("text")
                yield sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": self._next_index - 1,
                    "delta": {"type": "text_delta", "text": text},
                })
        elif isinstance(seg, ToolUseSegment):
            yield from self._emit_tool_use_block(seg.name, seg.params)

    def _ensure_block(self, block_type: str) -> Iterator[bytes]:
        if self._current_block == block_type:
            return
        if self._current_block is not None:
            yield self._close_current_block_event()
        yield self._open_block_event(block_type)
        self._current_block = block_type

    def _close_current_block(self) -> Iterator[bytes]:
        if self._current_block is not None:
            yield self._close_current_block_event()

    def _close_current_block_event(self) -> bytes:
        event = sse("content_block_stop", {
            "type": "content_block_stop",
            "index": self._next_index - 1,
        })
        self._current_block = None
        return event

    def _open_block_event(self, block_type: str) -> bytes:
        index = self._next_index
        self._next_index += 1
        if block_type == "thinking":
            content_block = {"type": "thinking", "thinking": "", "signature": ""}
        elif block_type == "text":
            content_block = {"type": "text", "text": ""}
        else:
            raise ValueError(f"Unknown block type: {block_type}")
        return sse("content_block_start", {
            "type": "content_block_start",
            "index": index,
            "content_block": content_block,
        })

    def _emit_message_start(self) -> bytes:
        return sse("message_start", {
            "type": "message_start",
            "message": {
                "id": self._message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self._alias,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        })

    def _emit_message_delta(self) -> bytes:
        stop_reason = _STOP_REASON_MAP.get(self._finish_reason or "stop", "end_turn")
        if self._has_tool_use and stop_reason == "end_turn":
            stop_reason = "tool_use"
        output_tokens = 0
        if self._usage:
            output_tokens = int(self._usage.get("completion_tokens", 0) or 0)
        return sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        })

    def _emit_message_stop(self) -> bytes:
        return sse("message_stop", {"type": "message_stop"})

    # --- Native OpenAI tool_calls handling ---

    def _accumulate_native_tool_calls(self, tool_calls: list[dict[str, Any]]) -> Iterator[bytes]:
        """Accumulate streaming tool_call deltas. Emit tool_use blocks when
        a new tool call starts (we get its name)."""
        for tc_delta in tool_calls:
            if not isinstance(tc_delta, dict):
                continue
            idx = tc_delta.get("index", 0)
            if idx not in self._native_tool_calls:
                self._native_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
            entry = self._native_tool_calls[idx]
            # id comes in the first chunk for this index
            tc_id = tc_delta.get("id")
            if isinstance(tc_id, str) and tc_id:
                entry["id"] = tc_id
            # function name + arguments
            fn = tc_delta.get("function")
            if isinstance(fn, dict):
                name = fn.get("name")
                if isinstance(name, str) and name:
                    entry["name"] = name
                args_chunk = fn.get("arguments")
                if isinstance(args_chunk, str):
                    entry["arguments"] += args_chunk

    def _emit_accumulated_tool_calls(self) -> Iterator[bytes]:
        """Emit all accumulated native tool calls as tool_use blocks."""
        for idx in sorted(self._native_tool_calls.keys()):
            entry = self._native_tool_calls[idx]
            name = entry.get("name")
            if not name:
                continue
            raw_args = entry.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if raw_args else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            tc_id = entry.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"
            yield from self._emit_tool_use_block(name, args if isinstance(args, dict) else {}, tc_id)

    def _emit_tool_use_block(
        self, name: str, params: dict[str, Any], tool_id: str | None = None
    ) -> Iterator[bytes]:
        # Close any open block first (text or thinking)
        if self._current_block is not None:
            yield self._close_current_block_event()
        index = self._next_index
        self._next_index += 1
        tid = tool_id or f"toolu_{uuid.uuid4().hex[:24]}"
        yield sse("content_block_start", {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": tid, "name": name, "input": {}},
        })
        partial_json = json.dumps(params, ensure_ascii=False)
        yield sse("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        })
        yield sse("content_block_stop", {"type": "content_block_stop", "index": index})
        self._has_tool_use = True
