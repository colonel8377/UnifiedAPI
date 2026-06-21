"""OpenAI Chat Completions response → Anthropic Messages response (non-stream).

Full v1 conversion:
  - reasoning_content → optional thinking block (config-controlled)
  - <think>...</think> blocks stripped from content
  - <function_calls> XML → tool_use content blocks
  - surrounding text → text content blocks
  - finish_reason → stop_reason mapping
  - usage mapping
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from ..models import AnthropicResponse, AnthropicUsage, OpenAIChatResponse
from ..tools.think_splitter import strip_think_complete
from ..tools.xml_parser import TextSegment, ToolUseSegment, parse_complete

logger = logging.getLogger(__name__)


_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
    "unknown": "end_turn",
}


def convert_response(
    oai_resp: OpenAIChatResponse,
    requested_model_alias: str,
    return_thinking: bool,
) -> AnthropicResponse:
    """Translate an OpenAI Chat response into an Anthropic Messages response."""
    content_blocks: list[dict[str, Any]] = []

    # Optional thinking from reasoning_content (the structured field)
    if return_thinking:
        reasoning = _extract_reasoning(oai_resp)
        if reasoning:
            content_blocks.append({"type": "thinking", "thinking": reasoning})

    # Pull visible content, then split out <think> blocks and <function_calls> XML
    raw_content = _extract_text(oai_resp)
    cleaned_content, think_text = strip_think_complete(raw_content)

    if return_thinking and think_text:
        content_blocks.append({"type": "thinking", "thinking": think_text})

    segments = parse_complete(cleaned_content)
    has_tool_use = False
    for seg in segments:
        if isinstance(seg, TextSegment):
            text = seg.text.strip()
            if text:
                content_blocks.append({"type": "text", "text": seg.text})
        elif isinstance(seg, ToolUseSegment):
            content_blocks.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": seg.name,
                "input": seg.params,
            })
            has_tool_use = True

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    finish_reason = _extract_finish_reason(oai_resp)
    # If the upstream said stop but we parsed tool_use blocks, prefer tool_use
    stop_reason = _STOP_REASON_MAP.get(finish_reason, "end_turn")
    if has_tool_use and stop_reason == "end_turn":
        stop_reason = "tool_use"

    usage = _extract_usage(oai_resp)

    return AnthropicResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        model=requested_model_alias,
        content=content_blocks,
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=usage,
    )


# --- field extractors ---


def _extract_text(resp: OpenAIChatResponse) -> str:
    if not resp.choices:
        return ""
    choice = resp.choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return ""


def _extract_reasoning(resp: OpenAIChatResponse) -> str:
    if not resp.choices:
        return ""
    choice = resp.choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        return ""
    # Some upstreams use "reasoning_content", others use "reasoning"
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    return reasoning if isinstance(reasoning, str) and reasoning else ""


def _extract_finish_reason(resp: OpenAIChatResponse) -> str:
    if not resp.choices:
        return "unknown"
    choice = resp.choices[0]
    if isinstance(choice, dict):
        reason = choice.get("finish_reason")
        if isinstance(reason, str) and reason:
            return reason
    return "unknown"


def _extract_usage(resp: OpenAIChatResponse) -> AnthropicUsage:
    usage = resp.usage or {}
    return AnthropicUsage(
        input_tokens=int(usage.get("prompt_tokens", 0) or 0),
        output_tokens=int(usage.get("completion_tokens", 0) or 0),
    )
