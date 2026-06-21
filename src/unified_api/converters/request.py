"""Anthropic Messages request → OpenAI Chat Completions request.

Full v1 conversion:
  - system prompt (string or content-block array) → first system message
  - tools injected as an XML spec appended to the system prompt
    (upstream doesn't honor OpenAI tools natively)
  - tool_choice mapped to a natural-language constraint
  - prior tool_use blocks → `<function_calls>` XML replay in assistant turns
  - prior tool_result blocks → `<tool_result>` text in user turns
  - text-only content blocks → concatenated string
  - image blocks → dropped (not in v1 scope)
  - model name passes through unchanged
"""
from __future__ import annotations

import logging
from typing import Any

from ..errors import ConversionError
from ..models import AnthropicMessageRequest, OpenAIChatRequest
from ..tools.prompt_builder import (
    build_tools_prompt,
    render_tool_result_as_text,
    render_tool_use_as_text,
)

logger = logging.getLogger(__name__)


def convert_request(
    anth_req: AnthropicMessageRequest,
    model_name: str,
) -> OpenAIChatRequest:
    """Translate an Anthropic Messages request into an OpenAI Chat request.

    model_name is passed through directly to the upstream without alias mapping.
    """
    openai_messages: list[dict[str, Any]] = []

    # system → first system message, plus injected tool spec if tools present
    system_text = _flatten_system(anth_req.system) if anth_req.system is not None else ""
    if anth_req.tools:
        tool_prompt = build_tools_prompt(anth_req.tools, anth_req.tool_choice)
        if tool_prompt:
            system_text = f"{system_text}\n\n{tool_prompt}" if system_text else tool_prompt
    if system_text:
        openai_messages.append({"role": "system", "content": system_text})

    for idx, msg in enumerate(anth_req.messages):
        role = msg.get("role")
        if role not in ("user", "assistant"):
            raise ConversionError(f"Unsupported message role at index {idx}: {role!r}")
        content = msg.get("content")
        openai_messages.append({"role": role, "content": _flatten_content(content, idx, role)})

    # DeepSeek reasoning tokens share the max_tokens budget with content.
    # Anthropic's max_tokens only counts visible output (thinking has a separate
    # budget). Add a buffer to ensure enough tokens for content after reasoning.
    _REASONING_BUFFER = 512
    upstream_max_tokens = (anth_req.max_tokens or 0) + _REASONING_BUFFER

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": openai_messages,
        "max_tokens": upstream_max_tokens,
    }
    if anth_req.temperature is not None:
        payload["temperature"] = anth_req.temperature
    if anth_req.top_p is not None:
        payload["top_p"] = anth_req.top_p
    if anth_req.stop_sequences:
        payload["stop"] = anth_req.stop_sequences

    # Pass native OpenAI-format tools to upstream alongside the XML prompt.
    # This lets upstreams that support OpenAI tool calling (e.g. HKUST) use
    # structured tool definitions directly.
    if anth_req.tools:
        oai_tools = _convert_tools_to_openai(anth_req.tools)
        if oai_tools:
            payload["tools"] = oai_tools
    if anth_req.tool_choice is not None:
        oai_choice = _convert_tool_choice_to_openai(anth_req.tool_choice)
        if oai_choice is not None:
            payload["tool_choice"] = oai_choice

    return OpenAIChatRequest(**payload)


def _flatten_system(system: str | list[dict[str, Any]]) -> str:
    if isinstance(system, str):
        return system
    parts: list[str] = []
    for block in system:
        if isinstance(block, dict) and block.get("type") == "text":
            t = block.get("text")
            if isinstance(t, str) and t:
                parts.append(t)
        else:
            logger.warning("Dropping non-text system block: %r", block)
    return "\n\n".join(parts)


def _flatten_content(content: Any, idx: int, role: str) -> str:
    """Flatten an Anthropic message body into a single string.

    - string content → returned directly
    - text blocks → concatenated text
    - tool_use blocks (assistant) → `<function_calls>` XML
    - tool_result blocks (user) → `<tool_result>` text
    - image blocks → dropped (not in v1 scope)
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ConversionError(
            f"Unsupported content shape at message[{idx}] (role={role}): expected string or array"
        )

    parts: list[str] = []
    # Group consecutive tool_use blocks into one <function_calls> wrapper
    # so the model recognizes the format from its own prior turns.
    pending_invokes: list[tuple[str, dict[str, Any]]] = []

    def _flush_pending() -> None:
        if not pending_invokes:
            return
        # Render as one combined block
        from ..tools.prompt_builder import _render_invokes
        parts.append(_render_invokes(pending_invokes))
        pending_invokes.clear()

    for block in content:
        if not isinstance(block, dict):
            logger.warning("Dropping non-dict content block at message[%d]: %r", idx, block)
            continue
        btype = block.get("type")
        if btype == "text":
            _flush_pending()
            t = block.get("text")
            if isinstance(t, str) and t:
                parts.append(t)
        elif btype == "tool_use":
            name = block.get("name")
            tool_input = block.get("input") or {}
            tool_use_id = block.get("id", "")
            if not isinstance(name, str) or not name:
                logger.warning("Dropping tool_use without name at message[%d]", idx)
                continue
            _flush_pending()  # ensure prior invokes batched first
            pending_invokes.append((name, tool_input if isinstance(tool_input, dict) else {}))
            # id is not needed for replay — the model just sees its own output format
            _ = tool_use_id
        elif btype == "tool_result":
            _flush_pending()
            tool_use_id = block.get("tool_use_id", "")
            result_content = block.get("content")
            parts.append(render_tool_result_as_text(str(tool_use_id), result_content))
        elif btype == "image":
            logger.warning("Dropping image block at message[%d] — not in v1 scope", idx)
        else:
            logger.warning("Dropping unknown block type %r at message[%d]", btype, idx)

    _flush_pending()
    return "\n".join(p for p in parts if p)


# --- Anthropic → OpenAI tool schema conversion ---


def _convert_tools_to_openai(anth_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Anthropic tool definitions into OpenAI function-tool format."""
    result: list[dict[str, Any]] = []
    for tool in anth_tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        fn: dict[str, Any] = {"name": name}
        desc = tool.get("description")
        if isinstance(desc, str) and desc:
            fn["description"] = desc
        # Anthropic uses "input_schema"; OpenAI uses "parameters"
        schema = tool.get("input_schema") or tool.get("parameters")
        if isinstance(schema, dict) and schema:
            fn["parameters"] = schema
        result.append({"type": "function", "function": fn})
    return result


def _convert_tool_choice_to_openai(
    tool_choice: dict[str, Any],
) -> str | dict[str, Any] | None:
    """Translate Anthropic tool_choice into OpenAI tool_choice."""
    ctype = tool_choice.get("type")
    if ctype == "auto":
        return "auto"
    if ctype in ("any", "required"):
        return "required"
    if ctype == "tool":
        name = tool_choice.get("name")
        if isinstance(name, str) and name:
            return {"type": "function", "function": {"name": name}}
    return None
