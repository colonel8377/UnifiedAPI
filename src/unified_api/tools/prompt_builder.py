"""Build the system-prompt fragment that teaches the model about its tools.

The upstream service does NOT honor OpenAI's `tools` parameter — the model
just hallucinates function calls in free-form text. Empirical testing on
DeepSeek-V4-Pro showed the model natively emits a stable XML shape when
the tool spec is injected into the system prompt and the model is told
to use this exact format:

    <function_calls>
    <invoke name="TOOL_NAME">
    <parameter name="PARAM_NAME">VALUE</parameter>
    </invoke>
    </function_calls>

This module renders Anthropic tool schemas into that prompt fragment.
"""
from __future__ import annotations

import json
from typing import Any


def build_tools_prompt(tools: list[dict[str, Any]], tool_choice: dict[str, Any] | None = None) -> str:
    """Render an Anthropic tools array into a system-prompt fragment.

    Returns an empty string when `tools` is empty.
    """
    if not tools:
        return ""

    lines: list[str] = []
    lines.append("You have access to these tools:")
    lines.append("")
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        sig = _format_signature(tool)
        description = tool.get("description", "") or ""
        desc_suffix = f": {description}" if description else ""
        lines.append(f"- {sig}{desc_suffix}")
    lines.append("")
    lines.append("When you need to call a tool, output ONLY this XML block and nothing else:")
    lines.append("<function_calls>")
    lines.append('<invoke name="TOOL_NAME">')
    lines.append('<parameter name="PARAM_NAME">VALUE</parameter>')
    lines.append("</invoke>")
    lines.append("</function_calls>")
    lines.append("")
    lines.append("You may include multiple <invoke> blocks inside one <function_calls> block to call several tools in parallel.")
    lines.append("Escape special characters in parameter values using XML entities (&lt; &gt; &amp; &quot; &apos;).")

    choice_hint = _format_tool_choice_hint(tool_choice)
    if choice_hint:
        lines.append("")
        lines.append(choice_hint)

    return "\n".join(lines)


def _format_signature(tool: dict[str, Any]) -> str:
    """Render `- name(param: type, ...)` from an Anthropic input_schema."""
    name = tool["name"]
    schema = tool.get("input_schema") or tool.get("parameters") or {}
    if not isinstance(schema, dict):
        return name
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    if not isinstance(properties, dict) or not properties:
        return name

    params: list[str] = []
    for pname, pschema in properties.items():
        ptype = (pschema or {}).get("type", "any") if isinstance(pschema, dict) else "any"
        marker = "" if pname in required else "?"  # mark optional
        params.append(f"{pname}{marker}: {ptype}")
    return f"{name}({', '.join(params)})"


def _format_tool_choice_hint(tool_choice: dict[str, Any] | None) -> str:
    if not isinstance(tool_choice, dict):
        return ""
    ctype = tool_choice.get("type")
    if ctype == "any" or ctype == "required":
        return "You MUST call one of the available tools in your response. Do not respond with only text."
    if ctype == "tool":
        name = tool_choice.get("name")
        if isinstance(name, str):
            return f"You MUST call the tool named {name!r} in your response."
    # type == "auto" (default) → no extra hint
    return ""


# --- helpers for converting prior assistant tool_use + user tool_result into the conversation ---


def render_tool_use_as_text(name: str, tool_use_id: str, input_: dict[str, Any]) -> str:
    """Render a prior assistant tool_use content block as the XML the model emits.

    Used when replaying conversation history: an assistant turn that contained
    a tool_use block is replayed as the equivalent XML so the model recognizes
    its own prior output.
    """
    return _render_invokes([(name, input_)])


def render_tool_result_as_text(tool_use_id: str, content: Any) -> str:
    """Render a prior user tool_result content block as text."""
    text = _flatten_tool_result_content(content)
    return f"<tool_result tool_use_id=\"{tool_use_id}\">\n{text}\n</tool_result>"


def _render_invokes(invoke_pairs: list[tuple[str, dict[str, Any]]]) -> str:
    lines = ["<function_calls>"]
    for name, params in invoke_pairs:
        lines.append(f'<invoke name="{name}">')
        if isinstance(params, dict):
            for pname, pvalue in params.items():
                value_str = json.dumps(pvalue, ensure_ascii=False) if not isinstance(pvalue, str) else pvalue
                escaped = _xml_escape(value_str)
                lines.append(f'<parameter name="{pname}">{escaped}</parameter>')
        lines.append("</invoke>")
    lines.append("</function_calls>")
    return "\n".join(lines)


def _flatten_tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(block, dict) and block.get("type") == "image":
                parts.append("[image omitted]")
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&apos;")
    )
