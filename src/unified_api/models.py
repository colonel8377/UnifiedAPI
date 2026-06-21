"""Minimal request/response schemas for the skeleton.

Full Anthropic & OpenAI schemas are filled in during Task #2. For now we
define just enough to type the route handlers.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# --- Anthropic side (what the client sends / receives) ---


class AnthropicMessageRequest(BaseModel):
    """Subset of the Anthropic Messages API request we accept in v1.

    Unknown fields are allowed (and ignored) so Claude Code's full request
    payload doesn't 422.
    """
    model_config = {"extra": "allow"}

    model: str
    messages: list[dict[str, Any]]
    system: str | list[dict[str, Any]] | None = None
    max_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class AnthropicResponse(BaseModel):
    """Anthropic Messages API non-streaming response."""
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[dict[str, Any]]
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage


class AnthropicError(BaseModel):
    type: Literal["error"] = "error"
    error: dict[str, Any]


# --- OpenAI side (what we send to / receive from upstream) ---


class OpenAIChatRequest(BaseModel):
    """OpenAI Chat Completions request."""
    model_config = {"extra": "allow"}

    model: str
    messages: list[dict[str, Any]]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None


class OpenAIChatResponse(BaseModel):
    """OpenAI Chat Completions response (lenient: upstream has extra fields)."""
    model_config = {"extra": "allow"}

    id: str
    object: str | None = None
    created: int | str | None = None  # upstream returns string datetime non-stream, int stream
    model: str | None = None
    choices: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] | None = None
