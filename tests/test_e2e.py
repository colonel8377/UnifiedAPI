"""End-to-end tests against a live UnifiedAPI server.

These tests stand up the actual FastAPI app (via httpx.AsyncClient transport)
and exercise the full pipeline: Anthropic request → conversion → upstream →
conversion → Anthropic response.

Requires a real upstream. Skipped automatically when:
  - no .env / OPENAI_KEY is set
  - the upstream is unreachable

To run only these tests: pytest tests/test_e2e.py -v
To skip them: pytest --ignore=tests/test_e2e.py
"""
from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import httpx
import pytest

from unified_api.config import get_config, reset_config_cache
from unified_api.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _has_upstream_config() -> bool:
    """Check whether .env has credentials."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return False
    for line in env_file.read_text().splitlines():
        if line.startswith("OPENAI_KEY_") and len(line.strip()) > len("OPENAI_KEY_1="):
            return True
    return False


pytestmark = pytest.mark.skipif(
    not _has_upstream_config(),
    reason="No OPENAI_KEY_1/OPENAI_KEY_2 in .env — skipping live upstream tests",
)


@pytest.fixture
async def client():
    """Async HTTP client wired directly to the FastAPI app (no port binding)."""
    reset_config_cache()
    config = get_config()
    password = config.auth.password
    headers = {}
    if password:
        headers["x-api-key"] = password
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        timeout=120.0,
        headers=headers,
    ) as c:
        # Manually run the lifespan so upstream client + rate limiter are set up
        async with app.router.lifespan_context(app):
            yield c


@pytest.fixture
def sample_request() -> dict:
    return {
        "model": "DeepSeek-V4-Pro",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": f"Reply with only the word OK. ({uuid.uuid4().hex[:6]})"}
        ],
    }


# --- non-streaming ---


async def test_nonstream_basic_text(client, sample_request):
    resp = await client.post("/v1/messages", json=sample_request)
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["model"] == "DeepSeek-V4-Pro"
    assert data["id"].startswith("msg_")
    assert any(b["type"] == "text" for b in data["content"])
    assert data["stop_reason"] in {"end_turn", "max_tokens"}
    assert "input_tokens" in data["usage"]
    assert "output_tokens" in data["usage"]


async def test_nonstream_with_system(client):
    resp = await client.post("/v1/messages", json={
        "model": "DeepSeek-V4-Pro",
        "max_tokens": 50,
        "system": "You are a parrot. Reply with exactly what the user says, nothing else.",
        "messages": [{"role": "user", "content": "banana"}],
    })
    assert resp.status_code == 200
    text_blocks = [b for b in resp.json()["content"] if b["type"] == "text"]
    combined = "".join(b["text"] for b in text_blocks).lower()
    assert "banana" in combined


async def test_nonstream_tool_use(client):
    """Tool-call prompt injection should yield a tool_use block."""
    resp = await client.post("/v1/messages", json={
        "model": "DeepSeek-V4-Pro",
        "max_tokens": 500,
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo? Use the get_weather tool."}
        ],
        "tools": [{
            "name": "get_weather",
            "description": "Get current weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }],
        "tool_choice": {"type": "required"},
    })
    assert resp.status_code == 200
    data = resp.json()
    tool_uses = [b for b in data["content"] if b["type"] == "tool_use"]
    assert len(tool_uses) >= 1
    assert tool_uses[0]["name"] == "get_weather"
    assert "city" in tool_uses[0]["input"]
    assert data["stop_reason"] == "tool_use"


# --- streaming ---


async def test_stream_basic_text(client):
    """Streaming should emit the standard Anthropic SSE event sequence."""
    resp = await client.post("/v1/messages", json={
        "model": "DeepSeek-V4-Pro",
        "max_tokens": 50,
        "stream": True,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
    })
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = _parse_sse_stream(resp)
    types = [t for t, _ in events]

    # Required event sequence
    assert types[0] == "message_start"
    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert "content_block_stop" in types
    assert "message_delta" in types
    assert types[-1] == "message_stop"


async def test_stream_emits_text_deltas(client):
    """content_block_delta events with text_delta should aggregate to readable text."""
    resp = await client.post("/v1/messages", json={
        "model": "DeepSeek-V4-Pro",
        "max_tokens": 50,
        "stream": True,
        "messages": [{"role": "user", "content": "Count from 1 to 3."}],
    })
    events = _parse_sse_stream(resp)
    text_deltas = [
        data["delta"]["text"]
        for t, data in events
        if t == "content_block_delta" and data["delta"].get("type") == "text_delta"
    ]
    combined = "".join(text_deltas)
    # Should contain digits 1, 2, 3 somewhere
    assert any(c.isdigit() for c in combined)


async def test_stream_tool_use(client):
    """Streaming tool_use should emit content_block_start with type=tool_use + input_json_delta."""
    resp = await client.post("/v1/messages", json={
        "model": "DeepSeek-V4-Pro",
        "max_tokens": 500,
        "stream": True,
        "messages": [
            {"role": "user", "content": "What's the weather in Paris? Use the get_weather tool."}
        ],
        "tools": [{
            "name": "get_weather",
            "description": "Get current weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }],
        "tool_choice": {"type": "required"},
    })
    events = _parse_sse_stream(resp)
    tool_starts = [
        data for t, data in events
        if t == "content_block_start" and data["content_block"]["type"] == "tool_use"
    ]
    assert len(tool_starts) >= 1
    # message_delta should carry stop_reason=tool_use
    deltas = [data for t, data in events if t == "message_delta"]
    assert any(d["delta"].get("stop_reason") == "tool_use" for d in deltas)


async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --- multi-turn conversation ---


async def test_multi_turn_with_tool_result(client):
    """Full tool round-trip: model calls tool → user sends tool_result → model answers."""
    # Round 1: model emits tool_use
    resp1 = await client.post("/v1/messages", json={
        "model": "DeepSeek-V4-Pro",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": "What's the weather in Berlin? Use the tool."}],
        "tools": [{
            "name": "get_weather",
            "description": "Get current weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }],
        "tool_choice": {"type": "required"},
    })
    data1 = resp1.json()
    tool_uses = [b for b in data1["content"] if b["type"] == "tool_use"]
    assert tool_uses
    tool_use_id = tool_uses[0]["id"]

    # Round 2: send the tool_result back
    resp2 = await client.post("/v1/messages", json={
        "model": "DeepSeek-V4-Pro",
        "max_tokens": 500,
        "messages": [
            {"role": "user", "content": "What's the weather in Berlin? Use the tool."},
            {"role": "assistant", "content": data1["content"]},
            {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": [{"type": "text", "text": "sunny, 22C"}],
            }]},
        ],
        "tools": [{
            "name": "get_weather",
            "description": "Get current weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }],
    })
    assert resp2.status_code == 200
    data2 = resp2.json()
    text_blocks = [b for b in data2["content"] if b["type"] == "text"]
    combined = "".join(b["text"] for b in text_blocks).lower()
    # The model should incorporate the tool result
    assert any(token in combined for token in ["sunny", "22", "berlin", "weather"])


# --- helpers ---


def _parse_sse_stream(resp: httpx.Response) -> list[tuple[str, dict]]:
    """Parse the SSE response body into a list of (event_type, data).

    The server emits events with CRLF line endings and double-CRLF separators.
    """
    import json
    events: list[tuple[str, dict]] = []
    # Normalize CRLF → LF so we can split cleanly
    text = resp.text.replace("\r\n", "\n")
    for raw in text.split("\n\n"):
        raw = raw.strip()
        if not raw:
            continue
        et = None
        data = None
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("event:"):
                et = line[len("event:"):].strip()
            elif line.startswith("data:"):
                try:
                    data = json.loads(line[len("data:"):].strip())
                except json.JSONDecodeError:
                    pass
        if et and data:
            events.append((et, data))
    return events
