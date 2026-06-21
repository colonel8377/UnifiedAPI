"""Tests for /v1/chat/completions OpenAI pass-through route, KeyPool, and auth.

Covers:
  - KeyPool round-robin behavior
  - Config backward compatibility
  - Auth middleware (password check, public paths)
"""
from __future__ import annotations

import httpx
import pytest

from unified_api.config import KeyPool, get_config, reset_config_cache
from unified_api.main import app


# --- KeyPool ---


def test_keypool_round_robin():
    pool = KeyPool(["key_a", "key_b", "key_c"])
    assert pool.next_key() == "key_a"
    assert pool.next_key() == "key_b"
    assert pool.next_key() == "key_c"
    assert pool.next_key() == "key_a"


def test_keypool_single_key():
    pool = KeyPool(["only_key"])
    assert pool.next_key() == "only_key"
    assert pool.next_key() == "only_key"


def test_keypool_size():
    pool = KeyPool(["k1", "k2"])
    assert pool.size == 2


def test_keypool_empty_raises():
    import pytest
    with pytest.raises(ValueError):
        KeyPool([])


# --- Config backward compat ---


def test_upstream_config_single_key():
    """api_key alone should populate api_keys."""
    from unified_api.config import UpstreamConfig
    cfg = UpstreamConfig(base_url="http://test", api_key="solo_key")
    assert cfg.api_keys == ["solo_key"]


def test_upstream_config_multi_keys():
    from unified_api.config import UpstreamConfig
    cfg = UpstreamConfig(base_url="http://test", api_keys=["k1", "k2"])
    assert cfg.api_keys == ["k1", "k2"]


def test_upstream_config_no_keys_raises():
    import pytest
    from unified_api.config import UpstreamConfig
    with pytest.raises(Exception):
        UpstreamConfig(base_url="http://test")


# --- Auth middleware ---


@pytest.fixture
async def auth_client():
    """Async HTTP client wired to the FastAPI app for auth tests."""
    reset_config_cache()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        timeout=10.0,
    ) as c:
        async with app.router.lifespan_context(app):
            yield c


async def test_health_no_auth_required(auth_client):
    """/health is public and does not require auth."""
    resp = await auth_client.get("/health")
    assert resp.status_code == 200


async def test_messages_without_auth_returns_401(auth_client):
    """Anthropic route without x-api-key returns 401 with Anthropic error format."""
    resp = await auth_client.post("/v1/messages", json={
        "model": "test",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 401
    body = resp.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "authentication_error"


async def test_chat_without_auth_returns_401(auth_client):
    """OpenAI route without Bearer returns 401 with OpenAI error format."""
    resp = await auth_client.post("/v1/chat/completions", json={
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 401
    body = resp.json()
    assert "error" in body
    assert body["error"]["type"] == "authentication_error"


async def test_messages_with_x_api_key_passes_auth(auth_client):
    """Anthropic route: correct x-api-key passes auth."""
    config = get_config()
    password = config.auth.password
    resp = await auth_client.post(
        "/v1/messages",
        json={"model": "test", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
        headers={"x-api-key": password},
    )
    assert resp.status_code != 401


async def test_chat_with_bearer_passes_auth(auth_client):
    """OpenAI route: correct Authorization: Bearer passes auth."""
    config = get_config()
    password = config.auth.password
    resp = await auth_client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {password}"},
    )
    assert resp.status_code != 401


async def test_messages_bearer_does_not_pass_auth(auth_client):
    """Anthropic route: Authorization: Bearer should NOT authenticate."""
    config = get_config()
    password = config.auth.password
    resp = await auth_client.post(
        "/v1/messages",
        json={"model": "test", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {password}"},
    )
    assert resp.status_code == 401


async def test_chat_x_api_key_does_not_pass_auth(auth_client):
    """OpenAI route: x-api-key should NOT authenticate."""
    config = get_config()
    password = config.auth.password
    resp = await auth_client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        headers={"x-api-key": password},
    )
    assert resp.status_code == 401


async def test_messages_wrong_key_returns_401(auth_client):
    resp = await auth_client.post(
        "/v1/messages",
        json={"model": "test", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
        headers={"x-api-key": "wrong_password"},
    )
    assert resp.status_code == 401


async def test_chat_wrong_key_returns_401(auth_client):
    resp = await auth_client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer wrong_password"},
    )
    assert resp.status_code == 401


# --- Tool call passthrough ---


def test_openai_request_preserves_tool_fields():
    """OpenAIChatRequest explicitly declares tools, tool_choice, parallel_tool_calls."""
    from unified_api.models import OpenAIChatRequest

    req = OpenAIChatRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "What is weather in HK?"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }],
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    dumped = req.model_dump(exclude_none=True)
    assert "tools" in dumped
    assert dumped["tool_choice"] == "auto"
    assert dumped["parallel_tool_calls"] is True
    assert dumped["tools"][0]["function"]["name"] == "get_weather"


def test_openai_request_tool_choice_dict():
    """tool_choice can be a dict (e.g. {"type": "function", "function": {"name": "..."}}."""
    from unified_api.models import OpenAIChatRequest

    req = OpenAIChatRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
        tool_choice={"type": "function", "function": {"name": "f"}},
    )
    dumped = req.model_dump(exclude_none=True)
    assert isinstance(dumped["tool_choice"], dict)
    assert dumped["tool_choice"]["function"]["name"] == "f"


def test_openai_request_multiturn_with_tool_results():
    """Multi-turn with assistant tool_calls and tool result messages (Cursor/Codex pattern)."""
    from unified_api.models import OpenAIChatRequest

    req = OpenAIChatRequest(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "Weather in HK?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"HK"}'}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 30C"},
        ],
        tools=[{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
    )
    dumped = req.model_dump(exclude_none=True)
    assert len(dumped["messages"]) == 3
    assert dumped["messages"][1]["tool_calls"][0]["id"] == "call_1"
    assert dumped["messages"][2]["role"] == "tool"
    assert dumped["messages"][2]["tool_call_id"] == "call_1"


def test_openai_response_preserves_tool_calls():
    """OpenAIChatResponse preserves tool_calls in choices and finish_reason='tool_calls'."""
    from unified_api.models import OpenAIChatResponse

    resp = OpenAIChatResponse.model_validate({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"HK"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })
    dumped = resp.model_dump()
    assert dumped["choices"][0]["finish_reason"] == "tool_calls"
    tool_calls = dumped["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"

# --- IP-based client_id extraction ---


def test_extract_client_id_uses_ip():
    """_extract_client_id should use client IP, not auth header."""
    from unittest.mock import MagicMock
    from unified_api.routes.chat import _extract_client_id as chat_extract
    from unified_api.routes.messages import _extract_client_id as msg_extract

    req = MagicMock()
    req.headers = {}
    req.client = MagicMock()
    req.client.host = "192.168.1.42"

    assert chat_extract(req) == "192.168.1.42"
    assert msg_extract(req) == "192.168.1.42"


def test_extract_client_id_x_forwarded_for():
    """X-Forwarded-For header should take precedence over client.host."""
    from unittest.mock import MagicMock
    from unified_api.routes.chat import _extract_client_id as chat_extract

    req = MagicMock()
    req.headers = {"x-forwarded-for": "10.0.0.1, 10.0.0.2, 10.0.0.3"}
    req.client = MagicMock()
    req.client.host = "192.168.1.1"

    assert chat_extract(req) == "10.0.0.1"


def test_extract_client_id_no_client():
    """If client is None (e.g. unix socket), return 'unknown'."""
    from unittest.mock import MagicMock
    from unified_api.routes.chat import _extract_client_id as chat_extract

    req = MagicMock()
    req.headers = {}
    req.client = None

    assert chat_extract(req) == "unknown"


# --- Cursor / Claude Code / Codex integration patterns ---


def test_cursor_openai_request_with_stream_options():
    """Cursor sends extra fields like stream_options; should be accepted."""
    from unified_api.models import OpenAIChatRequest

    req = OpenAIChatRequest(
        model="DeepSeek-V4-Pro",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        stream_options={"include_usage": True},  # Cursor-specific extra field
        temperature=0.7,
    )
    dumped = req.model_dump(exclude_none=True)
    assert dumped["model"] == "DeepSeek-V4-Pro"
    # Extra fields should be preserved (extra="allow")
    assert dumped.get("stream_options") == {"include_usage": True}


def test_claude_code_anthropic_request_with_thinking():
    """Claude Code sends Anthropic request with thinking parameter."""
    from unified_api.models import AnthropicMessageRequest

    req = AnthropicMessageRequest(
        model="DeepSeek-V4-Pro",
        messages=[{"role": "user", "content": "solve this"}],
        max_tokens=8000,
        thinking={"type": "enabled", "budget_tokens": 4000},
    )
    assert req.thinking == {"type": "enabled", "budget_tokens": 4000}


def test_codex_openai_request_with_tools_multiturn():
    """Codex uses OpenAI protocol with tool calls and multi-turn results."""
    from unified_api.models import OpenAIChatRequest

    req = OpenAIChatRequest(
        model="DeepSeek-V4-Pro",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": '{"city":"HK"}'}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 30C"},
        ],
        tools=[{"type": "function", "function": {"name": "weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
    )
    dumped = req.model_dump(exclude_none=True)
    assert len(dumped["messages"]) == 4
    assert dumped["messages"][2]["tool_calls"][0]["function"]["name"] == "weather"
    assert dumped["messages"][3]["role"] == "tool"


def test_model_passthrough_in_request():
    """Verify model name passes through unchanged for both routes."""
    from unified_api.models import AnthropicMessageRequest, OpenAIChatRequest
    from unified_api.converters.request import convert_request

    # Anthropic route
    anth_req = AnthropicMessageRequest(
        model="my-custom-model-v2",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
    )
    oai = convert_request(anth_req, "my-custom-model-v2")
    assert oai.model == "my-custom-model-v2"

    # OpenAI route (model preserved directly)
    oai_req = OpenAIChatRequest(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert oai_req.model == "gpt-4o-mini"


# --- KeyPool thread safety ---


def test_keypool_thread_safety():
    """KeyPool should distribute keys correctly under concurrent access."""
    import threading
    from collections import Counter

    pool = KeyPool(["key_a", "key_b", "key_c"])
    results: list[str] = []
    lock = threading.Lock()

    def get_keys(n: int):
        local: list[str] = []
        for _ in range(n):
            local.append(pool.next_key())
        with lock:
            results.extend(local)

    threads = [threading.Thread(target=get_keys, args=(30,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 120
    counter = Counter(results)
    # Each key should be selected exactly 40 times (120 / 3)
    assert counter["key_a"] == 40
    assert counter["key_b"] == 40
    assert counter["key_c"] == 40
