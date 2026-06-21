"""POST /v1/messages — Anthropic Messages API endpoint.

Pipeline:
  1. Convert Anthropic → OpenAI request (model passes through)
  2. Acquire admission slot (queue + concurrency, per-client + global)
  3. Acquire rate-limit token (per-client + global RPM)
  4. Call upstream (non-stream: auto-retry; stream: retry first chunk only)
  5. Convert OpenAI → Anthropic response (non-stream or SSE)
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..config import AppConfig, get_config
from ..control.concurrency import AdmissionControl, OverloadedError
from ..control.rate_limiter import RateLimiter
from ..control.retry import make_retry_decorator
from ..converters.request import convert_request
from ..converters.response import convert_response
from ..converters.stream import StreamConverter, sse
from ..errors import AnthropicError, ConversionError
from ..models import AnthropicMessageRequest
from ..upstream.client import UpstreamClient
from ..upstream.errors import (
    UpstreamAuthError,
    UpstreamBadRequestError,
    UpstreamError,
    UpstreamNetworkError,
    UpstreamPermissionError,
    UpstreamRateLimitError,
    UpstreamServerError,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# --- TEMPORARY DEBUG INSTRUMENTATION (remove after diagnosing) ---
# Enable with env var UAPI_DEBUG=1. Writes JSONL to /tmp/uapi_debug.jsonl.
_DEBUG = os.environ.get("UAPI_DEBUG") == "1"
_DEBUG_PATH = Path("/tmp/uapi_debug.jsonl")


def _debug_dump(record: dict[str, Any]) -> None:
    if not _DEBUG:
        return
    record["ts"] = time.time()
    with _DEBUG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _capture_chunk(stats: dict[str, Any], chunk: dict[str, Any]) -> None:
    """Tally upstream chunk contents into stats (called only when UAPI_DEBUG=1)."""
    stats["chunks"] += 1
    choices = chunk.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return
    delta = choices[0].get("delta") or {}
    rc = delta.get("reasoning_content") or delta.get("reasoning") or ""
    if rc:
        stats["rc_chars"] += len(rc)
    c = delta.get("content")
    if isinstance(c, str):
        stats["content_chars"] += len(c)
    tcs = delta.get("tool_calls")
    if isinstance(tcs, list):
        stats["tool_delta_chars"] += len(json.dumps(tcs))
    fr = choices[0].get("finish_reason")
    if fr:
        stats["finish_reason"] = fr


def _summarize_anthropic_request(req: AnthropicMessageRequest) -> dict[str, Any]:
    """Capture key fields without dumping full content (could be huge)."""
    sys_size = 0
    if isinstance(req.system, str):
        sys_size = len(req.system)
    elif isinstance(req.system, list):
        sys_size = sum(len(b.get("text") or "") for b in req.system if isinstance(b, dict))
    msg_summary = []
    for m in req.messages:
        c = m.get("content")
        if isinstance(c, str):
            msg_summary.append({"role": m.get("role"), "chars": len(c), "shape": "str"})
        elif isinstance(c, list):
            msg_summary.append({"role": m.get("role"), "blocks": len(c), "shape": "list",
                                "types": [b.get("type") for b in c if isinstance(b, dict)]})
        else:
            msg_summary.append({"role": m.get("role"), "shape": type(c).__name__})
    return {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "stream": req.stream,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "thinking": getattr(req, "thinking", None),
        "system_chars": sys_size,
        "tools_count": len(req.tools) if req.tools else 0,
        "tool_choice": req.tool_choice,
        "messages": msg_summary,
        "extra_keys": [k for k in (req.__dict__ if hasattr(req, "__dict__") else {})
                       if k not in ("model", "max_tokens", "stream", "temperature", "top_p",
                                    "thinking", "system", "tools", "tool_choice", "messages",
                                    "stop_sequences", "top_k", "metadata")],
    }


@router.post("/v1/messages")
async def create_message(request: AnthropicMessageRequest, http_request: Request):
    config: AppConfig = get_config()

    _debug_dump({"event": "anthropic_request", **_summarize_anthropic_request(request),
                 "headers": {k: v for k, v in http_request.headers.items()
                             if k.lower() in ("x-api-key", "authorization", "anthropic-version",
                                              "user-agent", "x-forwarded-for")}})

    try:
        oai_request = convert_request(request, request.model)
    except ConversionError as e:
        raise AnthropicError(e.status_code, e.error_type, e.message) from e

    _debug_dump({
        "event": "openai_request_to_upstream",
        "model": oai_request.model,
        "max_tokens": oai_request.max_tokens,
        "messages_count": len(oai_request.messages),
        "system_chars": len(oai_request.messages[0]["content"]) if oai_request.messages and oai_request.messages[0]["role"] == "system" else 0,
        "has_tools": bool(getattr(oai_request, "tools", None)),
        "has_stream_options": "stream_options" in (oai_request.model_dump(exclude_none=True)),
        "payload_keys": sorted(oai_request.model_dump(exclude_none=True).keys()),
    })

    return_thinking = _should_return_thinking(request, config)
    client_id = _extract_client_id(http_request)
    admission: AdmissionControl = http_request.app.state.admission
    rate_limiter: RateLimiter = http_request.app.state.rate_limiter

    if request.stream:
        return await _stream_response(
            http_request, oai_request, request.model, return_thinking, client_id, admission, rate_limiter
        )
    return await _nonstream_response(
        http_request, oai_request, request.model, return_thinking, client_id, admission, rate_limiter
    )


# --- non-streaming path ---


async def _nonstream_response(
    http_request: Request,
    oai_request,
    alias: str,
    return_thinking: bool,
    client_id: str,
    admission: AdmissionControl,
    rate_limiter: RateLimiter,
) -> dict:
    client: UpstreamClient = http_request.app.state.upstream_client
    config = get_config()

    try:
        async with admission.slot(client_id):
            await rate_limiter.acquire(client_id)
            oai_response = await client.chat(oai_request)
    except OverloadedError as e:
        raise AnthropicError(503, "overloaded_error", str(e)) from e
    except UpstreamAuthError as e:
        raise AnthropicError(401, "authentication_error", str(e)) from e
    except UpstreamPermissionError as e:
        raise AnthropicError(403, "permission_error", str(e)) from e
    except UpstreamRateLimitError as e:
        raise AnthropicError(429, "rate_limit_error", str(e)) from e
    except UpstreamBadRequestError as e:
        raise AnthropicError(400, "invalid_request_error", str(e)) from e
    except UpstreamServerError as e:
        raise AnthropicError(502, "api_error", f"Upstream error: {e}") from e
    except UpstreamNetworkError as e:
        raise AnthropicError(503, "overloaded_error", f"Cannot reach upstream: {e}") from e
    except UpstreamError as e:
        raise AnthropicError(502, "api_error", str(e)) from e
    except Exception as e:
        logger.error("Unexpected error in non-streaming messages: %s", e, exc_info=True)
        raise AnthropicError(500, "api_error", f"Internal server error: {e}") from e

    _ = config  # reserved for future per-request hooks
    anthropic_response = convert_response(oai_response, alias, return_thinking)
    return anthropic_response.model_dump()


# --- streaming path ---


async def _stream_response(
    http_request: Request,
    oai_request,
    alias: str,
    return_thinking: bool,
    client_id: str,
    admission: AdmissionControl,
    rate_limiter: RateLimiter,
) -> StreamingResponse:
    """Open the upstream stream with retry on first chunk, then commit to a
    StreamingResponse that drains the rest.
    """
    client: UpstreamClient = http_request.app.state.upstream_client
    config = get_config()

    # Acquire admission + rate-limit tokens BEFORE peeking so they apply to
    # the streaming path too. Held for the entire stream duration.
    try:
        admission_ctx = admission.slot(client_id)
        await admission_ctx.__aenter__()
    except OverloadedError as e:
        raise AnthropicError(503, "overloaded_error", str(e)) from e
    await rate_limiter.acquire(client_id)

    try:
        # Peek first chunk with retry (no SSE emitted yet, so we can retry safely)
        retry_decorator = make_retry_decorator(config.retry)

        @retry_decorator
        async def _peek_first():
            gen = client.chat_stream(oai_request)
            try:
                first = await gen.__anext__()
            except StopAsyncIteration:
                gen = None
                first = None
            return gen, first

        try:
            upstream_gen, first_chunk = await _peek_first()
        except UpstreamAuthError as e:
            raise AnthropicError(401, "authentication_error", str(e)) from e
        except UpstreamPermissionError as e:
            raise AnthropicError(403, "permission_error", str(e)) from e
        except UpstreamRateLimitError as e:
            raise AnthropicError(429, "rate_limit_error", str(e)) from e
        except UpstreamBadRequestError as e:
            raise AnthropicError(400, "invalid_request_error", str(e)) from e
        except UpstreamServerError as e:
            raise AnthropicError(502, "api_error", f"Upstream error: {e}") from e
        except UpstreamNetworkError as e:
            raise AnthropicError(503, "overloaded_error", f"Cannot reach upstream: {e}") from e
        except UpstreamError as e:
            raise AnthropicError(502, "api_error", str(e)) from e
        except Exception as e:
            logger.error("Unexpected error during stream init: %s", e, exc_info=True)
            raise AnthropicError(500, "api_error", f"Internal server error: {e}") from e
    except AnthropicError:
        # Clean up admission slot before propagating
        await admission_ctx.__aexit__(None, None, None)
        raise

    converter = StreamConverter(requested_model_alias=alias, return_thinking=return_thinking)

    # Debug counters for upstream stream
    _dbg = {"chunks": 0, "rc_chars": 0, "content_chars": 0, "tool_delta_chars": 0,
            "first_chunk_at": None, "finish_reason": None, "started_at": time.time()} if _DEBUG else None

    async def event_gen() -> AsyncIterator[bytes]:
        try:
            if first_chunk is not None and upstream_gen is not None:
                if _dbg is not None:
                    _dbg["first_chunk_at"] = time.time()
                    _capture_chunk(_dbg, first_chunk)
                for ev in converter.feed(first_chunk):
                    yield ev
                try:
                    async for chunk in upstream_gen:
                        if _dbg is not None:
                            _capture_chunk(_dbg, chunk)
                        for ev in converter.feed(chunk):
                            yield ev
                except UpstreamError as e:
                    logger.warning("Upstream error mid-stream: %s", e)
                    if _dbg is not None:
                        _dbg["upstream_error"] = str(e)
                    yield sse("error", {
                        "type": "error",
                        "error": {"type": "api_error", "message": f"Upstream stream error: {e}"},
                    })
                    return
                except Exception as e:
                    logger.error("Unexpected error mid-stream: %s", e, exc_info=True)
                    if _dbg is not None:
                        _dbg["midstream_exception"] = f"{type(e).__name__}: {e}"
                    yield sse("error", {
                        "type": "error",
                        "error": {"type": "api_error", "message": f"Internal error during streaming: {e}"},
                    })
                    return
            for ev in converter.flush():
                yield ev
        except Exception as e:
            logger.error("Fatal error in stream event_gen: %s", e, exc_info=True)
            yield sse("error", {
                "type": "error",
                "error": {"type": "api_error", "message": f"Internal server error: {e}"},
            })
        finally:
            if _dbg is not None:
                _debug_dump({
                    "event": "stream_completed",
                    "upstream_chunks": _dbg["chunks"],
                    "upstream_reasoning_chars": _dbg["rc_chars"],
                    "upstream_content_chars": _dbg["content_chars"],
                    "upstream_tool_delta_chars": _dbg["tool_delta_chars"],
                    "upstream_finish_reason": _dbg["finish_reason"],
                    "converter_has_visible_content": converter._has_visible_content,
                    "converter_has_tool_use": converter._has_tool_use,
                    "converter_finish_reason": converter._finish_reason,
                    "duration_s": time.time() - _dbg["started_at"],
                })
            await admission_ctx.__aexit__(None, None, None)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- helpers ---


def _should_return_thinking(request: AnthropicMessageRequest, config: AppConfig) -> bool:
    if config.thinking.return_by_default:
        return True
    if config.thinking.return_when_client_enables:
        thinking = request.thinking
        if isinstance(thinking, dict) and thinking.get("type") == "enabled":
            return True
    return False


def _extract_client_id(http_request: Request) -> str:
    """Per-client identity for rate limit / concurrency — uses client IP."""
    forwarded = http_request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return http_request.client.host if http_request.client else "unknown"
