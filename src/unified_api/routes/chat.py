"""POST /v1/chat/completions — OpenAI Chat Completions pass-through endpoint.

Pipeline:
  1. Parse OpenAI request (model passes through unchanged)
  2. Acquire admission slot (queue + concurrency, per-client + global)
  3. Acquire rate-limit token (per-client + global RPM)
  4. Call upstream (non-stream: auto-retry; stream: retry first chunk only)
  5. Return OpenAI response directly (non-stream or SSE)
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..control.concurrency import AdmissionControl, OverloadedError
from ..control.rate_limiter import RateLimiter
from ..control.retry import make_retry_decorator
from ..models import OpenAIChatRequest
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


@router.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatRequest, http_request: Request):
    client_id = _extract_client_id(http_request)
    admission: AdmissionControl = http_request.app.state.admission
    rate_limiter: RateLimiter = http_request.app.state.rate_limiter

    if request.stream:
        return await _stream_response(http_request, request, client_id, admission, rate_limiter)
    return await _nonstream_response(http_request, request, client_id, admission, rate_limiter)


# --- non-streaming path ---


async def _nonstream_response(
    http_request: Request,
    oai_request: OpenAIChatRequest,
    client_id: str,
    admission: AdmissionControl,
    rate_limiter: RateLimiter,
) -> JSONResponse:
    client: UpstreamClient = http_request.app.state.upstream_client

    try:
        async with admission.slot(client_id):
            await rate_limiter.acquire(client_id)
            oai_response = await client.chat(oai_request)
    except OverloadedError as e:
        return _openai_error(503, "server_error", str(e))
    except UpstreamAuthError as e:
        return _openai_error(401, "authentication_error", str(e))
    except UpstreamPermissionError as e:
        return _openai_error(403, "permission_error", str(e))
    except UpstreamRateLimitError as e:
        return _openai_error(429, "rate_limit_error", str(e))
    except UpstreamBadRequestError as e:
        return _openai_error(400, "invalid_request_error", str(e))
    except UpstreamServerError as e:
        return _openai_error(502, "server_error", f"Upstream error: {e}")
    except UpstreamNetworkError as e:
        return _openai_error(503, "server_error", f"Cannot reach upstream: {e}")
    except UpstreamError as e:
        return _openai_error(502, "server_error", str(e))

    return JSONResponse(content=oai_response.model_dump())


# --- streaming path ---


async def _stream_response(
    http_request: Request,
    oai_request: OpenAIChatRequest,
    client_id: str,
    admission: AdmissionControl,
    rate_limiter: RateLimiter,
) -> StreamingResponse:
    """Open the upstream stream with retry on first chunk, then commit to a
    StreamingResponse that drains the rest.
    """
    from ..config import get_config

    client: UpstreamClient = http_request.app.state.upstream_client
    config = get_config()

    # Acquire admission + rate-limit tokens BEFORE peeking
    try:
        admission_ctx = admission.slot(client_id)
        await admission_ctx.__aenter__()
    except OverloadedError as e:
        return _openai_error(503, "server_error", str(e))
    await rate_limiter.acquire(client_id)

    try:
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
            await admission_ctx.__aexit__(None, None, None)
            return _openai_error(401, "authentication_error", str(e))
        except UpstreamPermissionError as e:
            await admission_ctx.__aexit__(None, None, None)
            return _openai_error(403, "permission_error", str(e))
        except UpstreamRateLimitError as e:
            await admission_ctx.__aexit__(None, None, None)
            return _openai_error(429, "rate_limit_error", str(e))
        except UpstreamBadRequestError as e:
            await admission_ctx.__aexit__(None, None, None)
            return _openai_error(400, "invalid_request_error", str(e))
        except UpstreamServerError as e:
            await admission_ctx.__aexit__(None, None, None)
            return _openai_error(502, "server_error", f"Upstream error: {e}")
        except UpstreamNetworkError as e:
            await admission_ctx.__aexit__(None, None, None)
            return _openai_error(503, "server_error", f"Cannot reach upstream: {e}")
        except UpstreamError as e:
            await admission_ctx.__aexit__(None, None, None)
            return _openai_error(502, "server_error", str(e))
    except Exception:
        await admission_ctx.__aexit__(None, None, None)
        raise

    async def event_gen() -> AsyncIterator[bytes]:
        try:
            if first_chunk is not None:
                yield _oai_sse(first_chunk)
                if upstream_gen is not None:
                    async for chunk in upstream_gen:
                        yield _oai_sse(chunk)
            yield b"data: [DONE]\n\n"
        except UpstreamError as e:
            logger.warning("Upstream error mid-stream: %s", e)
            yield _oai_sse_error(f"Upstream stream error: {e}")
        finally:
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


def _extract_client_id(http_request: Request) -> str:
    """Per-client identity for rate limit / concurrency — uses client IP."""
    forwarded = http_request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return http_request.client.host if http_request.client else "unknown"


def _openai_error(status_code: int, error_type: str, message: str) -> JSONResponse:
    """Return an OpenAI-format error response."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": None}},
    )


def _oai_sse(chunk: dict[str, Any]) -> bytes:
    """Format one OpenAI-style SSE data frame."""
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")


def _oai_sse_error(message: str) -> bytes:
    """Format an OpenAI SSE error frame."""
    payload = {"error": {"message": message, "type": "server_error"}}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
