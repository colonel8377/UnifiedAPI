"""POST /v1/messages — Anthropic Messages API endpoint.

Pipeline:
  1. Convert Anthropic → OpenAI request (model passes through)
  2. Acquire admission slot (queue + concurrency, per-client + global)
  3. Acquire rate-limit token (per-client + global RPM)
  4. Call upstream (non-stream: auto-retry; stream: retry first chunk only)
  5. Convert OpenAI → Anthropic response (non-stream or SSE)
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

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


@router.post("/v1/messages")
async def create_message(request: AnthropicMessageRequest, http_request: Request):
    config: AppConfig = get_config()

    try:
        oai_request = convert_request(request, request.model)
    except ConversionError as e:
        raise AnthropicError(e.status_code, e.error_type, e.message) from e

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

    async def event_gen() -> AsyncIterator[bytes]:
        try:
            if first_chunk is not None and upstream_gen is not None:
                for ev in converter.feed(first_chunk):
                    yield ev
                try:
                    async for chunk in upstream_gen:
                        for ev in converter.feed(chunk):
                            yield ev
                except UpstreamError as e:
                    logger.warning("Upstream error mid-stream: %s", e)
                    yield sse("error", {
                        "type": "error",
                        "error": {"type": "api_error", "message": f"Upstream stream error: {e}"},
                    })
                    return
                except Exception as e:
                    logger.error("Unexpected error mid-stream: %s", e, exc_info=True)
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
