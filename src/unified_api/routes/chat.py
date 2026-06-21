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
from ..tools.prompt_builder import build_tools_prompt
from ..tools.xml_parser import IncrementalXmlScanner, TextSegment, ToolUseSegment, parse_complete
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

    # Inject XML tool definitions into system prompt so DeepSeek can use tools
    augmented = _inject_tools_prompt(request)
    # Normalize multi-turn history: convert tool messages + assistant tool_calls to XML
    augmented = _normalize_oai_messages(augmented)

    if augmented.stream:
        return await _stream_response(http_request, augmented, client_id, admission, rate_limiter)
    return await _nonstream_response(http_request, augmented, client_id, admission, rate_limiter)


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

    return JSONResponse(content=_postprocess_tool_calls(oai_response.model_dump()))


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
        import uuid as _uuid
        scanner = IncrementalXmlScanner()
        tool_call_index = 0
        has_tool_calls = False
        try:
            if first_chunk is not None:
                # Process first chunk
                for ev in _process_stream_chunk(first_chunk, scanner, tool_call_index, has_tool_calls):
                    if ev.get("_is_tool"):
                        tool_call_index = ev["_next_index"]
                        has_tool_calls = True
                        yield _oai_sse(ev["chunk"])
                    else:
                        yield _oai_sse(ev.get("chunk", ev))
                if upstream_gen is not None:
                    async for chunk in upstream_gen:
                        for ev in _process_stream_chunk(chunk, scanner, tool_call_index, has_tool_calls):
                            if ev.get("_is_tool"):
                                tool_call_index = ev["_next_index"]
                                has_tool_calls = True
                                yield _oai_sse(ev["chunk"])
                            else:
                                yield _oai_sse(ev.get("chunk", ev))
            # Flush scanner for any remaining content
            for seg in scanner.flush():
                if isinstance(seg, TextSegment) and seg.text:
                    yield _oai_sse({
                        "choices": [{"delta": {"content": seg.text}, "index": 0}]
                    })
                elif isinstance(seg, ToolUseSegment):
                    tc_id = f"call_{_uuid.uuid4().hex[:24]}"
                    yield _oai_sse({
                        "choices": [{"delta": {"tool_calls": [{
                            "index": tool_call_index,
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": seg.name, "arguments": json.dumps(seg.params, ensure_ascii=False)},
                        }]}, "index": 0}]
                    })
                    tool_call_index += 1
                    has_tool_calls = True
            # Emit finish
            finish = "tool_calls" if has_tool_calls else "stop"
            yield _oai_sse({"choices": [{"delta": {}, "finish_reason": finish, "index": 0}]})
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


def _inject_tools_prompt(request: OpenAIChatRequest) -> OpenAIChatRequest:
    """Inject XML tool definitions into the system prompt for upstream models
    that don't natively support OpenAI tool calling (e.g. DeepSeek on HKUST).

    Converts OpenAI-format tools to Anthropic-format and uses build_tools_prompt
    to generate the XML instruction block, then prepends it to messages.
    """
    if not request.tools:
        return request

    # Convert OpenAI tools → Anthropic format for build_tools_prompt
    anth_tools: list[dict[str, Any]] = []
    for tc in request.tools:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        anth_tool: dict[str, Any] = {"name": fn.get("name", "")}
        if fn.get("description"):
            anth_tool["description"] = fn["description"]
        if fn.get("parameters"):
            anth_tool["input_schema"] = fn["parameters"]
        anth_tools.append(anth_tool)

    tool_prompt = build_tools_prompt(anth_tools)
    if not tool_prompt:
        return request

    # Prepend/inject into system message
    messages = list(request.messages)
    injected = False
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "system":
            existing = msg.get("content", "")
            new_content = f"{existing}\n\n{tool_prompt}" if existing else tool_prompt
            messages[i] = {**msg, "content": new_content}
            injected = True
            break
    if not injected:
        messages.insert(0, {"role": "system", "content": tool_prompt})

    return request.model_copy(update={"messages": messages})


def _normalize_oai_messages(request: OpenAIChatRequest) -> OpenAIChatRequest:
    """Normalize OpenAI-format conversation history for upstream models that
    don't understand native OpenAI tool calling.

    Converts:
      - ``role: "tool"`` → ``role: "user"`` with ``<tool_result>`` XML
      - assistant ``tool_calls`` → appended XML ``<function_calls>`` in content

    The upstream (DeepSeek) was prompted with XML tool format via system prompt,
    so it needs to see tool history in the same XML shape.
    """
    messages = list(request.messages)
    if not any(
        (isinstance(m, dict) and m.get("role") in ("tool", "assistant") and m.get("tool_calls"))
        for m in messages
    ):
        return request  # nothing to normalize

    from ..tools.prompt_builder import _render_invokes

    normalized: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            normalized.append(msg)
            continue
        role = msg.get("role")

        if role == "tool":
            # Convert tool result to XML user message
            tool_use_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            xml = (
                f'<tool_result tool_use_id="{tool_use_id}">'
                f"\n{content}\n"
                f"</tool_result>"
            )
            normalized.append({"role": "user", "content": xml})

        elif role == "assistant" and msg.get("tool_calls"):
            # Convert assistant tool_calls to XML appended to content
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = "".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            invokes: list[tuple[str, dict[str, Any]]] = []
            for tc in msg.get("tool_calls", []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, TypeError):
                    args = {}
                if name and isinstance(args, dict):
                    invokes.append((name, args))
            if invokes:
                xml = _render_invokes(invokes)
                content = f"{content}\n{xml}" if content else xml
            # Emit without tool_calls field — upstream doesn't understand it
            new_msg: dict[str, Any] = {"role": "assistant"}
            if content:
                new_msg["content"] = content
            normalized.append(new_msg)

        else:
            normalized.append(msg)

    return request.model_copy(update={"messages": normalized})


def _postprocess_tool_calls(resp: dict[str, Any]) -> dict[str, Any]:
    """Parse XML <function_calls> from the model's text content and convert
    them into native OpenAI tool_calls.

    This is needed because the upstream (DeepSeek) emits XML-format tool calls
    in its text content rather than using native OpenAI tool_calls.
    """
    choices = resp.get("choices") or []
    if not choices:
        return resp
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message")
    if not isinstance(message, dict):
        return resp
    content = message.get("content")
    if not isinstance(content, str) or not content:
        return resp
    # Check for XML tool calls
    segments = parse_complete(content)
    tool_uses = [s for s in segments if isinstance(s, ToolUseSegment)]
    if not tool_uses:
        return resp
    # Build native tool_calls
    import uuid
    tool_calls: list[dict[str, Any]] = []
    for tu in tool_uses:
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": tu.name,
                "arguments": json.dumps(tu.params, ensure_ascii=False),
            },
        })
    # Rebuild content text without XML blocks
    text_parts = [s.text for s in segments if isinstance(s, TextSegment)]
    clean_text = "".join(text_parts).strip()
    message["content"] = clean_text if clean_text else None
    message["tool_calls"] = tool_calls
    choice["finish_reason"] = "tool_calls"
    return resp


def _openai_error(status_code: int, error_type: str, message: str) -> JSONResponse:
    """Return an OpenAI-format error response."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": None}},
    )


def _process_stream_chunk(
    chunk: dict[str, Any],
    scanner: IncrementalXmlScanner,
    tool_call_index: int,
    has_tool_calls: bool,
) -> list[dict[str, Any]]:
    """Process one OpenAI streaming chunk: extract content, scan for XML tool calls,
    return a list of events to emit.

    Returns dicts with either:
      - {"chunk": <openai_chunk>} for pass-through events
      - {"chunk": <openai_chunk>, "_is_tool": True, "_next_index": N} for tool calls
    """
    import uuid as _uuid
    events: list[dict[str, Any]] = []
    choices = chunk.get("choices") or []
    if not choices:
        events.append({"chunk": chunk})
        return events
    choice = choices[0] if isinstance(choices[0], dict) else {}
    delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
    content = delta.get("content")
    finish_reason = choice.get("finish_reason")

    # Preserve chunk metadata (id, model, created, object) for client compatibility
    meta = {k: v for k, v in chunk.items() if k not in ("choices",)}

    # If there's content, scan for XML tool calls
    if isinstance(content, str) and content:
        for seg in scanner.feed(content):
            if isinstance(seg, TextSegment) and seg.text:
                events.append({"chunk": {
                    **meta,
                    "choices": [{"delta": {"content": seg.text}, "index": 0}]
                }})
            elif isinstance(seg, ToolUseSegment):
                tc_id = f"call_{_uuid.uuid4().hex[:24]}"
                events.append({
                    "chunk": {
                        **meta,
                        "choices": [{"delta": {"tool_calls": [{
                            "index": tool_call_index,
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": seg.name, "arguments": json.dumps(seg.params, ensure_ascii=False)},
                        }]}, "index": 0}]
                    },
                    "_is_tool": True,
                    "_next_index": tool_call_index + 1,
                })
                tool_call_index += 1
                has_tool_calls = True
    elif not finish_reason:
        # Pass through non-content chunks (role, etc.) but skip finish_reason
        # (we emit our own finish_reason at the end)
        events.append({"chunk": chunk})
    return events


def _oai_sse(chunk: dict[str, Any]) -> bytes:
    """Format one OpenAI-style SSE data frame."""
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")


def _oai_sse_error(message: str) -> bytes:
    """Format an OpenAI SSE error frame."""
    payload = {"error": {"message": message, "type": "server_error"}}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
