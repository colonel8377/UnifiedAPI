"""Async client for the upstream OpenAI-compatible service.

Wraps httpx with:
  - Shared connection pool
  - Normalization of the upstream's mixed error formats (HTTP 200 + body error)
  - Both non-streaming and streaming chat completion methods
  - Non-streaming calls are automatically retried (rate limit, 5xx, network)
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from ..config import KeyPool, RetryConfig, UpstreamConfig
from ..control.retry import make_retry_decorator
from ..models import OpenAIChatRequest, OpenAIChatResponse
from .errors import (
    UpstreamAuthError,
    UpstreamBadRequestError,
    UpstreamError,
    UpstreamNetworkError,
    UpstreamPermissionError,
    UpstreamRateLimitError,
    UpstreamServerError,
)

logger = logging.getLogger(__name__)


class UpstreamClient:
    """HTTP client for the upstream /chat/completions endpoint."""

    def __init__(
        self,
        config: UpstreamConfig,
        retry_config: RetryConfig | None = None,
        key_pool: KeyPool | None = None,
    ) -> None:
        self._config = config
        self._retry_config = retry_config
        self._key_pool = key_pool or KeyPool(config.api_keys)
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            base_url=self._config.base_url,
            timeout=httpx.Timeout(self._config.timeout_seconds, connect=10.0),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=50,
                keepalive_expiry=30,
            ),
            headers={"Content-Type": "application/json"},
        )

    def _auth_headers(self) -> dict[str, str]:
        """Return per-request Authorization header from the key pool."""
        return {"Authorization": f"Bearer {self._key_pool.next_key()}"}

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("UpstreamClient.start() must be called before use")
        return self._client

    async def chat(self, request: OpenAIChatRequest) -> OpenAIChatResponse:
        """Non-streaming chat completion with automatic retry."""
        if self._retry_config is not None:
            decorator = make_retry_decorator(self._retry_config)
            # Apply decorator to a fresh coroutine function so retry can wrap it.
            return await decorator(self._chat_once)(request)
        return await self._chat_once(request)

    async def _chat_once(self, request: OpenAIChatRequest) -> OpenAIChatResponse:
        payload = request.model_dump(exclude_none=True)
        try:
            resp = await self.client.post(
                "/chat/completions", json=payload, headers=self._auth_headers()
            )
        except httpx.NetworkError as e:
            raise UpstreamNetworkError(f"Network error calling upstream: {e}") from e
        except httpx.TimeoutException as e:
            raise UpstreamNetworkError(f"Timeout calling upstream: {e}") from e

        body = _safe_json(resp)
        if body is not None and _looks_like_error(body):
            raise _error_from_body(body)
        if resp.status_code >= 400:
            raise _error_from_status(resp.status_code, body)
        if body is None:
            raise UpstreamError(f"Upstream returned non-JSON body (HTTP {resp.status_code})")
        return OpenAIChatResponse.model_validate(body)

    async def chat_stream(self, request: OpenAIChatRequest) -> AsyncIterator[dict[str, Any]]:
        """Streaming chat completion.

        Yields parsed delta dicts from each `data: {...}` SSE frame.
        Upstream sends NO `data: [DONE]` terminator — iteration ends when
        the HTTP body ends.

        NOTE: This method is an async generator and cannot be retried mid-
        stream. The caller is responsible for retrying the FIRST chunk
        acquisition (before any SSE bytes have been emitted to the client).
        """
        payload = request.model_dump(exclude_none=True)
        payload["stream"] = True
        # Request usage from upstream (silently ignored if unsupported)
        payload["stream_options"] = {"include_usage": True}
        try:
            async with self.client.stream(
                "POST", "/chat/completions", json=payload, headers=self._auth_headers()
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    try:
                        parsed = json.loads(body)
                    except json.JSONDecodeError:
                        parsed = None
                    raise _error_from_status(resp.status_code, parsed)

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if not data_str:
                        continue
                    if data_str == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning("Skipping unparseable SSE line: %r", data_str)
                        continue
                    if _looks_like_error(chunk):
                        raise _error_from_body(chunk)
                    yield chunk
        except httpx.NetworkError as e:
            raise UpstreamNetworkError(f"Network error during stream: {e}") from e
        except httpx.TimeoutException as e:
            raise UpstreamNetworkError(f"Timeout during stream: {e}") from e


# --- helpers ---


def _safe_json(resp: httpx.Response) -> dict[str, Any] | None:
    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _looks_like_error(body: dict[str, Any]) -> bool:
    if "error" in body and isinstance(body["error"], (dict, str)):
        return True
    if "code" in body and "msg" in body and isinstance(body.get("code"), int):
        return True
    # FastAPI / HKUST upstream may return {"detail": "..."}
    if "detail" in body and isinstance(body["detail"], (str, dict, list)):
        return True
    return False


def _error_from_body(body: dict[str, Any]) -> UpstreamError:
    err = body.get("error")
    if isinstance(err, dict):
        message = str(err.get("message") or "upstream error")
        code = err.get("code")
        etype = err.get("type", "")
        status = _coerce_status(code) or _status_from_msg(message)
        return _typed_error(status, message, etype)

    if isinstance(err, str) and err:
        status = _status_from_msg(err)
        return _typed_error(status, err, "")

    if "code" in body and "msg" in body:
        message = str(body.get("msg") or "upstream error")
        status = _coerce_status(body.get("code"))
        return _typed_error(status, message, str(body.get("type", "")))

    # FastAPI / HKUST style: {"detail": "..."}
    detail = body.get("detail")
    if isinstance(detail, str) and detail:
        status = _status_from_msg(detail)
        return _typed_error(status, detail, "")
    if isinstance(detail, dict):
        message = str(detail.get("message") or detail.get("msg") or "upstream error")
        code = detail.get("code")
        etype = detail.get("type", "")
        status = _coerce_status(code) or _status_from_msg(message)
        return _typed_error(status, message, etype)

    return UpstreamError(f"Unrecognized upstream error shape: {body!r}")


def _error_from_status(status_code: int, body: dict[str, Any] | None) -> UpstreamError:
    msg = "upstream error"
    if body and isinstance(body.get("error"), dict):
        msg = str(body["error"].get("message") or msg)
    elif body and isinstance(body.get("error"), str):
        msg = str(body["error"])
    elif body and "msg" in body:
        msg = str(body["msg"])
    elif body and isinstance(body.get("detail"), str):
        msg = str(body["detail"])
    return _typed_error(status_code, msg, "")


def _coerce_status(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _status_from_msg(message: str) -> int | None:
    m = message.lower()
    if "unauthorized" in m or "invalid api key" in m:
        return 401
    if "not allowed to access model" in m or "permission" in m:
        return 403
    if "rate limit" in m or "too many" in m:
        return 429
    return None


def _typed_error(status: int | None, message: str, etype: str) -> UpstreamError:
    if status == 401:
        return UpstreamAuthError(message)
    if status == 403:
        return UpstreamPermissionError(message)
    if status == 429:
        return UpstreamRateLimitError(message)
    if status == 400:
        return UpstreamBadRequestError(message)
    if status is not None and 500 <= status < 600:
        return UpstreamServerError(message, status_code=status)
    return UpstreamError(message, status_code=status, retryable=False)
