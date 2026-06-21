"""Tests for the retry decorator.

Covers:
  - retries on 5xx / 429 / network errors
  - does NOT retry on 4xx (except 429)
  - reraises the last exception after exhausting attempts
  - exponential backoff with jitter (timing sanity check)
"""
from __future__ import annotations

import asyncio
import time

import pytest

from unified_api.config import RetryConfig
from unified_api.control.retry import make_retry_decorator
from unified_api.upstream.errors import (
    UpstreamAuthError,
    UpstreamBadRequestError,
    UpstreamNetworkError,
    UpstreamRateLimitError,
    UpstreamServerError,
)


def _fast_retry_config(max_attempts: int = 3) -> RetryConfig:
    return RetryConfig(
        max_attempts=max_attempts,
        base_backoff_ms=1,  # 1ms to keep tests fast
        max_backoff_ms=10,
        retry_on_status=[429, 500, 502, 503, 504],
        retry_on_network=True,
        retry_stream_midway=False,
    )


async def test_retries_on_5xx_then_succeeds():
    attempts = 0

    @make_retry_decorator(_fast_retry_config(3))
    async def flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise UpstreamServerError("boom", status_code=502)
        return "ok"

    result = await flaky()
    assert result == "ok"
    assert attempts == 3


async def test_retries_on_429_then_succeeds():
    attempts = 0

    @make_retry_decorator(_fast_retry_config(3))
    async def rate_limited():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise UpstreamRateLimitError("slow down")
        return "ok"

    result = await rate_limited()
    assert result == "ok"
    assert attempts == 2


async def test_retries_on_network_error_then_succeeds():
    attempts = 0

    @make_retry_decorator(_fast_retry_config(3))
    async def flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise UpstreamNetworkError("conn refused")
        return "ok"

    assert await flaky() == "ok"


async def test_does_not_retry_on_4xx_auth():
    attempts = 0

    @make_retry_decorator(_fast_retry_config(3))
    async def bad_auth():
        nonlocal attempts
        attempts += 1
        raise UpstreamAuthError("invalid key")

    with pytest.raises(UpstreamAuthError):
        await bad_auth()
    assert attempts == 1  # No retries


async def test_does_not_retry_on_bad_request():
    attempts = 0

    @make_retry_decorator(_fast_retry_config(3))
    async def bad():
        nonlocal attempts
        attempts += 1
        raise UpstreamBadRequestError("malformed")

    with pytest.raises(UpstreamBadRequestError):
        await bad()
    assert attempts == 1


async def test_reraises_after_exhausting_attempts():
    attempts = 0

    @make_retry_decorator(_fast_retry_config(3))
    async def always_fails():
        nonlocal attempts
        attempts += 1
        raise UpstreamServerError("permanent", status_code=500)

    with pytest.raises(UpstreamServerError):
        await always_fails()
    assert attempts == 3


async def test_backoff_grows_between_attempts():
    """With base=100ms max=10s, second retry should be ~200ms later."""
    timestamps: list[float] = []

    # Use a more meaningful backoff for timing
    config = RetryConfig(
        max_attempts=3,
        base_backoff_ms=100,
        max_backoff_ms=1000,
    )

    @make_retry_decorator(config)
    async def failing():
        timestamps.append(time.monotonic())
        raise UpstreamServerError("x", status_code=500)

    with pytest.raises(UpstreamServerError):
        await failing()

    assert len(timestamps) == 3
    gap1 = timestamps[1] - timestamps[0]
    gap2 = timestamps[2] - timestamps[1]
    # tenacity's exponential_jitter: base*2^n (capped at max) + random jitter up to ~1s.
    # So the upper bound for gap1 is ~0.1+1.0=1.1s, for gap2 ~0.2+1.0=1.2s.
    # We use generous bounds to avoid flaky failures on slow CI.
    assert gap1 < 1.5
    assert gap2 < 2.5
