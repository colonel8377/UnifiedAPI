"""Retry wrapper for upstream calls.

Builds a tenacity decorator from runtime config so retry policy can be
loaded from config.yaml. The decorator is applied per-call so we can
disable retry for streaming's mid-stream path.

Policy:
  - retry on: 429, 5xx, network errors
  - exponential backoff with jitter
  - max_attempts total (including the first try)
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable, TypeVar

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ..config import RetryConfig
from ..upstream.errors import (
    UpstreamNetworkError,
    UpstreamRateLimitError,
    UpstreamServerError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exceptions that warrant a retry
RETRYABLE_EXCEPTIONS = (
    UpstreamRateLimitError,
    UpstreamServerError,
    UpstreamNetworkError,
)


def make_retry_decorator(config: RetryConfig) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Build a tenacity retry decorator from runtime config."""
    return retry(
        stop=stop_after_attempt(config.max_attempts),
        wait=wait_exponential_jitter(
            initial=config.base_backoff_ms / 1000.0,
            max=config.max_backoff_ms / 1000.0,
        ),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
