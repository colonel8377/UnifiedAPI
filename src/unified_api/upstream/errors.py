"""Typed exceptions for upstream interactions.

These get translated to AnthropicError at the route boundary so callers
don't need to know about the upstream's mixed error formats.
"""
from __future__ import annotations


class UpstreamError(Exception):
    """Base class for upstream errors."""

    def __init__(self, message: str, *, status_code: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class UpstreamAuthError(UpstreamError):
    """API key invalid or not authorized."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=401, retryable=False)


class UpstreamPermissionError(UpstreamError):
    """Key works but not allowed to use this model."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=403, retryable=False)


class UpstreamRateLimitError(UpstreamError):
    """429 from upstream."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message, status_code=429, retryable=True)
        self.retry_after = retry_after


class UpstreamBadRequestError(UpstreamError):
    """Upstream rejected the request body."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400, retryable=False)


class UpstreamServerError(UpstreamError):
    """5xx from upstream."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message, status_code=status_code, retryable=True)


class UpstreamNetworkError(UpstreamError):
    """Connection / timeout / DNS — no response received."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=None, retryable=True)
