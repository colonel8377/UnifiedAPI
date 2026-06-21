"""Application-level errors shaped for the Anthropic API response contract."""
from __future__ import annotations


class AnthropicError(Exception):
    """Raised to short-circuit a request with an Anthropic-shaped error response.

    Caught by the global exception handler in main.py and serialized as
    `{"type":"error","error":{"type":..., "message":...}}` with the configured
    HTTP status.
    """

    def __init__(self, status_code: int, error_type: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message


class ConversionError(AnthropicError):
    """Raised by converters when the input cannot be faithfully translated."""

    def __init__(self, message: str) -> None:
        super().__init__(400, "invalid_request_error", message)
