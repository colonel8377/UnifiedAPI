"""FastAPI app entrypoint."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from .config import KeyPool, get_config, PROJECT_ROOT
from .control.concurrency import AdmissionControl
from .control.rate_limiter import RateLimiter
from .errors import AnthropicError
from .routes import chat, messages
from .upstream.client import UpstreamClient

_LOG_DIR = PROJECT_ROOT / "log"


def _configure_logging(level: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s %(levelname)s %(name)s - %(message)s"

    # Ensure log directory exists
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(fmt))
    root_logger.addHandler(console)

    # Rotating file handler: 10MB per file, keep 5 backups
    log_file = _LOG_DIR / "unified_api.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(fmt))
    root_logger.addHandler(file_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    _configure_logging(config.logging.level)

    key_pool = KeyPool(config.upstream.api_keys)
    upstream = UpstreamClient(config.upstream, retry_config=config.retry, key_pool=key_pool)
    await upstream.start()
    app.state.upstream_client = upstream

    app.state.rate_limiter = RateLimiter(
        global_rpm=config.limits.global_rpm,
        per_client_rpm=config.limits.per_client_rpm,
    )
    app.state.admission = AdmissionControl(
        max_concurrent=config.limits.global_concurrency,
        max_per_client=config.limits.per_client_concurrency,
        max_waiting=config.limits.queue_max_size,
    )

    logging.getLogger(__name__).info(
        "UnifiedAPI starting: upstream=%s, key_pool_size=%d, "
        "routes=[/v1/messages, /v1/chat/completions], "
        "global_concurrency=%d, global_rpm=%d",
        config.upstream.base_url,
        key_pool.size,
        config.limits.global_concurrency,
        config.limits.global_rpm,
    )
    try:
        yield
    finally:
        await upstream.aclose()


# Paths that do not require authentication
_PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}


def _check_auth(request: Request, password: str) -> bool:
    """Accept password from either x-api-key OR Authorization: Bearer header.

    This ensures compatibility with Claude Code (which may use either field),
    Cursor, Codex, and other clients.
    """
    # Check x-api-key (Anthropic native)
    api_key = request.headers.get("x-api-key")
    if api_key == password:
        return True
    # Check Authorization: Bearer (OpenAI style, also used by some Anthropic clients)
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer ") and auth[7:].strip() == password:
        return True
    return False


def _auth_error_response(request: Request) -> JSONResponse:
    """Return a 401 error in the protocol's native format."""
    if request.url.path.startswith("/v1/messages"):
        return JSONResponse(
            status_code=401,
            content={
                "type": "error",
                "error": {"type": "authentication_error", "message": "Invalid or missing API key"},
            },
        )
    return JSONResponse(
        status_code=401,
        content={"error": {"message": "Invalid or missing API key", "type": "authentication_error", "code": None}},
    )


app = FastAPI(
    title="UnifiedAPI",
    description="Anthropic & OpenAI protocol conversion gateway",
    version="0.2.0",
    lifespan=lifespan,
)


# Register auth middleware after lifespan so config is loaded
@app.middleware("http")
async def auth_middleware(request: Request, call_next: RequestResponseEndpoint) -> Response:
    config = get_config()
    password = config.auth.password
    if request.url.path in _PUBLIC_PATHS:
        return await call_next(request)
    if not password:
        return await call_next(request)
    if not _check_auth(request, password):
        return _auth_error_response(request)
    return await call_next(request)


@app.exception_handler(AnthropicError)
async def anthropic_error_handler(request: Request, exc: AnthropicError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": "error",
            "error": {"type": exc.error_type, "message": exc.message},
        },
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


app.include_router(messages.router)
app.include_router(chat.router)
