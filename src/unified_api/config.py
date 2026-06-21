"""Configuration loader.

Loads `.env` first (via python-dotenv), then `config.yaml` with `${VAR}` env
interpolation. Result is a singleton validated against a Pydantic model.
"""
from __future__ import annotations

import itertools
import os
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")

# Project root: this file is at src/unified_api/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class UpstreamConfig(BaseModel):
    base_url: str
    api_key: str = ""
    api_keys: list[str] = Field(default_factory=list)
    timeout_seconds: int = 300

    @model_validator(mode="after")
    def _ensure_keys(self) -> "UpstreamConfig":
        if not self.api_keys:
            if self.api_key:
                self.api_keys = [self.api_key]
            else:
                raise ValueError("UpstreamConfig requires 'api_key' or 'api_keys'")
        return self

    @property
    def chat_completions_url(self) -> str:
        """Full URL for the /chat/completions endpoint."""
        base = self.base_url.rstrip("/")
        return f"{base}/chat/completions"


class KeyPool:
    """Thread-safe round-robin key selector."""

    def __init__(self, keys: list[str]) -> None:
        if not keys:
            raise ValueError("KeyPool requires at least one key")
        self._keys = keys
        self._cycle = itertools.cycle(keys)
        self._lock = threading.Lock()

    def next_key(self) -> str:
        with self._lock:
            return next(self._cycle)

    @property
    def size(self) -> int:
        return len(self._keys)


class LimitsConfig(BaseModel):
    global_concurrency: int = 10
    global_rpm: int = 60
    per_client_concurrency: int = 5
    per_client_rpm: int = 30
    queue_max_size: int = 100


class RetryConfig(BaseModel):
    max_attempts: int = 3
    base_backoff_ms: int = 500
    max_backoff_ms: int = 10000
    retry_on_status: list[int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])
    retry_on_network: bool = True
    retry_stream_midway: bool = False


class ThinkingConfig(BaseModel):
    return_by_default: bool = False
    return_when_client_enables: bool = True


class AuthConfig(BaseModel):
    password: str = ""


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    upstream: UpstreamConfig
    auth: AuthConfig = Field(default_factory=AuthConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    thinking: ThinkingConfig = Field(default_factory=ThinkingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _interpolate_env(text: str) -> str:
    """Replace ${VAR} with os.environ[VAR]. Raises if missing."""
    def _sub(match: re.Match[str]) -> str:
        var = match.group(1)
        val = os.environ.get(var)
        if val is None:
            raise KeyError(f"Environment variable '{var}' referenced in config.yaml is not set")
        return val

    return _ENV_VAR_PATTERN.sub(_sub, text)


def _load_raw_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    interpolated = _interpolate_env(raw)
    data = yaml.safe_load(interpolated)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} did not parse to a dict")
    return data


@lru_cache(maxsize=1)
def get_config(config_path: Path | None = None) -> AppConfig:
    """Load and cache the application config.

    Loads `.env` from project root first so its values are available for
    `${VAR}` interpolation in config.yaml.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    path = config_path or (PROJECT_ROOT / "config.yaml")
    data = _load_raw_yaml(path)
    return AppConfig.model_validate(data)


def reset_config_cache() -> None:
    """Drop cached config (used in tests)."""
    get_config.cache_clear()
