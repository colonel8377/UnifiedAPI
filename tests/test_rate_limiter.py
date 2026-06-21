"""Tests for the token bucket rate limiter.

Covers:
  - bucket refills over time
  - blocks until token available
  - per-client isolation
  - global budget shared across clients
  - "go negative" trick allows concurrent waiters
"""
from __future__ import annotations

import asyncio
import time

import pytest

from unified_api.control.rate_limiter import RateLimiter, TokenBucket


# --- TokenBucket unit ---


async def test_bucket_starts_full():
    bucket = TokenBucket(rate_per_sec=10.0, capacity=5)
    # Should be able to acquire 5 immediately
    for _ in range(5):
        start = time.monotonic()
        await bucket.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.05  # all immediate


async def test_bucket_blocks_when_empty():
    bucket = TokenBucket(rate_per_sec=10.0, capacity=1)
    await bucket.acquire()  # drain the 1 token
    # Next acquire should block ~0.1s (1 token / 10 per sec)
    start = time.monotonic()
    await bucket.acquire()
    elapsed = time.monotonic() - start
    assert 0.05 < elapsed < 0.5


async def test_bucket_partial_consumption():
    """acquire(tokens=0.5) should only consume half a token."""
    bucket = TokenBucket(rate_per_sec=100.0, capacity=2)
    await bucket.acquire(0.5)
    await bucket.acquire(0.5)
    await bucket.acquire(0.5)
    await bucket.acquire(0.5)
    # 4 × 0.5 = 2 tokens consumed; should be exhausted now
    start = time.monotonic()
    await bucket.acquire(0.5)
    elapsed = time.monotonic() - start
    assert elapsed > 0.001  # had to wait


def test_bucket_rejects_invalid_rate():
    with pytest.raises(ValueError):
        TokenBucket(rate_per_sec=0, capacity=1)


def test_bucket_rejects_invalid_capacity():
    with pytest.raises(ValueError):
        TokenBucket(rate_per_sec=1.0, capacity=0)


async def test_concurrent_waiters_serialize_at_distinct_times():
    """The 'go negative' trick should let multiple waiters queue at distinct times."""
    # 1 token, 10/sec → each token takes 0.1s after the first
    bucket = TokenBucket(rate_per_sec=10.0, capacity=1)
    # Fire 3 concurrent acquires; first is instant, others should wait
    start = time.monotonic()
    results = await asyncio.gather(*(bucket.acquire() for _ in range(3)))
    elapsed = time.monotonic() - start
    # 3 tokens at 10/sec = ~0.2s minimum (1 free + 2 waited)
    assert elapsed >= 0.15


# --- RateLimiter (global + per-client) ---


async def test_rate_limiter_isolates_clients():
    """Each client has its own bucket."""
    rl = RateLimiter(global_rpm=100000, per_client_rpm=600)
    # Client A burns through its small budget; client B should not be affected
    # Per-client 600 RPM = 10 tokens/sec, capacity=600
    for _ in range(600):
        await rl.acquire("A")
    # B should still have its full budget — fast
    start = time.monotonic()
    for _ in range(100):
        await rl.acquire("B")
    elapsed_b = time.monotonic() - start
    assert elapsed_b < 0.5


async def test_rate_limiter_global_applies_to_all():
    """A tight global budget blocks all clients."""
    rl = RateLimiter(global_rpm=600, per_client_rpm=100000)
    # Burn through global budget (600 RPM = 10 tokens/sec, capacity=600)
    for _ in range(600):
        await rl.acquire("A")
    # B should be blocked by global (~0.1s wait for one token)
    start = time.monotonic()
    await asyncio.wait_for(rl.acquire("B"), timeout=2.0)
    elapsed = time.monotonic() - start
    assert elapsed > 0.05


async def test_per_client_bucket_reused():
    rl = RateLimiter(global_rpm=1000, per_client_rpm=1000)
    # Two acquires for same client should return the same bucket internally
    await rl.acquire("X")
    await rl.acquire("X")
    # Just check no errors; harder to assert bucket identity externally
    assert "X" in rl._clients


async def test_global_rpm_zero_rejected():
    with pytest.raises(ValueError):
        RateLimiter(global_rpm=0, per_client_rpm=10)
