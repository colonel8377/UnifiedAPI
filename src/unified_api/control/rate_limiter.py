"""Token bucket rate limiter.

Each bucket refills at `rate_per_sec` tokens/second up to `capacity`.
`acquire()` blocks (awaits) until a token is available.

We maintain:
  - one global bucket (global RPM / 60)
  - one bucket per client_id (per_client_rpm / 60)

Both must be satisfied for a request to proceed. The "go negative on
over-subscribe" trick lets concurrent waiters queue at distinct times
without serializing through the lock.
"""
from __future__ import annotations

import asyncio
import time


class TokenBucket:
    """Async token bucket. capacity tokens, refills at rate_per_sec."""

    def __init__(self, rate_per_sec: float, capacity: int) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._rate = float(rate_per_sec)
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Wait until `tokens` tokens are available, then consume them.

        Uses the "go negative" trick: when the bucket is empty, we decrement
        below zero to reserve a future token, then sleep for the time it
        takes the bucket to recover. Concurrent callers each reserve their
        own future token (going more negative), so they serialize at
        distinct future times without contending on the lock.
        """
        async with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return
            # Compute wait based on current deficit, then reserve by going
            # negative. No re-check loop needed: the act of going negative
            # IS the consumption; refill will bring the balance back over time.
            deficit = tokens - self._tokens
            wait_time = deficit / self._rate
            self._tokens -= tokens  # may go negative
        # Sleep outside the lock so other tasks can proceed.
        if wait_time > 0:
            await asyncio.sleep(wait_time)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_refill = now


class RateLimiter:
    """Global + per-client token-bucket rate limiter."""

    def __init__(self, global_rpm: int, per_client_rpm: int) -> None:
        self._global = TokenBucket(rate_per_sec=global_rpm / 60.0, capacity=global_rpm)
        self._per_client_rpm = per_client_rpm
        self._clients: dict[str, TokenBucket] = {}
        self._clients_lock = asyncio.Lock()

    async def acquire(self, client_id: str) -> None:
        """Block until both per-client and global budgets allow one request."""
        client_bucket = await self._get_or_create_client_bucket(client_id)
        # Acquire client first (usually smaller budget) to fail fast on per-client overage.
        await client_bucket.acquire()
        await self._global.acquire()

    async def _get_or_create_client_bucket(self, client_id: str) -> TokenBucket:
        # Fast path: read without lock
        bucket = self._clients.get(client_id)
        if bucket is not None:
            return bucket
        async with self._clients_lock:
            bucket = self._clients.get(client_id)
            if bucket is None:
                bucket = TokenBucket(
                    rate_per_sec=self._per_client_rpm / 60.0,
                    capacity=self._per_client_rpm,
                )
                self._clients[client_id] = bucket
            return bucket
