"""Concurrency + queue admission control.

Bounded in-flight slots (per-client + global) with a bounded waiting room.
If the waiting room is full, raise `OverloadedError` so the caller can map
to a 503 Anthropic error.

Semantics:
  - max_concurrent: how many requests can be IN the upstream call at once (global)
  - max_per_client: same, per client_id
  - max_waiting: how many requests can be WAITING for a slot (not yet running)
  - max_concurrent + max_waiting is the hard cap on simultaneous in-slot tasks

Usage:
    async with admission.slot(client_id):
        # at most max_concurrent (globally) and max_per_client (this client)
        # are inside this block at once
        await upstream_call()
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class OverloadedError(Exception):
    """Raised when the queue of waiting requests exceeds the configured limit."""


class AdmissionControl:
    """In-flight + waiting-room admission gate."""

    def __init__(self, *, max_concurrent: int, max_per_client: int, max_waiting: int) -> None:
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        if max_per_client <= 0:
            raise ValueError("max_per_client must be positive")
        if max_waiting < 0:
            raise ValueError("max_waiting must be >= 0")
        self._global_sem = asyncio.Semaphore(max_concurrent)
        self._max_per_client = max_per_client
        self._max_waiting = max_waiting

        self._waiting = 0
        self._state_lock = asyncio.Lock()

        self._client_sems: dict[str, asyncio.Semaphore] = {}
        self._client_sems_lock = asyncio.Lock()

    @asynccontextmanager
    async def slot(self, client_id: str) -> AsyncIterator[None]:
        """Acquire a slot; raises OverloadedError if the waiting room is full."""
        # 1. Reserve a spot in the waiting room (or fail fast)
        async with self._state_lock:
            if self._waiting >= self._max_waiting:
                raise OverloadedError(
                    f"Server overloaded: {self._waiting} requests already waiting "
                    f"(max {self._max_waiting})"
                )
            self._waiting += 1

        client_sem: asyncio.Semaphore | None = None
        client_acquired = False
        global_acquired = False
        try:
            client_sem = await self._get_or_create_client_sem(client_id)
            await client_sem.acquire()
            client_acquired = True
            await self._global_sem.acquire()
            global_acquired = True
            # 2. Leave the waiting room — we're running now
            async with self._state_lock:
                self._waiting -= 1
            # 3. Run the body
            yield
        finally:
            if global_acquired:
                self._global_sem.release()
            if client_acquired and client_sem is not None:
                client_sem.release()
            if not global_acquired:
                # Never finished acquiring — release our waiting reservation
                async with self._state_lock:
                    self._waiting -= 1

    @property
    def currently_waiting(self) -> int:
        return self._waiting

    async def _get_or_create_client_sem(self, client_id: str) -> asyncio.Semaphore:
        sem = self._client_sems.get(client_id)
        if sem is not None:
            return sem
        async with self._client_sems_lock:
            sem = self._client_sems.get(client_id)
            if sem is None:
                sem = asyncio.Semaphore(self._max_per_client)
                self._client_sems[client_id] = sem
            return sem
