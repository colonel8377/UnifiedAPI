"""Tests for AdmissionControl (concurrency + waiting room).

Covers:
  - max_concurrent enforced globally
  - max_per_client enforced per client
  - max_waiting rejected with OverloadedError when full
  - slots released on completion (even on exception)
  - different clients don't share per-client budget
"""
from __future__ import annotations

import asyncio

import pytest

from unified_api.control.concurrency import AdmissionControl, OverloadedError


async def test_single_slot_acquired_and_released():
    ac = AdmissionControl(max_concurrent=1, max_per_client=1, max_waiting=10)
    async with ac.slot("client"):
        assert ac.currently_waiting == 0  # we transitioned out of waiting
    # After exit, slot is free


async def test_global_concurrency_limit_enforced():
    ac = AdmissionControl(max_concurrent=2, max_per_client=10, max_waiting=10)
    in_flight = 0
    observed_max = 0
    lock = asyncio.Lock()

    async def worker():
        nonlocal in_flight, observed_max
        async with ac.slot("c"):
            async with lock:
                in_flight += 1
                observed_max = max(observed_max, in_flight)
            await asyncio.sleep(0.05)
            async with lock:
                in_flight -= 1

    await asyncio.gather(*(worker() for _ in range(6)))
    assert observed_max <= 2


async def test_per_client_concurrency_limit_enforced():
    ac = AdmissionControl(max_concurrent=100, max_per_client=2, max_waiting=100)
    in_flight = 0
    observed_max = 0
    lock = asyncio.Lock()

    async def worker():
        nonlocal in_flight, observed_max
        async with ac.slot("same_client"):
            async with lock:
                in_flight += 1
                observed_max = max(observed_max, in_flight)
            await asyncio.sleep(0.05)
            async with lock:
                in_flight -= 1

    await asyncio.gather(*(worker() for _ in range(6)))
    assert observed_max <= 2


async def test_different_clients_have_independent_budgets():
    ac = AdmissionControl(max_concurrent=100, max_per_client=1, max_waiting=100)
    # Two different clients should both be able to hold a slot simultaneously
    async with ac.slot("A"):
        async with ac.slot("B"):
            await asyncio.sleep(0.01)


async def test_waiting_room_full_raises_overloaded():
    ac = AdmissionControl(max_concurrent=1, max_per_client=10, max_waiting=1)
    holder_done = asyncio.Event()

    async def hold():
        async with ac.slot("c"):
            await holder_done.wait()

    holder = asyncio.create_task(hold())
    await asyncio.sleep(0.05)  # let holder acquire

    # One slot is running, one is waiting, the third should fail
    waiter = asyncio.create_task(ac.slot("c").__aenter__())
    await asyncio.sleep(0.05)
    # Now waiting = 1, max_waiting = 1 → next should raise
    with pytest.raises(OverloadedError):
        async with ac.slot("c"):
            pass

    holder_done.set()
    await waiter
    holder.cancel()
    try:
        await holder
    except asyncio.CancelledError:
        pass


async def test_slot_released_on_exception():
    """If the body raises, the slot must still be released."""
    ac = AdmissionControl(max_concurrent=1, max_per_client=1, max_waiting=10)

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        async with ac.slot("c"):
            raise Boom()

    # Should be able to acquire immediately after
    async with ac.slot("c"):
        pass


async def test_slot_released_on_cancellation():
    ac = AdmissionControl(max_concurrent=1, max_per_client=1, max_waiting=10)

    started = asyncio.Event()
    cancel_me = asyncio.create_task(_hold_until_cancelled(ac, started))
    await started.wait()
    cancel_me.cancel()
    try:
        await cancel_me
    except asyncio.CancelledError:
        pass

    # Slot should be freed
    async with ac.slot("c"):
        pass


async def _hold_until_cancelled(ac: AdmissionControl, started: asyncio.Event):
    async with ac.slot("c"):
        started.set()
        await asyncio.sleep(60)  # will be cancelled


async def test_invalid_args_rejected():
    with pytest.raises(ValueError):
        AdmissionControl(max_concurrent=0, max_per_client=1, max_waiting=1)
    with pytest.raises(ValueError):
        AdmissionControl(max_concurrent=1, max_per_client=0, max_waiting=1)
    with pytest.raises(ValueError):
        AdmissionControl(max_concurrent=1, max_per_client=1, max_waiting=-1)


async def test_max_waiting_zero_rejects_all():
    """With max_waiting=0, every request is rejected because the waiting-room
    check fires before semaphore acquisition. This is a degenerate config
    (real deployments use max_waiting >= 1) but the behavior should be
    deterministic — no request can squeeze through.
    """
    ac = AdmissionControl(max_concurrent=1, max_per_client=1, max_waiting=0)
    with pytest.raises(OverloadedError):
        async with ac.slot("c"):
            pass
