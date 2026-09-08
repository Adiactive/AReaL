# SPDX-License-Identifier: Apache-2.0

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from areal.infra.processor_cache import ProcessorCacheRegistry, ProcessorCallCache


def test_concurrent_sync_calls_compute_once():
    cache = ProcessorCallCache()
    entered = Event()
    release = Event()
    calls = []

    def factory():
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return object()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(cache.get_or_compute, "key", factory) for _ in range(4)
        ]
        try:
            assert entered.wait(5)
        finally:
            release.set()
        results = [future.result(timeout=5) for future in futures]
    assert calls == [1]
    assert all(result is results[0] for result in results)


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_shared_computation():
    cache = ProcessorCallCache()
    release = Event()
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    value = object()

    def factory():
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return value

    owner = asyncio.create_task(cache.aget_or_compute("key", factory))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        waiter = asyncio.create_task(cache.aget_or_compute("key", factory))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
    finally:
        release.set()
    assert await asyncio.wait_for(owner, 5) is value
    assert await cache.aget_or_compute("key", lambda: None) is value


@pytest.mark.asyncio
async def test_failed_processor_call_can_retry():
    cache = ProcessorCallCache()

    def fail():
        raise ValueError("processor failed")

    with pytest.raises(ValueError, match="processor failed"):
        await cache.aget_or_compute("key", fail)
    assert await cache.aget_or_compute("key", lambda: 42) == 42


def test_registry_keeps_cache_between_staggered_sessions_and_closes_at_end():
    registry = ProcessorCacheRegistry()
    first = registry.acquire("group", 2)
    value = first.get_or_compute("key", object)
    registry.release("group")
    second = registry.acquire("group", 2)
    assert second is first
    assert second.get_or_compute("key", object) is value
    registry.release("group")
    assert first.get_or_compute("key", object) is not value
    assert registry.acquire("group", 2) is not first


def test_registry_discard_closes_aborted_group():
    registry = ProcessorCacheRegistry()
    cache = registry.acquire("group", 4)
    value = cache.get_or_compute("key", object)
    registry.discard("group")
    assert cache.get_or_compute("key", object) is not value
    registry.release("group")
