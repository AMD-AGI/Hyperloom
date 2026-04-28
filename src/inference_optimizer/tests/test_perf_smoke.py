"""Performance smoke — IMPL-CHECKLIST §13.7.

Lightweight stress sanity check, NOT a benchmark. The goal is to catch
quadratic regressions in the bus / cursor / lock paths before they ship.

Thresholds are intentionally generous so this test can run on any CI
worker (Windows, slow disk, etc.) without flaking.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.cursor_store import CursorStore
from inference_optimizer.orchestrator.message_bus import Message, MessageBus
from inference_optimizer.orchestrator.resource_lock import (
    ResourceLockManager,
    SqliteLeaseBackend,
    LaneBusy,
)
from inference_optimizer.storage.connection import SqliteConnection


# Total wall-clock cap (very generous to keep CI green on slow disk).
HARD_CAP_S: float = 10.0
EVENT_COUNT: int = 500
REACTOR_COUNT: int = 8


@pytest.mark.asyncio
async def test_emit_500_events_well_under_10s(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    bus = MessageBus(db)
    started = time.monotonic()
    for i in range(EVENT_COUNT):
        await bus.append_and_seq(
            Message.new(
                from_agent="bench",
                to_agent="*",
                topic="event",
                payload={"i": i},
                priority=1,
            )
        )
    elapsed = time.monotonic() - started
    assert elapsed < HARD_CAP_S, f"event emit took {elapsed:.2f}s"
    rows = await bus.tail(n=EVENT_COUNT + 1)
    assert len(rows) == EVENT_COUNT
    db.close()


@pytest.mark.asyncio
async def test_replay_for_many_agents_remains_linear(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    bus = MessageBus(db)
    cursors = CursorStore(db)
    # seed events
    for i in range(EVENT_COUNT):
        await bus.append_and_seq(
            Message.new(
                from_agent="bench",
                to_agent="*",
                topic="event",
                payload={"i": i},
            )
        )

    started = time.monotonic()
    for k in range(REACTOR_COUNT):
        agent = f"agent-{k}"
        msgs = await bus.replay_for(agent, after_seq=0)
        if msgs:
            last = msgs[-1]
            await cursors.advance(agent, seq=last.seq, msg_id=last.msg_id)
    elapsed = time.monotonic() - started
    assert elapsed < HARD_CAP_S, f"reactor warm-up took {elapsed:.2f}s"
    db.close()


@pytest.mark.asyncio
async def test_repeated_lease_round_trip_no_leak(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    backend = SqliteLeaseBackend(db)
    locks = ResourceLockManager(backend)
    started = time.monotonic()
    for i in range(50):
        lease = await locks.acquire(
            ["benchmark_lane"],
            holder_id=f"holder-{i}",
            task_id=f"task-{i}",
            action="bench_runner",
            ttl_sec=5,
        )
        await backend.release(lease)
    elapsed = time.monotonic() - started
    assert elapsed < HARD_CAP_S, f"lease churn took {elapsed:.2f}s"
    # No active leases left.
    active = await backend.active_lanes()
    assert active == []
    db.close()


@pytest.mark.asyncio
async def test_concurrent_lane_acquisition_is_mutually_exclusive(tmp_path: Path):
    """Two coroutines hammer the same lane — exactly one should succeed
    on the *initial* acquire, the second should block / retry."""
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    backend = SqliteLeaseBackend(db)

    async def try_acquire(holder: str) -> bool:
        try:
            await backend.acquire_many(
                ["benchmark_lane"],
                holder_id=holder,
                task_id=f"t-{holder}",
                action="bench_runner",
                ttl_sec=5,
            )
            return True
        except LaneBusy:
            return False

    a, b = await asyncio.gather(try_acquire("A"), try_acquire("B"))
    assert (a, b).count(True) == 1
    db.close()
