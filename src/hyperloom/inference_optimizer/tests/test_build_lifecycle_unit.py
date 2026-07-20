# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the off-loop targeted-build lifecycle collaborator (S2).

Drives the pump/reaper against a minimal fake coordinator (real TaskRegistry +
ResourceLockManager + SharedState) with fake ``build_command`` argv, proving the
build runs off the tick loop, ``build_lane`` serializes, timeout kills the group,
and idempotency avoids a double spawn.
"""

from __future__ import annotations

import os
import signal
import sys
import time

import pytest

from hyperloom.orchestrator.bus.resource_lock import ResourceLockManager, SqliteLeaseBackend
from hyperloom.orchestrator.bus.storage import SqliteConnection
from hyperloom.orchestrator.bus.storage.schema import ensure_schema
from hyperloom.orchestrator.framework.build_actions import TargetedBuildAction
from hyperloom.orchestrator.loop.build_lifecycle import BuildLifecycleCollaborator
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import TaskRegistry


class _FakeCoordinator:
    def __init__(self, session_dir, db):
        self.session_dir = session_dir
        self.tasks = TaskRegistry(db)
        self.locks = ResourceLockManager(SqliteLeaseBackend(db))
        self.shared_state = SharedState()


@pytest.fixture
def coord(tmp_path):
    db = SqliteConnection(tmp_path / "coordinator.db")
    ensure_schema(db.raw)
    fc = _FakeCoordinator(tmp_path, db)
    yield fc
    db.close()


@pytest.fixture
def bl(coord):
    return BuildLifecycleCollaborator(coord)


def _action(cmd, **kw):
    base = dict(
        gap_id="g", framework="vllm", component="aiter", capability="fp4_moe", build_command=tuple(cmd)
    )
    base.update(kw)
    return TargetedBuildAction(**base)


async def _drain(bl, *, ticks=200, sleep=0.05):
    """Pump + reap across simulated ticks until the build row is terminal."""
    for i in range(ticks):
        await bl._maybe_reap_targeted_build(tick=i)
        await bl._maybe_pump_targeted_build(tick=i)
        running = [t for t in await bl.tasks.by_state("running") if t.kind == "targeted_build"]
        succeeded = [t for t in await bl.tasks.by_state("succeeded") if t.kind == "targeted_build"]
        failed = [t for t in await bl.tasks.by_state("failed") if t.kind == "targeted_build"]
        if (succeeded or failed) and not running:
            return succeeded, failed
        time.sleep(sleep)
    raise AssertionError("build did not reach a terminal state")


@pytest.mark.asyncio
async def test_enqueue_then_build_succeeds(coord, bl):
    tid = await bl.enqueue_targeted_build(_action([sys.executable, "-c", "print('ok')"]))
    assert tid
    succeeded, failed = await _drain(bl)
    assert succeeded and not failed
    # Manifest recorded, sentinel cleared, lane released.
    assert coord.shared_state.enablement_build_manifest
    assert coord.shared_state.pending_targeted_build == {}
    assert await coord.locks.lane_holders() == {} or "build_lane" not in await coord.locks.lane_holders()


@pytest.mark.asyncio
async def test_no_tick_freeze_build_runs_off_loop(coord, bl):
    """Ticks keep advancing while a long build is in flight (no freeze)."""
    await bl.enqueue_targeted_build(_action([sys.executable, "-c", "import time; time.sleep(3)"]))
    # First pump starts the build; subsequent pump/reap calls return promptly.
    ticks_observed = 0
    started = time.monotonic()
    for i in range(20):
        t0 = time.monotonic()
        await bl._maybe_reap_targeted_build(tick=i)
        await bl._maybe_pump_targeted_build(tick=i)
        # Each tick body must be fast (never blocks on the 3s build).
        assert time.monotonic() - t0 < 1.0
        ticks_observed += 1
        running = [t for t in await bl.tasks.by_state("running") if t.kind == "targeted_build"]
        if i >= 1:
            assert running, "build should be in flight without blocking the tick"
        if time.monotonic() - started > 1.0:
            break
    assert ticks_observed >= 2
    # Build still running after several fast ticks (proves off-loop).
    running = [t for t in await bl.tasks.by_state("running") if t.kind == "targeted_build"]
    assert running


@pytest.mark.asyncio
async def test_nonzero_build_fails_and_records_failure(coord, bl):
    await bl.enqueue_targeted_build(_action([sys.executable, "-c", "import sys; sys.exit(2)"]))
    succeeded, failed = await _drain(bl)
    assert failed and not succeeded
    lbf = coord.shared_state.enablement_last_build_failure
    assert lbf["failure_class"] == "compile_error"


@pytest.mark.asyncio
async def test_timeout_kills_and_marks_failed(coord, bl):
    action = _action([sys.executable, "-c", "import time; time.sleep(600)"], build_budget_sec=1)
    tid = await bl.enqueue_targeted_build(action)
    succeeded, failed = await _drain(bl, ticks=400)
    assert failed and not succeeded
    assert coord.shared_state.enablement_last_build_failure["failure_class"] == "timeout"
    # Lane freed after failure.
    holders = await coord.locks.lane_holders()
    assert holders.get("build_lane", 0) in (0, None) or "build_lane" not in holders


@pytest.mark.asyncio
async def test_build_lane_serializes_two_builds(coord, bl):
    """Capacity-1 build_lane: the second build stays queued until the first ends."""
    a1 = await bl.enqueue_targeted_build(_action([sys.executable, "-c", "import time; time.sleep(2)"], ref="v1"))
    a2 = await bl.enqueue_targeted_build(_action([sys.executable, "-c", "print('two')"], ref="v2"))
    assert a1 != a2
    await bl._maybe_pump_targeted_build(tick=0)
    running = [t.task_id for t in await bl.tasks.by_state("running") if t.kind == "targeted_build"]
    queued = [t.task_id for t in await bl.tasks.queued() if t.kind == "targeted_build"]
    assert len(running) == 1
    assert len(queued) == 1  # second is held back by build_lane


@pytest.mark.asyncio
async def test_idempotent_enqueue_no_double_row(coord, bl):
    a = _action([sys.executable, "-c", "print('x')"], ref="v1", gpu_arch="gfx950")
    t1 = await bl.enqueue_targeted_build(a)
    t2 = await bl.enqueue_targeted_build(a)  # same novelty tuple
    assert t1 == t2
    all_builds = [t for t in await bl.tasks.queued() if t.kind == "targeted_build"]
    assert len(all_builds) == 1


@pytest.mark.asyncio
async def test_build_lane_empty_conflict_does_not_block_serving(coord, bl):
    """build_lane must not mutex the serving/benchmark lanes."""
    await bl.enqueue_targeted_build(_action([sys.executable, "-c", "import time; time.sleep(2)"]))
    await bl._maybe_pump_targeted_build(tick=0)
    # A benchmark lease must still be grantable while a build holds build_lane.
    lease = await coord.locks.try_acquire_many(
        ["benchmark_lane"], holder_id="bench", task_id="bench", action="bench", ttl_sec=60
    )
    assert lease is not None
    await coord.locks.release(lease)
