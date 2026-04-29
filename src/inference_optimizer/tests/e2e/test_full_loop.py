"""End-to-end loops that exercise multiple subsystems together.

These cover the IMPL-CHECKLIST §14 E2E* slots that *don't* require real
GPUs. We rely on MockBackend + the bundled action registry + Sqlite +
the Conductor's normal ``run`` entrypoint.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from inference_optimizer.paths import asset_actions_dir
from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.backends import MockBackend
from inference_optimizer.orchestrator.backends.mock import ScriptStep
from inference_optimizer.orchestrator.checkpoint import (
    Checkpoint,
    Verdict,
    evidence_check_matrix,
    resume_from_session_dir,
)
from inference_optimizer.orchestrator.conductor import Conductor, StopReason
from inference_optimizer.orchestrator.cursor_store import CursorStore
from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.message_bus import MessageBus
from inference_optimizer.orchestrator.resource_lock import (
    ResourceLockManager,
    SqliteLeaseBackend,
)
from inference_optimizer.orchestrator.task_registry import TaskRegistry
from inference_optimizer.storage.connection import SqliteConnection


PACKAGE_ACTIONS_DIR = asset_actions_dir()


# ---------------------------------------------------------------------------
# E2E2-ish: queued delegate → SubAgentRunner → succeeded
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_delegate_dispatch_loop_succeeds(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    registry = ActionRegistry(PACKAGE_ACTIONS_DIR).load()

    succ_intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"current_tput": 6000.0}},
    )
    backend = MockBackend(script=[ScriptStep(intents=[succ_intent])])

    conductor = Conductor(
        tmp_path,
        backend=backend,
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        action_registry=registry,
        reactor_tick_s=0.05,
        clock_tick_s=0.05,
        enable_dispatcher=True,
    )

    # Pre-queue a delegate via a helper coroutine.
    async def queue_delegate():
        # wait until conductor is bootstrapped
        for _ in range(50):
            if conductor.ctx is not None:
                break
            await asyncio.sleep(0.01)
        ctx = conductor.ctx
        await ctx.tasks.create(
            kind="delegate",
            params={"action_name": "bench_runner", "params": {}},
            idempotency_key="e2e2-task",
            requires_lanes=["benchmark_lane"],
            allowed_tools=["emit_intent", "Read", "Bash"],
            side_effects=["reads_server"],
            lease_ttl_sec=900,
        )

    asyncio.create_task(queue_delegate())
    await asyncio.wait_for(conductor.run(), timeout=15.0)

    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    tr = TaskRegistry(db)
    succeeded = await tr.list_by_state("succeeded")
    assert len(succeeded) == 1
    assert succeeded[0].kind == "delegate"
    db.close()


# ---------------------------------------------------------------------------
# E2E4-ish: crash mid-bench → resume → evidence_check → succeeded recovery
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_crash_resume_succeeded_via_evidence_check(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    backend = SqliteLeaseBackend(db)
    locks = ResourceLockManager(backend)
    tasks = TaskRegistry(db)

    # Phase 1: "before crash" — task is queued + cursors recorded.
    cs = CursorStore(db)
    await cs.advance("executor", seq=20, msg_id="m20")
    t = await tasks.create(
        kind="delegate",
        params={"action_name": "bench_runner", "params": {}},
        idempotency_key="e2e4-task",
        requires_lanes=["benchmark_lane"],
        allowed_tools=["emit_intent"],
        side_effects=["reads_server"],
        lease_ttl_sec=900,
    )
    await tasks.transition(t.task_id, "running", evidence={"started": True})

    # Bench wrote real metrics before the crash.
    metrics = tmp_path / "results" / t.task_id / "metrics.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text('{"tput_per_gpu": 6000}', encoding="utf-8")
    db.close()

    # Phase 2: fresh process — open the DB again, run resume.
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    backend = SqliteLeaseBackend(db)
    locks = ResourceLockManager(backend)
    state = await resume_from_session_dir(tmp_path, db, locks)
    assert state.cursors == {"executor": 20}
    assert len(state.found_inflight_tasks) == 1
    found = state.found_inflight_tasks[0]
    assert evidence_check_matrix(found, tmp_path) == Verdict.SUCCEEDED
    db.close()


# ---------------------------------------------------------------------------
# E2E10-ish: emergency stop on crash_count >= 2
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_emergency_stop_via_set_stopping(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "10"},
        db=db,
        reactor_tick_s=0.05,
        clock_tick_s=0.05,
        enable_dispatcher=False,
    )

    async def raise_emergency():
        for _ in range(50):
            if conductor.ctx is not None:
                break
            await asyncio.sleep(0.05)
        conductor.ctx.state.crash_count = 2
        conductor.ctx.state.set_stopping(StopReason.EMERGENCY)

    asyncio.create_task(raise_emergency())
    await asyncio.wait_for(conductor.run(), timeout=10.0)

    bus = MessageBus(db)
    events = await bus.tail(n=200)
    # graceful_stop event with reason emergency must be present
    grace = [
        e for e in events
        if e.topic == "graceful_stop"
        and isinstance(e.payload, dict)
        and e.payload.get("reason") == StopReason.EMERGENCY
    ]
    assert grace, "expected graceful_stop with reason=emergency"
    db.close()


# ---------------------------------------------------------------------------
# E2E checkpoint+resume round-trip
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_checkpoint_then_resume_round_trip(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    backend = SqliteLeaseBackend(db)
    locks = ResourceLockManager(backend)
    tasks = TaskRegistry(db)
    cs = CursorStore(db)

    # State before checkpoint
    await cs.advance("executor", seq=12, msg_id="seed")
    await tasks.create(
        kind="delegate",
        params={"action_name": "bench_runner"},
        idempotency_key="cp-resume",
        requires_lanes=["benchmark_lane"],
        allowed_tools=["emit_intent"],
        side_effects=[],
        lease_ttl_sec=900,
    )

    from inference_optimizer.orchestrator.shared_state import SharedState
    state = SharedState(
        session_id="t",
        max_minutes=60.0,
        execution_mode=ExecutionMode.GUIDED_KERNEL_OPT,
    )
    handle = await Checkpoint.create(tmp_path, db, state)
    assert handle.path.is_dir()
    db.close()

    # Re-open the DB and resume.
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    backend = SqliteLeaseBackend(db)
    locks = ResourceLockManager(backend)
    rs = await resume_from_session_dir(tmp_path, db, locks)
    assert rs.cursors == {"executor": 12}
    assert len(rs.found_inflight_tasks) == 1
    db.close()
