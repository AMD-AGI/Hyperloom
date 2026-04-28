"""End-to-end dry-run smoke test (Phase D minimum-viable set).

What this proves:
    1. ``Conductor.run()`` boots, runs reactors + clock, and stops cleanly
       on ``time_exhausted`` when ``MAX_HOURS`` elapses.
    2. The MockBackend produces ``send_message`` intents that flow through
       the bus and back into the reactor inbox without feedback-looping.
    3. The cursor is advanced past every message the reactor saw.
    4. ``state.json`` is written and contains the expected fields.
    5. Self-emitted messages do **not** trigger another backend call.

This is the offline twin of ``python -m inference_optimizer --backend mock``.
Once the Claude/Codex backends land (Phase 6), the same test pattern can
swap MockBackend for ClaudeBackend with no Conductor changes.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from inference_optimizer.orchestrator.backends import MockBackend
from inference_optimizer.orchestrator.conductor import Conductor, StopReason
from inference_optimizer.orchestrator.cursor_store import CursorStore
from inference_optimizer.orchestrator.message_bus import MessageBus
from inference_optimizer.storage.connection import SqliteConnection


# 0.0005h = 1.8s wall time. Tight enough to keep the test fast,
# loose enough that the clock has time to fire several ticks.
TINY_MAX_HOURS = "0.0005"


@pytest.mark.asyncio
async def test_dry_run_completes_with_time_exhausted(session_dir):
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    backend = MockBackend()
    conductor = Conductor(
        session_dir,
        backend=backend,
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": TINY_MAX_HOURS},
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
    )

    ctx = await asyncio.wait_for(conductor.run(), timeout=10.0)

    assert ctx.state.stop_reason == StopReason.TIME_EXHAUSTED
    assert ctx.state.elapsed_minutes >= float(TINY_MAX_HOURS) * 60.0 * 0.5
    assert ctx.state.execution_mode.value == "quick_param_sweep"
    assert backend.calls, "MockBackend was never invoked"
    db.close()


@pytest.mark.asyncio
async def test_dry_run_writes_expected_events(session_dir):
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    conductor = Conductor(
        session_dir,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": TINY_MAX_HOURS},
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
    )
    await asyncio.wait_for(conductor.run(), timeout=10.0)

    # Re-open through a public API to make sure data is durable.
    bus = MessageBus(db)
    all_events = await bus.tail(n=1000)
    topics = {e.topic for e in all_events}
    assert "event" in topics, "boot 'run_started' event missing"
    assert "graceful_stop" in topics, "graceful_stop event not appended"
    assert "reflection_tick" in topics, "clock never fired"
    assert "heartbeat" in topics, "executor never emitted heartbeat"
    db.close()


@pytest.mark.asyncio
async def test_dry_run_no_self_message_feedback_loop(session_dir):
    """Heartbeats target ``to=*`` so without a filter the executor would
    consume its own output and infinitely loop; ratio must stay bounded."""
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    conductor = Conductor(
        session_dir,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": TINY_MAX_HOURS},
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
    )
    await asyncio.wait_for(conductor.run(), timeout=10.0)

    bus = MessageBus(db)
    all_events = await bus.tail(n=1000)
    n_ticks = sum(1 for e in all_events if e.topic == "reflection_tick")
    n_beats = sum(1 for e in all_events if e.topic == "heartbeat")
    # In the bug case n_beats grew quadratically (each beat triggered another
    # beat). With the self-message filter, n_beats <= n_ticks + small slack.
    assert n_ticks > 0
    assert n_beats <= n_ticks + 2, (
        f"feedback loop suspected: {n_beats} beats vs {n_ticks} ticks"
    )
    db.close()


@pytest.mark.asyncio
async def test_dry_run_advances_cursor(session_dir):
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    conductor = Conductor(
        session_dir,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": TINY_MAX_HOURS},
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
    )
    await asyncio.wait_for(conductor.run(), timeout=10.0)

    cursors = CursorStore(db)
    state = await cursors.load("executor")
    assert state.last_processed_seq > 0
    db.close()


@pytest.mark.asyncio
async def test_dry_run_writes_state_json(session_dir):
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    conductor = Conductor(
        session_dir,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": TINY_MAX_HOURS},
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
    )
    await asyncio.wait_for(conductor.run(), timeout=10.0)

    snap_path = session_dir / "state.json"
    assert snap_path.exists()
    snap = json.loads(snap_path.read_text())
    assert snap["session_id"] == session_dir.name
    assert snap["model_path"] == "fake/model"
    assert snap["execution_mode"] == "quick_param_sweep"
    assert snap["stop_reason"] == StopReason.TIME_EXHAUSTED
    assert snap["elapsed_minutes"] >= 0
    db.close()


@pytest.mark.asyncio
async def test_dry_run_with_target_gain_objective(session_dir):
    """Building TargetGainObjective from env should not interfere with the
    dry-run loop; cumulative_gain stays at 0 because the mock never sets
    bench results."""
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    conductor = Conductor(
        session_dir,
        backend=MockBackend(),
        env={
            "MODEL_PATH": "fake/model",
            "MAX_HOURS": TINY_MAX_HOURS,
            "TARGET_GAIN_PCT": "5",
        },
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
    )
    ctx = await asyncio.wait_for(conductor.run(), timeout=10.0)

    assert ctx.objective.kind() == "gain_pct"
    assert ctx.state.cumulative_gain == 0.0
    assert ctx.state.stop_reason == StopReason.TIME_EXHAUSTED
    db.close()


@pytest.mark.asyncio
async def test_emergency_stop_via_set_stopping(session_dir):
    """An external request to stop early should short-circuit the clock."""
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    conductor = Conductor(
        session_dir,
        backend=MockBackend(),
        # Generous budget — would otherwise run for 36s.
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": "0.01"},
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
    )

    async def _kick():
        await asyncio.sleep(0.5)
        assert conductor.ctx is not None
        conductor.ctx.state.set_stopping(StopReason.EMERGENCY)

    ctx, _ = await asyncio.gather(
        asyncio.wait_for(conductor.run(), timeout=10.0),
        _kick(),
    )
    assert ctx.state.stop_reason == StopReason.EMERGENCY
    assert ctx.state.elapsed_minutes < 0.6  # fired before time_exhausted
    db.close()
