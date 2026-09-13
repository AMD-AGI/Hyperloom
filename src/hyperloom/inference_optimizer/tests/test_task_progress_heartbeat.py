# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A long task's heartbeat: reported by the executor, landed on its own row."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.session import paths
from hyperloom.orchestrator.bus.resource_lock import ResourceLockManager, SqliteLeaseBackend
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
from hyperloom.orchestrator.loop.sub_agent_runner import (
    PROGRESS_OWNER_AGENT,
    SubAgentRunner,
    _format_progress,
)
from hyperloom.orchestrator.state.task_registry import TaskRegistry
from hyperloom.orchestrator.trace import task_progress
from hyperloom.orchestrator.trace.task_progress import (
    OutputActivity,
    heartbeat_while_output_flows,
    progress_scope,
    report_progress,
)


def _runner(tmp_path: Path, monkeypatch) -> SubAgentRunner:
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir()
    db = SqliteConnection(tmp_path / "coord.db")
    return SubAgentRunner(
        ResourceLockManager(SqliteLeaseBackend(db)),
        TaskRegistry(db),
        session_dir=sd,
        policy=None,
    )


def _progress_notes(task) -> list[dict]:
    return [row["progress"] for row in task.history if "progress" in row]


def _iso_ago(seconds: float) -> str:
    """Build the ISO timestamp a task that started ``seconds`` ago would carry."""
    return datetime.fromtimestamp(time.time() - seconds, tz=timezone.utc).isoformat()


# Bound on every wait paced by the heartbeat driver, so a regression that stops the driver fails in seconds instead of
# hanging the suite.
_HEARTBEAT_BACKSTOP_S = 10.0

# Bound on every wait for a worker thread the rollback tests hold a lock in, so a regression that never frees it fails
# instead of hanging the suite.
_ROLLBACK_BACKSTOP_S = 5.0

# How long a cancelled write is given to come back before it is called early.
_RETURNED_EARLY_WINDOW_S = 0.2


async def _await_notes(notes: list[dict], label: str, count: int, *, interval_s: float) -> None:
    """Wait until the driver stamping ``label`` has reported ``count`` notes."""

    async def _reached() -> None:
        while sum(1 for note in notes if note.get("label") == label) < count:
            await asyncio.sleep(interval_s)

    await asyncio.wait_for(_reached(), timeout=_HEARTBEAT_BACKSTOP_S)


async def _await_cancelled(task: asyncio.Task) -> None:
    """Await a cancelled task so the caller can inspect leftover state."""
    try:
        await task
        pytest.fail(f"expected CancelledError; the task ended as {task.exception()!r}")
    except asyncio.CancelledError:
        # Expected: cancel is the success path; the caller asserts leftover state.
        return


@pytest.mark.asyncio
async def test_a_task_that_reports_units_leaves_a_trail_on_its_own_row(tmp_path, monkeypatch):
    """The heartbeat is what separates a working long task from a hung one."""
    sub = _runner(tmp_path, monkeypatch)

    async def _grid(_ctx) -> dict:
        for i in (1, 2, 3):
            await report_progress(unit="variant", label=f"v{i}", index=i, total=3)
        return {"status": "ok"}

    sub.register_executor("explore", _grid)
    task = await sub.tasks.create(kind="explore", params={}, idempotency_key="grid-0")
    res = await sub.run_task(task)

    assert res.state == "succeeded"
    done = await sub.tasks.get(task.task_id)
    assert [note["label"] for note in _progress_notes(done)] == ["v1", "v2", "v3"]


@pytest.mark.asyncio
async def test_every_note_names_the_agent_it_vouches_for(tmp_path, monkeypatch):
    """Without an owner the heartbeat would excuse whichever agent is quiet."""
    sub = _runner(tmp_path, monkeypatch)

    async def _grid(_ctx) -> dict:
        await report_progress(unit="variant", label="v1")
        return {"status": "ok"}

    sub.register_executor("explore", _grid)
    task = await sub.tasks.create(kind="explore", params={}, idempotency_key="grid-owner")
    await sub.run_task(task)

    done = await sub.tasks.get(task.task_id)
    assert [note["agent"] for note in _progress_notes(done)] == [PROGRESS_OWNER_AGENT]


@pytest.mark.asyncio
async def test_a_heartbeat_leaves_the_running_mark_where_it_was(tmp_path, monkeypatch):
    """``updated_at`` says when the task started running, not when it last spoke."""
    sub = _runner(tmp_path, monkeypatch)
    started_iso = _iso_ago(3600)
    seen: dict[str, Any] = {}

    async def _slow(ctx) -> dict:
        await sub.tasks.db.execute(
            "UPDATE tasks SET updated_at=? WHERE task_id=?",
            (started_iso, ctx.task.task_id),
        )
        await report_progress(unit="baseline_round", label="warmup")
        beating = await sub.tasks.get(ctx.task.task_id)
        seen["updated_at"] = beating.updated_at
        seen["notes"] = len(_progress_notes(beating))
        assert beating.state == "running"
        return {"status": "ok"}

    sub.register_executor("baseline", _slow)
    task = await sub.tasks.create(kind="baseline", params={}, idempotency_key="baseline-0")
    await sub.run_task(task)

    assert seen["updated_at"] == started_iso
    assert seen["notes"] == 1


@pytest.mark.asyncio
async def test_a_task_that_heartbeats_all_along_is_still_reclaimed_at_its_lease(tmp_path, monkeypatch):
    """The R6 watchdog is a runtime budget, not an inactivity timeout."""
    sub = _runner(tmp_path, monkeypatch)
    task = await sub.tasks.create(
        kind="roofline",
        params={},
        idempotency_key="roofline-lease",
        lease_ttl_sec=2700,
    )
    await sub.tasks.transition(task.task_id, "running")
    await sub.tasks.db.execute(
        "UPDATE tasks SET updated_at=? WHERE task_id=?",
        (_iso_ago(3106), task.task_id),
    )
    for unit in range(3):
        await sub.tasks.record_progress(task.task_id, {"unit": "roofline_step", "index": unit})

    reclaimed = await sub.tasks.reclaim_expired_running(reason="test_watchdog")

    assert reclaimed == [task.task_id]
    assert (await sub.tasks.get(task.task_id)).state == "failed"


@pytest.mark.asyncio
async def test_a_sink_that_raises_is_swallowed_not_propagated(tmp_path, monkeypatch):
    """The reporter is an observability path; it owns its own failures."""

    async def _broken(**_note) -> None:
        raise RuntimeError("sink is down")

    with progress_scope(_broken):
        await report_progress(unit="variant")


@pytest.mark.asyncio
async def test_progress_reported_outside_a_task_is_a_no_op() -> None:
    """Executors call the reporter unconditionally, including from unit tests."""
    await report_progress(unit="variant", label="unscoped")


@pytest.mark.asyncio
async def test_the_scope_does_not_outlive_the_task_that_opened_it(tmp_path, monkeypatch):
    """Otherwise one task's units would land on another task's row."""
    sub = _runner(tmp_path, monkeypatch)

    async def _noop(_ctx) -> dict:
        return {"status": "ok"}

    sub.register_executor("explore", _noop)
    task = await sub.tasks.create(kind="explore", params={}, idempotency_key="grid-2")
    await sub.run_task(task)

    await report_progress(unit="variant", label="after")
    done = await sub.tasks.get(task.task_id)
    assert _progress_notes(done) == []


@pytest.mark.asyncio
async def test_a_long_step_keeps_reporting_while_its_child_talks() -> None:
    """The 55-minute analysis has one await and no completion report to lean on."""
    notes: list[dict] = []

    async def _sink(**note) -> None:
        notes.append(note)

    interval = 0.02
    with progress_scope(_sink):
        async with heartbeat_while_output_flows(
            unit="kernel_tool",
            label="trace_analyze",
            interval_s=interval,
        ) as activity:
            for line in range(1, 4):
                activity.note()
                await _await_notes(notes, "trace_analyze", line, interval_s=interval)

    assert len(notes) == 3
    assert {n["label"] for n in notes} == {"trace_analyze"}
    assert all(n["status"] == "running" for n in notes)
    assert [n["output_lines"] for n in notes] == sorted(n["output_lines"] for n in notes)


@pytest.mark.asyncio
async def test_a_step_whose_child_went_quiet_is_allowed_to_go_stale() -> None:
    """A timer would keep vouching for a wedged process; that is the failure to catch."""
    notes: list[dict] = []

    async def _sink(**note) -> None:
        notes.append(note)

    interval = 0.02
    with progress_scope(_sink):
        async with heartbeat_while_output_flows(
            unit="kernel_tool",
            label="trace_analyze",
            interval_s=interval,
        ) as activity:
            activity.note()
            await _await_notes(notes, "trace_analyze", 1, interval_s=interval)
            async with heartbeat_while_output_flows(label="pacer", interval_s=interval) as pacer:
                for tick in range(1, 4):
                    pacer.note()
                    await _await_notes(notes, "pacer", tick, interval_s=interval)

    assert [note["label"] for note in notes] == ["trace_analyze", "pacer", "pacer", "pacer"]


@pytest.mark.asyncio
async def test_the_driver_stops_with_the_step_it_watches() -> None:
    """A leaked driver task would report a step that already returned."""
    notes: list[dict] = []

    async def _sink(**note) -> None:
        notes.append(note)

    interval = 0.02
    with progress_scope(_sink):
        async with heartbeat_while_output_flows(label="t", interval_s=interval) as activity:
            activity.note()
            await _await_notes(notes, "t", 1, interval_s=interval)
        activity.note()  # the step it belonged to is over; nothing may report this
        async with heartbeat_while_output_flows(label="pacer", interval_s=interval) as pacer:
            for tick in range(1, 4):
                pacer.note()
                await _await_notes(notes, "pacer", tick, interval_s=interval)

    assert [note["label"] for note in notes] == ["t", "pacer", "pacer", "pacer"]


@pytest.mark.asyncio
async def test_teardown_lets_the_note_in_flight_finish_before_giving_up() -> None:
    """A note is a ``tasks`` write; cancelling one mid-write wedges the connection."""
    events: list[str] = []
    writing = asyncio.Event()

    async def _slow_sink(**_note) -> None:
        events.append("begin")
        writing.set()
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            events.append("cancelled")
            raise
        events.append("end")

    interval = 0.01
    with progress_scope(_slow_sink):
        async with heartbeat_while_output_flows(label="t", interval_s=interval) as activity:
            activity.note()
            await asyncio.wait_for(writing.wait(), timeout=_HEARTBEAT_BACKSTOP_S)

    assert events == ["begin", "end"]


@pytest.mark.asyncio
async def test_a_wedged_sink_cannot_hold_the_step_open(monkeypatch) -> None:
    """Cooperative shutdown is bounded: past the grace the driver is cancelled."""
    monkeypatch.setattr(task_progress, "_DRIVER_STOP_GRACE_S", 0.05)
    events: list[str] = []
    entered = asyncio.Event()

    async def _wedged_sink(**_note) -> None:
        entered.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            events.append("cancelled")
            raise

    interval = 0.01

    async def _step() -> None:
        with progress_scope(_wedged_sink):
            async with heartbeat_while_output_flows(label="t", interval_s=interval) as activity:
                activity.note()
                await entered.wait()

    await asyncio.wait_for(_step(), timeout=_HEARTBEAT_BACKSTOP_S)

    await asyncio.sleep(0)
    # The cancellation is the whole claim: a teardown that waited on the sink instead would still be inside the
    # ``async with``, not here.
    assert events == ["cancelled"]


@pytest.mark.asyncio
async def test_a_cancel_landing_in_teardown_still_cancels_the_step() -> None:
    """Suppressing the driver's cancellation used to absorb the caller's too."""
    entered = asyncio.Event()

    async def _slow_to_die_sink(**_note) -> None:
        entered.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
            raise

    async def _step() -> str:
        with progress_scope(_slow_to_die_sink):
            async with heartbeat_while_output_flows(label="t", interval_s=0.01) as activity:
                activity.note()
                await entered.wait()
        return "finished"

    step = asyncio.create_task(_step())
    await entered.wait()
    await asyncio.sleep(0.02)  # the body has returned; teardown is waiting
    step.cancel()
    await _await_cancelled(step)


@pytest.mark.asyncio
async def test_a_cancelled_progress_write_leaves_the_connection_usable(tmp_path, monkeypatch):
    """A cancel mid-``BEGIN IMMEDIATE`` must not wedge the shared connection."""
    sub = _runner(tmp_path, monkeypatch)
    task = await sub.tasks.create(kind="explore", params={}, idempotency_key="wedge")
    await sub.tasks.transition(task.task_id, "running")

    begun = threading.Event()
    real_begin = sub.tasks.db._begin_immediate

    def _slow_begin():
        """Widen the window between ``BEGIN IMMEDIATE`` and the caller resuming."""
        cur = real_begin()
        if not begun.is_set():
            begun.set()
            time.sleep(0.2)
        return cur

    monkeypatch.setattr(sub.tasks.db, "_begin_immediate", _slow_begin)
    writing = asyncio.create_task(sub.tasks.record_progress(task.task_id, {"unit": "variant"}))
    await asyncio.to_thread(begun.wait, 5.0)
    writing.cancel()
    await _await_cancelled(writing)

    assert not sub.tasks.db.raw.in_transaction
    await sub.tasks.record_progress(task.task_id, {"unit": "variant", "label": "after"})
    assert [note.get("label") for note in _progress_notes(await sub.tasks.get(task.task_id))] == ["after"]


@pytest.mark.asyncio
async def test_the_loop_keeps_running_while_a_cancelled_write_rolls_back(tmp_path, monkeypatch):
    """The rollback waits on a lock a worker thread holds; the loop must not wait with it."""
    sub = _runner(tmp_path, monkeypatch)
    task = await sub.tasks.create(kind="explore", params={}, idempotency_key="loop-liveness")
    await sub.tasks.transition(task.task_id, "running")

    tick_s = 0.01
    ticks_required = 5
    db = sub.tasks.db
    holding = threading.Event()
    cancelled = threading.Event()
    loop_alive = threading.Event()
    real_begin = db._begin_immediate

    def _begin_and_keep_the_lock():
        """Begin, then hold ``_sync_lock`` past the cancel like a contended BEGIN does."""
        with db._sync_lock:
            cur = real_begin()
            holding.set()
            loop_alive.wait(_ROLLBACK_BACKSTOP_S)
            return cur

    monkeypatch.setattr(db, "_begin_immediate", _begin_and_keep_the_lock)

    async def _tick_until_the_rollback_may_proceed() -> None:
        """Count loop ticks once the cancel has landed, then free the worker's lock."""
        ticks = 0
        while ticks < ticks_required:
            await asyncio.sleep(tick_s)
            if cancelled.is_set():
                ticks += 1
        loop_alive.set()

    ticker = asyncio.create_task(_tick_until_the_rollback_may_proceed())
    writing = asyncio.create_task(sub.tasks.record_progress(task.task_id, {"unit": "variant"}))
    await asyncio.to_thread(holding.wait, _ROLLBACK_BACKSTOP_S)
    writing.cancel()
    cancelled.set()
    await _await_cancelled(writing)
    ticker.cancel()
    await asyncio.gather(ticker, return_exceptions=True)

    assert loop_alive.is_set(), "the event loop stopped while the rollback waited for the worker's lock"
    # And the rollback is awaited, not fired and forgotten: ``transaction()`` cannot return while the connection is
    # still inside one, or the next ``BEGIN IMMEDIATE`` would fail the way it did before the rollback existed.
    assert not db.raw.in_transaction
    await sub.tasks.record_progress(task.task_id, {"unit": "variant", "label": "after"})


def _rollback_gated_on(db, entered: threading.Event, release: threading.Event, monkeypatch) -> None:
    """Make ``db``'s rollback announce itself and then wait for ``release``."""
    real_rollback = db._rollback

    def _gated_rollback() -> None:
        entered.set()
        release.wait(_ROLLBACK_BACKSTOP_S)
        real_rollback()

    monkeypatch.setattr(db, "_rollback", _gated_rollback)


@pytest.mark.asyncio
async def test_a_cancel_landing_on_the_rollback_does_not_release_the_lock_early(tmp_path, monkeypatch):
    """The second cancel must not abandon the wait the first one created."""
    sub = _runner(tmp_path, monkeypatch)
    task = await sub.tasks.create(kind="explore", params={}, idempotency_key="second-cancel")
    await sub.tasks.transition(task.task_id, "running")

    db = sub.tasks.db
    holding = threading.Event()
    release = threading.Event()
    rolling_back = threading.Event()
    real_begin = db._begin_immediate

    def _begin_and_keep_the_lock():
        """Hold ``_sync_lock`` past the cancel, like a contended BEGIN IMMEDIATE."""
        with db._sync_lock:
            cur = real_begin()
            holding.set()
            release.wait(_ROLLBACK_BACKSTOP_S)
            return cur

    monkeypatch.setattr(db, "_begin_immediate", _begin_and_keep_the_lock)
    _rollback_gated_on(db, rolling_back, release, monkeypatch)

    writing = asyncio.create_task(sub.tasks.record_progress(task.task_id, {"unit": "variant"}))
    await asyncio.to_thread(holding.wait, _ROLLBACK_BACKSTOP_S)
    writing.cancel()
    await asyncio.to_thread(rolling_back.wait, _ROLLBACK_BACKSTOP_S)
    writing.cancel()
    await asyncio.wait({writing}, timeout=_RETURNED_EARLY_WINDOW_S)

    assert not writing.done(), "the write returned with its rollback still queued behind the abandoned worker"
    release.set()
    await _await_cancelled(writing)

    assert not db.raw.in_transaction
    assert not db._async_lock.locked()
    await sub.tasks.record_progress(task.task_id, {"unit": "variant", "label": "after"})


@pytest.mark.asyncio
async def test_a_cancel_arriving_while_the_rollback_runs_is_not_lost(tmp_path, monkeypatch):
    """One cancel is enough, and it must not be swallowed by the rollback handler."""
    sub = _runner(tmp_path, monkeypatch)
    task = await sub.tasks.create(kind="explore", params={}, idempotency_key="lost-cancel")
    await sub.tasks.transition(task.task_id, "running")
    await sub.tasks.db.execute(
        "UPDATE tasks SET history=? WHERE task_id=?",
        ("{ this will not parse", task.task_id),
    )

    db = sub.tasks.db
    rolling_back = threading.Event()
    release = threading.Event()
    _rollback_gated_on(db, rolling_back, release, monkeypatch)

    writing = asyncio.create_task(sub.tasks.record_progress(task.task_id, {"unit": "variant"}))
    await asyncio.to_thread(rolling_back.wait, _ROLLBACK_BACKSTOP_S)
    assert writing.cancel()
    release.set()
    await asyncio.wait({writing})

    assert writing.cancelled(), f"the cancel was lost; the write ended as {writing.exception()!r}"
    assert not db.raw.in_transaction


@pytest.mark.asyncio
async def test_a_rollback_the_connection_cannot_do_is_logged_not_raised(tmp_path, monkeypatch, caplog):
    """A rollback that fails outright must not become the exception the caller sees."""
    db = SqliteConnection(tmp_path / "coord.db")

    def _rollback_on_a_closed_connection() -> None:
        raise sqlite3.ProgrammingError("Cannot operate on a closed database.")

    monkeypatch.setattr(db, "_rollback", _rollback_on_a_closed_connection)
    caught = False
    try:
        with caplog.at_level(logging.WARNING):
            try:
                async with db.transaction():
                    raise ValueError("body failed")
            except ValueError:
                caught = True
        assert caught, "the body's exception was lost behind the rollback"
        assert "rollback after a failed transaction did not complete" in caplog.text
        assert "Cannot operate on a closed database" in caplog.text
    finally:
        db.close()


def test_the_tally_is_safe_to_advance_from_reader_threads() -> None:
    """The pump threads write it; the event loop reads it."""
    activity = OutputActivity()
    threads = [threading.Thread(target=lambda: [activity.note() for _ in range(200)]) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert activity.count() == 800


def test_a_counter_reads_as_one_field_in_the_log_line() -> None:
    """``3/12`` is what an operator scans for; two separate keys are not."""
    line = _format_progress({"unit": "variant", "label": "v3", "index": 3, "total": 12, "status": None})
    assert "progress=3/12" in line
    assert "status" not in line

    assert "progress=2" in _format_progress({"index": 2})
