# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A long task's heartbeat: reported by the executor, landed on its own row.

Covers the plumbing that lets a multi-hour composite action say "unit 3 of 12
is done" — the ambient reporter, the runner scope that binds it, and the
registry write that makes the row look alive.
"""

from __future__ import annotations

import asyncio
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
    """``updated_at`` says when the task started running, not when it last spoke.

    The lease watchdog, the ``extend_lease`` budget math and the in-flight
    projection all measure elapsed runtime from it.
    """
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
    """The R6 watchdog is a runtime budget, not an inactivity timeout.

    A heartbeat that reset its clock would make the backstop unreachable for
    exactly the long-running rows that hold lanes and read as live work.
    """
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
async def test_a_heartbeat_never_fails_the_work_it_reports_on(tmp_path, monkeypatch):
    """A row reaped underneath a running executor must not turn into a task failure."""
    sub = _runner(tmp_path, monkeypatch)

    async def _reported_after_deletion(ctx) -> dict:
        async with sub.tasks.db.transaction() as cur:
            cur.execute("DELETE FROM tasks WHERE task_id=?", (ctx.task.task_id,))
        await report_progress(unit="variant", label="orphan")
        return {"status": "ok"}

    sub.register_executor("explore", _reported_after_deletion)
    task = await sub.tasks.create(kind="explore", params={}, idempotency_key="grid-1")
    res = await sub.run_task(task)

    assert res.state == "succeeded"


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
            for _ in range(3):
                activity.note()
                await asyncio.sleep(interval * 1.5)

    assert len(notes) >= 2
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
            await asyncio.sleep(interval * 2)
            reported_while_alive = len(notes)
            await asyncio.sleep(interval * 5)

    assert reported_while_alive == 1
    assert len(notes) == 1


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
            await asyncio.sleep(interval * 2)
        reported_during = len(notes)
        activity.note()
        await asyncio.sleep(interval * 3)

    assert reported_during == 1
    assert len(notes) == reported_during


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
