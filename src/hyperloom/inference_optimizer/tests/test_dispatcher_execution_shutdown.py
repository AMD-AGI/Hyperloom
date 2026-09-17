# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Execution ownership survives caller cancellation and event-loop shutdown."""

from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3
import threading
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hyperloom.orchestrator.actions.cancel_channel import current_cancel_scope
from hyperloom.orchestrator.bus.message_bus import MessageBus
from hyperloom.orchestrator.bus.resource_lock import ResourceLockManager, SqliteLeaseBackend
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
from hyperloom.orchestrator.loop import dispatcher as dispatcher_module
from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator
from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner
from hyperloom.orchestrator.state.task_registry import TaskRegistry


def _dispatcher(tmp_path):
    db = SqliteConnection(tmp_path / "shutdown.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tasks = TaskRegistry(db)
    state = SimpleNamespace(phase="PRELUDE", macro_cycle=0, tick=0, session_budget_usable_sec=lambda: None)
    coord = SimpleNamespace(
        db=db,
        locks=locks,
        tasks=tasks,
        shared_state=state,
        bus=MessageBus(db),
        sub=SubAgentRunner(locks, tasks),
        _stop=asyncio.Event(),
        _dispatcher_poll_sec=0.01,
        _BUDGET_GATED_DISPATCH_PHASES=frozenset(),
        _promote_to_shared_state=AsyncMock(),
        _fact_write_hook=AsyncMock(),
        _is_promotable_result=lambda *_args: True,
    )
    dispatcher = DispatcherCollaborator(coord)
    dispatcher._cancel_queued_task_over_budget = AsyncMock(return_value=False)
    return dispatcher


async def _close(dispatcher):
    # Exercise both sides of the synchronous-to-async shutdown regression.
    closing = dispatcher.close_db_after_executions()
    if inspect.isawaitable(closing):
        await closing


def test_asyncio_run_shutdown_waits_for_execution_and_completion(tmp_path, monkeypatch):
    """The loop exits immediately after shutdown, not after a test-only worker join."""
    dispatcher = _dispatcher(tmp_path)
    monkeypatch.setattr(dispatcher_module, "_CANCEL_NOTICE_SEC", 0)
    entered = threading.Event()
    stop_worker = threading.Event()
    worker_done = threading.Event()
    observed = []
    real_close = dispatcher.db.close

    def close_db():
        observed.append((worker_done.is_set(), dispatcher.db.fetchone_sync("SELECT state FROM tasks")[0]))
        real_close()

    monkeypatch.setattr(dispatcher.db, "close", close_db)

    def work():
        entered.set()
        assert stop_worker.wait(5)
        dispatcher.db.fetchone_sync("SELECT 1")
        worker_done.set()
        return {"status": "ok"}

    async def execute(_ctx):
        return await asyncio.to_thread(work)

    async def run():
        dispatcher.sub.register_executor("shutdown_test", execute)
        task = await dispatcher.tasks.create(
            kind="shutdown_test", params={}, idempotency_key="shutdown", requires_lanes=["research_lane"]
        )
        pump = asyncio.create_task(dispatcher._pump_dispatcher_once())
        assert await asyncio.to_thread(entered.wait, 5)
        pump.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pump
        # Release from outside the event loop as shutdown begins. No await after close.
        stop_worker.set()
        await _close(dispatcher)
        return task.task_id

    try:
        task_id = asyncio.run(run())
        assert observed == [(True, "succeeded")]
        with closing(sqlite3.connect(dispatcher.db.db_path)) as db:
            assert db.execute("SELECT state FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0] == "succeeded"
            assert db.execute("SELECT count(*) FROM leases").fetchone()[0] == 0
            rows = db.execute("SELECT payload FROM events WHERE topic='delegated_result'").fetchall()
        assert len(rows) == 1
        assert json.loads(rows[0][0])["task_id"] == task_id
        assert dispatcher._promote_to_shared_state.await_count == 1
    finally:
        stop_worker.set()
        real_close()


def test_cancelled_pump_late_success_is_reaped_once(tmp_path, monkeypatch):
    dispatcher = _dispatcher(tmp_path)
    monkeypatch.setattr(dispatcher_module, "_CANCEL_NOTICE_SEC", 0)

    async def run():
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def execute(_ctx):
            entered.set()
            await finish.wait()
            return {"status": "ok"}

        dispatcher.sub.register_executor("shutdown_test", execute)
        await dispatcher.tasks.create(kind="shutdown_test", params={}, idempotency_key="late-success")
        pump = asyncio.create_task(dispatcher._pump_dispatcher_once())
        await asyncio.wait_for(entered.wait(), 5)
        pump.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pump
        executions = tuple(dispatcher._executions)
        finish.set()
        await asyncio.gather(*executions)
        await dispatcher._pump_dispatcher_once()
        events = await dispatcher.db.fetchall("SELECT payload FROM events WHERE topic='delegated_result'")
        assert len(events) == 1
        assert dispatcher._promote_to_shared_state.await_count == 1
        await _close(dispatcher)

    try:
        asyncio.run(run())
    finally:
        dispatcher.db.close()


def test_shutdown_requests_scope_and_keeps_unconfirmed_database_open(tmp_path, monkeypatch):
    dispatcher = _dispatcher(tmp_path)
    monkeypatch.setattr(dispatcher_module, "_CANCEL_NOTICE_SEC", 0)
    monkeypatch.setattr(dispatcher_module, "_COOPERATIVE_CANCEL_GRACE_SEC", 0)
    entered = threading.Event()
    finish = threading.Event()
    scopes = []

    def work():
        scope = current_cancel_scope()
        scopes.append(scope)
        entered.set()
        assert finish.wait(5)
        return {"status": "ok"}

    async def execute(_ctx):
        return await asyncio.to_thread(work)

    async def run():
        dispatcher.sub.register_executor("shutdown_test", execute)
        task = await dispatcher.tasks.create(
            kind="shutdown_test", params={}, idempotency_key="unconfirmed", requires_lanes=["research_lane"]
        )
        action = asyncio.create_task(dispatcher.run_task_registered(task))
        assert await asyncio.to_thread(entered.wait, 5)
        await _close(dispatcher)
        assert scopes[0].cancelled
        assert (await dispatcher.tasks.get(task.task_id)).state == "running"
        assert await dispatcher.locks.lane_holders()
        finish.set()
        return action

    try:
        asyncio.run(run())
        assert dispatcher.db.fetchone_sync("SELECT 1")[0] == 1
    finally:
        finish.set()
        dispatcher.db.close()


def test_normal_pump_completion_is_not_reaped_twice(tmp_path, monkeypatch):
    dispatcher = _dispatcher(tmp_path)

    async def run():
        dispatcher.sub.register_executor("shutdown_test", AsyncMock(return_value={"status": "ok"}))
        await dispatcher.tasks.create(kind="shutdown_test", params={}, idempotency_key="normal-completion")
        await dispatcher._pump_dispatcher_once()
        await dispatcher._pump_dispatcher_once()
        events = await dispatcher.db.fetchall("SELECT payload FROM events WHERE topic='delegated_result'")
        assert len(events) == 1
        assert dispatcher._promote_to_shared_state.await_count == 1
        assert not dispatcher._executions
        await _close(dispatcher)

    try:
        asyncio.run(run())
    finally:
        dispatcher.db.close()


def test_cancelled_shutdown_drain_retains_live_execution(tmp_path, monkeypatch):
    dispatcher = _dispatcher(tmp_path)
    monkeypatch.setattr(dispatcher_module, "_CANCEL_NOTICE_SEC", 0)

    async def run():
        entered = asyncio.Event()
        finish = asyncio.Event()
        draining = asyncio.Event()
        original_wait = asyncio.wait

        async def observe_drain(*args, **kwargs):
            draining.set()
            return await original_wait(*args, **kwargs)

        async def execute(_ctx):
            entered.set()
            await finish.wait()
            return {"status": "ok"}

        dispatcher.sub.register_executor("shutdown_test", execute)
        task = await dispatcher.tasks.create(
            kind="shutdown_test", params={}, idempotency_key="cancelled-drain", requires_lanes=["research_lane"]
        )
        action = asyncio.create_task(dispatcher.run_task_registered(task))
        await asyncio.wait_for(entered.wait(), 5)
        monkeypatch.setattr(asyncio, "wait", observe_drain)
        shutdown = asyncio.create_task(_close(dispatcher))
        await asyncio.wait_for(draining.wait(), 5)
        shutdown.cancel()
        with pytest.raises(asyncio.CancelledError):
            await shutdown
        assert (await dispatcher.tasks.get(task.task_id)).state == "running"
        assert await dispatcher.locks.lane_holders()
        finish.set()
        await action
        await _close(dispatcher)

    try:
        asyncio.run(run())
    finally:
        dispatcher.db.close()


def test_unconfirmed_physical_cleanup_prevents_database_close(tmp_path, monkeypatch):
    from hyperloom.orchestrator.loop.sub_agent_runner import ExecutionCleanupUnconfirmed

    dispatcher = _dispatcher(tmp_path)

    async def run():
        dispatcher.sub.register_executor("shutdown_test", AsyncMock(return_value={"status": "ok"}))
        task = await dispatcher.tasks.create(
            kind="shutdown_test", params={}, idempotency_key="physical-cleanup", requires_lanes=["research_lane"]
        )
        with pytest.raises(ExecutionCleanupUnconfirmed):
            await dispatcher.run_task_registered(task, gpu_specialist_lease=SimpleNamespace(close=lambda: False))
        await _close(dispatcher)
        assert await dispatcher.locks.lane_holders()
        assert dispatcher.db.fetchone_sync("SELECT 1")[0] == 1

    try:
        asyncio.run(run())
    finally:
        dispatcher.db.close()


def test_specialist_budget_uses_shared_benchmark_timeout(tmp_path, monkeypatch):
    dispatcher = _dispatcher(tmp_path)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "9000")
    try:
        assert dispatcher._specialist_wall_budget_sec(needs_gpu=False) == 600
        assert (
            dispatcher._specialist_wall_budget_sec(
                needs_gpu=True, params={"scope": "domain", "mode": "patch", "bench": True}
            )
            == 9600
        )
    finally:
        dispatcher.db.close()
