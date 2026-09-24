# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Execution ownership survives caller cancellation and event-loop shutdown."""

from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3
import sys
import threading
from contextlib import closing
from concurrent.futures import CancelledError as FuturesCancelledError
from dataclasses import asdict
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hyperloom.orchestrator.actions.cancel_channel import current_cancel_scope
from hyperloom.orchestrator.actions.executors import _ray_serving as ray_serving
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


def test_forced_specialist_actor_kill_releases_capacity_and_lane(tmp_path, monkeypatch):
    class RayError(Exception):
        pass

    killed = []
    fake_ray = SimpleNamespace(
        get=lambda ref, **_kwargs: ref,
        kill=killed.append,
        exceptions=SimpleNamespace(RayError=RayError),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    dispatcher = _dispatcher(tmp_path)
    dispatcher.gpu_specialist_pool = SimpleNamespace(release=AsyncMock())
    actor = SimpleNamespace(stop=SimpleNamespace(remote=lambda: False))
    specialist_lease = ray_serving.GpuSpecialistLease(num_gpus=1)
    specialist_lease._actor = actor
    specialist_lease._start_ref = object()
    gpu_lease = object()

    async def run():
        dispatcher.sub.register_executor("shutdown_test", AsyncMock(return_value={"status": "ok"}))
        task = await dispatcher.tasks.create(
            kind="shutdown_test", params={}, idempotency_key="forced-actor-kill", requires_lanes=["research_lane"]
        )
        result = await dispatcher.run_task_registered(
            task,
            gpu_specialist_lease=specialist_lease,
            gpu_lease=gpu_lease,
        )
        assert result.state == "succeeded"
        assert killed == [actor]
        assert specialist_lease._actor is None
        assert specialist_lease._start_ref is None
        dispatcher.gpu_specialist_pool.release.assert_awaited_once_with(gpu_lease)
        assert await dispatcher.locks.lane_holders() == {}
        assert (await dispatcher.tasks.get(task.task_id)).state == "succeeded"
        assert dispatcher._executions == set()
        assert dispatcher._inflight_actions == {}

    try:
        asyncio.run(run())
    finally:
        dispatcher.db.close()


@pytest.mark.parametrize("cleanup", ["false", "raises"])
@pytest.mark.parametrize("outcome", ["succeeded", "failed", "cancelled"])
def test_cleanup_unconfirmed_preserves_outcome_without_completion(tmp_path, cleanup, outcome):
    from hyperloom.orchestrator.loop.sub_agent_runner import ExecutionCleanupUnconfirmed

    dispatcher = _dispatcher(tmp_path)
    payload = {"status": "ok", "decision": "KEEP", "nested": {"answer": [42]}}
    completed = AsyncMock()
    dispatcher.gpu_specialist_pool = SimpleNamespace(release=AsyncMock())

    async def execute(_ctx):
        if outcome == "failed":
            raise ValueError("executor failed")
        if outcome == "cancelled":
            raise FuturesCancelledError("session_time_exhausted")
        return payload

    def close():
        if cleanup == "raises":
            raise OSError("cleanup failed")
        return False

    async def run():
        dispatcher.sub.register_executor("shutdown_test", execute)
        task = await dispatcher.tasks.create(
            kind="shutdown_test", params={}, idempotency_key="outcome", requires_lanes=["research_lane"]
        )
        with pytest.raises(ExecutionCleanupUnconfirmed) as caught:
            await dispatcher.run_task_registered(
                task,
                gpu_specialist_lease=SimpleNamespace(close=close),
                gpu_lease=object(),
                on_complete=completed,
            )
        result = caught.value.result
        assert result.state == outcome
        assert result.task_id == task.task_id
        if outcome == "succeeded":
            assert result.result == payload
        else:
            assert result.error and result.error_class
        if cleanup == "raises":
            assert isinstance(caught.value.__cause__, OSError)
        for index in range(125):
            await dispatcher.tasks.record_progress(task.task_id, {"index": index})
        stored = await dispatcher.tasks.get(task.task_id)
        evidence = [entry["evidence"] for entry in stored.history if "evidence" in entry][-1]
        assert evidence["outcome"] == asdict(result)
        assert evidence["cleanup_confirmed"] is False
        assert evidence["cleanup_error"]
        assert completed.await_count == 0
        assert dispatcher.gpu_specialist_pool.release.await_count == 0
        assert dispatcher._promote_to_shared_state.await_count == 0
        assert dispatcher._fact_write_hook.await_count == 0
        assert await dispatcher.locks.lane_holders() == {"research_lane": 1}
        assert not await dispatcher.bus.tail(topic="delegated_result")
        await _close(dispatcher)
        assert dispatcher.db.fetchone_sync("SELECT 1")[0] == 1

    try:
        asyncio.run(run())
    finally:
        dispatcher.db.close()


def test_executor_cleanup_unconfirmed_keeps_result_and_ownership(tmp_path):
    from hyperloom.orchestrator.loop.sub_agent_runner import ExecutionCleanupUnconfirmed, SubAgentResult

    dispatcher = _dispatcher(tmp_path)
    close = AsyncMock(return_value=True)

    async def run():
        task = await dispatcher.tasks.create(
            kind="shutdown_test", params={}, idempotency_key="executor-cleanup", requires_lanes=["research_lane"]
        )
        result = SubAgentResult(task.task_id, "cancelled", {"status": "cancelled"}, "stop", "cancelled")
        dispatcher.sub.register_executor(
            "shutdown_test", AsyncMock(side_effect=ExecutionCleanupUnconfirmed("worker alive", result=result))
        )
        lease = await dispatcher.locks.try_acquire_many(
            ["research_lane"], holder_id=task.task_id, task_id=task.task_id, action=task.kind, ttl_sec=60
        )
        with pytest.raises(ExecutionCleanupUnconfirmed) as caught:
            await dispatcher.sub.run_task(task, prebound_lease=lease, release_resources=close)
        assert caught.value.result is result
        assert close.await_count == 0
        assert await dispatcher.locks.lane_holders() == {"research_lane": 1}
        stored = await dispatcher.tasks.get(task.task_id)
        assert stored.history[-1]["evidence"]["outcome"] == asdict(result)
        assert stored.history[-1]["evidence"]["cleanup_confirmed"] is False

    try:
        asyncio.run(run())
    finally:
        dispatcher.db.close()


@pytest.mark.parametrize("named_tree", [True, False])
def test_an_unconfirmed_cleanup_records_the_group_for_the_operator(tmp_path, named_tree):
    """The lead an operator gets for a retained lane: the group, as a number, not prose.

    The lane stays held here on purpose, so the one thing that can ever release
    it is an observation that nothing of the execution is left -- and this row
    is the only durable place its process group survives the process that saw
    it. A raise site with no local group to name records none, and that lane is
    then held for good.
    """
    from hyperloom.orchestrator.loop.sub_agent_runner import ExecutionCleanupUnconfirmed, SubAgentResult

    dispatcher = _dispatcher(tmp_path)

    async def run():
        task = await dispatcher.tasks.create(
            kind="shutdown_test", params={}, idempotency_key="tree-root", requires_lanes=["research_lane"]
        )
        result = SubAgentResult(task.task_id, "failed", {}, "tree cleanup unconfirmed", "cleanup")
        dispatcher.sub.register_executor(
            "shutdown_test",
            AsyncMock(
                side_effect=ExecutionCleanupUnconfirmed(
                    "specialist pid=4242: tree cleanup unconfirmed",
                    result=result,
                    tree_pgid=4242 if named_tree else None,
                )
            ),
        )
        lease = await dispatcher.locks.try_acquire_many(
            ["research_lane"], holder_id=task.task_id, task_id=task.task_id, action=task.kind, ttl_sec=60
        )
        with pytest.raises(ExecutionCleanupUnconfirmed):
            await dispatcher.sub.run_task(task, prebound_lease=lease, release_resources=AsyncMock(return_value=True))
        assert await dispatcher.locks.lane_holders() == {"research_lane": 1}
        evidence = (await dispatcher.tasks.get(task.task_id)).history[-1]["evidence"]
        assert evidence["cleanup_confirmed"] is False
        assert evidence.get("cleanup_tree_pgid") == (4242 if named_tree else None)

    try:
        asyncio.run(run())
    finally:
        dispatcher.db.close()


@pytest.mark.parametrize("outcome", ["succeeded", "failed", "cancelled"])
@pytest.mark.parametrize("callback_error", [RuntimeError("completion failed"), asyncio.CancelledError()])
def test_confirmed_cleanup_unregisters_even_when_completion_raises(tmp_path, outcome, callback_error):
    dispatcher = _dispatcher(tmp_path)
    completed = AsyncMock(side_effect=callback_error)

    async def execute(_ctx):
        if outcome == "failed":
            raise ValueError("executor failed")
        if outcome == "cancelled":
            raise FuturesCancelledError("session_time_exhausted")
        return {"status": "ok"}

    async def run():
        dispatcher.sub.register_executor("shutdown_test", execute)
        task = await dispatcher.tasks.create(
            kind="shutdown_test", params={}, idempotency_key="callback", requires_lanes=["research_lane"]
        )
        with pytest.raises(type(callback_error)):
            await dispatcher.run_task_registered(task, on_complete=completed)
        assert completed.await_count == 1
        assert completed.await_args.args[0].state == outcome
        assert not dispatcher._executions and not dispatcher._inflight_actions
        assert not await dispatcher.locks.lane_holders()
        await _close(dispatcher)
        with pytest.raises(sqlite3.ProgrammingError):
            dispatcher.db.fetchone_sync("SELECT 1")

    try:
        asyncio.run(run())
    finally:
        dispatcher.db.close()


def test_confirmed_cancellation_records_once_without_promotion_or_retry(tmp_path):
    dispatcher = _dispatcher(tmp_path)
    dispatcher._maybe_auto_retry_specialist = AsyncMock(return_value=True)
    dispatcher._record_specialist_result = AsyncMock()
    dispatcher._record_framework_agent_authoring_empty_outcome = lambda **_kwargs: None
    dispatcher._ingest_candidate_discovery = lambda **_kwargs: None
    dispatcher._handle_unpromotable_result = AsyncMock()

    async def run():
        dispatcher.sub.register_executor("specialist", AsyncMock(side_effect=FuturesCancelledError("stop")))
        task = await dispatcher.tasks.create(
            kind="specialist", params={}, idempotency_key="cancelled", requires_lanes=["research_lane"]
        )
        result = await dispatcher.run_task_registered(
            task, on_complete=partial(dispatcher._reap_dispatched_task, task, gpu_lease=None)
        )
        assert result.state == "cancelled"
        assert (await dispatcher.tasks.get(task.task_id)).state == "cancelled"
        events = await dispatcher.bus.tail(topic="delegated_result")
        assert len(events) == 1 and events[0].payload["state"] == "cancelled"
        assert dispatcher._maybe_auto_retry_specialist.await_count == 0
        assert dispatcher._record_specialist_result.await_count == 1
        assert dispatcher._promote_to_shared_state.await_count == 0
        assert dispatcher._fact_write_hook.await_count == 0
        assert not dispatcher._executions and not dispatcher._inflight_actions
        assert not await dispatcher.locks.lane_holders()

    try:
        asyncio.run(run())
    finally:
        dispatcher.db.close()


def test_shutdown_drain_closes_after_clean_callback_failure(tmp_path, monkeypatch):
    dispatcher = _dispatcher(tmp_path)
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def completed(_result):
        entered.set()
        await finish.wait()
        raise RuntimeError("completion failed")

    async def run():
        dispatcher.sub.register_executor("shutdown_test", AsyncMock(return_value={"status": "ok"}))
        task = await dispatcher.tasks.create(kind="shutdown_test", params={}, idempotency_key="drain-callback")
        caller = asyncio.create_task(dispatcher.run_task_registered(task, on_complete=completed))
        await entered.wait()
        asyncio.get_running_loop().call_soon(finish.set)
        await _close(dispatcher)
        with pytest.raises(RuntimeError, match="completion failed"):
            await caller
        assert not dispatcher._executions
        with pytest.raises(sqlite3.ProgrammingError):
            dispatcher.db.fetchone_sync("SELECT 1")

    try:
        asyncio.run(run())
    finally:
        dispatcher.db.close()


def test_terminal_race_preserves_unconfirmed_outcome(tmp_path):
    from hyperloom.orchestrator.loop.sub_agent_runner import ExecutionCleanupUnconfirmed

    dispatcher = _dispatcher(tmp_path)

    async def execute(ctx):
        await dispatcher.tasks.transition(ctx.task.task_id, "cancelled", evidence={"reason": "external"})
        return {"status": "ok", "answer": 42}

    async def run():
        dispatcher.sub.register_executor("shutdown_test", execute)
        task = await dispatcher.tasks.create(kind="shutdown_test", params={}, idempotency_key="terminal-race")
        with pytest.raises(ExecutionCleanupUnconfirmed) as caught:
            await dispatcher.run_task_registered(task, gpu_specialist_lease=SimpleNamespace(close=lambda: False))
        for index in range(125):
            await dispatcher.tasks.record_progress(task.task_id, {"index": index})
        stored = await dispatcher.tasks.get(task.task_id)
        assert stored.state == "cancelled"
        outcomes = [entry["evidence"] for entry in stored.history if "outcome" in entry.get("evidence", {})]
        assert len(outcomes) == 1
        assert outcomes[0]["outcome"] == asdict(caught.value.result)
        assert outcomes[0]["cleanup_confirmed"] is False

    try:
        asyncio.run(run())
    finally:
        dispatcher.db.close()


@pytest.mark.parametrize("error", [RuntimeError("runner interrupted"), asyncio.CancelledError()])
def test_unknown_runner_exit_retains_execution_ownership(tmp_path, monkeypatch, error):
    dispatcher = _dispatcher(tmp_path)
    monkeypatch.setattr(dispatcher.sub, "run_task", AsyncMock(side_effect=error))
    completed = AsyncMock()

    async def run():
        task = await dispatcher.tasks.create(
            kind="shutdown_test", params={}, idempotency_key="unknown-runner", requires_lanes=["research_lane"]
        )
        with pytest.raises(type(error)):
            await dispatcher.run_task_registered(task, on_complete=completed)
        assert completed.await_count == 0
        assert len(dispatcher._executions) == len(dispatcher._inflight_actions) == 1
        assert await dispatcher.locks.lane_holders() == {"research_lane": 1}
        await _close(dispatcher)
        assert dispatcher.db.fetchone_sync("SELECT 1")[0] == 1

    try:
        asyncio.run(run())
    finally:
        dispatcher.db.close()


@pytest.mark.parametrize(
    "phase",
    [
        "cancel_running",
        "cancel_pending",
        "cancel_done_grace",
        "direct_running",
        "direct_pending",
        "direct_done_grace",
        "timeout",
        "done",
    ],
)
@pytest.mark.parametrize("confirmed", [True, False])
def test_specialist_followup_ack_reaches_dispatcher_cleanup(tmp_path, monkeypatch, phase, confirmed):
    from unittest.mock import Mock

    from hyperloom.common.deadline import Deadline
    from hyperloom.orchestrator.actions.cancel_channel import CancelScope
    from hyperloom.orchestrator.loop.sub_agent_runner import ExecutionCleanupUnconfirmed
    from hyperloom.orchestrator.specialists import subprocess_ as ss

    dispatcher = _dispatcher(tmp_path)
    specialist = ss.SpecialistSubprocessDispatcher(
        ss.SpecialistSubprocessConfig(agent_backend="claude", poll_interval_seconds=0.01)
    )
    monkeypatch.setattr(specialist, "_build_claude_cmd", lambda **kw: ["unused"])
    harvest = Mock(return_value=([], {}))
    monkeypatch.setattr(specialist, "_collect_patches", harvest)
    scope = CancelScope()
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(ss, "time", SimpleNamespace(monotonic=lambda: clock.now, time=lambda: 100.0))
    polls = 0
    cancelling = phase.startswith(("cancel", "direct"))
    direct = phase.startswith("direct")
    real_sleep = asyncio.sleep

    async def advance(_delay):
        nonlocal polls
        polls += 1
        clock.now += 31.0
        if cancelling and (not phase.endswith("done_grace") or polls == 2):
            if direct:
                asyncio.current_task().cancel()
                await real_sleep(0)
            else:
                scope.cancel(reason="session_time_exhausted")

    monkeypatch.setattr(ss.asyncio, "sleep", advance)
    lease = Mock()
    lease.poll_started.return_value = None if phase.endswith("pending") else 4242
    actor = SimpleNamespace(closed=False)
    lease.is_alive.side_effect = lambda: not actor.closed
    lease.exit_code.side_effect = lambda: None if actor.closed else 7
    lease.stop.return_value = False
    close_calls = 0

    def close():
        nonlocal close_calls
        close_calls += 1
        if cancelling and close_calls == 1:
            return False
        actor.closed = confirmed
        return confirmed

    lease.close.side_effect = close
    completed = AsyncMock()
    payload = {"summary": "preserved", "proposal_set": []}
    is_done = phase == "done" or phase.endswith("done_grace")
    filename = "specialist_done.json" if is_done else "specialist_done.partial.json"
    (tmp_path / filename).write_text(json.dumps(payload), encoding="utf-8")

    async def execute(ctx):
        result = await specialist.run(
            task_id=ctx.task.task_id,
            workspace=tmp_path,
            worktree=None,
            worktree_base=None,
            system_prompt="system",
            user_prompt="user",
            max_turns=1,
            gpu_lease=lease,
            deadline=Deadline.after(0 if phase == "timeout" else 600, now=100),
        )
        return asdict(result)

    async def run():
        dispatcher.sub.register_executor("shutdown_test", execute)
        task = await dispatcher.tasks.create(
            kind="shutdown_test", params={}, idempotency_key="followup-ack", requires_lanes=["research_lane"]
        )
        if confirmed and not direct:
            result = await dispatcher.run_task_registered(
                task, gpu_specialist_lease=lease, cancel_scope=scope, on_complete=completed
            )
            assert result.state == ("cancelled" if phase.startswith("cancel") else "succeeded")
            if phase.startswith("cancel"):
                assert result.error_class == "cancelled"
                assert result.result["reason"] == "session_time_exhausted"
                harvest.assert_not_called()
            else:
                assert result.result["exit_code"] == 7
                assert result.result["timed_out"] is (phase == "timeout")
                assert result.result["done_payload"]["summary"] == "preserved"
                assert result.result["done_payload"].get("_recovered_from_partial", False) is (phase == "timeout")
            assert completed.await_count == 1
            assert not await dispatcher.locks.lane_holders()
            assert not dispatcher._executions
            assert lease.close.call_count == (3 if phase.startswith("cancel") else 2)
        else:
            expected_error = asyncio.CancelledError if confirmed and direct else ExecutionCleanupUnconfirmed
            with pytest.raises(expected_error):
                await dispatcher.run_task_registered(
                    task, gpu_specialist_lease=lease, cancel_scope=scope, on_complete=completed
                )
            assert (await dispatcher.tasks.get(task.task_id)).state == "running"
            assert await dispatcher.locks.lane_holders() == {"research_lane": 1}
            assert len(dispatcher._executions) == 1
            assert all(execution.done() for execution in dispatcher._executions)
            assert completed.await_count == 0
            assert lease.close.call_count == (2 if cancelling else 1)
            harvest.assert_not_called()
        assert lease.stop.call_count == int(not cancelling)
        await _close(dispatcher)
        if confirmed and not direct:
            with pytest.raises(sqlite3.ProgrammingError):
                dispatcher.db.fetchone_sync("SELECT 1")
        else:
            assert dispatcher.db.fetchone_sync("SELECT 1")[0] == 1

    try:
        asyncio.run(run())
    finally:
        dispatcher.db.close()


def test_local_unknown_cleanup_cannot_use_empty_gpu_release_ack(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from hyperloom.orchestrator.loop.sub_agent_runner import ExecutionCleanupUnconfirmed
    from hyperloom.orchestrator.specialists import subprocess_ as ss

    dispatcher = _dispatcher(tmp_path)
    proc = Mock(pid=4242)
    proc.poll.return_value = 0
    completed = AsyncMock()
    release = AsyncMock(return_value=True)
    monkeypatch.setattr(dispatcher.locks, "release", release)

    async def execute(_ctx):
        ss.SpecialistSubprocessDispatcher._kill(proc)

    async def run():
        dispatcher.sub.register_executor("shutdown_test", execute)
        task = await dispatcher.tasks.create(
            kind="shutdown_test", params={}, idempotency_key="local-unknown", requires_lanes=["research_lane"]
        )
        with pytest.raises(ExecutionCleanupUnconfirmed, match="exited root"):
            await dispatcher.run_task_registered(task, on_complete=completed)
        assert release.await_count == 0
        assert completed.await_count == 0
        assert await dispatcher.locks.lane_holders() == {"research_lane": 1}
        assert len(dispatcher._executions) == 1
        await _close(dispatcher)
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
