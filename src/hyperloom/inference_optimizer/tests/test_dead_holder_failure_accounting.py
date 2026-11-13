# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A lease-reaped task death must count as a failure for its action."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator
from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentResult
from hyperloom.orchestrator.roles import (
    MockBackend,
    MockCriticBackend,
    ScriptedPlan,
)
from hyperloom.orchestrator.state.task_registry import TaskSpec

# Far above any real pid on Linux, so the liveness probe proves it dead.
_DEAD_PID = 2_147_483_646


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    from hyperloom.orchestrator.bus.resource_lock import SqliteLeaseBackend

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setattr("hyperloom.orchestrator.bus.resource_lock.local_owner_scope", lambda: "test-node")
    monkeypatch.setattr(SqliteLeaseBackend, "_pid_alive", staticmethod(lambda pid: pid != _DEAD_PID))
    return make_session_dir()


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(
        turns=[],
        default_intent=Intent(
            type=IntentType.SEND_MESSAGE,
            payload={"topic": "heartbeat", "body_md": "ok"},
        ),
    )
    return {
        "orchestration": MockBackend(silent, name="orch"),
        "kernel_agent": MockBackend(silent, name="kernel_agent"),
        "critic": MockCriticBackend(),
    }


async def _running_task_with_dead_lease(
    coord: Coordinator,
    *,
    kind: str = "baseline",
    key: str,
    params: dict[str, Any] | None = None,
):
    """Create a running task holding a lease from a provably dead process."""
    task, _ = await coord.tasks.create_or_return_existing(TaskSpec(kind=kind, params=params or {}, idempotency_key=key))
    await coord.tasks.transition(task.task_id, "running")
    coord.db.raw.execute(
        "INSERT INTO leases(lane, holder_id, task_id, action, pid, acquired_at, expires_at) VALUES (?,?,?,?,?,?,?)",
        (
            "benchmark_lane",
            task.task_id,
            task.task_id,
            kind,
            _DEAD_PID,
            "2026-01-01T00:00:00+00:00",
            "2099-12-31T23:59:59+00:00",
        ),
    )
    coord.db.raw.execute("UPDATE leases SET owner_scope='test-node' WHERE task_id=?", (task.task_id,))
    coord.db.raw.commit()
    return task


@pytest.mark.asyncio
async def test_pump_counts_lease_reaped_baseline_as_failure(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        task = await _running_task_with_dead_lease(c, key="k-dead-1")
        await c.dispatcher.pump_dispatcher_once()
        assert (await c.tasks.get(task.task_id)).state == "failed"
        assert c.shared_state.baseline_failure_streak == 1
        assert c.shared_state.baseline_total_failures == 1
        entry = c.shared_state.last_action_failures[-1]
        assert entry["action"] == "baseline"
        assert entry["task_id"] == task.task_id
        assert entry["error_class"] == "dead_holder_reaped"
        assert c.shared_state.baseline_attempts[-1]["status"] == "failed"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_pump_accounts_for_the_reconcilers_confirmed_deaths(session_dir):
    import time

    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        task = await _running_task_with_dead_lease(c, key="reconciler-death")
        report = await c.reconciler.run(time.time())
        assert report.failed_tasks == [task.task_id]
        await c.dispatcher.pump_dispatcher_once()
        await c.dispatcher.pump_dispatcher_once()
        assert c.shared_state.baseline_total_failures == 1
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_a_death_survives_a_second_reconcile_before_the_pump(session_dir):
    import time

    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        task = await _running_task_with_dead_lease(c, key="reconciler-death-twice")
        assert (await c.reconciler.run(time.time())).failed_tasks == [task.task_id]
        assert (await c.reconciler.run(time.time())).failed_tasks == []
        await c.dispatcher.pump_dispatcher_once()
        assert c.shared_state.baseline_failure_streak == 1
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_three_lease_reaped_baselines_trip_the_streak_stop(session_dir):
    """--enablement=off: three lease-reaped baselines trip the three-strike stop."""
    c = Coordinator(session_dir, backends=_silent_backends())
    c.shared_state.enablement_mode = "off"
    try:
        for i in range(3):
            await _running_task_with_dead_lease(c, key=f"k-dead-streak-{i}")
            await c.dispatcher.pump_dispatcher_once()
        assert c.shared_state.baseline_failure_streak == 3
        assert c.shared_state.stop_reason == "baseline_failed"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_accounting_is_idempotent_per_task(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        task = await _running_task_with_dead_lease(c, key="k-dead-once")
        await c.dispatcher.pump_dispatcher_once()
        await c.dispatcher._account_dead_holder_failures([task.task_id], reason="dead_holder_pump")
        assert c.shared_state.baseline_failure_streak == 1
        assert len(c.shared_state.last_action_failures) == 1
    finally:
        await c.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("result_owed", [False, True])
async def test_a_reaped_discovery_round_settles_once(session_dir, monkeypatch, result_owed):
    """An orphaned discovery round settles as failed, or the pump re-fetches it as in flight for the whole phase.

    It settles at the reclaim even while an execution still owes a result: a late result is dropped, not booked.
    """
    c = Coordinator(session_dir, backends=_silent_backends())
    monkeypatch.setattr(c.dispatcher, "owes_result", lambda _task_id: result_owed)
    try:
        task = await _running_task_with_dead_lease(
            c,
            kind="specialist",
            key="candidate-discovery:candidate_pool_empty:r0",
            params={"candidate_discovery": True, "domain": "candidate_discovery_specialist"},
        )
        await c.dispatcher.pump_dispatcher_once()
        await c.dispatcher.pump_dispatcher_once()
        assert (await c.tasks.get(task.task_id)).state == "failed"
        assert c.shared_state.framework_agent_discover_failures == 1
    finally:
        await c.stop()


async def _running_task_leased_by_this_process(
    coord: Coordinator, *, key: str, kind: str = "specialist", expires_at: str = "2026-01-01T00:00:00+00:00"
):
    """Create a running task whose research_lane lease this very process holds, expiring at ``expires_at``."""
    import os

    task, _ = await coord.tasks.create_or_return_existing(TaskSpec(kind=kind, params={}, idempotency_key=key))
    await coord.tasks.transition(task.task_id, "running")
    coord.db.raw.execute(
        "INSERT INTO leases(lane, holder_id, task_id, action, pid, acquired_at, expires_at, owner_scope)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (
            "research_lane",
            task.task_id,
            task.task_id,
            kind,
            os.getpid(),
            "2026-01-01T00:00:00+00:00",
            expires_at,
            "test-node",
        ),
    )
    coord.db.raw.commit()
    return task


def _lease_rows(coord: Coordinator, task_id: str) -> list[str]:
    return [row[0] for row in coord.db.raw.execute("SELECT lane FROM leases WHERE task_id=?", (task_id,))]


@pytest.mark.asyncio
async def test_a_quiet_specialist_of_this_process_whose_lease_ran_out_is_failed_and_frees_its_lane(session_dir):
    """The holder pid is the coordinator's own, so no pid probe can find it dead; its lease and its CLI's silence do."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        task = await _running_task_leased_by_this_process(c, key="spec-quiet")
        await c.dispatcher.pump_dispatcher_once()
        row = await c.tasks.get(task.task_id)
        assert row.state == "failed"
        assert row.history[-1]["evidence"] == {
            "reason": "lapsed_lease_pump",
            "lease_expired_at": "2026-01-01T00:00:00+00:00",
        }
        assert _lease_rows(c, task.task_id) == []
        assert await c.tasks.running() == []
        assert c.dispatcher.has_unbooked_completions() is False
    finally:
        await c.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("alive_because", ["lease_still_valid", "cli_still_writing", "holds_gpu", "not_a_specialist"])
async def test_an_execution_of_this_process_that_still_shows_life_is_left_running(session_dir, alive_because):
    import time

    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        task = await _running_task_leased_by_this_process(
            c,
            key=f"spec-{alive_because}",
            kind="baseline" if alive_because == "not_a_specialist" else "specialist",
            expires_at="2099-12-31T23:59:59+00:00"
            if alive_because == "lease_still_valid"
            else "2026-01-01T00:00:00+00:00",
        )
        if alive_because == "cli_still_writing":
            workspace = runs_dir(c.session_dir, "specialist", task.task_id)
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "process.log").write_text("{}\n")
            assert time.time() - (workspace / "process.log").stat().st_mtime < 60
        if alive_because == "holds_gpu":
            c.db.raw.execute("INSERT INTO gpu_leases(gpu_id, holder_id, task_id) VALUES (0, ?, ?)", (task.task_id,) * 2)
            c.db.raw.commit()
        await c.dispatcher.pump_dispatcher_once()
        assert (await c.tasks.get(task.task_id)).state == "running"
        assert _lease_rows(c, task.task_id) == ["research_lane"]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_a_late_result_of_a_reclaimed_task_is_not_counted(session_dir, monkeypatch):
    c = Coordinator(session_dir, backends=_silent_backends())
    booked: list[str] = []

    async def book(task, result):
        booked.append(task.task_id)

    monkeypatch.setattr(c.dispatcher, "reap_dispatched_task", book)
    try:
        task = await _running_task_leased_by_this_process(c, key="spec-late")
        await c.dispatcher.pump_dispatcher_once()
        late = SubAgentResult(task_id=task.task_id, state="succeeded", result={"runner_status": "succeeded"})
        await c.dispatcher._queue_completion(task, late)
        await c.dispatcher.pump_dispatcher_once()
        assert booked == []
        assert (await c.tasks.get(task.task_id)).state == "failed"
        assert c.dispatcher.has_unbooked_completions() is False
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_a_specialist_that_died_with_its_holder_is_dispatched_again(session_dir):
    """A specialist killed with its process gets the bounded infra retry a crash gets, on the same gap."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        params = {"domain": "serving_specialist", "mode": "patch", "gap_canonical_id": "gap.front_end_input_cost"}
        task = await _running_task_with_dead_lease(c, kind="specialist", key="spec-front-end", params=params)
        await c.dispatcher.pump_dispatcher_once()

        assert (await c.tasks.get(task.task_id)).state == "failed"
        live = [*await c.tasks.queued(), *await c.tasks.running()]
        retries = [
            (t.idempotency_key, t.params.get("gap_canonical_id"), t.params.get("_auto_retry_attempt")) for t in live
        ]
        assert retries == [("spec-front-end-autoretry1", "gap.front_end_input_cost", 1)]
        assert c.shared_state.last_action_failures == []
    finally:
        await c.stop()


class _ReapStub:
    """Minimal coordinator shell for the reap-path double-count guard."""

    def __init__(self) -> None:
        self.unpromotable: list[str] = []
        self.gpu_specialist_pool = SimpleNamespace(release=self._noop_async)
        self.bus = SimpleNamespace(append_and_seq=self._noop_async, record_observation=self._noop_async)
        self.shared_state = SimpleNamespace(phase="PRELUDE", macro_cycle=0)
        self.recipe_journal = SimpleNamespace(fact_write_hook=self._noop_async)
        _outer = self
        self.writeback = SimpleNamespace(
            handle_unpromotable_result=lambda task, result: _outer._handle_unpromotable_result(task, result)
        )

    async def _noop_async(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def _handle_unpromotable_result(self, task: Any, _result: Any) -> None:
        self.unpromotable.append(task.task_id)

    def record_exception(self, **_kwargs: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_reap_skips_failure_accounting_already_charged():
    stub = _ReapStub()
    disp = DispatcherCollaborator(stub)
    task = SimpleNamespace(task_id="t-dead", kind="baseline", params={})
    result = SubAgentResult(task_id=task.task_id, state="failed", result={"status": "failed"})

    await disp.reap_dispatched_task(task, result)
    assert stub.unpromotable == ["t-dead"]

    disp._dead_holder_accounted.add(task.task_id)
    await disp.reap_dispatched_task(task, result)
    assert stub.unpromotable == ["t-dead"]
