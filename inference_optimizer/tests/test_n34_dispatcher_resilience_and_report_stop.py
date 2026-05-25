"""N34 (May 2026) bug-fix tests: dispatcher resilience to disappearing
``tasks`` rows + report-task triggers run-loop exit.

Two regressions surfaced from the MiMo-7B production run on
2026-05-21:

* **Bug #1 / #2** — A long-running grid task (params combo step, ~30
  min wall-clock) raised ``TaskNotFound`` on its terminal
  ``transition(running → succeeded)`` because the SQLite ``tasks``
  row had vanished mid-flight (root cause not yet isolated;
  hypothesis: WAL transaction edge race). The dispatcher's
  ``except Exception ... continue`` then dropped the executor's
  result on the floor, so ``params_search`` ledger never recorded
  the 12 variant runs and the N19c ``kernel_opt`` unlock never
  fired -- the whole optimization loop silently stalled.

* **Bug #4** — A successful ``report`` task should be the canonical
  terminal signal (operator-facing ``final.md`` / ``final.json``
  are already on disk). Pre-fix the main loop kept iterating after
  the report was written; the LLM kept proposing fresh
  ``params`` / ``backends`` rounds and the session burned the rest
  of its wall-clock budget producing useless data.

The fixes live in:

* ``sub_agent_runner._transition_resilient`` — wraps every
  ``transition()`` so ``TaskNotFound`` is downgraded to a structured
  warning and the executor result still propagates to
  ``Coordinator._promote_to_shared_state``.
* ``coordinator._pump_dispatcher_once`` — sets
  ``shared_state.stop_reason = "report_emitted"`` after a
  ``report`` task succeeds so the next loop iteration breaks out.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors import report_executor
from inference_optimizer.orchestrator.backends import (
    Backend,
    MockBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.resource_lock import (
    ResourceLockManager,
    SqliteLeaseBackend,
)
from inference_optimizer.orchestrator.sub_agent_runner import (
    SubAgentResult,
    SubAgentRunner,
)
from inference_optimizer.orchestrator.task_registry import (
    Task,
    TaskRegistry,
)
from inference_optimizer.paths import make_session_dir
from inference_optimizer.session_paths import target_baseline_json
from inference_optimizer.storage import SqliteConnection


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _heartbeat() -> Intent:
    return Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    )


def _silent_backends() -> dict[str, Backend]:
    plan = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        name: MockBackend(plan, name=name)
        for name in ("orchestration", "kernel", "critic", "robustness")
    }


def _write_marker_target_baseline(session_dir: Path) -> None:
    """Satisfy the unconditional ``target_analysis`` hard gate so the
    Coordinator's main loop doesn't bail before reaching the rest of
    the pipeline."""
    path = target_baseline_json(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "status": "no_target",
            "reason": "no_target_gpu_configured",
            "row_count": 0,
        }),
        encoding="utf-8",
    )


# ===========================================================================
# Bug #1 + #2: SubAgentRunner tolerates disappearing ``tasks`` rows
# ===========================================================================
@pytest.mark.asyncio
async def test_sub_agent_runner_swallows_tasknotfound_on_final_transition(
    tmp_path,
):
    """When the ``tasks`` row vanishes mid-execution the final
    ``transition(running → succeeded)`` raises ``TaskNotFound``.
    Pre-fix this propagated up, the dispatcher dropped the result,
    and the params ledger never got the 12 variant measurements.
    Post-fix the runner returns ``SubAgentResult.state == "succeeded"``
    with the full ``result_payload`` so the caller can still promote
    it to SharedState."""
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)

    payload = {
        "tput": 4632.8,
        "params_search_update": {"tested": {"fp1": {"name": "v1"}}},
    }

    async def long_runner(ctx):
        # Mid-flight the tasks row vanishes (production race).
        db.raw.execute("DELETE FROM tasks WHERE task_id=?", (ctx.task.task_id,))
        db.raw.commit()
        return payload

    sub.register_executor("params", long_runner)
    task = await tr.create(kind="params", params={}, idempotency_key="k-params-1")
    res = await sub.run_task(task)

    assert res.state == "succeeded"
    assert res.result == payload, (
        "executor result must survive the TaskNotFound -- the dispatcher "
        "uses it to update params_search ledger; losing it stalls N19c"
    )
    db.close()


@pytest.mark.asyncio
async def test_sub_agent_runner_swallows_tasknotfound_on_initial_transition(
    tmp_path, caplog,
):
    """Even the FIRST ``transition(queued → running)`` can race with a
    vanished row. The runner must still attempt the executor work --
    if the row is gone the audit-trail is shot but the measurement
    is more valuable than the audit row."""
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)

    ran = {"called": False}

    async def runner(ctx):
        ran["called"] = True
        return {"tput": 1.0}

    sub.register_executor("baseline", runner)
    task = await tr.create(
        kind="baseline", params={}, idempotency_key="k-baseline-1",
    )
    # Simulate the row vanishing BEFORE the dispatcher pulls it.
    db.raw.execute("DELETE FROM tasks WHERE task_id=?", (task.task_id,))
    db.raw.commit()

    with caplog.at_level("WARNING"):
        res = await sub.run_task(task)

    assert ran["called"] is True, (
        "executor must still run; losing audit doesn't excuse losing "
        "the measurement"
    )
    assert res.state == "succeeded"
    assert res.result == {"tput": 1.0}
    # Structured warning surfaced so an operator can correlate with
    # the disappearing-row hypothesis.
    assert any(
        "vanished" in rec.message.lower()
        and "_transition_resilient" in rec.message
        for rec in caplog.records
    ), "expected the disappearing-row warning to fire"
    db.close()


@pytest.mark.asyncio
async def test_sub_agent_runner_normal_path_still_records_transitions(
    tmp_path,
):
    """Sanity: the happy path still moves the row queued → running →
    succeeded. The resilience layer must NOT silently drop transitions
    when the row exists."""
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    sub.register_executor("baseline", lambda ctx: _async_return({"tput": 1.0}))
    task = await tr.create(kind="baseline", params={}, idempotency_key="k-ok")
    res = await sub.run_task(task)
    assert res.state == "succeeded"
    after = await tr.get(task.task_id)
    assert after.state == "succeeded"
    db.close()


async def _async_return(value: Any) -> Any:
    return value


# ===========================================================================
# Bug #4: report success terminates the Coordinator run loop
# ===========================================================================
@pytest.mark.asyncio
async def test_report_success_sets_stop_reason(session_dir):
    """A successful ``report`` task is the canonical terminal signal
    -- final.md is on disk so any further LLM-driven exploration is
    waste. Verify ``stop_reason`` flips to ``"report_emitted"`` the
    moment the dispatcher processes the report success."""
    _write_marker_target_baseline(session_dir)
    c = Coordinator(session_dir, backends=_silent_backends())
    c.sub.register_executor("report", report_executor)
    c.shared_state.baseline_tput = 100.0
    c.shared_state.save(session_dir)
    try:
        # Hand-create a report task that mimics the closing-phase /
        # robustness-delegated emission path used in production.
        task = await c.tasks.create(
            kind="report",
            params={"session_dir": str(session_dir)},
            idempotency_key="k-report-1",
        )
        await c._pump_dispatcher_once()
        after = await c.tasks.get(task.task_id)
        assert after.state == "succeeded"
        assert c.shared_state.stop_reason == "report_emitted"
        # Persisted to disk so a launcher / monitor can see it.
        on_disk = json.loads((session_dir / "state.json").read_text())
        assert on_disk["stop_reason"] == "report_emitted"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_report_success_does_not_overwrite_prior_stop_reason(session_dir):
    """If ``stop_reason`` is already set (e.g. signal / target_reached /
    earlier failure), the report-success guard must NOT paper over it
    with the cheery ``"report_emitted"`` reason."""
    _write_marker_target_baseline(session_dir)
    c = Coordinator(session_dir, backends=_silent_backends())
    c.sub.register_executor("report", report_executor)
    c.shared_state.baseline_tput = 100.0
    c.shared_state.stop_reason = "target_reached"
    c.shared_state.save(session_dir)
    try:
        task = await c.tasks.create(
            kind="report",
            params={"session_dir": str(session_dir)},
            idempotency_key="k-report-pre-set",
        )
        await c._pump_dispatcher_once()
        after = await c.tasks.get(task.task_id)
        assert after.state == "succeeded"
        # Pre-existing stop_reason wins -- this preserves diagnostic
        # signal when a real failure terminates the run *before* the
        # report write happens.
        assert c.shared_state.stop_reason == "target_reached"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_loop_exits_after_report_emitted(session_dir):
    """End-to-end: drop a report task on the queue, start the long
    loop, verify it terminates via the ``report_emitted`` exit instead
    of running until ``max_ticks``."""
    _write_marker_target_baseline(session_dir)
    c = Coordinator(session_dir, backends=_silent_backends())
    c.sub.register_executor("report", report_executor)
    c.shared_state.baseline_tput = 100.0
    c.shared_state.save(session_dir)
    try:
        await c.tasks.create(
            kind="report",
            params={"session_dir": str(session_dir)},
            idempotency_key="k-report-loop",
        )
        reason = await c.run(max_ticks=20, tick_interval_sec=0.0)
        # ``run`` returns the resolved stop reason. ``report_emitted``
        # must win over the ``max_ticks`` fallback.
        assert reason == "report_emitted", reason
    finally:
        await c.stop()
