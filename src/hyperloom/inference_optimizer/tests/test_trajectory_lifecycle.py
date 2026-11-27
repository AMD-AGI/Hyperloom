# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lifecycle events on the trajectory ledger: phase, intent, task, and retry linkage."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases import machine_state
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import TaskRegistry, TaskSpec
from hyperloom.orchestrator.tests.mock_backend import MockBackend, MockTurn, ScriptedPlan
from hyperloom.inference_optimizer.trace import trajectory_projection as trajmap
from hyperloom.inference_optimizer.trace import trajectory_trace as tt


def _rows(session_dir: Path, event_type: str) -> list[dict]:
    return [r for r in tt.load_events(session_dir) if r["event_type"] == event_type]


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _backends(orchestration: ScriptedPlan) -> dict[str, MockBackend]:
    return {"orchestration": MockBackend(orchestration, name="o")}


def _propose(action_name: str) -> Intent:
    return Intent(type=IntentType.PROPOSE_ACTION, payload={"action_name": action_name, "predicted_gain_pct": 1.5})


@pytest.mark.asyncio
async def test_task_rows_follow_the_registry_state_machine(tmp_path):
    registry = TaskRegistry(SqliteConnection(tmp_path / "db.sqlite"))
    try:
        with tt.trajectory_scope(session_dir=tmp_path, parent_span_id="creator"):
            task, _ = await registry.create_or_return_existing(
                TaskSpec(kind="sweep", params={}, idempotency_key="k1", requires_lanes=["a"])
            )
            again, _ = await registry.create_or_return_existing(TaskSpec(kind="sweep", params={}, idempotency_key="k1"))
            with tt.trajectory_scope(task_id=task.task_id, parent_span_id=task.task_id):
                await registry.transition(task.task_id, "running")
                await registry.transition(task.task_id, "failed", evidence={"reason": "oom", "blob": {"x": 1}})
            other, _ = await registry.create_or_return_existing(
                TaskSpec(kind="profile", params={}, idempotency_key="k2")
            )
            assert await registry.cancel_family(["profile"], reason="prune") == [other.task_id]
    finally:
        registry.db.close()

    assert again.task_id == task.task_id
    rows = _rows(tmp_path, tt.EVENT_TASK)
    mine = [r for r in rows if r["span_id"] == task.task_id]
    assert [r["status"] for r in mine] == [tt.STATUS_QUEUED, tt.STATUS_STARTED, tt.STATUS_FAILED]
    assert {r["task_id"] for r in mine} == {task.task_id}
    assert [r["parent_span_id"] for r in mine] == ["creator", None, None]
    assert mine[0]["attributes"] == {"name": "sweep", "kind": "sweep", "requires_lanes": ["a"]}
    assert mine[2]["attributes"]["reason"] == "oom"
    assert "blob" not in mine[2]["attributes"]
    cancelled = [r for r in rows if r["span_id"] == other.task_id]
    assert [r["status"] for r in cancelled] == [tt.STATUS_QUEUED, tt.STATUS_CANCELLED]

    openings = trajmap.span_openings(rows)
    spec = trajmap.project_row(mine[2], openings)
    assert spec is not None
    assert spec.name == "task:sweep"
    assert spec.metadata["parent_span_id"] == "creator"
    assert spec.metadata["queued_ts"] == mine[0]["ts"]
    assert spec.start == trajmap.lfmap.parse_ts(mine[1]["ts"])


@pytest.mark.asyncio
async def test_a_queued_task_row_carries_its_dispatch_class(tmp_path):
    registry = TaskRegistry(SqliteConnection(tmp_path / "db.sqlite"))
    try:
        with tt.trajectory_scope(session_dir=tmp_path):
            task, _ = await registry.create_or_return_existing(
                TaskSpec(
                    kind="explore",
                    params={},
                    idempotency_key="k1",
                    dispatch_class="llm",
                    dispatch_origin={"phase": "EXPLORE", "macro_cycle": 0, "tick": 3},
                ),
            )
    finally:
        registry.db.close()

    (queued,) = [r for r in _rows(tmp_path, tt.EVENT_TASK) if r["span_id"] == task.task_id]
    assert queued["attributes"]["dispatch_class"] == "llm"


def test_a_phase_transition_is_a_point_event_in_the_new_phase(tmp_path):
    state = SharedState()
    state.phase = "BASELINE"
    with tt.trajectory_scope(session_dir=tmp_path, component="coordinator"):
        machine_state.record_phase_transition(state, to_phase="EXPLORE", reason="baseline_done")
    (row,) = _rows(tmp_path, tt.EVENT_PHASE)
    assert row["status"] == tt.STATUS_POINT
    assert row["phase"] == "EXPLORE"
    assert row["attributes"]["from_phase"] == "BASELINE"
    assert row["attributes"]["reason"] == "baseline_done"


@pytest.mark.asyncio
async def test_a_proposed_action_links_call_intent_and_task(session_dir):
    c = Coordinator(session_dir, backends=_backends(ScriptedPlan(turns=[MockTurn(intents=[_propose("baseline")])])))
    try:
        with tt.trajectory_scope(session_dir=session_dir, component="coordinator"):
            await c.tick(1)
    finally:
        await c.stop()

    (call,) = [
        r
        for r in _rows(session_dir, tt.EVENT_LLM_CALL)
        if r["status"] == tt.STATUS_COMPLETED and r["agent"] == "orchestration"
    ]
    intents = _rows(session_dir, tt.EVENT_INTENT)
    (propose_intent,) = [r for r in intents if r["attributes"]["name"] == IntentType.PROPOSE_ACTION.value]
    assert propose_intent["parent_span_id"] == call["span_id"]
    assert propose_intent["call_id"] == call["call_id"]
    assert propose_intent["attributes"]["admitted"] is True

    (queued,) = [r for r in _rows(session_dir, tt.EVENT_TASK) if r["status"] == tt.STATUS_QUEUED]
    assert queued["parent_span_id"] == propose_intent["span_id"]
    assert queued["attributes"]["kind"] == "baseline"


@pytest.mark.asyncio
async def test_a_dispatched_task_runs_under_its_own_scope(session_dir):
    seen: dict[str, object] = {}

    async def _executor(ctx):
        seen["context"] = tt.current_context()
        return {"tput": 1}

    delegate = Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": "baseline", "params": {"runs": 1}, "idempotency_key": "k-traj"},
    )
    c = Coordinator(session_dir, backends=_backends(ScriptedPlan(turns=[MockTurn(intents=[delegate])])))
    c.sub.register_executor("baseline", _executor)
    try:
        with tt.trajectory_scope(session_dir=session_dir, component="coordinator"):
            await c.tick(1)
    finally:
        await c.stop()

    rows = _rows(session_dir, tt.EVENT_TASK)
    (task_id,) = {r["span_id"] for r in rows}
    assert [r["status"] for r in rows] == [tt.STATUS_QUEUED, tt.STATUS_STARTED, tt.STATUS_COMPLETED]
    (delegate_intent,) = [r for r in _rows(session_dir, tt.EVENT_INTENT) if r["attributes"]["name"] == "delegate"]
    assert rows[0]["parent_span_id"] == delegate_intent["span_id"]
    ctx = seen["context"]
    assert (ctx.task_id, ctx.parent_span_id) == (task_id, task_id)


@pytest.mark.asyncio
async def test_an_auto_retry_hangs_off_the_failed_task(session_dir, monkeypatch):
    from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentResult
    from hyperloom.orchestrator.state.task_registry import Task

    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY", "1")
    failed = Task(task_id="spec-r", kind="specialist", state="running", params={}, idempotency_key="spec-r-key")
    result = SubAgentResult(task_id="spec-r", state="failed", result={"runner_status": "stale"}, error="timeout")
    c = Coordinator(session_dir, backends=_backends(ScriptedPlan(turns=[], default_intent=_heartbeat())))
    try:
        with tt.trajectory_scope(session_dir=session_dir, component="coordinator"):
            assert await c.specialist_dispatch.maybe_auto_retry_specialist(failed, result) is True
    finally:
        await c.stop()

    (retry,) = _rows(session_dir, tt.EVENT_TASK_RETRY)
    assert (retry["parent_span_id"], retry["task_id"]) == ("spec-r", "spec-r")
    assert retry["attributes"]["attempt"] == 1
    (queued,) = [r for r in _rows(session_dir, tt.EVENT_TASK) if r["span_id"] == retry["attributes"]["retry_task_id"]]
    assert queued["parent_span_id"] == "spec-r"


@pytest.mark.asyncio
async def test_a_denied_intent_is_recorded_as_not_admitted(session_dir):
    c = Coordinator(session_dir, backends=_backends(ScriptedPlan(turns=[], default_intent=_heartbeat())))
    try:
        with tt.trajectory_scope(session_dir=session_dir, component="coordinator"):
            await c.router.handle_intent("nobody", _propose("baseline"))
    finally:
        await c.stop()

    (row,) = _rows(session_dir, tt.EVENT_INTENT)
    assert row["attributes"]["admitted"] is False
    assert "nobody" in row["attributes"]["denied"]
    assert not _rows(session_dir, tt.EVENT_TASK)
