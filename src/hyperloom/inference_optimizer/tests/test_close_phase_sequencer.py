# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE phase seven-step sequencer tests."""

from __future__ import annotations

import asyncio
import json
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.common.deadline import Deadline
from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane
from hyperloom.inference_optimizer.breakdown.collectors.v6_close import collect_v6_close
from hyperloom.inference_optimizer.breakdown.recorder import phase_event
from hyperloom.inference_optimizer.breakdown.recorder.assembler import assemble_parts, phase_event_parts
from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.orchestrator.knowledge.config import KnowledgeConfig, KnowledgeStoreMode
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.roles.mock_backend import (
    MockBackend,
    MockTurn,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases import machine_state
from hyperloom.orchestrator.phases.machine import Transition
from hyperloom.orchestrator.phases.close import (
    _CLOSE_STEP_WAIT_CEILING_SEC,
    _CLOSE_STEP_WAIT_FLOOR_SEC,
)
from hyperloom.orchestrator.actions.executors.session_breakdown import session_breakdown_executor
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import task_dispatch_record

from .conftest import make_coordinator


@dataclass
class _BareState(SharedState):
    """SharedState in the CLOSE phase that counts saves instead of writing them."""

    phase: str = "CLOSE"
    framework: str = "sglang"
    closing_report_task_id: str = ""
    recipe_kb_session_id: str = ""
    recipe_kb_session_summary: dict[str, Any] = field(default_factory=dict)
    stop_reason: str = ""
    close_sequence_done: bool = False
    recipe_finalize_status: str = ""
    recipe_finalize_attempts: int = 0
    recipe_finalize_outcome: dict[str, Any] = field(default_factory=dict)
    phase_history: list[dict[str, Any]] = field(default_factory=list)
    max_minutes: int = 0
    closing_grace_sec: float | None = None
    save_count: int = 0

    def save(self, _session_dir: Path | None = None) -> None:
        self.save_count += 1


@dataclass
class _StubTaskRow:
    task_id: str
    kind: str
    state: str
    params: dict
    idempotency_key: str
    # Mirrors the real ``Task``: the dispatch path reads both before running a row.
    requires_lanes: list = field(default_factory=list)
    lease_ttl_sec: int = 0


class _StubTaskRegistry:
    """Task registry double; tracks insertion order to assert step 1 before step 2."""

    def __init__(self):
        self._by_key: dict[str, _StubTaskRow] = {}
        self._by_id: dict[str, _StubTaskRow] = {}
        self.insertion_order: list[str] = []

    async def create_or_return_existing(
        self,
        *,
        kind: str,
        params: dict,
        idempotency_key: str,
        requires_lanes: list | None = None,
        allowed_tools: list | None = None,
        side_effects: list | None = None,
        lease_ttl_sec: int = 0,
        task_id: str | None = None,
        dispatch_class: str | None = None,
        dispatch_origin: dict | None = None,
    ):
        existing = self._by_key.get(idempotency_key)
        if existing is not None:
            return existing, True
        import uuid as _uuid

        tid = task_id or _uuid.uuid4().hex
        row = _StubTaskRow(
            task_id=tid,
            kind=kind,
            # Matches the registry's INSERT: a new row is always ``queued``.
            state="queued",
            params=dict(params),
            idempotency_key=idempotency_key,
        )
        self._by_key[idempotency_key] = row
        self._by_id[tid] = row
        self.insertion_order.append(idempotency_key)
        return row, False

    async def get(self, task_id):
        from hyperloom.orchestrator.state.task_registry import TaskNotFound

        row = self._by_id.get(task_id)
        if row is None:
            raise TaskNotFound(task_id)
        return row


class _StubRecipeKB:
    """Recipe KB double for the CLOSE fact-finalize step and the T4 hook."""

    enabled: bool = True

    def __init__(
        self,
        *,
        drain_remaining: int = 0,
        drain_raises: BaseException | None = None,
    ):
        self.drain_calls: int = 0
        self._drain_remaining = drain_remaining
        self._drain_raises = drain_raises

    def drain_pending(self, *, timeout_sec: float = 60.0) -> dict:
        self.drain_calls += 1
        if self._drain_raises is not None:
            raise self._drain_raises
        return {"remaining": self._drain_remaining}

    def read_recipe_exact(self, *, model: str, hardware: str) -> dict:
        return {}

    def update_recipe(self, **kwargs) -> dict:
        return {"status": "auto_accepted"}

    def put_recipe(self, **kwargs) -> dict:
        return {"status": "written", "canonical_id": kwargs.get("canonical_id", "")}

    def get_authoritative_recipe(self, *, canonical_id: str) -> dict:
        return {}


class _StubSubResult:
    """Minimal sub-agent run result: only ``.state`` is read by the CLOSE sequencer."""

    def __init__(self, state: str = "succeeded"):
        self.state = state


class _StubSubAgentRunner:
    """``Coordinator.sub`` double returning terminal-succeeded so the sequencer advances."""

    def __init__(self):
        self.run_calls: list[Any] = []

    async def run_task(self, task, *args, **kwargs):
        self.run_calls.append(task)
        return _StubSubResult(state="succeeded")


@pytest.fixture
def coord(tmp_path: Path):
    """Lean Coordinator stub for hook unit tests."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _BareState()
    c.tasks = _StubTaskRegistry()
    c.sub = _StubSubAgentRunner()
    c.knowledge_plane = KnowledgePlane(recipe_kb=None)
    c.knowledge_plane = None
    c.role_registry = {}
    return c


def _close_phase_history_row() -> dict[str, Any]:
    return {"to_phase": "CLOSE", "reason": "sweep_done", "evidence": {}}


@pytest.mark.asyncio
async def test_record_close_step_appends_to_evidence_close_steps(coord):
    coord.shared_state.phase_history = [_close_phase_history_row()]
    await coord.phase_close._record_close_step("report", status="done", task_id="t-1")
    rows = coord.shared_state.phase_history[-1]["evidence"]["close_steps"]
    assert len(rows) == 1
    assert rows[0]["step"] == "report"
    assert rows[0]["status"] == "done"
    assert rows[0]["task_id"] == "t-1"
    assert "ts" in rows[0]
    assert "detail" not in rows[0]
    assert coord.shared_state.save_count == 1


@pytest.mark.asyncio
async def test_record_close_step_optional_detail(coord):
    coord.shared_state.phase_history = [_close_phase_history_row()]
    await coord.phase_close._record_close_step(
        "recipe_kb_commit",
        status="failed",
        detail="recipe kb unreachable",
    )
    row = coord.shared_state.phase_history[-1]["evidence"]["close_steps"][0]
    assert row["detail"] == "recipe kb unreachable"


@pytest.mark.asyncio
async def test_record_close_step_creates_missing_evidence_dict(coord):
    """Phase_history row with no ``evidence`` key gets one installed."""
    coord.shared_state.phase_history = [
        {"to_phase": "CLOSE", "reason": "sweep_done"},
    ]
    await coord.phase_close._record_close_step("done", status="done")
    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert "close_steps" in evidence


@pytest.mark.asyncio
async def test_record_close_step_replaces_non_list_close_steps(coord):
    """Defensive: malformed pre-existing close_steps (not a list) gets replaced."""
    coord.shared_state.phase_history = [
        {
            "to_phase": "CLOSE",
            "evidence": {"close_steps": "broken"},
        }
    ]
    await coord.phase_close._record_close_step("report", status="done")
    assert isinstance(coord.shared_state.phase_history[-1]["evidence"]["close_steps"], list)


@pytest.mark.asyncio
async def test_record_close_step_no_op_when_history_empty(coord):
    coord.shared_state.phase_history = []
    await coord.phase_close._record_close_step("report", status="done")


@pytest.mark.asyncio
async def test_enqueue_internal_report_task_fresh(coord):
    task = await coord.phase_close._enqueue_internal_report_task(reason="close_phase_entry")
    assert task.kind == "report"
    assert task.idempotency_key == "internal-report-close_phase_entry"
    assert task.params["source"] == "coordinator_internal"
    assert task.params["reason"] == "close_phase_entry"
    assert coord.shared_state.closing_report_task_id == task.task_id


@pytest.mark.asyncio
async def test_enqueue_internal_report_task_reuses_existing(coord):
    """When the wall-clock deadline already enqueued a report task, reuse it."""
    existing = _StubTaskRow(
        task_id="wallclock-report",
        kind="report",
        state="succeeded",
        params={},
        idempotency_key="closing-report-1234",
    )
    coord.tasks._by_id["wallclock-report"] = existing
    coord.shared_state.closing_report_task_id = "wallclock-report"

    task = await coord.phase_close._enqueue_internal_report_task(reason="close_phase_entry")
    assert task is existing
    assert "internal-report-close_phase_entry" not in coord.tasks._by_key


@pytest.mark.asyncio
async def test_enqueue_internal_report_task_replaces_a_cancelled_one(coord):
    """A report the deadline path enqueued and then cancelled cannot be run; the sequencer needs a live one."""
    dead = _StubTaskRow(
        task_id="wallclock-report",
        kind="report",
        state="cancelled",
        params={},
        idempotency_key="closing-report-1234",
    )
    coord.tasks._by_id["wallclock-report"] = dead
    coord.shared_state.closing_report_task_id = "wallclock-report"

    task = await coord.phase_close._enqueue_internal_report_task(reason="close_phase_entry")

    assert task is not dead
    assert task.state == "queued"
    assert coord.shared_state.closing_report_task_id == task.task_id


@pytest.mark.asyncio
async def test_enqueue_internal_report_task_retries_past_a_dead_idempotent_row(coord):
    """The idempotency key itself can resolve to a corpse; the retry key mints a runnable row."""
    coord.tasks._by_key["internal-report-close_phase_entry"] = _StubTaskRow(
        task_id="dead-idempotent",
        kind="report",
        state="cancelled",
        params={},
        idempotency_key="internal-report-close_phase_entry",
    )

    task = await coord.phase_close._enqueue_internal_report_task(reason="close_phase_entry")

    assert task.task_id != "dead-idempotent"
    assert task.idempotency_key == "internal-report-close_phase_entry-retry"
    assert task.state == "queued"


@pytest.mark.asyncio
async def test_close_sequencer_still_reports_when_the_first_report_task_was_cancelled(coord):
    """End to end: a session that hits its deadline is the one whose report matters most."""
    coord.shared_state.phase_history = [_close_phase_history_row()]
    coord.tasks._by_id["wallclock-report"] = _StubTaskRow(
        task_id="wallclock-report",
        kind="report",
        state="cancelled",
        params={},
        idempotency_key="closing-report-1234",
    )
    coord.shared_state.closing_report_task_id = "wallclock-report"

    await coord.phase_close.on_enter_close(
        Transition(from_phase="SWEEP", to_phase="CLOSE", reason="test", evidence={}, loopback=False)
    )

    rows = coord.shared_state.phase_history[-1]["evidence"]["close_steps"]
    by_step = {r["step"]: r for r in rows}
    assert by_step["report"]["status"] == "done"
    assert [t.task_id for t in coord.sub.run_calls] != ["wallclock-report"]


@pytest.mark.asyncio
async def test_a_terminal_task_is_reported_not_run(coord):
    """Backstop: nothing hands a terminal row to ``run_task``, whose ``queued -> running`` would raise."""
    done = _StubTaskRow(
        task_id="already-done",
        kind="report",
        state="succeeded",
        params={},
        idempotency_key="internal-report-close_phase_entry",
    )

    state = await coord.phase_close._run_close_task(done, step="1 (report)")

    assert state == "succeeded"
    assert coord.sub.run_calls == []


class _FinishesWhileWaiting(_StubTaskRegistry):
    """Registry whose running row lands terminal after ``lands_on`` lookups."""

    def __init__(self, terminal_state: str, *, lands_on: int = 2):
        super().__init__()
        self._terminal_state = terminal_state
        self._lands_on = lands_on
        self.gets = 0

    async def get(self, task_id):
        row = await super().get(task_id)
        self.gets += 1
        if self.gets >= self._lands_on:
            row.state = self._terminal_state
        return row


def _running_report_row(coord, *, kind: str = "report") -> _StubTaskRow:
    """Register a close-step task the wall-clock deadline path already enqueued AND dispatched."""
    row = _StubTaskRow(
        task_id="wallclock-report",
        kind=kind,
        state="running",
        params={},
        idempotency_key="closing-report-1234",
    )
    coord.tasks._by_id[row.task_id] = row
    return row


def _clock_advancing_by(monkeypatch: pytest.MonkeyPatch, step_sec: float) -> None:
    """Give the CLOSE module a monotonic clock that jumps ``step_sec`` per read."""
    from hyperloom.orchestrator.phases import close as close_mod

    now = 0.0

    def _monotonic() -> float:
        nonlocal now
        now += step_sec
        return now

    monkeypatch.setattr(
        close_mod,
        "time",
        SimpleNamespace(monotonic=_monotonic, time=time.time),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_state", ["succeeded", "failed"])
async def test_a_running_task_is_waited_for_not_re_run(coord, terminal_state: str):
    """``running -> running`` is not a transition the registry has; asking for it kills the step."""
    coord.tasks = _FinishesWhileWaiting(terminal_state)
    coord.dispatcher.poll_sec = 0.01
    coord.shared_state.max_minutes = 60

    state = await coord.phase_close._run_close_task(_running_report_row(coord), step="1 (report)")

    assert state == terminal_state
    assert coord.sub.run_calls == []


@pytest.mark.asyncio
async def test_a_running_task_that_never_lands_is_reported_not_waited_on_forever(
    coord,
    monkeypatch: pytest.MonkeyPatch,
):
    """The wait is patient, not unbounded: a step that never lands is recorded, not awaited forever."""
    coord.dispatcher.poll_sec = 0.0
    coord.shared_state.max_minutes = 60
    _clock_advancing_by(monkeypatch, step_sec=30.0)

    state = await coord.phase_close._run_close_task(_running_report_row(coord), step="1 (report)")

    assert state == "running"
    assert coord.sub.run_calls == []


@pytest.mark.asyncio
async def test_a_fresh_report_that_never_lands_is_not_awaited_forever(coord):
    """The close-step timeout cancels its waiter, not the execution's ownership."""
    from hyperloom.orchestrator.bus.resource_lock import ResourceLockManager, SqliteLeaseBackend
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
    from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner
    from hyperloom.orchestrator.state.task_registry import TaskRegistry

    db = SqliteConnection(coord.session_dir / "close-task.db")
    coord.tasks = TaskRegistry(db)
    coord.locks = ResourceLockManager(SqliteLeaseBackend(db))
    coord.sub = SubAgentRunner(coord.locks, coord.tasks)
    coord.shared_state.max_minutes = 60
    coord.phase_close._close_step_wait_sec = lambda _task: 0.05  # type: ignore[method-assign]
    finish = asyncio.Event()
    calls = []

    async def execute(ctx):
        calls.append(ctx.task.task_id)
        await finish.wait()
        return {"status": "ok"}

    coord.sub.register_executor("report", execute)
    queued = await coord.tasks.create(kind="report", params={}, idempotency_key="internal-report-close_phase_entry")
    try:
        started = time.monotonic()
        state = await coord.phase_close._run_close_task(queued, step="1 (report)")
        elapsed = time.monotonic() - started

        assert state == "running"
        assert elapsed < 2.0
        assert calls == [queued.task_id]
        assert (await coord.tasks.get(queued.task_id)).state == "running"
        handle = coord.dispatcher._inflight_actions[queued.task_id]
        assert handle.scope.cancelled
        assert handle.scope.reason == "caller_cancelled"
        assert coord.dispatcher._executions
        assert all(not execution.done() for execution in coord.dispatcher._executions)
    finally:
        executions = tuple(coord.dispatcher._executions)
        finish.set()
        await asyncio.gather(*executions)
    try:
        assert (await coord.tasks.get(queued.task_id)).state == "succeeded"
        assert not coord.dispatcher._inflight_actions
        assert not coord.dispatcher._executions
    finally:
        db.close()


@pytest.mark.asyncio
async def test_the_wait_for_a_running_report_outlives_a_short_session_reserve(
    coord,
    monkeypatch: pytest.MonkeyPatch,
):
    """A ten-minute session reserves twelve seconds for CLOSE; no report is written in twelve seconds."""
    coord.tasks = _FinishesWhileWaiting("succeeded", lands_on=5)
    coord.dispatcher.poll_sec = 0.0
    coord.shared_state.max_minutes = 10
    assert coord.shared_state.closing_reserve_sec() == pytest.approx(12.0)
    # Five looks at five simulated seconds apiece: past the reserve, inside the two minutes the catalogue prices a
    # report at.
    _clock_advancing_by(monkeypatch, step_sec=5.0)

    state = await coord.phase_close._run_close_task(_running_report_row(coord), step="1 (report)")

    assert state == "succeeded"
    assert coord.sub.run_calls == []


def test_the_wait_is_the_step_s_own_expected_runtime(coord):
    bound = coord.phase_close._close_step_wait_sec(_running_report_row(coord))

    assert bound == pytest.approx(ACTION_CATALOGUE["report"].typical_runtime_min * 60.0)


def test_a_step_the_catalogue_prices_at_almost_nothing_still_gets_the_floor(coord):
    """``session_breakdown`` is priced at 12s; giving up on it after 12s is giving up on it."""
    row = _running_report_row(coord, kind="session_breakdown")

    assert coord.phase_close._close_step_wait_sec(row) == pytest.approx(_CLOSE_STEP_WAIT_FLOOR_SEC)


def test_an_uncatalogued_step_gets_the_floor_too(coord):
    row = _running_report_row(coord, kind="not_an_action")

    assert coord.phase_close._close_step_wait_sec(row) == pytest.approx(_CLOSE_STEP_WAIT_FLOOR_SEC)


def test_an_extravagantly_priced_step_is_capped(coord):
    """A wedged step must not hold the process open for as long as its action might legitimately run."""
    coord.action_registry = {"report": SimpleNamespace(typical_runtime_min=1000.0)}

    bound = coord.phase_close._close_step_wait_sec(_running_report_row(coord))

    assert bound == pytest.approx(_CLOSE_STEP_WAIT_CEILING_SEC)


@pytest.mark.asyncio
async def test_the_sequencer_records_the_state_a_running_report_ended_in(coord):
    """End to end: the waited-for report is reported like any other outcome."""
    coord.tasks = _FinishesWhileWaiting("succeeded")
    coord.dispatcher.poll_sec = 0.01
    coord.shared_state.max_minutes = 60
    coord.shared_state.phase_history = [_close_phase_history_row()]
    _running_report_row(coord)
    coord.shared_state.closing_report_task_id = "wallclock-report"

    await coord.phase_close.on_enter_close(
        Transition(from_phase="SWEEP", to_phase="CLOSE", reason="test", evidence={}, loopback=False)
    )

    rows = coord.shared_state.phase_history[-1]["evidence"]["close_steps"]
    report = next(r for r in rows if r["step"] == "report")
    assert report["status"] == "done"
    assert "wallclock-report" not in [t.task_id for t in coord.sub.run_calls]


@pytest.mark.asyncio
async def test_enqueue_internal_session_breakdown_task(coord):
    task = await coord.phase_close._enqueue_internal_session_breakdown_task(
        reason="close_phase_entry",
    )
    assert task.kind == "session_breakdown"
    assert task.idempotency_key == "internal-session_breakdown-close_phase_entry"
    assert task.params["source"] == "coordinator_internal"


@pytest.mark.asyncio
async def test_a_resumed_leg_writes_its_own_report_and_breakdown(coord, tmp_path, monkeypatch):
    """An earlier leg's signal close already ran both steps; the resumed leg must not keep their stale artifacts."""
    monkeypatch.setenv("HYPERLOOM_SESSION_PACKAGE_DEST", str(tmp_path / "session-packages"))
    coord.shared_state.phase_history = [_close_phase_history_row()]
    for kind in ("report", "session_breakdown"):
        key = f"internal-{kind}-close_phase_entry"
        row = _StubTaskRow(task_id=f"earlier-leg-{kind}", kind=kind, state="succeeded", params={}, idempotency_key=key)
        coord.tasks._by_key[key] = row
        coord.tasks._by_id[row.task_id] = row
    coord.shared_state.resumed_ts = "2026-09-25T09:46:24+00:00"

    await coord.phase_close.on_enter_close(
        Transition(from_phase="SWEEP", to_phase="CLOSE", reason="test", evidence={}, loopback=False)
    )

    ran = [(t.kind, t.idempotency_key) for t in coord.sub.run_calls if t.kind in {"report", "session_breakdown"}]
    assert ran == [
        ("report", "internal-report-close_phase_entry-leg-2026-09-25T09:46:24+00:00"),
        ("session_breakdown", "internal-session_breakdown-close_phase_entry-leg-2026-09-25T09:46:24+00:00"),
    ]


@pytest.mark.asyncio
async def test_close_sequencer_runs_all_steps_in_order_happy_path(
    coord,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "HYPERLOOM_SESSION_PACKAGE_DEST",
        str(tmp_path / "session-packages"),
    )
    coord.shared_state.phase_history = [_close_phase_history_row()]
    coord.knowledge_plane = KnowledgePlane(recipe_kb=_StubRecipeKB())
    coord.shared_state.recipe_kb_session_id = "sid-test"
    coord.shared_state.model_name = "model"
    coord.shared_state.gpu_type = "mi300x"
    # stop_reason is normally set by the phase machine before CLOSE is entered.
    coord.shared_state.stop_reason = "sweep_done"

    await coord.phase_close.on_enter_close(
        Transition(from_phase="SWEEP", to_phase="CLOSE", reason="test", evidence={}, loopback=False)
    )

    rows = coord.shared_state.phase_history[-1]["evidence"]["close_steps"]
    steps = [r["step"] for r in rows]
    assert steps == [
        "sequencer_started",
        "stack_revalidation",
        "post_opt_roofline",
        "fact_finalize",
        "report",
        "session_breakdown",
        "langfuse_flush",
        "artifact_package",
        "ndjson_drain",
        "done",
    ]
    by_step = {r["step"]: r for r in rows}
    assert by_step["stack_revalidation"]["status"] == "done"
    # No kernel patch in optimization_stack → post-opt roofline is skipped in a param-search session.
    assert by_step["post_opt_roofline"]["status"] == "skipped"
    assert by_step["report"]["status"] == "done"
    assert by_step["session_breakdown"]["status"] == "done"
    # fact_finalize now runs first and writes optimization_journal.json, so the artifact package has a curated file to
    # include.
    assert by_step["artifact_package"]["status"] == "done"
    assert by_step["fact_finalize"]["status"] == "done"
    assert "status=written" in by_step["fact_finalize"]["detail"]
    assert "backend=local" in by_step["fact_finalize"]["detail"]
    assert by_step["langfuse_flush"]["status"] == "done"
    assert by_step["ndjson_drain"]["status"] == "skipped"
    assert by_step["done"]["status"] == "done"
    assert coord.shared_state.close_sequence_done is True
    # A normal SWEEP completion's sweep_done reason must be preserved.
    assert coord.shared_state.stop_reason == "sweep_done"


@pytest.mark.asyncio
async def test_custom_close_packages_explicit_export_refusal(coord, tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_SESSION_PACKAGE_DEST", str(tmp_path / "packages"))
    coord.shared_state.framework = "custom"
    coord.shared_state.phase_history = [_close_phase_history_row()]
    coord.shared_state.to_dict = lambda: {"framework": "custom", "framework_repo_path": str(tmp_path / "missing")}
    await coord.phase_close.on_enter_close(
        Transition(from_phase="SWEEP", to_phase="CLOSE", reason="test", evidence={}, loopback=False)
    )
    rows = coord.shared_state.phase_history[-1]["evidence"]["close_steps"]
    export_step = next(row for row in rows if row["step"] == "inference_export")
    assert export_step["status"] == "failed"
    assert coord.shared_state.close_sequence_done is True
    packages = list((tmp_path / "packages").rglob("*.zip"))
    assert packages
    with zipfile.ZipFile(packages[0]) as archive:
        manifest = json.loads(archive.read("deployment/deployment.json"))
        assert manifest["status"] == "incomplete"


@pytest.mark.asyncio
async def test_close_sequencer_records_its_own_verdict_and_artifacts(
    coord,
    tmp_path,
    monkeypatch,
):
    """The sequencer states the close-out rather than leaving it to be inferred.

    The projection this replaces had to guess a verdict from which steps were
    present, and it ran while the sequence was still going, so it labelled a
    healthy session ``degraded``. Here the same happy path is asserted to
    settle at ``succeeded``, with the artifact paths named by the steps that
    produced them instead of recovered by probing the session tree.
    """
    monkeypatch.setenv("HYPERLOOM_SESSION_PACKAGE_DEST", str(tmp_path / "session-packages"))
    coord.shared_state.phase_history = [_close_phase_history_row()]
    coord.knowledge_plane = KnowledgePlane(recipe_kb=_StubRecipeKB())
    coord.shared_state.recipe_kb_session_id = "sid-test"
    coord.shared_state.model_name = "model"
    coord.shared_state.gpu_type = "mi300x"
    # stop_reason is normally set by the phase machine before CLOSE is entered.
    coord.shared_state.stop_reason = "sweep_done"

    await coord.phase_close.on_enter_close(
        Transition(from_phase="SWEEP", to_phase="CLOSE", reason="test", evidence={}, loopback=False)
    )

    recorded = assemble_parts(tmp_path, warnings=[])["close"]
    assert recorded["status"] == "succeeded"
    assert recorded["close_sequence_done"] is True
    assert recorded["stop_reason"] == "sweep_done"
    assert [row["step"] for row in recorded["steps"]] == [
        "sequencer_started",
        "stack_revalidation",
        "post_opt_roofline",
        "fact_finalize",
        "report",
        "session_breakdown",
        "langfuse_flush",
        "artifact_package",
        "ndjson_drain",
        "done",
    ]
    # Named by the artifact_package step from the path it was handed, not
    # parsed back out of that step's free-text detail.
    assert recorded["artifacts"]["artifact_package_path"].endswith(".zip")
    warnings: list[str] = []
    collect_v6_close(warnings, recorded=recorded)
    assert warnings == []


@pytest.mark.asyncio
async def test_close_sequencer_records_the_recipe_publication_under_close(
    coord,
    tmp_path,
    monkeypatch,
):
    """The publication lands in ``close.kb_write_back``, not the timeline.

    It is what the session does unconditionally on its way out, so whether it
    happened is always worth answering -- and a timeline event that did not
    happen is simply absent, leaving nowhere to answer it.
    """
    monkeypatch.setenv("HYPERLOOM_SESSION_PACKAGE_DEST", str(tmp_path / "session-packages"))
    coord.shared_state.phase_history = [_close_phase_history_row()]
    coord.knowledge_plane = KnowledgePlane(recipe_kb=_StubRecipeKB())
    coord.shared_state.recipe_kb_session_id = "sid-test"
    coord.shared_state.model_name = "model"
    coord.shared_state.gpu_type = "mi300x"

    await coord.phase_close.on_enter_close(
        Transition(from_phase="SWEEP", to_phase="CLOSE", reason="test", evidence={}, loopback=False)
    )

    write_back = assemble_parts(tmp_path, warnings=[])["close"]["kb_write_back"]
    assert write_back["status"] == "written"
    # Stated by the publisher at the exit it took, not recovered downstream by
    # matching substrings against the reason string.
    assert write_back["result_type"] == "written"
    assert write_back["backend"] == "local"
    # One attempt, opened by the CLOSE path and settled by it.
    assert [(row["attempt"], row["source"], row["status"]) for row in write_back["attempts"]] == [
        (1, "close", "written"),
    ]
    assert "queue" in write_back


@pytest.mark.asyncio
async def test_close_sequencer_records_degraded_when_a_step_fails(
    coord,
    tmp_path,
    monkeypatch,
):
    """``degraded`` now means a step failed, which is what it always read as."""
    monkeypatch.setenv("HYPERLOOM_SESSION_PACKAGE_DEST", str(tmp_path / "session-packages"))
    coord.shared_state.phase_history = [_close_phase_history_row()]
    coord.knowledge_plane = KnowledgePlane(recipe_kb=_StubRecipeKB())
    coord.shared_state.recipe_kb_session_id = "sid-test"

    class _FailingRunner(_StubSubAgentRunner):
        async def run_task(self, task, *args, **kwargs):
            self.run_calls.append(task)
            return _StubSubResult(state="failed")

    coord.sub = _FailingRunner()

    await coord.phase_close.on_enter_close(
        Transition(from_phase="SWEEP", to_phase="CLOSE", reason="test", evidence={}, loopback=False)
    )

    recorded = assemble_parts(tmp_path, warnings=[])["close"]
    assert recorded["status"] == "degraded"
    assert recorded["close_sequence_done"] is True
    assert [row["step"] for row in recorded["steps"] if row["status"] == "failed"] == [
        "report",
        "session_breakdown",
    ]


@pytest.mark.asyncio
async def test_close_sequencer_surfaces_remote_finalize_failure(
    coord,
    monkeypatch,
):
    coord.shared_state.phase_history = [_close_phase_history_row()]
    monkeypatch.setattr(
        coord.recipe_journal,
        "finalize_recipe_and_journal",
        lambda *, source: {
            "status": "error",
            "reason": "KBStoreError",
            "backend": "kb-store",
        },
    )

    await coord.phase_close.on_enter_close(
        Transition(from_phase="SWEEP", to_phase="CLOSE", reason="test", evidence={}, loopback=False)
    )

    rows = coord.shared_state.phase_history[-1]["evidence"]["close_steps"]
    fact = next(row for row in rows if row["step"] == "fact_finalize")
    assert fact["status"] == "failed"
    assert fact["detail"] == ("status=error reason=KBStoreError backend=kb-store")


@pytest.mark.asyncio
async def test_close_sequencer_does_not_overwrite_caller_set_stop_reason(
    coord,
):
    """An operator-set ``stop_reason`` (e.g. ``signal``) must survive step 5."""
    coord.shared_state.phase_history = [_close_phase_history_row()]
    coord.shared_state.stop_reason = "signal"

    await coord.phase_close.on_enter_close(
        Transition(from_phase="SWEEP", to_phase="CLOSE", reason="test", evidence={}, loopback=False)
    )

    assert coord.shared_state.stop_reason == "signal"


async def _advance_into_close(coord, monkeypatch, reason: str) -> None:
    async def _entered(*, from_phase, to_phase, reason="", evidence=None):
        return None

    monkeypatch.setattr(coord.phase_machine, "_on_phase_entered", _entered)
    monkeypatch.setattr(
        machine_state,
        "compute_next_phase",
        lambda *_a, **_k: (machine_state.PHASE_CLOSE, reason, {"terminal": True}),
    )
    coord.shared_state.phase = machine_state.PHASE_SWEEP
    await coord.phase_machine.advance_phase_if_needed()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["sweep_budget_exhausted", "sweep_budget_cap"])
async def test_a_sweep_budget_exit_closes_the_run_as_time_exhausted(tmp_path: Path, monkeypatch, reason: str):
    coord = make_coordinator(tmp_path)

    await _advance_into_close(coord, monkeypatch, reason)

    assert coord.shared_state.phase == machine_state.PHASE_CLOSE
    assert coord.shared_state.stop_reason == "time_exhausted"


@pytest.mark.asyncio
async def test_a_close_reason_outside_the_stop_vocabulary_is_refused(tmp_path: Path, monkeypatch):
    coord = make_coordinator(tmp_path)

    with pytest.raises(ValueError, match="not_a_real_vocab_reason"):
        await _advance_into_close(coord, monkeypatch, "not_a_real_vocab_reason")

    assert coord.shared_state.phase == machine_state.PHASE_SWEEP


@pytest.mark.asyncio
async def test_entering_close_keeps_the_stop_reason_already_set(tmp_path: Path, monkeypatch):
    coord = make_coordinator(tmp_path)
    coord.shared_state.set_stop_reason("signal")

    await _advance_into_close(coord, monkeypatch, "sweep_done")

    assert coord.shared_state.stop_reason == "signal"


@pytest.mark.asyncio
async def test_close_sequencer_report_before_session_breakdown(coord):
    """Report task MUST be enqueued before session_breakdown."""
    coord.shared_state.phase_history = [_close_phase_history_row()]
    await coord.phase_close.on_enter_close(
        Transition(from_phase="SWEEP", to_phase="CLOSE", reason="test", evidence={}, loopback=False)
    )
    insertion = coord.tasks.insertion_order
    report_idx = insertion.index("internal-report-close_phase_entry")
    bd_idx = insertion.index("internal-session_breakdown-close_phase_entry")
    assert report_idx < bd_idx


@pytest.mark.asyncio
async def test_close_sequencer_skips_recipe_kb_steps_when_no_recipe_kb(coord):
    """``--degraded-kb`` runs (recipe_kb=None): NDJSON drain recorded 'skipped', not silent."""
    coord.shared_state.phase_history = [_close_phase_history_row()]
    await coord.phase_close.on_enter_close(
        Transition(from_phase="SWEEP", to_phase="CLOSE", reason="test", evidence={}, loopback=False)
    )
    rows = coord.shared_state.phase_history[-1]["evidence"]["close_steps"]
    drain_row = next(r for r in rows if r["step"] == "ndjson_drain")
    assert drain_row["status"] == "skipped"
    assert coord.shared_state.close_sequence_done is True


@pytest.mark.asyncio
async def test_phase_transition_into_close_runs_sequencer_e2e(tmp_path: Path):
    """End-to-end: real Coordinator + TaskRegistry enqueue both internal tasks and flip close_sequence_done."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "critic": MockBackend(idle_plan),
    }
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        knowledge_plane=None,
    )
    # Seed state at SWEEP boundary.
    coord.shared_state.phase = "SWEEP"
    coord.shared_state.phase_history = [
        {"to_phase": "EXPLORE", "evidence": {}, "reason": "prelude_done"},
        {"to_phase": "SWEEP", "evidence": {}, "reason": "kernel_no_more_leverage"},
    ]
    machine_state.record_phase_transition(
        coord.shared_state,
        to_phase="CLOSE",
        reason="sweep_done",
        evidence={"trigger": "test_e2e"},
    )
    await coord.phase_machine._on_phase_entered(from_phase="SWEEP", to_phase="CLOSE")

    rows_report = await coord.tasks.db.fetchall(
        "SELECT * FROM tasks WHERE idempotency_key=?",
        ("internal-report-close_phase_entry",),
    )
    rows_bd = await coord.tasks.db.fetchall(
        "SELECT * FROM tasks WHERE idempotency_key=?",
        ("internal-session_breakdown-close_phase_entry",),
    )
    assert len(rows_report) == 1
    assert len(rows_bd) == 1

    assert coord.shared_state.close_sequence_done is True
    evidence = coord.shared_state.phase_history[-1]["evidence"]
    steps = [r["step"] for r in evidence.get("close_steps", [])]
    assert "sequencer_started" in steps
    assert "done" in steps


class TestEveryTerminalReachesAWrittenReport:
    """The close sequence must not depend on the phase machine advancing.

    A run that cannot advance a phase -- because the machine has no next phase,
    because the step raised, or because the deadline it would have advanced at
    has already passed -- used to end with no report at all, since the sequencer
    was reachable only as a side effect of entering CLOSE.
    """

    @staticmethod
    def _coordinator(session_dir: Path) -> Coordinator:
        """A coordinator whose roles say nothing, so only the loop drives it."""
        session_dir.mkdir(exist_ok=True)
        idle = ScriptedPlan(turns=[MockTurn(intents=[])])
        return Coordinator(
            session_dir=session_dir,
            backends={name: MockBackend(idle) for name in ("orchestration", "critic")},
            role_registry=default_role_registry(),
            knowledge_plane=None,
        )

    @pytest.mark.asyncio
    async def test_a_run_that_never_advances_a_phase_still_closes(self, tmp_path: Path, monkeypatch):
        coord = self._coordinator(tmp_path / "session")

        async def _cannot_advance() -> None:
            raise RuntimeError("the phase machine has no next phase")

        monkeypatch.setattr(coord.phase_machine, "advance_phase_if_needed", _cannot_advance)
        try:
            reason = await coord.run(max_ticks=1, max_minutes=60, closing_grace_sec=0.0)
        finally:
            await coord.stop()

        assert reason == "max_ticks"
        assert coord.shared_state.close_sequence_done is True
        assert coord.shared_state.phase == "CLOSE"
        assert coord.shared_state.phase_history[-1]["reason"] == "max_ticks"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "phase,hint,sweep_status,transition_reason",
        [
            ("PRELUDE", "", "", "signal"),
            # No baseline was measured, so an early close is a baseline failure.
            ("PRELUDE", "skip_to_close", "", "baseline_failed"),
            ("SWEEP", "skip_to_close", "completed", "signal"),
            ("FRAMEWORK_AGENT", "", "", "signal"),
            ("KERNEL_AGENT", "", "", "signal"),
        ],
    )
    async def test_terminal_close_records_the_phase_before_dispatch(
        self, tmp_path, phase, hint, sweep_status, transition_reason
    ):
        coord = self._coordinator(tmp_path / "session")
        coord.sub.register_executor("session_breakdown", session_breakdown_executor)
        state = coord.shared_state
        state.max_minutes = 60
        if state.phase != phase:
            machine_state.record_phase_transition(state, to_phase=phase, reason="phase_entered")
        if phase == "FRAMEWORK_AGENT":
            coord.phase_framework._open_framework_timeline()
        elif phase == "KERNEL_AGENT":
            coord.phase_kernel._open_kernel_timeline(
                route="forge", route_reason="kernel_optimizer=forge", from_phase="PRELUDE"
            )
        state.pending_escalate_hint = hint
        state.last_conc_sweep = {"status": sweep_status}
        state.set_stop_reason("signal")
        try:
            assert await coord.phase_close.ensure_close_sequence(reason="signal") is True
            assert state.stop_reason == "signal"
            transition = state.phase_history[-1]
            assert (transition["from_phase"], transition["to_phase"], transition["reason"]) == (
                phase,
                "CLOSE",
                transition_reason,
            )
            replay = machine_state.replay_next_phase(transition["evidence"]["predicate_inputs"])
            assert replay[:2] == ("CLOSE", transition_reason)
            assert transition["ts_unix"] == transition["evidence"]["predicate_inputs"]["now_unix"]
            assert state.pending_escalate_hint == ""
            assert assemble_parts(coord.session_dir)["outcome"]["stage_reached_recorded"] == "close"
            prior, _ = phase_event.assemble_phase_ext(phase_event_parts(), event=phase_event.phase_event_id(phase, 0))
            assert prior["open"] is False
            assert prior["duration_sec"] is not None
            if phase in {"FRAMEWORK_AGENT", "KERNEL_AGENT"}:
                event_type = "framework_agent" if phase == "FRAMEWORK_AGENT" else "kernel"
                internal = next(
                    event for event in read_timeline_events(coord.session_dir) if event["type"] == event_type
                )
                assert internal["end_time"]
                if phase == "FRAMEWORK_AGENT":
                    assert internal["ext"]["exit"]["reason"] == transition_reason
                else:
                    assert internal["ext"]["outcome"]["exit_reason"] == transition_reason
            close, _ = phase_event.assemble_phase_ext(phase_event_parts(), event="close:0:phase")
            actions = {row["task_id"]: row for row in close["actions"]["rows"]}
            receipts = transition["evidence"]["close_steps"]
            for step in ("report", "session_breakdown"):
                receipt = next(row for row in receipts if row["step"] == step)
                task = await coord.tasks.get(receipt["task_id"])
                assert task_dispatch_record(task)["phase"] == "CLOSE"
                assert actions[task.task_id]["action"] == step
            history_count = len(state.phase_history)
            assert await coord.phase_close.ensure_close_sequence(reason="signal") is False
            assert len(state.phase_history) == history_count
        finally:
            await coord.stop()

    @pytest.mark.asyncio
    async def test_deadline_close_observes_enablement_work_without_renewing_its_lease(self, tmp_path):
        coord = self._coordinator(tmp_path / "session")
        state = coord.shared_state
        machine_state.record_phase_transition(state, to_phase="ENABLEMENT", reason="phase_entered")
        try:
            holder = await coord.enablement_lane._open_authoring_round(params={}, lanes=[], lease_ttl_sec=300)
            assert holder
            before = await coord.rounds.held()
            assert before is not None
            assert await coord.enablement_lane._round_has_live_work(holder)

            await coord.phase_close.enter_closing_phase(grace_sec=30)

            assert state.phase == "CLOSE"
            assert state.phase_history[-1]["evidence"]["predicate_inputs"]["enablement_in_flight"] is True
            after = await coord.rounds.held()
            assert after is not None
            assert (after.renewed_unix, after.expires_unix) == (before.renewed_unix, before.expires_unix)
        finally:
            await coord.stop()

    @pytest.mark.asyncio
    async def test_deadline_report_is_created_in_close_and_reused(self, tmp_path, monkeypatch):
        coord = self._coordinator(tmp_path / "session")
        state = coord.shared_state
        state.max_minutes = 60
        roofline_contexts = []
        original_roofline = coord.phase_close._maybe_run_close_post_opt_roofline

        async def _roofline():
            roofline_contexts.append(state.closing_phase)
            return await original_roofline()

        monkeypatch.setattr(coord.phase_close, "_session_integrated_kernel_patch", lambda: True)
        monkeypatch.setattr(coord.phase_close, "_maybe_run_close_post_opt_roofline", _roofline)
        try:
            deadline = await coord.phase_close.enter_closing_phase(grace_sec=30)
            assert deadline.remaining() > 0
            assert state.phase == "CLOSE"
            assert state.close_sequence_done is False
            # The breakdown written during the sequence derives this from state.json, so it must land on entry.
            assert state.stop_reason == "time_exhausted"
            transition = state.phase_history[-1]
            assert transition["reason"] == "time_exhausted"
            assert machine_state.replay_next_phase(transition["evidence"]["predicate_inputs"])[:2] == (
                "CLOSE",
                "time_exhausted",
            )
            report = await coord.tasks.get(state.closing_report_task_id)
            assert task_dispatch_record(report)["phase"] == "CLOSE"
            history_count = len(state.phase_history)
            coord._closing_deadline = deadline
            await coord.await_within_session_bound(coord.phase_machine.advance_phase_if_needed, stage="advance_phase")
            assert state.close_sequence_done is True
            assert roofline_contexts == [True]
            receipts = state.phase_history[-1]["evidence"]["close_steps"]
            assert next(row for row in receipts if row["step"] == "report")["task_id"] == report.task_id
            assert len(state.phase_history) == history_count
        finally:
            await coord.stop()

    @pytest.mark.asyncio
    async def test_deadline_sequence_still_obeys_the_grace_bound(self, tmp_path, monkeypatch):
        coord = self._coordinator(tmp_path / "session")
        cancelled = asyncio.Event()

        async def _slow_close(_transition):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        monkeypatch.setattr(coord.phase_close, "on_enter_close", _slow_close)
        try:
            await coord.phase_close.enter_closing_phase(grace_sec=0.5)
            # Measured from here, so the entry's own writes do not spend the window under test.
            coord._closing_deadline = Deadline.after(0.5)
            await coord.await_within_session_bound(coord.phase_machine.advance_phase_if_needed, stage="advance_phase")
            assert cancelled.is_set()
            assert coord.shared_state.close_sequence_done is False
        finally:
            await coord.stop()

    @pytest.mark.asyncio
    async def test_a_spent_session_closes_even_though_every_step_is_skipped(self, tmp_path: Path):
        coord = self._coordinator(tmp_path / "session")
        coord.shared_state.max_minutes = 60
        coord.shared_state.elapsed_charged_sec = 120 * 60.0

        try:
            reason = await coord.run(max_minutes=60, closing_grace_sec=0.0, max_ticks=4)
        finally:
            await coord.stop()

        assert reason == "time_exhausted"
        assert coord.shared_state.close_sequence_done is True

    @pytest.mark.asyncio
    async def test_the_close_sequence_runs_once_and_not_again(self, tmp_path: Path):
        coord = self._coordinator(tmp_path / "session")
        machine_state.record_phase_transition(
            coord.shared_state,
            to_phase="CLOSE",
            reason="sweep_done",
            evidence={"trigger": "test"},
        )
        await coord.phase_machine._on_phase_entered(from_phase="SWEEP", to_phase="CLOSE")
        assert coord.shared_state.close_sequence_done is True

        assert await coord.phase_close.ensure_close_sequence(reason="terminal") is False

    @pytest.mark.asyncio
    async def test_a_closing_phase_with_no_report_task_does_not_wait_on_one(self, tmp_path: Path):
        coord = self._coordinator(tmp_path / "session")
        coord.shared_state.closing_phase = True
        coord.shared_state.closing_report_task_id = ""

        # Absence is finished, not pending: waiting on a task nobody created is
        # how a session sat in CLOSE until its grace ran out with no report.
        assert await coord.phase_close.closing_report_terminal() is True
        await coord.stop()


@pytest.mark.asyncio
async def test_the_sequencer_delivers_the_finished_close_section_in_the_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The package, not the session dir, is what external sync ships."""
    dest_root = tmp_path / "dest"
    monkeypatch.setenv("HYPERLOOM_SESSION_PACKAGE_DEST", str(dest_root))
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    # Stand in for the step-2 breakdown task, which does not run under the mock backends: a payload whose ``close``
    # key stops where step 2 can see.
    (session_dir / "session_breakdown.json").write_text(
        json.dumps(
            {
                "schema_version": "hyperloom.session_breakdown.v6.0",
                "close": {"status": "degraded", "steps": [{"step": "sequencer_started"}]},
            }
        ),
        encoding="utf-8",
    )

    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    coord = Coordinator(
        session_dir=session_dir,
        backends={
            "orchestration": MockBackend(idle_plan),
            "critic": MockBackend(idle_plan),
        },
        role_registry=default_role_registry(),
        knowledge_plane=None,
    )
    coord.shared_state.phase = "SWEEP"
    coord.shared_state.phase_history = [{"to_phase": "SWEEP", "evidence": {}, "reason": "kernel_no_more_leverage"}]
    machine_state.record_phase_transition(coord.shared_state, to_phase="CLOSE", reason="sweep_done", evidence={})

    await coord.phase_machine._on_phase_entered(from_phase="SWEEP", to_phase="CLOSE")

    zips = sorted((dest_root / "hyperloom-session-packages").glob("*.zip"))
    assert len(zips) == 1
    with zipfile.ZipFile(zips[0]) as bundle:
        zipped = json.loads(bundle.read("session_breakdown.json"))
    loose = json.loads((dest_root / "session_breakdown.json").read_text(encoding="utf-8"))

    for delivered in (zipped["close"], loose["close"]):
        # The steps recorded after step 2 are the whole point: they are what
        # the bundled copy was missing before the rebuild.
        assert delivered["close_sequence_done"] is True
        assert {"artifact_package", "ndjson_drain", "done"} <= {step["step"] for step in delivered["steps"]}
        # Both copies carry a settled verdict. Which one it is depends on the
        # internal tasks, which do not run under mock backends, but it is
        # never ``running``: that would mean the patch never delivered the
        # verdict the sequencer recorded, which is the failure this rebuild
        # exists to prevent.
        assert delivered["status"] in {"succeeded", "degraded"}


@pytest.mark.asyncio
async def test_recipe_kb_t4_hook_short_circuits_when_sequencer_done(tmp_path: Path):
    """If the CLOSE sequencer already drained, ``_recipe_kb_t4_hook`` must skip (no double drain)."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "critic": MockBackend(idle_plan),
    }
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        knowledge_plane=KnowledgePlane(recipe_kb=_StubRecipeKB()),
    )
    coord.shared_state.recipe_kb_session_id = "sid-stop-skip"
    coord.shared_state.close_sequence_done = True
    coord.shared_state.recipe_finalize_status = "written"

    await coord._recipe_kb_t4_hook()
    assert coord.recipe_kb.drain_calls == 0


@pytest.mark.asyncio
async def test_recipe_kb_t4_hook_still_runs_when_sequencer_not_done(tmp_path: Path):
    """Graceful teardown/Ctrl-C fallback calls finalize when CLOSE did not."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "critic": MockBackend(idle_plan),
    }
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        knowledge_plane=KnowledgePlane(recipe_kb=_StubRecipeKB()),
    )
    coord.shared_state.recipe_kb_session_id = "sid-fallback"
    coord.shared_state.close_sequence_done = False

    finalize_calls: list[str] = []

    def _spy(*, source: str) -> dict:
        finalize_calls.append(source)
        return {"status": "written"}

    coord.recipe_journal.finalize_recipe_and_journal = _spy  # type: ignore[method-assign]
    await coord._recipe_kb_t4_hook()
    assert finalize_calls == ["t4_fallback"]


@pytest.mark.asyncio
async def test_recipe_kb_t4_hook_remote_runs_without_recipe_kb_or_sid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "critic": MockBackend(idle_plan),
    }
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.setenv("KB_STORE_URL", "https://kb-store.example.test")
    monkeypatch.setenv("KB_STORE_TOKEN", "test-token")
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        knowledge_plane=None,
    )
    coord.shared_state.recipe_kb_session_id = ""
    coord.shared_state.close_sequence_done = False

    finalize_calls: list[str] = []
    save_calls: list[Path] = []

    def _finalize(*, source: str) -> dict:
        finalize_calls.append(source)
        return {"status": "written"}

    coord.recipe_journal.finalize_recipe_and_journal = _finalize  # type: ignore[method-assign]
    coord.shared_state.save = lambda path: save_calls.append(path)  # type: ignore[method-assign]

    await coord._recipe_kb_t4_hook()

    assert finalize_calls == ["t4_fallback"]
    assert save_calls
    assert all(path == session_dir for path in save_calls)


@pytest.mark.asyncio
async def test_recipe_kb_t4_hook_remote_skips_when_close_sequence_done(tmp_path: Path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "critic": MockBackend(idle_plan),
    }
    config = KnowledgeConfig(
        mode=KnowledgeStoreMode.REMOTE,
        local_root=str(tmp_path / "knowledge"),
        kb_store_url="https://kb-store.example.test",
        kb_store_token="test-token",
    )
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        knowledge_plane=KnowledgePlane(config=config),
    )
    coord.shared_state.recipe_kb_session_id = ""
    coord.shared_state.close_sequence_done = True
    coord.shared_state.recipe_finalize_status = "written"

    recipe_finalize_attempts = 0

    def _finalize(*, source: str) -> dict:
        nonlocal recipe_finalize_attempts
        recipe_finalize_attempts += 1
        return {"status": "written"}

    coord.recipe_journal.finalize_recipe_and_journal = _finalize  # type: ignore[method-assign]

    await coord._recipe_kb_t4_hook()

    assert recipe_finalize_attempts == 0


@pytest.mark.asyncio
async def test_recipe_kb_t4_hook_retries_failed_finalize_after_close(
    tmp_path: Path,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "critic": MockBackend(idle_plan),
    }
    config = KnowledgeConfig(
        mode=KnowledgeStoreMode.REMOTE,
        local_root=str(tmp_path / "knowledge"),
        kb_store_url="https://kb-store.example.test",
        kb_store_token="test-token",
    )
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        knowledge_plane=KnowledgePlane(config=config),
    )
    coord.shared_state.close_sequence_done = True
    coord.shared_state.recipe_finalize_status = "failed"
    coord.shared_state.recipe_finalize_outcome = {"status": "error"}
    finalize_calls: list[str] = []

    def _finalize(*, source: str) -> dict:
        finalize_calls.append(source)
        return {"status": "written"}

    coord.recipe_journal.finalize_recipe_and_journal = _finalize  # type: ignore[method-assign]

    await coord._recipe_kb_t4_hook()

    assert finalize_calls == ["t4_fallback"]
    assert coord.shared_state.recipe_finalize_status == "written"
    assert coord.shared_state.recipe_finalize_attempts == 1


@pytest.mark.asyncio
async def test_recipe_kb_t4_hook_local_skips_without_recipe_kb(tmp_path: Path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "critic": MockBackend(idle_plan),
    }
    config = KnowledgeConfig(
        mode=KnowledgeStoreMode.LOCAL,
        local_root=str(tmp_path / "knowledge"),
    )
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        knowledge_plane=KnowledgePlane(config=config),
    )
    coord.shared_state.recipe_kb_session_id = "local-session"
    coord.shared_state.close_sequence_done = False

    finalize_calls: list[str] = []

    def _finalize(*, source: str) -> dict:
        finalize_calls.append(source)
        return {"status": "written"}

    coord.recipe_journal.finalize_recipe_and_journal = _finalize  # type: ignore[method-assign]

    await coord._recipe_kb_t4_hook()

    assert finalize_calls == []


@pytest.mark.asyncio
async def test_recipe_kb_t4_hook_local_skips_without_recipe_kb_sid(tmp_path: Path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "critic": MockBackend(idle_plan),
    }
    config = KnowledgeConfig(
        mode=KnowledgeStoreMode.LOCAL,
        local_root=str(tmp_path / "knowledge"),
    )
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        knowledge_plane=KnowledgePlane(config=config, recipe_kb=_StubRecipeKB()),
    )
    coord.shared_state.recipe_kb_session_id = "  "
    coord.shared_state.close_sequence_done = False

    finalize_calls: list[str] = []

    def _finalize(*, source: str) -> dict:
        finalize_calls.append(source)
        return {"status": "written"}

    coord.recipe_journal.finalize_recipe_and_journal = _finalize  # type: ignore[method-assign]

    await coord._recipe_kb_t4_hook()

    assert finalize_calls == []


@pytest.mark.asyncio
async def test_recipe_kb_t4_hook_degraded_is_complete_noop() -> None:
    coordinator = SimpleNamespace(
        knowledge_plane=SimpleNamespace(kb_disabled=True),
    )

    await Coordinator._recipe_kb_t4_hook(coordinator)


@pytest.fixture
def roofline_coord(coord):
    from hyperloom.orchestrator.bus.resource_lock import ResourceLockManager, SqliteLeaseBackend
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
    from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner
    from hyperloom.orchestrator.state.shared_state import SharedState
    from hyperloom.orchestrator.state.task_registry import TaskRegistry

    coord.db = SqliteConnection(coord.session_dir / "roofline-close.db")
    coord.tasks = TaskRegistry(coord.db)
    coord.locks = ResourceLockManager(SqliteLeaseBackend(coord.db))
    coord.sub = SubAgentRunner(coord.locks, coord.tasks)
    coord.shared_state = SharedState(enable_roofline=True, optimization_stack=[{"action": "geak_e2e"}])
    coord._dispatcher_poll_sec = 0.001
    yield coord
    coord.db.close()


@pytest.mark.asyncio
async def test_close_roofline_retries_prior_failed_legs_and_reuses_success(roofline_coord):
    from unittest.mock import AsyncMock

    c = roofline_coord
    execute = AsyncMock(return_value={"status": "ok"})
    c.sub.register_executor("roofline", execute)
    old_ids = []
    for state in ("failed", "cancelled", "failed"):
        task = await c.phase_prelude.enqueue_internal_analysis_task(reason="close_post_opt")
        await c.tasks.transition(task.task_id, "running")
        await c.tasks.transition(task.task_id, state)
        old_ids.append(task.task_id)
    c.shared_state.current_best = {"extra_envs": {"VJEPA2_ATTN_IMPL": "aiter_triton"}}

    await c.phase_close._maybe_run_close_post_opt_roofline()
    await c.phase_close._maybe_run_close_post_opt_roofline()

    execute.assert_awaited_once()
    executed = execute.await_args.args[0].task
    assert executed.task_id not in old_ids
    assert executed.params["reason"] == "close_post_opt"
    assert executed.params["base_extra_envs"]["VJEPA2_ATTN_IMPL"] == "aiter_triton"
    assert executed.requires_lanes == list(ACTION_CATALOGUE["roofline"].requires_lanes)
    assert (await c.tasks.get(executed.task_id)).state == "succeeded"
    assert [(await c.tasks.get(task_id)).state for task_id in old_ids] == ["failed", "cancelled", "failed"]
    assert not await c.locks.lane_holders()
    assert not c.dispatcher._executions
    await c.dispatcher.close_db_after_executions()


@pytest.mark.asyncio
async def test_close_roofline_waits_for_existing_execution(roofline_coord):
    c = roofline_coord
    entered, finish = asyncio.Event(), asyncio.Event()
    calls = []

    async def execute(ctx):
        calls.append(ctx.task.task_id)
        entered.set()
        await finish.wait()
        return {"status": "ok"}

    c.sub.register_executor("roofline", execute)
    task = await c.phase_prelude.enqueue_internal_analysis_task(reason="close_post_opt")
    action = asyncio.create_task(c.dispatcher.run_task_registered(task))
    await asyncio.wait_for(entered.wait(), 5)
    close = asyncio.create_task(c.phase_close._maybe_run_close_post_opt_roofline())
    try:
        await asyncio.sleep(0.01)
        assert not close.done()
    finally:
        finish.set()
        await asyncio.gather(action, close)
    assert calls == [task.task_id]
    assert not await c.locks.lane_holders()
    await c.dispatcher.close_db_after_executions()


@pytest.mark.asyncio
async def test_close_roofline_timeout_does_not_terminalize_live_worker(roofline_coord, monkeypatch):
    from hyperloom.orchestrator.loop import dispatcher as dispatcher_module

    c = roofline_coord
    monkeypatch.setattr(dispatcher_module, "_CANCEL_NOTICE_SEC", 0)
    c.phase_close.CLOSE_POST_OPT_ROOFLINE_TIMEOUT_SEC = 0.01
    finish = asyncio.Event()

    async def execute(_ctx):
        await finish.wait()
        return {"status": "ok"}

    c.sub.register_executor("roofline", execute)
    task = await c.phase_prelude.enqueue_internal_analysis_task(reason="close_post_opt")
    try:
        await c.phase_close._maybe_run_close_post_opt_roofline()
        assert (await c.tasks.get(task.task_id)).state == "running"
        assert await c.locks.lane_holders()
        assert c.dispatcher._executions
    finally:
        finish.set()
        await asyncio.gather(*c.dispatcher._executions)
    assert (await c.tasks.get(task.task_id)).state == "succeeded"
    assert not await c.locks.lane_holders()
    assert not c.dispatcher._inflight_actions
    await c.dispatcher.close_db_after_executions()


@pytest.mark.asyncio
async def test_resume_regenerates_close_reports_once_per_leg(roofline_coord):
    from hyperloom.inference_optimizer.cli.bootstrap import _begin_resume_leg
    from hyperloom.orchestrator.actions.executors.report import ReportExecutor
    from hyperloom.orchestrator.actions.executors.session_breakdown import SessionBreakdownExecutor
    from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner

    c = roofline_coord
    c.sub = SubAgentRunner(c.locks, c.tasks, session_dir=c.session_dir, shared_state=c.shared_state)
    report = ReportExecutor()
    c.sub.register_executor("report", report)
    c.sub.register_executor("session_breakdown", SessionBreakdownExecutor())
    c.shared_state.baseline_tput = 14.4
    c.shared_state.framework = "custom"
    c.shared_state.model_type = "pytorch"
    task_ids = {"report": [], "session_breakdown": []}
    for leg, latency in enumerate((21.7, 19.74, 19.5)):
        if leg:
            _begin_resume_leg(c.shared_state)
        c.shared_state.current_best = {
            "action": "geak_e2e" if leg else "explore",
            "tput": 1000 / latency,
            "e2el_mean_ms": latency,
        }
        c.shared_state.stop_reason = "sweep_done"
        c.shared_state.save(c.session_dir)
        for kind, enqueue in (
            ("report", c.phase_close._enqueue_internal_report_task),
            ("session_breakdown", c.phase_close._enqueue_internal_session_breakdown_task),
        ):
            task = await enqueue(reason="close_phase_entry")
            assert await c.phase_close._run_close_task(task, step=kind) == "succeeded"
            task_ids[kind].append(task.task_id)
            reused = await enqueue(reason="close_phase_entry")
            assert reused.task_id == task.task_id
            assert reused.state == "succeeded"
        saved = json.loads((c.session_dir / "reports/final.json").read_text())
        assert saved["current_best"]["e2el_mean_ms"] == latency
        assert f"{latency:.1f} ms" in (c.session_dir / "reports/final.md").read_text()
        assert (c.session_dir / "session_breakdown.json").is_file()
    assert all(len(set(ids)) == 3 for ids in task_ids.values())


@pytest.mark.asyncio
async def test_report_failure_emits_lifecycle_error_and_records_failed_step(tmp_path: Path):
    coord = make_coordinator(tmp_path)
    close = coord.phase_close
    lifecycle: list[dict[str, Any]] = []
    steps: list[tuple[str, str, str]] = []

    def _emit(**kwargs: Any) -> None:
        lifecycle.append(kwargs)

    async def _record(step: str, *, status: str, task_id: str = "", detail: str = "") -> None:
        steps.append((step, status, detail))

    async def _enqueue_fails(*, reason: str) -> None:
        raise RuntimeError("db down")

    coord.writeback.emit_lifecycle = _emit  # type: ignore[method-assign]
    close._record_close_step = _record  # type: ignore[method-assign]
    close._enqueue_internal_report_task = _enqueue_fails  # type: ignore[method-assign]

    await close._run_close_step("report", close._do_report())

    assert [(row["step"], row["status"]) for row in lifecycle] == [("report", "START"), ("report", "ERROR")]
    assert "db down" in lifecycle[-1]["detail"]
    assert ("report", "failed") in [(step, status) for step, status, _ in steps]
