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

from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.orchestrator.knowledge.config import KnowledgeConfig, KnowledgeStoreMode
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.roles.mock_backend import (
    MockBackend,
    MockTurn,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.close import (
    _CLOSE_STEP_WAIT_CEILING_SEC,
    _CLOSE_STEP_WAIT_FLOOR_SEC,
)
from hyperloom.orchestrator.policy.gate import CORE_STATE_FIELDS
from hyperloom.orchestrator.state.shared_state import effective_closing_grace_sec


@dataclass
class _BareState:
    """SharedState stand-in covering every attribute the CLOSE sequencer reads/writes."""

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

    def save(self, _session_dir: Path | None) -> None:
        self.save_count += 1

    def set_stop_reason(self, reason: str) -> None:
        self.stop_reason = reason

    def closing_reserve_sec(self) -> float:
        return effective_closing_grace_sec(self.max_minutes, self.closing_grace_sec)


@dataclass
class _StubTaskRow:
    task_id: str
    kind: str
    state: str
    params: dict
    idempotency_key: str
    # Mirrors the real ``Task``: the dispatch path reads both before running a
    # row. Closing steps take no lanes.
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
    c.recipe_kb = None
    c.knowledge_plane = None
    c.role_registry = {}
    return c


def _close_phase_history_row() -> dict[str, Any]:
    return {"to_phase": "CLOSE", "reason": "sweep_done", "evidence": {}}


@pytest.mark.asyncio
async def test_record_close_step_appends_to_evidence_close_steps(coord):
    coord.shared_state.phase_history = [_close_phase_history_row()]
    await coord._record_close_step("report", status="done", task_id="t-1")
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
    await coord._record_close_step(
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
    await coord._record_close_step("done", status="done")
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
    await coord._record_close_step("report", status="done")
    assert isinstance(coord.shared_state.phase_history[-1]["evidence"]["close_steps"], list)


@pytest.mark.asyncio
async def test_record_close_step_no_op_when_history_empty(coord):
    coord.shared_state.phase_history = []
    await coord._record_close_step("report", status="done")


@pytest.mark.asyncio
async def test_enqueue_internal_report_task_fresh(coord):
    task = await coord._enqueue_internal_report_task(reason="close_phase_entry")
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

    task = await coord._enqueue_internal_report_task(reason="close_phase_entry")
    assert task is existing
    assert "internal-report-close_phase_entry" not in coord.tasks._by_key


@pytest.mark.asyncio
async def test_enqueue_internal_report_task_replaces_a_cancelled_one(coord):
    """A report the deadline path enqueued and then cancelled cannot be run; the sequencer needs a live one.

    Regression for the ``cannot transition from 'cancelled' to 'running'``
    crash: the helper used to answer "was one enqueued?" when the caller needs
    "is one runnable?".
    """
    dead = _StubTaskRow(
        task_id="wallclock-report",
        kind="report",
        state="cancelled",
        params={},
        idempotency_key="closing-report-1234",
    )
    coord.tasks._by_id["wallclock-report"] = dead
    coord.shared_state.closing_report_task_id = "wallclock-report"

    task = await coord._enqueue_internal_report_task(reason="close_phase_entry")

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

    task = await coord._enqueue_internal_report_task(reason="close_phase_entry")

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

    await coord._on_enter_close(from_phase="SWEEP")

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

    state = await coord._run_close_task(done, step="1 (report)")

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
    """Give the CLOSE module a monotonic clock that jumps ``step_sec`` per read.

    The wait under test is measured in minutes, so a test that spent it would
    be a test nobody runs. Only the CLOSE module's view of the clock is
    replaced, which leaves the event loop's own timekeeping alone.
    """
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
    """``running -> running`` is not a transition the registry has; asking for it kills the step.

    The deadline path dispatches the report before CLOSE is entered, so the
    sequencer routinely meets its own step already under way. It was documented
    as "the sequencer will wait for it" and implemented as a second dispatch.
    """
    coord.tasks = _FinishesWhileWaiting(terminal_state)
    coord._dispatcher_poll_sec = 0.01
    coord.shared_state.max_minutes = 60

    state = await coord._run_close_task(_running_report_row(coord), step="1 (report)")

    assert state == terminal_state
    assert coord.sub.run_calls == []


@pytest.mark.asyncio
async def test_a_running_task_that_never_lands_is_reported_not_waited_on_forever(
    coord,
    monkeypatch: pytest.MonkeyPatch,
):
    """The wait is patient, not unbounded: a step that never lands is recorded, not awaited forever."""
    coord._dispatcher_poll_sec = 0.0
    coord.shared_state.max_minutes = 60
    _clock_advancing_by(monkeypatch, step_sec=30.0)

    state = await coord._run_close_task(_running_report_row(coord), step="1 (report)")

    assert state == "running"
    assert coord.sub.run_calls == []


class _HangsUntilCancelled(_StubSubAgentRunner):
    """A report writer that never returns unless the waiter cancels it."""

    def __init__(self):
        super().__init__()
        self.cancelled = False

    async def run_task(self, task, *args, **kwargs):
        self.run_calls.append(task)
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return _StubSubResult(state="succeeded")


@pytest.mark.asyncio
async def test_a_fresh_report_that_never_lands_is_not_awaited_forever(coord):
    """The in-flight wait had a bound; a newly started report must too."""
    coord.sub = _HangsUntilCancelled()
    coord.shared_state.max_minutes = 60
    coord.phase_close._close_step_wait_sec = lambda _task: 0.05  # type: ignore[method-assign]
    queued = _StubTaskRow(
        task_id="fresh-report",
        kind="report",
        state="queued",
        params={},
        idempotency_key="internal-report-close_phase_entry",
    )

    started = time.monotonic()
    state = await coord._run_close_task(queued, step="1 (report)")
    elapsed = time.monotonic() - started

    assert state == "running"
    assert elapsed < 2.0
    assert coord.sub.run_calls == [queued]
    assert coord.sub.cancelled is True


@pytest.mark.asyncio
async def test_the_wait_for_a_running_report_outlives_a_short_session_reserve(
    coord,
    monkeypatch: pytest.MonkeyPatch,
):
    """A ten-minute session reserves twelve seconds for CLOSE; no report is written in twelve seconds.

    Bounding the wait by the reserve made it a wait on paper only — the step
    the deadline path dispatched was declared failed while it was still
    running, and the session that ran out of time is the one whose report is
    worth the most. The bound belongs to the work, so it is the report's own
    expected runtime.
    """
    coord.tasks = _FinishesWhileWaiting("succeeded", lands_on=5)
    coord._dispatcher_poll_sec = 0.0
    coord.shared_state.max_minutes = 10
    assert coord.shared_state.closing_reserve_sec() == pytest.approx(12.0)
    # Five looks at five simulated seconds apiece: past the reserve, inside the
    # two minutes the catalogue prices a report at.
    _clock_advancing_by(monkeypatch, step_sec=5.0)

    state = await coord._run_close_task(_running_report_row(coord), step="1 (report)")

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
    coord._dispatcher_poll_sec = 0.01
    coord.shared_state.max_minutes = 60
    coord.shared_state.phase_history = [_close_phase_history_row()]
    _running_report_row(coord)
    coord.shared_state.closing_report_task_id = "wallclock-report"

    await coord._on_enter_close(from_phase="SWEEP")

    rows = coord.shared_state.phase_history[-1]["evidence"]["close_steps"]
    report = next(r for r in rows if r["step"] == "report")
    assert report["status"] == "done"
    assert "wallclock-report" not in [t.task_id for t in coord.sub.run_calls]


@pytest.mark.asyncio
async def test_enqueue_internal_session_breakdown_task(coord):
    task = await coord._enqueue_internal_session_breakdown_task(
        reason="close_phase_entry",
    )
    assert task.kind == "session_breakdown"
    assert task.idempotency_key == "internal-session_breakdown-close_phase_entry"
    assert task.params["source"] == "coordinator_internal"


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
    coord.recipe_kb = _StubRecipeKB()
    coord.shared_state.recipe_kb_session_id = "sid-test"
    coord.shared_state.model_name = "model"
    coord.shared_state.gpu_type = "mi300x"

    await coord._on_enter_close(from_phase="SWEEP")

    rows = coord.shared_state.phase_history[-1]["evidence"]["close_steps"]
    steps = [r["step"] for r in rows]
    assert steps == [
        "sequencer_started",
        "fact_finalize",
        "report",
        "session_breakdown",
        "artifact_package",
        "ndjson_drain",
        "done",
    ]
    by_step = {r["step"]: r for r in rows}
    assert by_step["report"]["status"] == "done"
    assert by_step["session_breakdown"]["status"] == "done"
    # fact_finalize now runs first and writes optimization_journal.json, so the
    # artifact package has a curated file to include.
    assert by_step["artifact_package"]["status"] == "done"
    assert by_step["fact_finalize"]["status"] == "done"
    assert "status=written" in by_step["fact_finalize"]["detail"]
    assert "backend=local" in by_step["fact_finalize"]["detail"]
    assert by_step["ndjson_drain"]["status"] == "skipped"
    assert by_step["done"]["status"] == "done"
    assert coord.shared_state.close_sequence_done is True
    # A normal SWEEP completion's sweep_done reason must be preserved.
    assert coord.shared_state.stop_reason == "sweep_done"


@pytest.mark.asyncio
async def test_close_sequencer_surfaces_remote_finalize_failure(
    coord,
    monkeypatch,
):
    coord.shared_state.phase_history = [_close_phase_history_row()]
    monkeypatch.setattr(
        coord.writeback,
        "finalize_recipe_and_journal",
        lambda *, source: {
            "status": "error",
            "reason": "KBStoreError",
            "backend": "kb-store",
        },
    )

    await coord._on_enter_close(from_phase="SWEEP")

    rows = coord.shared_state.phase_history[-1]["evidence"]["close_steps"]
    fact = next(row for row in rows if row["step"] == "fact_finalize")
    assert fact["status"] == "failed"
    assert fact["detail"] == ("status=error reason=KBStoreError backend=kb-store")


@pytest.mark.asyncio
async def test_close_sequencer_falls_back_to_time_exhausted(coord):
    """Falls back to ``stop_reason='time_exhausted'`` when CLOSE had no usable phase-exit reason."""
    coord.shared_state.phase_history = [
        {"to_phase": "CLOSE", "reason": "", "evidence": {}},
    ]
    assert coord.shared_state.stop_reason == ""

    await coord._on_enter_close(from_phase="SWEEP")

    assert coord.shared_state.stop_reason == "time_exhausted"


@pytest.mark.asyncio
async def test_close_sequencer_derives_sweep_done_from_phase_history(coord):
    """Regression: sequencer derives the phase_history reason rather than blanket-stamping time_exhausted."""
    coord.shared_state.phase_history = [
        {"to_phase": "CLOSE", "reason": "sweep_done", "evidence": {}},
    ]
    assert coord.shared_state.stop_reason == ""

    await coord._on_enter_close(from_phase="SWEEP")

    assert coord.shared_state.stop_reason == "sweep_done"


@pytest.mark.asyncio
async def test_close_sequencer_preserves_failed_conc_sweep_reason(coord):
    """Failed conc_sweep closeout should stay distinguishable in final stop_reason."""
    coord.shared_state.phase_history = [
        {"to_phase": "CLOSE", "reason": "sweep_failed", "evidence": {"sweep_status": "failed"}},
    ]
    assert coord.shared_state.stop_reason == ""

    await coord._on_enter_close(from_phase="SWEEP")

    assert coord.shared_state.stop_reason == "sweep_failed"


@pytest.mark.asyncio
async def test_close_sequencer_does_not_overwrite_existing_stop_reason(
    coord,
):
    """An operator-set ``stop_reason`` must survive the final step's setter."""
    coord.shared_state.phase_history = [_close_phase_history_row()]
    coord.shared_state.set_stop_reason("recipe_kb_drain_failed")

    await coord._on_enter_close(from_phase="SWEEP")

    assert coord.shared_state.stop_reason == "recipe_kb_drain_failed"


@pytest.mark.asyncio
async def test_close_sequencer_does_not_overwrite_caller_set_stop_reason(
    coord,
):
    """An operator-set ``stop_reason`` (e.g. ``signal``) must survive step 5."""
    coord.shared_state.phase_history = [_close_phase_history_row()]
    coord.shared_state.stop_reason = "signal"

    await coord._on_enter_close(from_phase="SWEEP")

    assert coord.shared_state.stop_reason == "signal"


@pytest.mark.asyncio
async def test_close_sequencer_report_before_session_breakdown(coord):
    """Report task MUST be enqueued before session_breakdown."""
    coord.shared_state.phase_history = [_close_phase_history_row()]
    await coord._on_enter_close(from_phase="SWEEP")
    insertion = coord.tasks.insertion_order
    report_idx = insertion.index("internal-report-close_phase_entry")
    bd_idx = insertion.index("internal-session_breakdown-close_phase_entry")
    assert report_idx < bd_idx


@pytest.mark.asyncio
async def test_close_sequencer_skips_recipe_kb_steps_when_no_recipe_kb(coord):
    """``--degraded-kb`` runs (recipe_kb=None): NDJSON drain recorded 'skipped', not silent."""
    coord.shared_state.phase_history = [_close_phase_history_row()]
    await coord._on_enter_close(from_phase="SWEEP")
    rows = coord.shared_state.phase_history[-1]["evidence"]["close_steps"]
    drain_row = next(r for r in rows if r["step"] == "ndjson_drain")
    assert drain_row["status"] == "skipped"
    assert coord.shared_state.close_sequence_done is True


def test_close_and_recipe_finalize_fields_in_core_state_fields():
    """LLM update_state must not flip close_sequence_done and bypass cli.finally's safety net."""
    assert {
        "close_sequence_done",
        "recipe_finalize_status",
        "recipe_finalize_attempts",
        "recipe_finalize_outcome",
    } <= CORE_STATE_FIELDS


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("close_sequence_done", True),
        ("recipe_finalize_status", "written"),
        ("recipe_finalize_attempts", 99),
        ("recipe_finalize_outcome", {"status": "skipped"}),
    ],
)
def test_policy_blocks_llm_recipe_finalize_state_write(field, value):
    from hyperloom.orchestrator.roles.agent_role import (
        default_role_registry,
    )
    from hyperloom.inference_optimizer.protocol.intent import (
        Intent,
        IntentType,
    )
    from hyperloom.orchestrator.policy.gate import (
        PolicyDenied,
        PolicyGate,
    )

    gate = PolicyGate(role_registry=default_role_registry())
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {field: value}},
    )
    with pytest.raises(PolicyDenied):
        gate.validate_intent("orchestration", intent)


@pytest.mark.asyncio
async def test_phase_transition_into_close_runs_sequencer_e2e(tmp_path: Path):
    """End-to-end: real Coordinator + TaskRegistry enqueue both internal tasks and flip close_sequence_done."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "critic": MockBackend(idle_plan),
        "robustness": MockBackend(idle_plan),
    }
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        recipe_kb=None,
        knowledge_plane=None,
    )
    # Seed state at SWEEP boundary.
    coord.shared_state.phase = "SWEEP"
    coord.shared_state.phase_history = [
        {"to_phase": "EXPLORE", "evidence": {}, "reason": "prelude_done"},
        {"to_phase": "SWEEP", "evidence": {}, "reason": "plateau_kernel"},
    ]
    coord.shared_state.record_phase_transition(
        to_phase="CLOSE",
        reason="sweep_done",
        evidence={"trigger": "test_e2e"},
    )
    await coord._on_phase_entered(from_phase="SWEEP", to_phase="CLOSE")

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


@pytest.mark.asyncio
async def test_the_sequencer_delivers_the_finished_close_section_in_the_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The package, not the session dir, is what external sync ships.

    ``session_breakdown`` is step 2, so the copy it writes describes the
    close-out only as far as itself, and the artifact package built at step 5
    bundles that partial copy. The sequencer's last act patches the ``close``
    key and rebuilds the bundle; this asserts on the two copies a consumer
    actually reads — inside the zip, and the loose tree beside it.
    """
    dest_root = tmp_path / "dest"
    monkeypatch.setenv("HYPERLOOM_SESSION_PACKAGE_DEST", str(dest_root))
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    # Stand in for the step-2 breakdown task, which does not run under the mock
    # backends: a payload whose ``close`` key stops where step 2 can see.
    (session_dir / "session_breakdown.json").write_text(
        json.dumps(
            {
                "schema_version": "hyperloom.session_breakdown.v5.0",
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
            "robustness": MockBackend(idle_plan),
        },
        role_registry=default_role_registry(),
        recipe_kb=None,
        knowledge_plane=None,
    )
    coord.shared_state.phase = "SWEEP"
    coord.shared_state.phase_history = [{"to_phase": "SWEEP", "evidence": {}, "reason": "plateau_kernel"}]
    coord.shared_state.record_phase_transition(to_phase="CLOSE", reason="sweep_done", evidence={})

    await coord._on_phase_entered(from_phase="SWEEP", to_phase="CLOSE")

    zips = sorted((dest_root / "hyperloom-session-packages").glob("*.zip"))
    assert len(zips) == 1
    with zipfile.ZipFile(zips[0]) as bundle:
        zipped = json.loads(bundle.read("session_breakdown.json"))
    loose = json.loads((dest_root / "session_breakdown.json").read_text(encoding="utf-8"))

    for delivered in (zipped["close"], loose["close"]):
        # The steps recorded after step 2 are the whole point: they are what
        # the bundled copy was missing before the rebuild. The stage status is
        # not asserted here because the internal tasks do not run under mock
        # backends; ``test_sbd_v6_stages.py`` pins the ``succeeded`` ladder on
        # a finished sequence.
        assert delivered["close_sequence_done"] is True
        assert {"artifact_package", "ndjson_drain", "done"} <= {step["step"] for step in delivered["steps"]}


@pytest.mark.asyncio
async def test_recipe_kb_t4_hook_short_circuits_when_sequencer_done(tmp_path: Path):
    """If the CLOSE sequencer already drained, ``_recipe_kb_t4_hook`` must skip (no double drain)."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "critic": MockBackend(idle_plan),
        "robustness": MockBackend(idle_plan),
    }
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        recipe_kb=_StubRecipeKB(),
        knowledge_plane=None,
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
        "robustness": MockBackend(idle_plan),
    }
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        recipe_kb=_StubRecipeKB(),
        knowledge_plane=None,
    )
    coord.shared_state.recipe_kb_session_id = "sid-fallback"
    coord.shared_state.close_sequence_done = False

    finalize_calls: list[str] = []

    def _spy(*, source: str) -> dict:
        finalize_calls.append(source)
        return {"status": "written"}

    coord.writeback.finalize_recipe_and_journal = _spy  # type: ignore[method-assign]
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
        "robustness": MockBackend(idle_plan),
    }
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.setenv("KB_STORE_URL", "https://kb-store.example.test")
    monkeypatch.setenv("KB_STORE_TOKEN", "test-token")
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        recipe_kb=None,
        knowledge_plane=None,
    )
    coord.shared_state.recipe_kb_session_id = ""
    coord.shared_state.close_sequence_done = False

    finalize_calls: list[str] = []
    save_calls: list[Path] = []

    def _finalize(*, source: str) -> dict:
        finalize_calls.append(source)
        return {"status": "written"}

    coord.writeback.finalize_recipe_and_journal = _finalize  # type: ignore[method-assign]
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
        "robustness": MockBackend(idle_plan),
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
        recipe_kb=None,
        knowledge_plane=SimpleNamespace(config=config),
    )
    coord.shared_state.recipe_kb_session_id = ""
    coord.shared_state.close_sequence_done = True
    coord.shared_state.recipe_finalize_status = "written"

    finalize_calls: list[int] = []
    coord.finalize_recipe_and_journal = lambda: finalize_calls.append(1)  # type: ignore[method-assign]

    await coord._recipe_kb_t4_hook()

    assert finalize_calls == []


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
        "robustness": MockBackend(idle_plan),
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
        recipe_kb=None,
        knowledge_plane=SimpleNamespace(config=config),
    )
    coord.shared_state.close_sequence_done = True
    coord.shared_state.recipe_finalize_status = "failed"
    coord.shared_state.recipe_finalize_outcome = {"status": "error"}
    finalize_calls: list[str] = []

    def _finalize(*, source: str) -> dict:
        finalize_calls.append(source)
        return {"status": "written"}

    coord.writeback.finalize_recipe_and_journal = _finalize  # type: ignore[method-assign]

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
        "robustness": MockBackend(idle_plan),
    }
    config = KnowledgeConfig(
        mode=KnowledgeStoreMode.LOCAL,
        local_root=str(tmp_path / "knowledge"),
    )
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        recipe_kb=None,
        knowledge_plane=SimpleNamespace(config=config),
    )
    coord.shared_state.recipe_kb_session_id = "local-session"
    coord.shared_state.close_sequence_done = False

    finalize_calls: list[int] = []
    coord.finalize_recipe_and_journal = lambda: finalize_calls.append(1)  # type: ignore[method-assign]

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
        "robustness": MockBackend(idle_plan),
    }
    config = KnowledgeConfig(
        mode=KnowledgeStoreMode.LOCAL,
        local_root=str(tmp_path / "knowledge"),
    )
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        recipe_kb=_StubRecipeKB(),
        knowledge_plane=SimpleNamespace(config=config),
    )
    coord.shared_state.recipe_kb_session_id = "  "
    coord.shared_state.close_sequence_done = False

    finalize_calls: list[int] = []
    coord.finalize_recipe_and_journal = lambda: finalize_calls.append(1)  # type: ignore[method-assign]

    await coord._recipe_kb_t4_hook()

    assert finalize_calls == []


@pytest.mark.asyncio
async def test_recipe_kb_t4_hook_degraded_is_complete_noop() -> None:
    coordinator = SimpleNamespace(
        knowledge_plane=SimpleNamespace(kb_disabled=True),
    )

    await Coordinator._recipe_kb_t4_hook(coordinator)
