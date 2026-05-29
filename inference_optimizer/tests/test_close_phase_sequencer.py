"""v0.8 §3.2 §5.5 / KB_gaps/Gap-06 — CLOSE phase 5-step sequencer tests.

KB_gaps/Gap-06 root cause: CLOSE phase had three uncoordinated exit
paths (``_enter_closing_phase`` wall-clock deadline, cli.finally
emergency breakdown, ``Coordinator.stop()::_cortex_t4_hook``) with
no guaranteed ordering and no shared "done" flag. The design (KB_design
§3.2 §5.5) wants a fixed 5-step sequence with breadcrumbs:

  1. ``report``
  2. ``session_breakdown``
  3. NDJSON drain
  4. Cortex ``session commit``
  5. mark ``close_sequence_done``

This file covers:

* Each step in isolation (status='done' on happy path; correct skip /
  failure status on each error branch).
* End-to-end 5-step ordering — every row in
  ``phase_history[-1].evidence.close_steps`` is present in the right
  sequence.
* Idempotence: report task already enqueued by wall-clock path → CLOSE
  sequencer reuses it.
* Single Cortex commit: stop()'s _cortex_t4_hook short-circuits when
  ``close_sequence_done=True``.
* cli.finally short-circuit when the sequencer wrote the breakdown.
* ``close_sequence_done`` is locked in CORE_STATE_FIELDS.
* ``_record_close_step`` helper edge cases (missing evidence dict /
  non-list close_steps / empty history).
* ``_wait_for_task_terminal`` polling helper (terminal states / timeout
  / unknown task).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.backends.mock_backend import (
    MockBackend, MockTurn, ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.policy import CORE_STATE_FIELDS


# ===========================================================================
# Fixtures
# ===========================================================================
@dataclass
class _BareState:
    """SharedState stand-in covering every attribute the CLOSE sequencer
    + helpers read / write."""

    closing_report_task_id: str = ""
    cortex_session_id: str = ""
    cortex_session_summary: dict[str, Any] = field(default_factory=dict)
    stop_reason: str = ""
    close_sequence_done: bool = False
    phase_history: list[dict[str, Any]] = field(default_factory=list)
    save_count: int = 0

    def save(self, _session_dir: Path | None) -> None:
        self.save_count += 1

    def set_stop_reason(self, reason: str) -> None:
        # Mirror SharedState.set_stop_reason ENUM-validated writer
        # signature; tests pass valid v0.8 vocab values.
        self.stop_reason = reason


@dataclass
class _StubTaskRow:
    task_id: str
    kind: str
    state: str
    params: dict
    idempotency_key: str


class _StubTaskRegistry:
    """create_or_return_existing + get double for CLOSE sequencer
    tests. Tracks insertion order so we can assert step 1 enqueued
    before step 2."""

    def __init__(self):
        self._by_key: dict[str, _StubTaskRow] = {}
        self._by_id: dict[str, _StubTaskRow] = {}
        self.insertion_order: list[str] = []  # idempotency_keys

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
            state="succeeded",  # terminal so _wait_for_task_terminal returns immediately
            params=dict(params),
            idempotency_key=idempotency_key,
        )
        self._by_key[idempotency_key] = row
        self._by_id[tid] = row
        self.insertion_order.append(idempotency_key)
        return row, False

    async def get(self, task_id):
        from inference_optimizer.orchestrator.task_registry import TaskNotFound

        row = self._by_id.get(task_id)
        if row is None:
            raise TaskNotFound(task_id)
        return row


class _StubCortex:
    """Cortex KB double with controllable behaviour for steps 3+4."""

    enabled: bool = True

    def __init__(
        self,
        *,
        drain_remaining: int = 0,
        drain_raises: BaseException | None = None,
        commit_raises: BaseException | None = None,
        commit_summary: dict | None = None,
    ):
        self.drain_calls: int = 0
        self.commit_calls: int = 0
        self._drain_remaining = drain_remaining
        self._drain_raises = drain_raises
        self._commit_raises = commit_raises
        self._commit_summary = commit_summary or {"status": "committed"}

    def drain_pending(self, *, timeout_sec: float = 60.0) -> dict:
        self.drain_calls += 1
        if self._drain_raises is not None:
            raise self._drain_raises
        return {"remaining": self._drain_remaining}

    def session_commit(self, sid: str) -> dict:
        self.commit_calls += 1
        if self._commit_raises is not None:
            raise self._commit_raises
        return dict(self._commit_summary)


@pytest.fixture
def coord(tmp_path: Path):
    """Lean Coordinator stub for hook unit tests."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _BareState()
    c.tasks = _StubTaskRegistry()
    c.cortex_kb = None
    c.knowledge_plane = None
    c.role_registry = {}
    return c


def _close_phase_history_row() -> dict[str, Any]:
    return {"to_phase": "CLOSE", "reason": "sweep_done", "evidence": {}}


# ===========================================================================
# 1. _record_close_step helper edges
# ===========================================================================
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
    assert "detail" not in rows[0]  # omitted when empty
    assert coord.shared_state.save_count == 1


@pytest.mark.asyncio
async def test_record_close_step_optional_detail(coord):
    coord.shared_state.phase_history = [_close_phase_history_row()]
    await coord._record_close_step(
        "cortex_commit", status="failed", detail="cortex unreachable",
    )
    row = coord.shared_state.phase_history[-1]["evidence"]["close_steps"][0]
    assert row["detail"] == "cortex unreachable"


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
    """Defensive: malformed pre-existing close_steps (not a list)
    gets replaced with a fresh list before appending."""
    coord.shared_state.phase_history = [{
        "to_phase": "CLOSE", "evidence": {"close_steps": "broken"},
    }]
    await coord._record_close_step("report", status="done")
    assert isinstance(
        coord.shared_state.phase_history[-1]["evidence"]["close_steps"], list
    )


@pytest.mark.asyncio
async def test_record_close_step_no_op_when_history_empty(coord):
    coord.shared_state.phase_history = []
    # Must not raise:
    await coord._record_close_step("report", status="done")


# ===========================================================================
# 2. _wait_for_task_terminal helper
# ===========================================================================
@pytest.mark.asyncio
async def test_wait_for_task_terminal_returns_immediately_when_succeeded(
    coord,
):
    # _StubTaskRegistry inserts rows with state='succeeded' so the very
    # first poll wins.
    await coord.tasks.create_or_return_existing(
        kind="report", params={}, idempotency_key="k1", task_id="t-x",
    )
    state = await coord._wait_for_task_terminal("t-x", timeout_sec=1.0)
    assert state == "succeeded"


@pytest.mark.asyncio
async def test_wait_for_task_terminal_returns_none_for_unknown(coord):
    state = await coord._wait_for_task_terminal("missing", timeout_sec=1.0)
    assert state is None


@pytest.mark.asyncio
async def test_wait_for_task_terminal_timeout(coord):
    """Task never reaches terminal → returns None after timeout."""
    class _StuckRegistry:
        async def get(self, task_id):
            return _StubTaskRow(
                task_id=task_id, kind="report", state="running",
                params={}, idempotency_key="",
            )
    coord.tasks = _StuckRegistry()
    state = await coord._wait_for_task_terminal("t-stuck", timeout_sec=0.05)
    assert state is None


# ===========================================================================
# 3. Step 1: report task enqueue
# ===========================================================================
@pytest.mark.asyncio
async def test_enqueue_internal_report_task_fresh(coord):
    task = await coord._enqueue_internal_report_task(reason="close_phase_entry")
    assert task.kind == "report"
    assert task.idempotency_key == "internal-report-close_phase_entry"
    assert task.params["source"] == "coordinator_internal"
    assert task.params["reason"] == "close_phase_entry"
    # closing_report_task_id mirror populated for back-compat with the
    # wall-clock deadline path inspectors.
    assert coord.shared_state.closing_report_task_id == task.task_id


@pytest.mark.asyncio
async def test_enqueue_internal_report_task_reuses_existing(coord):
    """When wall-clock deadline already enqueued a report task, the
    sequencer must reuse it instead of inserting a duplicate."""
    # Pre-seed: a report task from the wall-clock path.
    existing = _StubTaskRow(
        task_id="wallclock-report", kind="report", state="succeeded",
        params={}, idempotency_key="closing-report-1234",
    )
    coord.tasks._by_id["wallclock-report"] = existing
    coord.shared_state.closing_report_task_id = "wallclock-report"

    task = await coord._enqueue_internal_report_task(reason="close_phase_entry")
    assert task is existing
    # No new key inserted.
    assert "internal-report-close_phase_entry" not in coord.tasks._by_key


# ===========================================================================
# 4. Step 2: session_breakdown task enqueue
# ===========================================================================
@pytest.mark.asyncio
async def test_enqueue_internal_session_breakdown_task(coord):
    task = await coord._enqueue_internal_session_breakdown_task(
        reason="close_phase_entry",
    )
    assert task.kind == "session_breakdown"
    assert task.idempotency_key == "internal-session_breakdown-close_phase_entry"
    assert task.params["source"] == "coordinator_internal"


# ===========================================================================
# 5. End-to-end 5-step sequencer ordering
# ===========================================================================
@pytest.mark.asyncio
async def test_close_sequencer_runs_all_5_steps_in_order_happy_path(
    coord,
):
    coord.shared_state.phase_history = [_close_phase_history_row()]
    coord.cortex_kb = _StubCortex(commit_summary={"status": "committed"})
    coord.shared_state.cortex_session_id = "sid-test"

    await coord._on_enter_close(from_phase="SWEEP")

    rows = coord.shared_state.phase_history[-1]["evidence"]["close_steps"]
    # Step sequence: sequencer_started, report, session_breakdown,
    # ndjson_drain, cortex_commit, done.
    steps = [r["step"] for r in rows]
    assert steps == [
        "sequencer_started", "report", "session_breakdown",
        "ndjson_drain", "cortex_commit", "done",
    ]
    # All five effective steps succeeded.
    assert rows[1]["status"] == "done"   # report
    assert rows[2]["status"] == "done"   # session_breakdown
    assert rows[3]["status"] == "done"   # ndjson_drain
    assert rows[4]["status"] == "done"   # cortex_commit
    assert rows[5]["status"] == "done"   # done
    # close_sequence_done flag set.
    assert coord.shared_state.close_sequence_done is True
    # Cortex commit ran exactly once.
    assert coord.cortex_kb.commit_calls == 1
    assert coord.cortex_kb.drain_calls == 1
    # session_summary populated from the commit return.
    assert coord.shared_state.cortex_session_summary == {"status": "committed"}
    # v0.8 §3.2 §5.5 — sequencer step 5 must set stop_reason so the
    # main loop terminates on the next tick. Without this the loop
    # keeps re-firing the report executor + robustness alerts (the
    # bug we're closing here).
    assert coord.shared_state.stop_reason == "time_exhausted"


@pytest.mark.asyncio
async def test_close_sequencer_sets_time_exhausted_stop_reason(coord):
    """Step 5 stamps ``stop_reason='time_exhausted'`` when no other
    step set it first. ``time_exhausted`` is the canonical CLOSE
    vocab term in :data:`STOP_REASON_VOCAB` and matches the
    wall-clock-deadline path's terminator (Coordinator.run).
    """
    coord.shared_state.phase_history = [_close_phase_history_row()]
    assert coord.shared_state.stop_reason == ""

    await coord._on_enter_close(from_phase="SWEEP")

    assert coord.shared_state.stop_reason == "time_exhausted"


@pytest.mark.asyncio
async def test_close_sequencer_does_not_overwrite_existing_stop_reason(
    coord,
):
    """If step 4 already set ``stop_reason='cortex_commit_failed'``
    (or any other value), step 5's setter must leave it alone — the
    earlier observation carries the more diagnostic information."""
    coord.shared_state.phase_history = [_close_phase_history_row()]
    coord.cortex_kb = _StubCortex(commit_raises=RuntimeError("synthetic"))
    coord.shared_state.cortex_session_id = "sid"

    await coord._on_enter_close(from_phase="SWEEP")

    # Step 4 set this; step 5 must NOT overwrite to time_exhausted.
    assert coord.shared_state.stop_reason == "cortex_commit_failed"


@pytest.mark.asyncio
async def test_close_sequencer_does_not_overwrite_caller_set_stop_reason(
    coord,
):
    """An operator-set ``stop_reason`` (e.g. ``signal`` from a
    SIGTERM that arrived during the sequencer) must survive step 5."""
    coord.shared_state.phase_history = [_close_phase_history_row()]
    coord.shared_state.stop_reason = "signal"

    await coord._on_enter_close(from_phase="SWEEP")

    assert coord.shared_state.stop_reason == "signal"


@pytest.mark.asyncio
async def test_close_sequencer_report_before_session_breakdown(coord):
    """Step 1 (report) MUST be enqueued before step 2
    (session_breakdown). Asserted via TaskRegistry insertion order."""
    coord.shared_state.phase_history = [_close_phase_history_row()]
    await coord._on_enter_close(from_phase="SWEEP")
    insertion = coord.tasks.insertion_order
    report_idx = insertion.index("internal-report-close_phase_entry")
    bd_idx = insertion.index("internal-session_breakdown-close_phase_entry")
    assert report_idx < bd_idx


@pytest.mark.asyncio
async def test_close_sequencer_skips_cortex_steps_when_no_cortex(coord):
    """``--degraded-kb`` runs have cortex_kb=None: steps 3 + 4 must be
    explicitly recorded as 'skipped' (not silent no-ops, so operators
    can see the run was Cortex-less)."""
    coord.shared_state.phase_history = [_close_phase_history_row()]
    await coord._on_enter_close(from_phase="SWEEP")
    rows = coord.shared_state.phase_history[-1]["evidence"]["close_steps"]
    drain_row = next(r for r in rows if r["step"] == "ndjson_drain")
    commit_row = next(r for r in rows if r["step"] == "cortex_commit")
    assert drain_row["status"] == "skipped"
    assert commit_row["status"] == "skipped"
    # close_sequence_done still set so cli.finally short-circuits.
    assert coord.shared_state.close_sequence_done is True


@pytest.mark.asyncio
async def test_close_sequencer_records_drain_incomplete(coord):
    """NDJSON drain returns ``remaining > 0`` → step 3 status='incomplete'
    with detail of remaining count. Sequencer continues to step 4 / 5."""
    coord.shared_state.phase_history = [_close_phase_history_row()]
    coord.cortex_kb = _StubCortex(drain_remaining=12)
    coord.shared_state.cortex_session_id = "sid"
    await coord._on_enter_close(from_phase="SWEEP")
    rows = coord.shared_state.phase_history[-1]["evidence"]["close_steps"]
    drain_row = next(r for r in rows if r["step"] == "ndjson_drain")
    assert drain_row["status"] == "incomplete"
    assert "remaining=12" in drain_row["detail"]
    # Commit still attempted.
    assert coord.cortex_kb.commit_calls == 1
    # done step still reached.
    assert coord.shared_state.close_sequence_done is True


@pytest.mark.asyncio
async def test_close_sequencer_records_commit_failure(coord):
    """Cortex commit raises → step 4 status='failed', summary records
    ``status='commit_failed'``, stop_reason set to ``cortex_commit_failed``
    if not already set (matches _cortex_t4_hook semantics)."""
    coord.shared_state.phase_history = [_close_phase_history_row()]
    coord.cortex_kb = _StubCortex(commit_raises=RuntimeError("synthetic outage"))
    coord.shared_state.cortex_session_id = "sid"

    await coord._on_enter_close(from_phase="SWEEP")

    rows = coord.shared_state.phase_history[-1]["evidence"]["close_steps"]
    commit_row = next(r for r in rows if r["step"] == "cortex_commit")
    assert commit_row["status"] == "failed"
    assert "synthetic outage" in commit_row["detail"]
    assert coord.shared_state.cortex_session_summary["status"] == "commit_failed"
    assert coord.shared_state.stop_reason == "cortex_commit_failed"
    # done step still reached.
    assert coord.shared_state.close_sequence_done is True


@pytest.mark.asyncio
async def test_close_sequencer_records_drain_failure(coord):
    """Drain raises → step 3 status='failed', sequencer continues to
    step 4 / 5 anyway."""
    coord.shared_state.phase_history = [_close_phase_history_row()]
    coord.cortex_kb = _StubCortex(drain_raises=OSError("disk full"))
    coord.shared_state.cortex_session_id = "sid"

    await coord._on_enter_close(from_phase="SWEEP")

    rows = coord.shared_state.phase_history[-1]["evidence"]["close_steps"]
    drain_row = next(r for r in rows if r["step"] == "ndjson_drain")
    assert drain_row["status"] == "failed"
    assert "disk full" in drain_row["detail"]
    # Cortex commit still attempted despite drain failure.
    assert coord.cortex_kb.commit_calls == 1


# ===========================================================================
# 6. CORE_STATE_FIELDS lock
# ===========================================================================
def test_close_sequence_done_in_core_state_fields():
    """LLM update_state must not be able to flip close_sequence_done
    and trick cli.finally into skipping its safety net for a non-CLOSE
    termination."""
    assert "close_sequence_done" in CORE_STATE_FIELDS


def test_policy_blocks_llm_close_sequence_done_write():
    from inference_optimizer.orchestrator.agent_role import (
        default_role_registry,
    )
    from inference_optimizer.orchestrator.intent_parser import (
        Intent, IntentType,
    )
    from inference_optimizer.orchestrator.policy import (
        PolicyDenied, PolicyGate,
    )
    gate = PolicyGate(role_registry=default_role_registry())
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"close_sequence_done": True}},
    )
    with pytest.raises(PolicyDenied):
        gate.validate_intent("orchestration", intent)


# ===========================================================================
# 7. End-to-end via real Coordinator (real TaskRegistry + bus)
# ===========================================================================
@pytest.mark.asyncio
async def test_phase_transition_into_close_runs_sequencer_e2e(tmp_path: Path):
    """End-to-end: real Coordinator + real TaskRegistry. After
    transitioning to CLOSE + invoking the hook dispatcher, the
    sequencer enqueues both internal tasks under the v0.8 keys and
    flips ``close_sequence_done=True``."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "kernel":        MockBackend(idle_plan),
        "critic":        MockBackend(idle_plan),
        "robustness":    MockBackend(idle_plan),
    }
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        cortex_kb=None,
        knowledge_plane=None,
    )
    # Shrink wait timeouts — real dispatcher isn't ticking in this
    # test so the queued tasks never reach a terminal state; we want
    # the sequencer to record 'failed' (timeout) and move on, not
    # hang on the production 5-10 min defaults.
    coord.CLOSE_REPORT_TIMEOUT_SEC = 0.1
    coord.CLOSE_SESSION_BREAKDOWN_TIMEOUT_SEC = 0.1

    # Seed state at SWEEP boundary as if sweep_done just fired.
    coord.shared_state.phase = "SWEEP"
    coord.shared_state.phase_history = [
        {"to_phase": "EXPLORE", "evidence": {}, "reason": "prelude_done"},
        {"to_phase": "SWEEP",   "evidence": {}, "reason": "plateau_kernel"},
    ]
    coord.shared_state.record_phase_transition(
        to_phase="CLOSE", reason="sweep_done",
        evidence={"trigger": "test_e2e"},
    )
    await coord._on_phase_entered(from_phase="SWEEP", to_phase="CLOSE")

    # Both internal tasks were inserted into the real registry.
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

    # close_sequence_done flipped despite the queued tasks (steps 3+4
    # skipped because no cortex_kb; step 5 always sets the flag).
    assert coord.shared_state.close_sequence_done is True
    # Evidence section recorded steps.
    evidence = coord.shared_state.phase_history[-1]["evidence"]
    steps = [r["step"] for r in evidence.get("close_steps", [])]
    assert "sequencer_started" in steps
    assert "done" in steps


# ===========================================================================
# 8. stop()'s _cortex_t4_hook short-circuits when sequencer already ran
# ===========================================================================
@pytest.mark.asyncio
async def test_cortex_t4_hook_short_circuits_when_sequencer_done(tmp_path: Path):
    """KB_gaps/Gap-06 acceptance: Cortex session_commit MUST be called
    exactly once. If the CLOSE sequencer already committed, stop()'s
    legacy _cortex_t4_hook must skip."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "kernel":        MockBackend(idle_plan),
        "critic":        MockBackend(idle_plan),
        "robustness":    MockBackend(idle_plan),
    }
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        cortex_kb=_StubCortex(),
        knowledge_plane=None,
    )
    coord.shared_state.cortex_session_id = "sid-stop-skip"
    coord.shared_state.close_sequence_done = True

    await coord._cortex_t4_hook()
    # Drain + commit both untouched.
    assert coord.cortex_kb.drain_calls == 0
    assert coord.cortex_kb.commit_calls == 0


@pytest.mark.asyncio
async def test_cortex_t4_hook_still_runs_when_sequencer_not_done(tmp_path: Path):
    """Crash / Ctrl-C path: sequencer didn't run, so stop()'s
    _cortex_t4_hook MUST still drain + commit as a safety net."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "kernel":        MockBackend(idle_plan),
        "critic":        MockBackend(idle_plan),
        "robustness":    MockBackend(idle_plan),
    }
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        cortex_kb=_StubCortex(),
        knowledge_plane=None,
    )
    coord.shared_state.cortex_session_id = "sid-fallback"
    coord.shared_state.close_sequence_done = False

    await coord._cortex_t4_hook()
    assert coord.cortex_kb.drain_calls == 1
    assert coord.cortex_kb.commit_calls == 1
