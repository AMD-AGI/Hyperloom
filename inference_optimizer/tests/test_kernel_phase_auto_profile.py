"""v0.8 §3.2 §5.3 / KB_gaps/Gap-04 — KERNEL phase auto-profile tests.

KB_gaps/Gap-04 root cause: ``_advance_phase_if_needed`` updated the
``phase`` field on EXPLORE → KERNEL transition but did not enqueue
anything. The design says KERNEL entry should fire a deterministic
``profile`` task ("固定动作, 不需要 LLM propose") so
``last_profile_trace`` is always populated for the rest of the
KERNEL phase — without it, every ``select_kernels`` / ``kernel_opt``
gets stuck behind the sequence gate.

This file covers:

* Direct ``_on_enter_kernel`` unit tests (skip / enqueue / evidence).
* End-to-end Coordinator path: EXPLORE → KERNEL transition fires the
  hook and lands a kind='profile' task on the TaskRegistry.
* Idempotence on phase re-entry (defensive — Inv-2.1 monotonic
  forbids re-entry in practice, but the test asserts the
  ``idempotency_key='internal-profile-kernel_phase_entry'`` works
  via ``create_or_return_existing``).
* ``--no-kernel`` mode skip (defense in depth: ``compute_next_phase``
  routes around KERNEL, but the hook re-asserts).
* Resume scenario: ``last_profile_trace`` already set → skip.
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


# ===========================================================================
# Fixtures — lean Coordinator stand-in for unit tests
# ===========================================================================
@dataclass
class _BareState:
    """SharedState stand-in carrying only the fields the Gap-04 hook
    touches. Mirrors :class:`SharedState`'s relevant attribute names
    + a ``phase_history`` list so the evidence mutation has somewhere
    to land."""

    kernel_enabled: bool = True
    last_profile_trace: str = ""
    last_profile_status: str = ""
    last_profile_args: str = ""
    baseline_config_path: str = ""
    baseline_tput: float = 0.0
    current_best: dict[str, Any] = field(default_factory=dict)
    last_baseline: dict[str, Any] = field(default_factory=dict)
    phase_history: list[dict[str, Any]] = field(default_factory=list)
    # Roofline-related fields used by ``_needs_fresh_roofline``.
    use_roofline_composite: bool = False
    last_trace_analyze: dict[str, Any] = field(default_factory=dict)
    cumulative_gain: float = 0.0
    cumulative_gain_validated: float = 0.0
    auto_roofline_pending_task_id: str = ""
    save_count: int = 0
    # Audit ledger touched by Coordinator._promote_to_shared_state.
    action_attempts: list[dict[str, Any]] = field(default_factory=list)

    def save(self, _session_dir: Path | None) -> None:
        self.save_count += 1

    def record_action_attempt(self, **kwargs: Any) -> None:
        self.action_attempts.append(dict(kwargs))


class _StubTaskRegistry:
    """Minimal TaskRegistry double mirroring the
    :meth:`TaskRegistry.create_or_return_existing` contract — keyed
    by ``idempotency_key`` for the idempotence test."""

    def __init__(self):
        self._tasks: dict[str, Any] = {}

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
        from inference_optimizer.orchestrator.task_registry import Task

        existing = self._tasks.get(idempotency_key)
        if existing is not None:
            return existing, True
        import uuid as _uuid
        task = Task(
            task_id=task_id or _uuid.uuid4().hex,
            kind=kind,
            state="queued",
            params=dict(params),
            idempotency_key=idempotency_key,
        )
        self._tasks[idempotency_key] = task
        return task, False


@pytest.fixture
def coord(tmp_path: Path):
    """Build a Coordinator-shaped stub with just enough surface for
    ``_on_enter_kernel`` to run. Uses ``Coordinator.__new__`` so we
    skip the full constructor (which needs a sqlite db / backends /
    role registry / bus)."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _BareState()
    c.tasks = _StubTaskRegistry()
    c.knowledge_plane = None
    # _kernel_enabled() consults both role_registry presence + state flag.
    c.role_registry = {"kernel": object()}
    return c


# ===========================================================================
# 1. Hook skip conditions
# ===========================================================================
@pytest.mark.asyncio
async def test_on_enter_kernel_skips_when_kernel_disabled(coord):
    """``--no-kernel`` runs (or any path where the kernel role isn't
    registered) must NOT enqueue an auto-profile."""
    coord.role_registry = {}   # no 'kernel' role → _kernel_enabled() False
    coord.shared_state.kernel_enabled = False
    coord.shared_state.phase_history = [
        {"to_phase": "KERNEL", "evidence": {}, "reason": "plateau_explore"},
    ]
    await coord._on_enter_kernel(from_phase="EXPLORE")
    assert coord.tasks._tasks == {}
    # No evidence stamp either — we left the row untouched.
    assert "auto_profile_enqueued" not in coord.shared_state.phase_history[-1]["evidence"]


@pytest.mark.asyncio
async def test_on_enter_kernel_skips_on_resume_with_existing_trace(coord):
    """Resume case (composite OFF): a prior session already landed a
    profile trace. The hook must skip re-profiling (5–10 min wasted
    bench)."""
    coord.shared_state.use_roofline_composite = False
    coord.shared_state.last_profile_trace = (
        "/tmp/session/runs/profile/abc/torch_trace/run-1.trace.json.gz"
    )
    coord.shared_state.phase_history = [
        {"to_phase": "KERNEL", "evidence": {}, "reason": "resume"},
    ]
    await coord._on_enter_kernel(from_phase="EXPLORE")
    assert coord.tasks._tasks == {}
    # Evidence still records the skip reason for audit.
    assert (
        coord.shared_state.phase_history[-1]["evidence"]
        .get("auto_profile_skipped") == "trace_exists"
    )


@pytest.mark.asyncio
async def test_on_enter_kernel_enqueues_profile_happy_path(coord):
    """Fresh KERNEL entry with no prior trace and composite OFF: hook
    enqueues an internal profile task + stamps phase_history evidence.

    The hook deliberately does NOT propagate
    ``state.baseline_config_path`` into the task params — that would
    point ProfileExecutor at a baseline YAML where
    ``profiler.torch_profiler.enabled`` is false, silently disabling
    Magpie's torch profiler. ProfileExecutor's
    ``_resolve_default_config`` picks ``profile_sglang.yaml`` instead;
    workload contract still flows via env-driven materialization.
    """
    coord.shared_state.use_roofline_composite = False
    coord.shared_state.baseline_config_path = "/tmp/baseline.yaml"
    coord.shared_state.current_best = {
        "extra_sglang_args": "--mla 1",
    }
    coord.shared_state.last_baseline = {
        "benchmark_script": "sglang_mi300x.sh",
    }
    coord.shared_state.phase_history = [
        {"to_phase": "KERNEL", "evidence": {}, "reason": "plateau_explore"},
    ]
    await coord._on_enter_kernel(from_phase="EXPLORE")
    # Exactly one task enqueued, keyed by the phase-scoped idem key.
    assert "internal-profile-kernel_phase_entry" in coord.tasks._tasks
    task = coord.tasks._tasks["internal-profile-kernel_phase_entry"]
    assert task.kind == "profile"
    assert task.state == "queued"
    assert task.params["source"] == "coordinator_internal"
    assert task.params["reason"] == "kernel_phase_entry"
    # config_path is intentionally NOT propagated — see helper docstring.
    assert "config_path" not in task.params
    assert task.params["base_extra_args"] == "--mla 1"
    assert task.params["benchmark_script"] == "sglang_mi300x.sh"
    # Evidence stamp visible on the phase_history row.
    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert evidence.get("auto_profile_enqueued") is True
    assert evidence.get("auto_profile_task_id") == task.task_id


@pytest.mark.asyncio
async def test_on_enter_kernel_omits_optional_params_when_state_empty(coord):
    """Bare state (no current_best / last_baseline), composite OFF:
    hook still enqueues the task. The profile executor falls back to
    its own defaults; params only carry source + reason."""
    coord.shared_state.use_roofline_composite = False
    coord.shared_state.phase_history = [
        {"to_phase": "KERNEL", "evidence": {}, "reason": "plateau_explore"},
    ]
    await coord._on_enter_kernel(from_phase="EXPLORE")
    task = coord.tasks._tasks["internal-profile-kernel_phase_entry"]
    assert task.params == {
        "source": "coordinator_internal",
        "reason": "kernel_phase_entry",
    }


# ===========================================================================
# 2. Idempotence — phase re-entry doesn't double-enqueue
# ===========================================================================
@pytest.mark.asyncio
async def test_on_enter_kernel_idempotent_on_reentry(coord):
    """Calling the hook twice with the same idempotency key returns
    the existing task instead of inserting a duplicate. Defensive —
    Inv-2.1 monotonic phase forbids real re-entry, but tests +
    operator scripts might trigger it."""
    coord.shared_state.phase_history = [
        {"to_phase": "KERNEL", "evidence": {}, "reason": "plateau_explore"},
    ]
    await coord._on_enter_kernel(from_phase="EXPLORE")
    task1 = coord.tasks._tasks["internal-profile-kernel_phase_entry"]
    # Reset the evidence to mimic a fresh phase_history row
    # (re-entry would have appended a new history row in production).
    coord.shared_state.phase_history.append({
        "to_phase": "KERNEL", "evidence": {}, "reason": "re_entry_test",
    })
    await coord._on_enter_kernel(from_phase="KERNEL")
    task2 = coord.tasks._tasks["internal-profile-kernel_phase_entry"]
    assert task1 is task2, (
        "second call must reuse existing task via idempotency_key"
    )
    # Single key → single task.
    assert len(coord.tasks._tasks) == 1


# ===========================================================================
# 3. _enqueue_internal_profile_task — params construction in isolation
# ===========================================================================
@pytest.mark.asyncio
async def test_enqueue_internal_profile_task_uses_explicit_reason(coord):
    coord.shared_state.phase_history = []
    task = await coord._enqueue_internal_profile_task(reason="manual_smoke")
    assert task.idempotency_key == "internal-profile-manual_smoke"
    assert task.params["reason"] == "manual_smoke"


@pytest.mark.asyncio
async def test_enqueue_internal_profile_task_omits_empty_strings(coord):
    """``current_best.extra_sglang_args=''`` should NOT land in params
    (avoids polluting profile executor's diff with empty fields)."""
    coord.shared_state.current_best = {"extra_sglang_args": ""}
    coord.shared_state.last_baseline = {"benchmark_script": ""}
    task = await coord._enqueue_internal_profile_task(
        reason="kernel_phase_entry",
    )
    assert "base_extra_args" not in task.params
    assert "benchmark_script" not in task.params


# ===========================================================================
# 4. _record_phase_entry_evidence helper
# ===========================================================================
@pytest.mark.asyncio
async def test_record_phase_entry_evidence_merges_into_latest_row(coord):
    coord.shared_state.phase_history = [
        {"to_phase": "KERNEL", "evidence": {"x": 1}, "reason": "r"},
    ]
    coord._record_phase_entry_evidence(y=2, z="three")
    assert coord.shared_state.phase_history[-1]["evidence"] == {
        "x": 1, "y": 2, "z": "three",
    }
    # Persisted.
    assert coord.shared_state.save_count == 1


def test_record_phase_entry_evidence_noop_when_history_empty(coord):
    coord.shared_state.phase_history = []
    coord._record_phase_entry_evidence(anything=1)
    # No crash; nothing persisted.
    assert coord.shared_state.save_count == 0


def test_record_phase_entry_evidence_handles_missing_evidence_dict(coord):
    """A row with no ``evidence`` key (or a non-dict evidence) gets a
    fresh dict installed before merging."""
    coord.shared_state.phase_history = [{"to_phase": "KERNEL", "reason": "r"}]
    coord._record_phase_entry_evidence(auto_profile_enqueued=True)
    assert coord.shared_state.phase_history[-1]["evidence"] == {
        "auto_profile_enqueued": True,
    }


# ===========================================================================
# 5. End-to-end via real Coordinator
# ===========================================================================
# ===========================================================================
# 6. Auto-roofline path (use_roofline_composite=True)
# ===========================================================================
@pytest.mark.asyncio
async def test_on_enter_kernel_enqueues_roofline_when_composite_on(coord):
    """When ``use_roofline_composite=True`` and no fresh snapshot
    exists, the KERNEL entry hook MUST enqueue a ``roofline`` task,
    not a ``profile`` task. The idempotency key is phase-scoped under
    ``internal-roofline-kernel_phase_entry`` so a re-entry dedupes
    against this exact slot."""
    coord.shared_state.use_roofline_composite = True
    coord.shared_state.last_trace_analyze = {}   # no snapshot yet
    coord.shared_state.baseline_config_path = "/tmp/baseline.yaml"
    coord.shared_state.phase_history = [
        {"to_phase": "KERNEL", "evidence": {}, "reason": "plateau_explore"},
    ]
    await coord._on_enter_kernel(from_phase="EXPLORE")

    # Roofline key, not profile.
    assert "internal-roofline-kernel_phase_entry" in coord.tasks._tasks
    assert "internal-profile-kernel_phase_entry" not in coord.tasks._tasks
    task = coord.tasks._tasks["internal-roofline-kernel_phase_entry"]
    assert task.kind == "roofline"
    assert task.params["source"] == "coordinator_internal"
    assert task.params["reason"] == "kernel_phase_entry"
    # config_path is intentionally NOT propagated so ProfileExecutor
    # picks profile_sglang.yaml (the one with torch_profiler enabled).
    assert "config_path" not in task.params

    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert evidence.get("auto_roofline_enqueued") is True
    assert evidence.get("auto_roofline_task_id") == task.task_id


@pytest.mark.asyncio
async def test_on_enter_kernel_skips_when_snapshot_fresh(coord):
    """When ``use_roofline_composite=True`` AND the existing snapshot
    is within the gain-drift threshold (10%), no new roofline fires —
    the existing analysis.md is still representative. Evidence stamps
    ``auto_roofline_skipped='fresh_snapshot'`` so the audit trail
    shows why."""
    coord.shared_state.use_roofline_composite = True
    coord.shared_state.last_trace_analyze = {
        "analysis_md_text": "## Executive Summary\n| Compute % | 60% |",
        "roofline_baseline_gain_at_snapshot": 12.0,
    }
    coord.shared_state.cumulative_gain_validated = 15.0   # delta 3% < 10%
    coord.shared_state.phase_history = [
        {"to_phase": "KERNEL", "evidence": {}, "reason": "plateau_explore"},
    ]
    await coord._on_enter_kernel(from_phase="EXPLORE")

    assert coord.tasks._tasks == {}
    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert evidence.get("auto_roofline_skipped") == "fresh_snapshot"


@pytest.mark.asyncio
async def test_on_enter_kernel_fires_roofline_when_gain_drifts(coord):
    """Existing snapshot's gain anchor is now 12% behind current
    cumulative_gain — re-roof so analysis.md aligns with stack."""
    coord.shared_state.use_roofline_composite = True
    coord.shared_state.last_trace_analyze = {
        "analysis_md_text": "## Executive Summary\n| Compute % | 60% |",
        "roofline_baseline_gain_at_snapshot": 5.0,
    }
    coord.shared_state.cumulative_gain_validated = 17.5   # delta = 12.5% > 10%
    coord.shared_state.phase_history = [
        {"to_phase": "KERNEL", "evidence": {}, "reason": "plateau_explore"},
    ]
    await coord._on_enter_kernel(from_phase="EXPLORE")

    assert "internal-roofline-kernel_phase_entry" in coord.tasks._tasks
    task = coord.tasks._tasks["internal-roofline-kernel_phase_entry"]
    assert task.kind == "roofline"


# ===========================================================================
# 7. _needs_fresh_roofline freshness gate
# ===========================================================================
def test_needs_fresh_roofline_off_when_composite_disabled(coord):
    coord.shared_state.use_roofline_composite = False
    coord.shared_state.last_trace_analyze = {}
    assert coord._needs_fresh_roofline() is False


def test_needs_fresh_roofline_true_when_no_snapshot(coord):
    coord.shared_state.use_roofline_composite = True
    coord.shared_state.last_trace_analyze = {}
    assert coord._needs_fresh_roofline() is True


def test_needs_fresh_roofline_true_when_analysis_text_empty(coord):
    coord.shared_state.use_roofline_composite = True
    coord.shared_state.last_trace_analyze = {
        "analysis_md_text": "",
        "roofline_baseline_gain_at_snapshot": 0.0,
    }
    assert coord._needs_fresh_roofline() is True


def test_needs_fresh_roofline_false_inside_drift_band(coord):
    coord.shared_state.use_roofline_composite = True
    coord.shared_state.last_trace_analyze = {
        "analysis_md_text": "## hello",
        "roofline_baseline_gain_at_snapshot": 8.0,
    }
    coord.shared_state.cumulative_gain_validated = 12.0   # delta 4% < 10%
    assert coord._needs_fresh_roofline() is False


def test_needs_fresh_roofline_true_beyond_drift_band(coord):
    coord.shared_state.use_roofline_composite = True
    coord.shared_state.last_trace_analyze = {
        "analysis_md_text": "## hello",
        "roofline_baseline_gain_at_snapshot": 2.0,
    }
    coord.shared_state.cumulative_gain_validated = 14.0   # delta 12% > 10%
    assert coord._needs_fresh_roofline() is True


# ===========================================================================
# 8. _enqueue_internal_roofline_task — params construction
# ===========================================================================
@pytest.mark.asyncio
async def test_enqueue_internal_roofline_task_uses_explicit_reason(coord):
    coord.shared_state.use_roofline_composite = True
    task = await coord._enqueue_internal_roofline_task(reason="explore_entry")
    assert task.idempotency_key == "internal-roofline-explore_entry"
    assert task.kind == "roofline"
    assert task.params["reason"] == "explore_entry"
    assert task.params["source"] == "coordinator_internal"


@pytest.mark.asyncio
async def test_enqueue_internal_roofline_task_carries_extra_args(coord):
    """Roofline task inherits current_best.extra_sglang_args +
    last_baseline.benchmark_script, but deliberately does NOT carry
    ``state.baseline_config_path`` — the profile sub-step needs the
    profile YAML (with torch_profiler enabled) and finds it via
    ``ProfileExecutor._resolve_default_config``. Passing baseline YAML
    here silently disables Magpie's torch profiler → no trace files."""
    coord.shared_state.use_roofline_composite = True
    coord.shared_state.baseline_config_path = "/tmp/baseline.yaml"
    coord.shared_state.current_best = {"extra_sglang_args": "--enable-cuda-graph"}
    coord.shared_state.last_baseline = {"benchmark_script": "sglang_mi300x.sh"}
    task = await coord._enqueue_internal_roofline_task(reason="kernel_phase_entry")
    assert "config_path" not in task.params
    assert task.params["base_extra_args"] == "--enable-cuda-graph"
    assert task.params["benchmark_script"] == "sglang_mi300x.sh"


# ===========================================================================
# 9. E2E test with --no-kernel
# ===========================================================================
@pytest.mark.asyncio
async def test_phase_transition_into_kernel_enqueues_roofline_e2e(
    tmp_path: Path,
):
    """E2E with the default ``use_roofline_composite=True``: a KERNEL
    transition lands a ``roofline`` task (not the legacy ``profile``)
    under the new internal idempotency key."""
    from inference_optimizer.orchestrator.shared_state import SharedState

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
    coord.shared_state.phase = "EXPLORE"
    coord.shared_state.kernel_enabled = True
    coord.shared_state.use_roofline_composite = True
    coord.shared_state.last_trace_analyze = {}  # no snapshot yet
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.cumulative_gain = 5.0
    coord.shared_state.phase_history = [
        {"to_phase": "EXPLORE", "evidence": {}, "reason": "prelude_done"},
    ]
    coord.shared_state.record_phase_transition(
        to_phase="KERNEL",
        reason="plateau_explore",
        evidence={"trigger": "test_e2e_roofline"},
    )
    await coord._on_phase_entered(from_phase="EXPLORE", to_phase="KERNEL")

    queued = await coord.tasks.db.fetchall(
        "SELECT * FROM tasks WHERE idempotency_key=?",
        ("internal-roofline-kernel_phase_entry",),
    )
    assert len(queued) == 1
    row = queued[0]
    assert row["kind"] == "roofline"
    assert row["state"] == "queued"

    # phase_history evidence stamped for roofline path.
    last_row = coord.shared_state.phase_history[-1]
    assert last_row["to_phase"] == "KERNEL"
    evidence = last_row.get("evidence") or {}
    assert evidence.get("auto_roofline_enqueued") is True


@pytest.mark.asyncio
async def test_phase_transition_into_kernel_enqueues_profile_when_composite_off(
    tmp_path: Path,
):
    """When ``use_roofline_composite=False`` the legacy auto-profile
    fallback still lands a ``profile`` task on KERNEL entry."""
    from inference_optimizer.orchestrator.shared_state import SharedState

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
    coord.shared_state.phase = "EXPLORE"
    coord.shared_state.kernel_enabled = True
    coord.shared_state.use_roofline_composite = False
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.cumulative_gain = 5.0
    coord.shared_state.phase_history = [
        {"to_phase": "EXPLORE", "evidence": {}, "reason": "prelude_done"},
    ]
    coord.shared_state.record_phase_transition(
        to_phase="KERNEL",
        reason="plateau_explore",
        evidence={"trigger": "test_e2e_legacy"},
    )
    await coord._on_phase_entered(from_phase="EXPLORE", to_phase="KERNEL")

    queued = await coord.tasks.db.fetchall(
        "SELECT * FROM tasks WHERE idempotency_key=?",
        ("internal-profile-kernel_phase_entry",),
    )
    assert len(queued) == 1
    assert queued[0]["kind"] == "profile"


@pytest.mark.asyncio
async def test_phase_transition_skips_when_no_kernel_mode(tmp_path: Path):
    """``--no-kernel`` runs (kernel role absent from registry) must
    not enqueue an auto-profile even if something forces a KERNEL
    transition (defense in depth — compute_next_phase normally
    routes around KERNEL)."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "critic":        MockBackend(idle_plan),
        "robustness":    MockBackend(idle_plan),
    }
    # role_registry without 'kernel' — mirrors cli's --no-kernel branch.
    role_registry = {
        k: v for k, v in default_role_registry().items() if k != "kernel"
    }
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=role_registry,
        cortex_kb=None,
        knowledge_plane=None,
    )
    coord.shared_state.kernel_enabled = False
    coord.shared_state.phase_history = [
        {"to_phase": "KERNEL", "evidence": {}, "reason": "forced_test"},
    ]
    await coord._on_phase_entered(from_phase="EXPLORE", to_phase="KERNEL")

    # No profile task enqueued — registry stays empty for the key.
    rows = await coord.tasks.db.fetchall(
        "SELECT * FROM tasks WHERE idempotency_key=?",
        ("internal-profile-kernel_phase_entry",),
    )
    assert len(rows) == 0


# ===========================================================================
# 10. Roofline-failure fallback (operator-requested 2026-05-25)
# ===========================================================================
def _make_roofline_task(*, reason: str, task_id: str = "rl-task-1"):
    """Build a Task object resembling a Coordinator-internal roofline
    task (kind='roofline', params['reason']=<...>)."""
    from inference_optimizer.orchestrator.task_registry import Task

    return Task(
        task_id=task_id,
        kind="roofline",
        state="succeeded",   # promote runs after task reached terminal state
        params={"source": "coordinator_internal", "reason": reason},
        idempotency_key=f"internal-roofline-{reason}",
    )


@pytest.mark.asyncio
async def test_promote_explore_roofline_failure_skips_no_fallback(coord):
    """EXPLORE-entry roofline failure: NO fallback profile enqueued,
    pending gate cleared, audit row records 'discarded'."""
    coord.shared_state.auto_roofline_pending_task_id = "rl-explore-1"
    coord.shared_state.phase_history = [
        {"to_phase": "EXPLORE", "evidence": {}, "reason": "prelude_done"},
    ]
    task = _make_roofline_task(reason="explore_entry", task_id="rl-explore-1")
    result = {
        "status": "failed",
        "error_class": "trace_analyze_failed",
        "error": "RuntimeError: trace_split_no_steady_state",
        "phase": "trace_analyze",
    }
    await coord._promote_to_shared_state("roofline", result, task=task)

    # Gate cleared so subsequent specialists can dispatch.
    assert coord.shared_state.auto_roofline_pending_task_id == ""
    # NO fallback profile task created.
    assert "internal-profile-kernel_phase_entry_roofline_failed" not in coord.tasks._tasks
    assert "internal-profile-kernel_phase_entry" not in coord.tasks._tasks
    # Audit row recorded the discard.
    assert any(a.get("action") == "roofline" and a.get("decision") == "discarded"
               for a in coord.shared_state.action_attempts)


@pytest.mark.asyncio
async def test_promote_kernel_roofline_failure_enqueues_fallback_profile(coord):
    """KERNEL-entry roofline failure: fallback profile task enqueued
    with the canonical idempotency key + phase_history evidence stamp."""
    coord.shared_state.phase_history = [
        {"to_phase": "KERNEL", "evidence": {}, "reason": "plateau_explore"},
    ]
    task = _make_roofline_task(
        reason="kernel_phase_entry", task_id="rl-kernel-1",
    )
    result = {
        "status": "failed",
        "error_class": "trace_analyze_failed",
        "error": "RuntimeError: trace_split_no_steady_state",
        "phase": "trace_analyze",
    }
    await coord._promote_to_shared_state("roofline", result, task=task)

    # Fallback profile lands under the canonical idempotency key.
    fallback_key = "internal-profile-kernel_phase_entry_roofline_failed"
    assert fallback_key in coord.tasks._tasks
    fallback = coord.tasks._tasks[fallback_key]
    assert fallback.kind == "profile"
    assert fallback.params["reason"] == "kernel_phase_entry_roofline_failed"
    # config_path intentionally NOT propagated (lets ProfileExecutor pick
    # profile_sglang.yaml — see _enqueue_internal_profile_task docstring).
    assert "config_path" not in fallback.params

    # Phase-history evidence stamped.
    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert evidence.get("auto_profile_fallback_enqueued") is True
    assert evidence.get("auto_profile_fallback_task_id") == fallback.task_id
    assert evidence.get("auto_profile_fallback_reason") == "roofline_failed"


@pytest.mark.asyncio
async def test_promote_kernel_roofline_fallback_idempotent_on_double_failure(coord):
    """A second KERNEL-roofline failure with the same reason reuses
    the same fallback task (idempotency_key dedup)."""
    coord.shared_state.phase_history = [
        {"to_phase": "KERNEL", "evidence": {}, "reason": "plateau_explore"},
    ]
    task1 = _make_roofline_task(
        reason="kernel_phase_entry", task_id="rl-kernel-1",
    )
    result_fail = {
        "status": "failed", "error_class": "trace_analyze_failed",
        "error": "...", "phase": "trace_analyze",
    }
    await coord._promote_to_shared_state("roofline", result_fail, task=task1)
    fallback1 = coord.tasks._tasks["internal-profile-kernel_phase_entry_roofline_failed"]

    # Simulate a second KERNEL-entry roofline failure (e.g., resume edge).
    task2 = _make_roofline_task(
        reason="kernel_phase_entry", task_id="rl-kernel-2",
    )
    await coord._promote_to_shared_state("roofline", result_fail, task=task2)
    fallback2 = coord.tasks._tasks["internal-profile-kernel_phase_entry_roofline_failed"]

    # Same task — idempotency key dedup.
    assert fallback1 is fallback2
    # Only one entry in the registry under that key.
    matching = [k for k in coord.tasks._tasks
                if k == "internal-profile-kernel_phase_entry_roofline_failed"]
    assert len(matching) == 1


@pytest.mark.asyncio
async def test_promote_successful_roofline_does_not_fire_fallback(coord):
    """A succeeded roofline must never trigger the fallback profile —
    that path is only for the failure branch."""
    coord.shared_state.phase_history = [
        {"to_phase": "KERNEL", "evidence": {}, "reason": "plateau_explore"},
    ]
    task = _make_roofline_task(reason="kernel_phase_entry", task_id="rl-ok-1")
    result_ok = {
        "status": "succeeded",
        "snapshot_id": 1,
        "last_profile_trace": "/tmp/trace.json.gz",
        "analysis_md_path": "/tmp/analysis.md",
        "profile_workspace": "/tmp/ws",
    }
    await coord._promote_to_shared_state("roofline", result_ok, task=task)

    # No fallback.
    assert "internal-profile-kernel_phase_entry_roofline_failed" not in coord.tasks._tasks
    # Audit row records 'promoted'.
    assert any(a.get("action") == "roofline" and a.get("decision") == "promoted"
               for a in coord.shared_state.action_attempts)


@pytest.mark.asyncio
async def test_promote_explore_roofline_success_does_not_skip_log(coord):
    """A succeeded EXPLORE-entry roofline clears the gate WITHOUT
    logging the skip message (no fallback either)."""
    coord.shared_state.auto_roofline_pending_task_id = "rl-explore-ok-1"
    coord.shared_state.phase_history = [
        {"to_phase": "EXPLORE", "evidence": {}, "reason": "prelude_done"},
    ]
    task = _make_roofline_task(
        reason="explore_entry", task_id="rl-explore-ok-1",
    )
    result_ok = {
        "status": "succeeded",
        "snapshot_id": 1,
        "last_profile_trace": "/tmp/trace.json.gz",
        "analysis_md_path": "/tmp/analysis.md",
        "profile_workspace": "/tmp/ws",
    }
    await coord._promote_to_shared_state("roofline", result_ok, task=task)

    # Gate cleared, audit promoted, no fallback.
    assert coord.shared_state.auto_roofline_pending_task_id == ""
    assert "internal-profile-kernel_phase_entry_roofline_failed" not in coord.tasks._tasks
    assert any(a.get("action") == "roofline" and a.get("decision") == "promoted"
               for a in coord.shared_state.action_attempts)
