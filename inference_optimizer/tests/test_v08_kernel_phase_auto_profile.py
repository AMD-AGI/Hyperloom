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
    baseline_config_path: str = ""
    current_best: dict[str, Any] = field(default_factory=dict)
    last_baseline: dict[str, Any] = field(default_factory=dict)
    phase_history: list[dict[str, Any]] = field(default_factory=list)
    save_count: int = 0

    def save(self, _session_dir: Path | None) -> None:
        self.save_count += 1


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
    """Resume case: a prior session already landed a profile trace.
    The hook must skip re-profiling (5–10 min wasted bench)."""
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
    """Fresh KERNEL entry with no prior trace: hook enqueues an
    internal profile task + stamps phase_history evidence."""
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
    # Params inherit baseline workload contract.
    assert task.params["source"] == "coordinator_internal"
    assert task.params["reason"] == "kernel_phase_entry"
    assert task.params["config_path"] == "/tmp/baseline.yaml"
    assert task.params["base_extra_args"] == "--mla 1"
    assert task.params["benchmark_script"] == "sglang_mi300x.sh"
    # Evidence stamp visible on the phase_history row.
    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert evidence.get("auto_profile_enqueued") is True
    assert evidence.get("auto_profile_task_id") == task.task_id


@pytest.mark.asyncio
async def test_on_enter_kernel_omits_optional_params_when_state_empty(coord):
    """Bare state (no baseline_config_path / current_best / last_baseline):
    hook still enqueues the task. The profile executor falls back to
    its own defaults; params only carry source + reason."""
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
@pytest.mark.asyncio
async def test_phase_transition_into_kernel_enqueues_profile_e2e(
    tmp_path: Path,
):
    """End-to-end: drive the Coordinator's phase machine from PRELUDE
    → EXPLORE → KERNEL by faking the signals ``compute_next_phase``
    reads. After the KERNEL transition, the dispatcher must see a
    queued profile task with the v0.8 idempotency_key.
    """
    from inference_optimizer.orchestrator.shared_state import SharedState

    session_dir = tmp_path / "session"
    session_dir.mkdir()

    # Mock backends for all primary roles — none of them get invoked
    # in this test (we drive the phase machine directly).
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

    # Seed SharedState as if PRELUDE + EXPLORE already completed.
    # The phase state machine fires KERNEL when EXPLORE plateau hits
    # AND kernel_enabled — we cheat by writing the phase directly
    # then invoking the hook. (Driving compute_next_phase requires a
    # full plateau signal set; not needed for hook-coverage purposes.)
    coord.shared_state.phase = "EXPLORE"
    coord.shared_state.kernel_enabled = True
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.cumulative_gain = 5.0
    coord.shared_state.phase_history = [
        {"to_phase": "EXPLORE", "evidence": {}, "reason": "prelude_done"},
    ]

    # Simulate the EXPLORE → KERNEL transition that
    # `_advance_phase_if_needed` would commit:
    coord.shared_state.record_phase_transition(
        to_phase="KERNEL",
        reason="plateau_explore",
        evidence={"trigger": "test_e2e"},
    )
    await coord._on_phase_entered(from_phase="EXPLORE", to_phase="KERNEL")

    # Assertion 1: profile task on the registry under the v0.8 key.
    queued = await coord.tasks.db.fetchall(
        "SELECT * FROM tasks WHERE idempotency_key=?",
        ("internal-profile-kernel_phase_entry",),
    )
    assert len(queued) == 1
    row = queued[0]
    assert row["kind"] == "profile"
    assert row["state"] == "queued"

    # Assertion 2: phase_history evidence stamped.
    last_row = coord.shared_state.phase_history[-1]
    assert last_row["to_phase"] == "KERNEL"
    evidence = last_row.get("evidence") or {}
    assert evidence.get("auto_profile_enqueued") is True
    assert evidence.get("auto_profile_task_id"), \
        "evidence should record the enqueued task_id for audit"


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
