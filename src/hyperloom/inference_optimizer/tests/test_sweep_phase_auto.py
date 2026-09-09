# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SWEEP phase auto-dispatch tests.

SWEEP entry dispatches ``conc_sweep`` directly; the full-workload ``sweep``
helper is covered as a manual compatibility path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.roles.mock_backend import (
    MockBackend,
    MockTurn,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.shared_state import SharedState


# Fixtures
@dataclass
class _BareState:
    """SharedState stand-in covering every attribute the SWEEP hook + helper read."""

    warm_start_recipe: dict | None = None
    baseline_config_path: str = ""
    current_best: dict[str, Any] = field(default_factory=dict)
    last_baseline: dict[str, Any] = field(default_factory=dict)
    phase_history: list[dict[str, Any]] = field(default_factory=list)
    pending_stack_validation_result: dict[str, Any] = field(default_factory=dict)
    pending_stack_validation_apply_results: list[dict[str, Any]] = field(default_factory=list)
    kernel_integrate_attempts: dict[str, Any] = field(default_factory=dict)
    optimization_stack: list[dict[str, Any]] = field(default_factory=list)
    last_conc_sweep: dict[str, Any] = field(default_factory=dict)
    last_conc_sweep_watermark: dict[str, Any] = field(default_factory=dict)
    cumulative_gain_validated: float = 0.0
    conc_sweep_enabled: bool = True
    conc_sweep_concs: list[int] = field(default_factory=lambda: [1, 2, 4])
    conc_sweep_total_budget_sec: int = 60
    conc_sweep_variant_timeout_sec: int = 30
    save_count: int = 0
    stop_reason: str = ""
    usable_sec: float | None = None

    def session_budget_usable_sec(self, *, reserve_sec=None) -> float | None:
        return self.usable_sec

    def save(self, _session_dir: Path | None) -> None:
        self.save_count += 1

    def record_conc_sweep(self, result: dict[str, Any]) -> None:
        self.last_conc_sweep = {
            "status": str(result.get("status") or "succeeded"),
            "skip_reason": str(result.get("skip_reason") or ""),
            "was_skipped": bool(result.get("was_skipped", False)),
        }


class _StubTaskRegistry:
    """create_or_return_existing double, keyed by idempotency_key."""

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
        from hyperloom.orchestrator.state.task_registry import Task

        self.last_lease_ttl_sec = lease_ttl_sec
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
    """Lean Coordinator stub for hook unit tests."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _BareState()
    c.tasks = _StubTaskRegistry()
    c.knowledge_plane = None
    return c


@pytest.mark.asyncio
async def test_drain_pending_keep_integrates_records_result_once(
    tmp_path: Path,
    monkeypatch,
):
    """SWEEP entry drain must record integrate results so the same KEEP is not retried until cap."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState(
        baseline_tput=100.0,
        current_best={"action": "baseline", "tput": 100.0},
    )
    c.shared_state.kernel_opt_attempts = {
        "k004": {
            "last_decision": "KEEP",
            "last_micro_speedup": 4.21,
            "last_source_file": "/tmp/kernel.cu",
        },
    }
    calls: list[str] = []

    async def _fake_integrate_handler(payload, *, session_dir):
        calls.append(payload["kernel_id"])
        return {
            "status": "ok",
            "decision": "KEEP",
            "kernel_id": payload["kernel_id"],
            "patch_path": "/tmp/optimized.cu",
            "target_file": "/tmp/kernel.cu",
            "base_tput": 100.0,
            "new_tput": 102.0,
            "gain_pct": 2.0,
            "workspace": str(tmp_path / "integrate-k004"),
        }

    async def _noop_roofline(*, reason: str):
        return None

    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers.integrate_handler",
        _fake_integrate_handler,
    )
    c.phase_kernel._maybe_enqueue_watermark_roofline = _noop_roofline

    await c._drain_pending_keep_integrates()

    assert calls == ["k004"]
    assert c.shared_state.kernel_integrate_attempts
    assert c.shared_state.next_pending_keep_kernel_id() == ""
    assert c.shared_state.current_best["action"] == "integrate"
    assert c.shared_state.current_best["variant_name"] == "k004"


def test_pending_keep_kernel_ids_prioritize_trace_impact_over_micro():
    """E2E integrate order should prefer trace impact over isolated micro speedup."""
    state = SharedState()
    state.last_trace_analyze = {
        "hot_kernels_top15": [
            {"kernel_id": "k001", "gpu_pct": 60.0},
            {"kernel_id": "k004", "gpu_pct": 10.0},
        ],
    }
    state.kernel_opt_attempts = {
        "k004": {
            "last_decision": "KEEP",
            "last_micro_speedup": 4.21,
            "last_source_file": "/tmp/rmsnorm.cu",
        },
        "k001": {
            "last_decision": "KEEP",
            "last_micro_speedup": 1.51,
            "last_source_file": "/tmp/moe.cu",
        },
    }

    assert state.pending_keep_kernel_ids() == ["k001", "k004"]
    assert state.next_pending_keep_kernel_id() == "k001"


def test_pending_keep_kernel_ids_do_not_retry_needs_review():
    """A recorded NEEDS_REVIEW attempt should not auto-rerun the same patch."""
    state = SharedState()
    state.kernel_opt_attempts = {
        "k004": {
            "last_decision": "KEEP",
            "last_micro_speedup": 4.21,
            "last_source_file": "/tmp/rmsnorm.cu",
        },
        "k001": {
            "last_decision": "KEEP",
            "last_micro_speedup": 1.51,
            "last_source_file": "/tmp/moe.cu",
        },
    }
    state.record_kernel_integrate_result(
        {
            "status": "ok",
            "decision": "NEEDS_REVIEW",
            "kernel_id": "k004",
            "patch_path": "/tmp/k004_opt.cu",
            "target_file": "/tmp/rmsnorm.cu",
            "new_tput": 100.8,
            "gain_pct": 0.8,
            "workspace": "/tmp/integrate-k004",
        }
    )

    assert state.pending_keep_kernel_ids() == ["k001"]
    assert state.next_pending_keep_kernel_id() == "k001"


def _patch_stack_validation_internals(monkeypatch, *, new_tput: float, revert_status: str = "ok"):
    """Stub apply/revert/bench so the real stack-validation decision path runs."""
    import hyperloom.orchestrator.kernel.request_handlers as krh
    import hyperloom.orchestrator.actions.executors.baseline as baseline_mod
    import hyperloom.orchestrator.actions.executors.benchmark_result as br

    def _fake_apply(payload, *, session_dir, kernel_id):
        return {"status": "ok", "kernel_id": kernel_id, "manifest_path": None}

    def _fake_revert(applied):
        return {"status": revert_status}

    class _FakeBaselineExecutor:
        default_timeout_sec = baseline_mod.BASELINE_DEFAULT_TIMEOUT_SEC

        def __init__(self, *, session_dir):
            self.session_dir = session_dir

        async def __call__(self, ctx):
            return {
                "output_throughput": new_tput,
                "report_path": "/tmp/report",
                "workspace": "/tmp/workspace",
            }

    monkeypatch.setattr(krh, "_maybe_apply_kernel_patch", _fake_apply)
    monkeypatch.setattr(krh, "_maybe_revert_kernel_patch", _fake_revert)
    monkeypatch.setattr(baseline_mod, "BaselineExecutor", _FakeBaselineExecutor)
    monkeypatch.setattr(br, "is_valid_measurement", lambda result: True)


def _stack_validation_coordinator(tmp_path: Path) -> Coordinator:
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    # current_best already banks a +10% KEEP'd kernel
    c.shared_state = SharedState(
        baseline_tput=100.0,
        baseline_config_path=str(tmp_path / "base.yaml"),
        current_best={"action": "integrate", "tput": 110.0, "kernel_id": "k_prev"},
    )
    c.shared_state.optimization_stack = [
        {"action": "integrate", "kernel_id": "k_prev", "tput": 110.0},
    ]
    for kid, gain in (("k001", 0.6), ("k004", 0.8)):
        c.shared_state.record_kernel_integrate_result(
            {
                "status": "ok",
                "decision": "NEEDS_REVIEW",
                "kernel_id": kid,
                "patch_path": f"/tmp/{kid}_opt.cu",
                "target_file": f"/tmp/{kid}.cu",
                "new_tput": 100.0 + gain,
                "gain_pct": gain,
                "workspace": f"/tmp/integrate-{kid}",
            }
        )
    return c


@pytest.mark.asyncio
async def test_stack_validation_reverts_when_no_gain_over_current_best(
    tmp_path: Path,
    monkeypatch,
):
    """Stack worse than current_best (110) but above baseline (100) must REVERT.

    The KEEP decision is incremental over current_best, not total over baseline:
    new_tput=109 is +9% vs baseline yet -0.9% vs current_best, so the stack adds
    no value and must be reverted.
    """
    c = _stack_validation_coordinator(tmp_path)
    stack = c._stack_entries_for_validation(["k001", "k004"])
    _patch_stack_validation_internals(monkeypatch, new_tput=109.0)

    result = await c._run_kernel_stack_validation_e2e(stack)

    assert result["decision"] == "REVERT"
    assert result["gain_pct"] == pytest.approx(9.0)
    assert result["stack_incremental_gain_pct"] == pytest.approx(-0.9090909, rel=1e-3)
    assert result["revert_result"]["status"] == "ok"


@pytest.mark.asyncio
async def test_stack_validation_partial_revert_becomes_failed(
    tmp_path: Path,
    monkeypatch,
):
    """A partial inner revert means the patch may still be on a remote pod.

    Under the new patch lifecycle contract, partial non-KEEP reverts are not
    treated as successful: the top-level status becomes "failed" and
    patch_cleanup_status becomes "recovery_required" so the coordinator knows
    the tree may be in an unknown state.
    """
    c = _stack_validation_coordinator(tmp_path)
    stack = c._stack_entries_for_validation(["k001", "k004"])
    _patch_stack_validation_internals(monkeypatch, new_tput=109.0, revert_status="partial")

    result = await c._run_kernel_stack_validation_e2e(stack)

    assert result["decision"] == "REVERT"
    # partial -> failed at the aggregate level: patch still live on remote pod
    assert result["status"] == "failed"
    assert result["patch_cleanup_status"] == "recovery_required"
    assert result["patch_cleanup_action"] == "revert"
    assert all(r["status"] == "partial" for r in result["revert_result"]["stack_reverts"])


@pytest.mark.asyncio
async def test_stack_validation_keeps_on_positive_increment_over_current_best(
    tmp_path: Path,
    monkeypatch,
):
    """A real increment over current_best (110 -> 112, +1.8%) must KEEP."""
    c = _stack_validation_coordinator(tmp_path)
    stack = c._stack_entries_for_validation(["k001", "k004"])
    _patch_stack_validation_internals(monkeypatch, new_tput=112.0)

    result = await c._run_kernel_stack_validation_e2e(stack)

    assert result["decision"] == "KEEP"
    assert result["gain_pct"] == pytest.approx(12.0)
    assert result["stack_incremental_gain_pct"] == pytest.approx(1.8181818, rel=1e-3)
    assert result["revert_result"]["status"] == "skipped"


@pytest.mark.asyncio
async def test_positive_needs_review_stack_validation_promotes_combo(tmp_path: Path):
    """Two positive sub-threshold kernel patches should get one combined E2E validation."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState(
        baseline_tput=100.0,
        current_best={"action": "baseline", "tput": 100.0},
    )
    for kid, gain in (("k001", 0.6), ("k004", 0.8)):
        c.shared_state.record_kernel_integrate_result(
            {
                "status": "ok",
                "decision": "NEEDS_REVIEW",
                "kernel_id": kid,
                "patch_path": f"/tmp/{kid}_opt.cu",
                "target_file": f"/tmp/{kid}.cu",
                "new_tput": 100.0 + gain,
                "gain_pct": gain,
                "workspace": f"/tmp/integrate-{kid}",
            }
        )

    validation_calls = 0

    async def _fake_stack_validation(entries):
        nonlocal validation_calls
        validation_calls += 1
        assert {e["kernel_id"] for e in entries} == {"k001", "k004"}
        return {
            "status": "ok",
            "decision": "KEEP",
            "kernel_id": "k001+k004",
            "patch_path": "/tmp/k001_opt.cu+/tmp/k004_opt.cu",
            "target_file": "/tmp/k001.cu+/tmp/k004.cu",
            "base_tput": 100.0,
            "new_tput": 102.0,
            "gain_pct": 2.0,
            "workspace": str(tmp_path / "integrate-stack"),
            "apply_result": {"status": "ok"},
            "stack_kernel_ids": ["k001", "k004"],
            "stack_validation": True,
        }

    async def _noop_roofline(*, reason: str):
        return None

    c.phase_kernel_stack._run_kernel_stack_validation_e2e = _fake_stack_validation
    c.phase_kernel._maybe_enqueue_watermark_roofline = _noop_roofline

    await c._maybe_validate_positive_needs_review_stack()

    assert c.shared_state.current_best["action"] == "integrate"
    assert c.shared_state.current_best["variant_name"] == "k001+k004"
    assert c.shared_state.cumulative_gain_validated == pytest.approx(2.0)
    assert validation_calls == 1
    resolved_entries = [
        entry
        for entry in c.shared_state.kernel_integrate_attempts.values()
        if entry.get("kernel_id") in {"k001", "k004"}
    ]
    assert all(entry["stack_resolved"] is True for entry in resolved_entries)
    assert {entry["stack_validation_kernel_id"] for entry in resolved_entries} == {"k001+k004"}

    # Re-invoking must be a no-op (idempotent): the call count must not advance.
    calls_before_recall = validation_calls
    await c._maybe_validate_positive_needs_review_stack()

    assert validation_calls == calls_before_recall
    stack_entries = [
        item
        for item in c.shared_state.optimization_stack
        if isinstance(item, dict) and item.get("kernel_id") == "k001+k004"
    ]
    assert stack_entries
    assert stack_entries[0].get("stack_kernel_ids") == ["k001", "k004"]


@pytest.mark.asyncio
async def test_recovers_pending_stack_validation_after_crash(tmp_path: Path):
    """A saved pending stack result should finish promotion without re-applying."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState(
        baseline_tput=100.0,
        current_best={"action": "baseline", "tput": 100.0},
    )
    for kid, gain in (("k001", 0.6), ("k004", 0.8)):
        c.shared_state.record_kernel_integrate_result(
            {
                "status": "ok",
                "decision": "NEEDS_REVIEW",
                "kernel_id": kid,
                "patch_path": f"/tmp/{kid}_opt.cu",
                "target_file": f"/tmp/{kid}.cu",
                "new_tput": 100.0 + gain,
                "gain_pct": gain,
                "workspace": f"/tmp/integrate-{kid}",
            }
        )
    stack = c._stack_entries_for_validation(["k001", "k004"])
    c._mark_stack_validation_in_progress(stack, "k001+k004")
    c.shared_state.pending_stack_validation_result = {
        "status": "ok",
        "decision": "KEEP",
        "kernel_id": "k001+k004",
        "patch_path": "/tmp/k001_opt.cu+/tmp/k004_opt.cu",
        "target_file": "/tmp/k001.cu+/tmp/k004.cu",
        "base_tput": 100.0,
        "new_tput": 102.0,
        "gain_pct": 2.0,
        "workspace": str(tmp_path / "integrate-stack"),
        "apply_result": {"status": "ok"},
        "stack_kernel_ids": ["k001", "k004"],
        "stack_validation": True,
    }
    c.shared_state.save(tmp_path)

    validation_calls = 0

    async def _should_not_run(entries):
        nonlocal validation_calls
        validation_calls += 1
        raise AssertionError("stack validation should not re-run during recovery")

    async def _noop_roofline(*, reason: str):
        return None

    c.phase_kernel_stack._run_kernel_stack_validation_e2e = _should_not_run
    c.phase_kernel._maybe_enqueue_watermark_roofline = _noop_roofline

    await c._recover_interrupted_stack_validation()

    assert validation_calls == 0
    assert c.shared_state.current_best["variant_name"] == "k001+k004"
    assert not c.shared_state.pending_stack_validation_result
    resolved = [
        entry
        for entry in c.shared_state.kernel_integrate_attempts.values()
        if entry.get("kernel_id") in {"k001", "k004"}
    ]
    assert all(entry.get("stack_resolved") for entry in resolved)


def test_positive_needs_review_integrates_skip_in_progress_entries():
    """In-flight stack members must not be re-selected for another validation."""
    c = Coordinator.__new__(Coordinator)
    c.shared_state = SharedState()
    c.shared_state.kernel_integrate_attempts = {
        "k001": {
            "kernel_id": "k001",
            "patch_path": "/tmp/k001_opt.cu",
            "target_file": "/tmp/k001.cu",
            "last_decision": "NEEDS_REVIEW",
            "best_gain_pct": 0.6,
            "stack_validation_in_progress": True,
        },
        "k004": {
            "kernel_id": "k004",
            "patch_path": "/tmp/k004_opt.cu",
            "target_file": "/tmp/k004.cu",
            "last_decision": "NEEDS_REVIEW",
            "best_gain_pct": 0.8,
        },
    }

    eligible = c._positive_needs_review_integrates()
    assert len(eligible) == 1
    assert eligible[0]["kernel_id"] == "k004"


@pytest.mark.asyncio
async def test_on_enter_sweep_triggers_stack_validation_without_pending_keeps(
    tmp_path: Path,
    monkeypatch,
):
    """Stack validation must run even when has_keep_pending_integrate is False."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState(
        baseline_tput=100.0,
        current_best={"action": "baseline", "tput": 100.0},
    )
    c.tasks = _StubTaskRegistry()
    c.knowledge_plane = None
    # All KEEPs already integrated as NEEDS_REVIEW — no pending KEEP.
    for kid, gain in (("k001", 0.6), ("k004", 0.8)):
        c.shared_state.record_kernel_integrate_result(
            {
                "status": "ok",
                "decision": "NEEDS_REVIEW",
                "kernel_id": kid,
                "patch_path": f"/tmp/{kid}_opt.cu",
                "target_file": f"/tmp/{kid}.cu",
                "new_tput": 100.0 + gain,
                "gain_pct": gain,
                "workspace": f"/tmp/integrate-{kid}",
            }
        )
    assert not c.shared_state.has_keep_pending_integrate

    validation_calls = []

    async def _fake_stack_validation(entries):
        validation_calls.append([e["kernel_id"] for e in entries])
        return {
            "status": "ok",
            "decision": "KEEP",
            "kernel_id": "k001+k004",
            "patch_path": "/tmp/k001_opt.cu+/tmp/k004_opt.cu",
            "target_file": "/tmp/k001.cu+/tmp/k004.cu",
            "base_tput": 100.0,
            "new_tput": 102.0,
            "gain_pct": 2.0,
            "workspace": str(tmp_path / "integrate-stack"),
            "apply_result": {"status": "ok"},
            "stack_kernel_ids": ["k001", "k004"],
            "stack_validation": True,
        }

    async def _noop_roofline(*, reason: str):
        return None

    c.phase_kernel_stack._run_kernel_stack_validation_e2e = _fake_stack_validation
    c.phase_kernel._maybe_enqueue_watermark_roofline = _noop_roofline

    await c._on_enter_sweep(from_phase="KERNEL")

    assert len(validation_calls) == 1
    assert c.shared_state.current_best["variant_name"] == "k001+k004"


@pytest.mark.asyncio
async def test_drain_uses_current_best_tput_not_baseline(
    tmp_path: Path,
    monkeypatch,
):
    """Drain should pass current_best.tput (not baseline) so multi-KEEP gain is incremental."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState(
        baseline_tput=100.0,
        current_best={"action": "integrate", "tput": 110.0, "kernel_id": "k_prev"},
    )
    c.shared_state.optimization_stack = [
        {"action": "integrate", "kernel_id": "k_prev", "tput": 110.0},
    ]
    c.shared_state.kernel_opt_attempts = {
        "k_new": {
            "last_decision": "KEEP",
            "last_micro_speedup": 2.0,
            "last_source_file": "/tmp/new.cu",
        },
    }
    captured_payloads = []

    async def _fake_integrate_handler(payload, *, session_dir):
        captured_payloads.append(payload)
        return {
            "status": "ok",
            "decision": "KEEP",
            "kernel_id": payload["kernel_id"],
            "patch_path": "/tmp/new_opt.cu",
            "target_file": "/tmp/new.cu",
            "base_tput": payload.get("base_tput", 0.0),
            "new_tput": 112.0,
            "gain_pct": (112.0 / payload.get("base_tput", 100.0) - 1) * 100,
            "workspace": str(tmp_path / "integrate-k_new"),
        }

    async def _noop_roofline(*, reason: str):
        return None

    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers.integrate_handler",
        _fake_integrate_handler,
    )
    c.phase_kernel._maybe_enqueue_watermark_roofline = _noop_roofline

    await c._drain_pending_keep_integrates()

    assert len(captured_payloads) == 1
    # use current_best.tput (110.0), not baseline (100.0)
    assert captured_payloads[0]["base_tput"] == 110.0


# 3. _on_enter_sweep hook
@pytest.mark.asyncio
async def test_on_enter_sweep_enqueues_and_stamps_evidence(coord):
    """Happy path: the hook enqueues conc_sweep and stamps phase evidence."""
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "plateau_kernel", "evidence": {}},
    ]
    await coord._on_enter_sweep(from_phase="KERNEL")
    assert "internal-conc_sweep-phase_entry" in coord.tasks._tasks
    task = coord.tasks._tasks["internal-conc_sweep-phase_entry"]
    assert task.kind == "conc_sweep"

    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert evidence["auto_conc_sweep_enqueued"] is True
    assert evidence["auto_conc_sweep_task_id"] == task.task_id
    assert evidence["auto_conc_sweep_concs"] == [1, 2, 4]


@pytest.mark.asyncio
async def test_on_enter_sweep_ignores_full_sweep_recipe_for_auto_path(coord):
    """The automatic path goes straight to conc_sweep; recipe sweep_grid is manual-only."""
    coord.shared_state.warm_start_recipe = {
        "sweep_grid": {
            "conc_values": [8, 32],
            "isl_osl_configs": ["1024:1024", "4096:4096", "8192:1024"],
        },
    }
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "plateau_kernel", "evidence": {}},
    ]
    await coord._on_enter_sweep(from_phase="KERNEL")
    assert "internal-conc_sweep-phase_entry" in coord.tasks._tasks
    assert "internal-sweep-phase_entry" not in coord.tasks._tasks
    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert evidence["auto_conc_sweep_concs"] == [1, 2, 4]
    assert "auto_sweep_grid_source" not in evidence


@pytest.mark.asyncio
async def test_a_state_with_no_ladder_lets_the_workload_pick(coord):
    """An unseeded ladder must reach the engine as "unset", not as "none wanted".

    The executor reads an empty list as a deliberate choice and skips the whole
    sweep, so collapsing None into [] here would silently drop it for any state
    the CLI did not seed.
    """
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "plateau_kernel", "evidence": {}},
    ]
    coord.shared_state.conc_sweep_concs = []
    await coord._on_enter_sweep(from_phase="KERNEL")

    task = coord.tasks._tasks["internal-conc_sweep-phase_entry"]
    assert task.params["concs"] is None
    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert evidence["auto_conc_sweep_concs"] is None


@pytest.mark.asyncio
async def test_on_enter_sweep_idempotent_on_reentry(coord):
    """Re-entering SWEEP twice hits the same conc_sweep idempotency_key."""
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "plateau_kernel", "evidence": {}},
    ]
    await coord._on_enter_sweep(from_phase="KERNEL")
    task1 = coord.tasks._tasks["internal-conc_sweep-phase_entry"]
    coord.shared_state.phase_history.append(
        {"to_phase": "SWEEP", "reason": "re_entry_test", "evidence": {}},
    )
    await coord._on_enter_sweep(from_phase="SWEEP")
    task2 = coord.tasks._tasks["internal-conc_sweep-phase_entry"]
    assert task1 is task2
    assert len(coord.tasks._tasks) == 1


@pytest.mark.asyncio
async def test_on_enter_sweep_failure_records_evidence(coord, monkeypatch):
    """If conc_sweep enqueue raises, the hook records a terminal skip."""

    async def _boom(*args, **kwargs):
        raise RuntimeError("simulated DB outage")

    monkeypatch.setattr(coord.phase_sweep, "_enqueue_internal_conc_sweep_task", _boom)
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "plateau_kernel", "evidence": {}},
    ]
    # Should not raise
    await coord._on_enter_sweep(from_phase="KERNEL")
    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert "auto_conc_sweep_error" in evidence
    assert "simulated DB outage" in evidence["auto_conc_sweep_error"]
    # No task was enqueued
    assert coord.tasks._tasks == {}
    assert coord.shared_state.last_conc_sweep["status"] == "skipped"
    assert coord.shared_state.last_conc_sweep["skip_reason"] == "enqueue_failed"
    assert coord.shared_state.save_count >= 1


@pytest.mark.asyncio
async def test_on_enter_sweep_keeps_the_declines_own_skip_reason(coord):
    """The helper's budget decline is terminal; the hook must not restate it."""
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "plateau_kernel", "evidence": {}},
    ]
    coord.shared_state.remaining_minutes = lambda: 1.0

    await coord._on_enter_sweep(from_phase="KERNEL")

    assert coord.tasks._tasks == {}
    assert coord.shared_state.last_conc_sweep["skip_reason"] == "session_time_budget"


@pytest.mark.asyncio
async def test_enqueue_conc_sweep_declines_when_clamp_leaves_no_time(coord):
    """A clamp that leaves nothing declines: a 0 budget would read as unbounded."""
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "plateau_kernel", "evidence": {}},
    ]
    # 1 minute left, minus the 120 s CLOSE reserve, is a negative budget.
    coord.shared_state.remaining_minutes = lambda: 1.0

    task = await coord.phase_sweep._enqueue_internal_conc_sweep_task(reason="phase_entry")

    assert task is None
    assert coord.tasks._tasks == {}
    assert coord.shared_state.last_conc_sweep["skip_reason"] == "session_time_budget"


@pytest.mark.asyncio
async def test_conc_sweep_lease_follows_the_clamped_budget(coord):
    """The lease must bound the task that runs, not the configured value.

    With no configured budget and a long session the clamp produces a budget
    larger than the old 9000 s default, and the watchdog would have failed a
    sweep that was still making progress.
    """
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "plateau_kernel", "evidence": {}},
    ]
    coord.shared_state.conc_sweep_total_budget_sec = 0
    coord.shared_state.remaining_minutes = lambda: 300.0  # 5 h

    task = await coord.phase_sweep._enqueue_internal_conc_sweep_task(reason="phase_entry")

    expected_budget = 300 * 60 - 120
    assert task.params["total_budget_sec"] == expected_budget
    assert coord.tasks.last_lease_ttl_sec == expected_budget + 600


@pytest.mark.asyncio
async def test_conc_sweep_unbounded_budget_opts_out_of_the_lease(coord):
    """An unbounded sweep has no deadline, so it must not carry a finite lease."""
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "plateau_kernel", "evidence": {}},
    ]
    coord.shared_state.conc_sweep_total_budget_sec = 0

    task = await coord.phase_sweep._enqueue_internal_conc_sweep_task(reason="phase_entry")

    assert task.params["total_budget_sec"] is None
    assert coord.tasks.last_lease_ttl_sec == 0


@pytest.mark.asyncio
async def test_enqueue_conc_sweep_unbounded_budget_is_none(coord):
    """A non-positive configured budget means "no gate" and travels as None."""
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "plateau_kernel", "evidence": {}},
    ]
    coord.shared_state.conc_sweep_total_budget_sec = 0

    task = await coord.phase_sweep._enqueue_internal_conc_sweep_task(reason="phase_entry")

    assert task is not None
    assert task.params["total_budget_sec"] is None


@pytest.mark.asyncio
async def test_enqueue_conc_sweep_clamps_to_remaining_session_time(coord):
    """With a session cap, the budget is the remaining time minus the CLOSE reserve."""
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "plateau_kernel", "evidence": {}},
    ]
    coord.shared_state.conc_sweep_total_budget_sec = 9000
    coord.shared_state.remaining_minutes = lambda: 5.0

    task = await coord.phase_sweep._enqueue_internal_conc_sweep_task(reason="phase_entry")

    assert task is not None
    assert task.params["total_budget_sec"] == 180  # 5 min - 120 s reserve


@pytest.mark.asyncio
async def test_on_enter_sweep_skips_when_conc_sweep_disabled(coord):
    """If conc_sweep is disabled, SWEEP records a terminal skip instead of idling."""
    coord.shared_state.conc_sweep_enabled = False
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "cycle_reloop", "evidence": {}},
    ]
    await coord._on_enter_sweep(from_phase="KERNEL")
    assert coord.tasks._tasks == {}
    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert evidence["auto_conc_sweep_skipped"] == "disabled"
    assert "auto_sweep_enqueued" not in evidence
    assert coord.shared_state.last_conc_sweep["status"] == "skipped"
    assert coord.shared_state.last_conc_sweep["skip_reason"] == "disabled"
    assert coord.shared_state.last_conc_sweep["was_skipped"] is True
    assert coord.shared_state.save_count >= 1


@pytest.mark.asyncio
async def test_on_enter_sweep_skips_when_no_validated_gain_since_last_conc_sweep(coord):
    """Cyclic reloop does not rerun conc_sweep without a new validated gain."""
    coord.shared_state.cumulative_gain_validated = 12.5
    coord.shared_state.last_conc_sweep_watermark = {
        "ts": "2026-01-01T00:00:00Z",
        "cumulative_gain_validated_at_record": 12.5,
    }
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "cycle_reloop", "evidence": {}},
    ]
    await coord._on_enter_sweep(from_phase="KERNEL")
    assert coord.tasks._tasks == {}
    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert evidence["auto_conc_sweep_skipped"] == "no_validated_gain_since_last_conc_sweep"
    assert evidence["auto_conc_sweep_skipped_validated_gain"] == 12.5
    assert coord.shared_state.last_conc_sweep["status"] == "skipped"
    assert coord.shared_state.last_conc_sweep["skip_reason"] == "no_validated_gain_since_last_conc_sweep"
    assert coord.shared_state.save_count >= 1


@pytest.mark.asyncio
async def test_on_enter_sweep_skips_when_the_session_budget_cannot_fit_conc_sweep(coord):
    """A conc_sweep the clock cannot pay for must not be enqueued, or SWEEP idles."""
    coord.shared_state.usable_sec = 14 * 60.0
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "plateau_kernel", "evidence": {}},
    ]
    await coord._on_enter_sweep(from_phase="KERNEL")
    assert coord.tasks._tasks == {}
    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert evidence["auto_conc_sweep_skipped"] == "session_time_budget"
    assert coord.shared_state.last_conc_sweep["status"] == "skipped"
    assert coord.shared_state.last_conc_sweep["skip_reason"] == "session_time_budget"
    assert coord.shared_state.last_conc_sweep["was_skipped"] is True


@pytest.mark.asyncio
async def test_on_enter_sweep_still_enqueues_when_the_session_budget_fits(coord):
    """The session-budget skip must not fire when the catalogue cost still fits."""
    coord.shared_state.usable_sec = 60 * 60.0
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "plateau_kernel", "evidence": {}},
    ]
    await coord._on_enter_sweep(from_phase="KERNEL")
    assert "internal-conc_sweep-phase_entry" in coord.tasks._tasks
    assert coord.shared_state.last_conc_sweep == {}


@pytest.mark.asyncio
async def test_on_enter_sweep_runs_when_validated_gain_improved(coord):
    """A new validated gain after the last conc_sweep watermark dispatches conc_sweep."""
    coord.shared_state.cumulative_gain_validated = 15.0
    coord.shared_state.last_conc_sweep_watermark = {
        "ts": "2026-01-01T00:00:00Z",
        "cumulative_gain_validated_at_record": 12.5,
    }
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "cycle_reloop", "evidence": {}},
    ]
    await coord._on_enter_sweep(from_phase="KERNEL")
    assert "internal-conc_sweep-phase_entry" in coord.tasks._tasks
    evidence = coord.shared_state.phase_history[-1]["evidence"]
    assert evidence["auto_conc_sweep_enqueued"] is True


@pytest.mark.asyncio
async def test_on_enter_sweep_first_sweep_runs_without_prior_watermark(coord):
    """The first SWEEP entry dispatches conc_sweep directly."""
    coord.shared_state.cumulative_gain_validated = 0.0
    coord.shared_state.last_conc_sweep = {}
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "reason": "plateau_kernel", "evidence": {}},
    ]
    await coord._on_enter_sweep(from_phase="KERNEL")
    assert "internal-conc_sweep-phase_entry" in coord.tasks._tasks


# 4. End-to-end via real Coordinator
@pytest.mark.asyncio
async def test_phase_transition_into_sweep_enqueues_conc_sweep_e2e(tmp_path: Path):
    """End-to-end: a SWEEP transition persists the conc_sweep task."""
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
    # Seed state at KERNEL boundary as if a plateau_kernel just fired
    coord.shared_state.phase = "KERNEL"
    coord.shared_state.kernel_enabled = True
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.cumulative_gain_validated = 12.0
    coord.shared_state.last_profile_trace = "/tmp/dummy.trace.json.gz"
    coord.shared_state.phase_history = [
        {"to_phase": "EXPLORE", "evidence": {}, "reason": "prelude_done"},
        {"to_phase": "KERNEL", "evidence": {}, "reason": "plateau_explore"},
    ]

    coord.shared_state.record_phase_transition(
        to_phase="SWEEP",
        reason="plateau_kernel",
        evidence={"trigger": "test_e2e"},
    )
    await coord._on_phase_entered(from_phase="KERNEL", to_phase="SWEEP")

    rows = await coord.tasks.db.fetchall(
        "SELECT * FROM tasks WHERE idempotency_key=?",
        ("internal-conc_sweep-phase_entry",),
    )
    assert len(rows) == 1
    assert rows[0]["kind"] == "conc_sweep"
    assert rows[0]["state"] == "queued"

    last_history = coord.shared_state.phase_history[-1]
    assert last_history["to_phase"] == "SWEEP"
    evidence = last_history.get("evidence") or {}
    assert evidence.get("auto_conc_sweep_enqueued") is True
    assert evidence.get("auto_conc_sweep_task_id")


@pytest.mark.asyncio
async def test_phase_transition_explore_to_sweep_no_kernel_mode(tmp_path: Path):
    """``--no-kernel`` runs go EXPLORE → SWEEP directly; conc_sweep still enqueues."""
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
    coord.shared_state.kernel_enabled = False
    coord.shared_state.phase_history = [
        {"to_phase": "SWEEP", "evidence": {}, "reason": "test_forced"},
    ]
    await coord._on_phase_entered(from_phase="FRAMEWORK_AGENT", to_phase="SWEEP")
    rows = await coord.tasks.db.fetchall(
        "SELECT * FROM tasks WHERE idempotency_key=?",
        ("internal-conc_sweep-phase_entry",),
    )
    assert len(rows) == 1, "conc_sweep auto-enqueue must run in --no-kernel mode too"


# 5. Idempotency key structural cross-check
def test_internal_sweep_idempotency_key_does_not_collide_with_llm_path():
    """The manual sweep helper key must never collide with the LLM approved key."""
    internal_key = "internal-sweep-phase_entry"
    # Mirror the format _materialize_approved_proposal builds
    llm_key = "approved-msg_abc123"
    assert internal_key != llm_key
    assert not llm_key.startswith("internal-")
    assert not internal_key.startswith("approved-")


class _SweepPhaseState:
    """SharedState stand-in carrying just the phase rows the gate reads."""

    def __init__(self, phase_history=None):
        self.phase_history = list(phase_history or [])


def _sweep_phase_row(*, auto_sweep_task_id: str = "") -> dict:
    """Build a SWEEP phase row carrying the auto conc_sweep evidence."""
    evidence: dict = {}
    if auto_sweep_task_id:
        evidence["auto_conc_sweep_task_id"] = auto_sweep_task_id
        evidence["auto_conc_sweep_enqueued"] = True
    return {
        "to_phase": "SWEEP",
        "from_phase": "EXPLORE",
        "reason": "explore_done",
        "evidence": evidence,
    }


def _make_policy_gate(*, shared_state):
    """Plain PolicyGate wired to the role registry + the test's SharedState double."""
    from hyperloom.orchestrator.roles.agent_role import (
        default_role_registry,
    )
    from hyperloom.orchestrator.policy.gate import PolicyGate

    return PolicyGate(
        role_registry=default_role_registry(),
        shared_state=shared_state,
    )


# 6. The workload grid action is gone; SWEEP admits the ladder and nothing else
def test_the_retired_action_is_off_every_surface_it_was_on():
    from hyperloom.inference_optimizer.cli.executors import _REAL_EXECUTORS_FULL
    from hyperloom.inference_optimizer.protocol.action_surfaces import (
        ACTION_CATALOGUE,
        FULL_ENABLED_ACTIONS,
        NO_KERNEL_AGENT_ENABLED_ACTIONS,
    )
    from hyperloom.orchestrator.phases.machine_state import PHASE_ALLOWED_ACTIONS

    assert "sweep" not in ACTION_CATALOGUE
    assert "sweep" not in FULL_ENABLED_ACTIONS
    assert "sweep" not in NO_KERNEL_AGENT_ENABLED_ACTIONS
    assert "sweep" not in _REAL_EXECUTORS_FULL
    assert "sweep" not in PHASE_ALLOWED_ACTIONS["SWEEP"]
    assert "conc_sweep" in PHASE_ALLOWED_ACTIONS["SWEEP"]


# 7. conc_sweep is Coordinator-internal — dispatch re-validation must not
# collide the sole auto-enqueued conc_sweep with its own singleton evidence.


def test_validate_dispatched_task_allows_auto_conc_sweep_against_own_evidence():
    """Regression: the SWEEP-entry auto-enqueued conc_sweep must pass dispatch re-validation.

    Before the fix, ``validate_dispatched_task`` fell through to the
    delegate-body sweep-family singleton guard, which keys on
    ``auto_conc_sweep_task_id`` — the auto-enqueued task's OWN id — and denied
    the sole conc_sweep against itself, surfacing as a spurious
    ``sweep_failed`` that closed the session at 0% gain. Now conc_sweep is
    a Coordinator-internal action, so it receives path checks only and is not
    re-validated against the singleton guard.
    """
    state = _SweepPhaseState(
        phase_history=[_sweep_phase_row(auto_sweep_task_id="conc-sweep-self-id")],
    )
    gate = _make_policy_gate(shared_state=state)
    # Must NOT raise, even though SWEEP evidence already carries the auto id.
    gate.validate_dispatched_task(
        "conc_sweep",
        {"source": "coordinator_internal", "concs": [64, 32], "total_budget_sec": 9000},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Patch lifecycle convergence — new tests (P1-19 fix verification)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stack_validation_failed_revert_sets_status_failed(
    tmp_path: Path,
    monkeypatch,
):
    """A completely failed stack revert must set top-level status='failed'.

    Previously _stack_revert_status mapped {"status": "failed"} -> "ok",
    causing a silent false success. The new contract surfaces it as top-level
    failure so the coordinator does not promote a patch that may still be live.
    """
    c = _stack_validation_coordinator(tmp_path)
    stack = c._stack_entries_for_validation(["k001", "k004"])
    _patch_stack_validation_internals(monkeypatch, new_tput=109.0, revert_status="failed")

    result = await c._run_kernel_stack_validation_e2e(stack)

    assert result["decision"] == "REVERT"
    assert result["status"] == "failed"
    assert result["patch_cleanup_status"] == "recovery_required"
    assert result["patch_cleanup_action"] == "revert"
    assert result.get("error_class") == "patch_revert_incomplete"
    assert all(r["status"] == "failed" for r in result["revert_result"]["stack_reverts"])


@pytest.mark.asyncio
async def test_stack_validation_keep_calls_finalize(
    tmp_path: Path,
    monkeypatch,
):
    """A KEEP result must call _maybe_finalize_kernel_patch for each applied patch.

    Previously the KEEP path never called finalize because the synthetic
    apply_result had no manifest_path. The new contract calls it explicitly and
    records patch_cleanup_status.
    """
    import hyperloom.orchestrator.kernel.request_handlers as krh

    finalize_calls: list[dict] = []

    def _spy_finalize(apply_result):
        finalize_calls.append(apply_result)
        return {"status": "ok", "manifest_path": str(apply_result.get("manifest_path") or "")}

    monkeypatch.setattr(krh, "_maybe_finalize_kernel_patch", _spy_finalize)

    def _fake_apply_with_manifest(payload, *, session_dir, kernel_id):
        return {
            "status": "ok",
            "kernel_id": kernel_id,
            "manifest_path": f"/tmp/{kernel_id}.manifest",
        }

    monkeypatch.setattr(krh, "_maybe_apply_kernel_patch", _fake_apply_with_manifest)

    c = _stack_validation_coordinator(tmp_path)
    stack = c._stack_entries_for_validation(["k001", "k004"])
    _patch_stack_validation_internals(monkeypatch, new_tput=115.0)

    result = await c._run_kernel_stack_validation_e2e(stack)

    assert result["decision"] == "KEEP"
    assert result["status"] == "ok"
    assert result["patch_cleanup_status"] == "complete"
    # One finalize call per patch in the two-entry stack.
    assert len(finalize_calls) == 2


@pytest.mark.asyncio
async def test_stack_validation_keep_partial_finalize_requires_recovery(
    tmp_path: Path,
    monkeypatch,
):
    """KEEP + partial finalize must ask for recovery, not report cleanup complete.

    finalize returns "partial" when a backup could not be deleted, a backup path
    failed containment, or a remote pod's finalize failed. The patch itself is
    correctly on tree, so the top status stays "ok".
    """
    import hyperloom.orchestrator.kernel.request_handlers as krh

    monkeypatch.setattr(
        krh,
        "_maybe_finalize_kernel_patch",
        lambda apply_result: {"status": "partial", "issues": [{"kind": "multinode_finalize"}]},
    )
    monkeypatch.setattr(
        krh,
        "_maybe_apply_kernel_patch",
        lambda payload, *, session_dir, kernel_id: {
            "status": "ok",
            "kernel_id": kernel_id,
            "manifest_path": f"/tmp/{kernel_id}.manifest",
        },
    )

    c = _stack_validation_coordinator(tmp_path)
    stack = c._stack_entries_for_validation(["k001", "k004"])
    _patch_stack_validation_internals(monkeypatch, new_tput=115.0)

    result = await c._run_kernel_stack_validation_e2e(stack)

    assert result["decision"] == "KEEP"
    assert result["status"] == "ok"
    assert result["patch_cleanup_status"] == "recovery_required"
    assert result["patch_cleanup_action"] == "finalize"


@pytest.mark.asyncio
async def test_stack_validation_accuracy_regression_downgrades_to_needs_review(
    tmp_path: Path,
    monkeypatch,
):
    """An accuracy regression on a stack KEEP must drop decision to NEEDS_REVIEW.

    The eval already runs (RUN_EVAL=true by default, no defer_accuracy set for
    this lane), so calling the gate is zero extra GPU cost.
    """
    import hyperloom.orchestrator.kernel.request_handlers as krh

    seen: dict[str, object] = {}

    def _fake_accuracy_gate(bench_result, *, session_dir, workspace, server_args=""):
        # The lane must hand the gate the args the bench server ran under, or a
        # context too small to host an eval reads as a broken eval.
        seen["server_args"] = server_args
        return {
            "blocked": True,
            "accuracy_pass": False,
            "reason": "accuracy regression detected",
            "degraded": False,
            "accuracy": 0.70,
            "baseline_accuracy": 0.85,
            "task": "gsm8k",
            "metric": "exact_match",
            "source_file": "/tmp/result.json",
        }

    monkeypatch.setattr(krh, "_grade_integrate_accuracy", _fake_accuracy_gate)

    c = _stack_validation_coordinator(tmp_path)
    stack = c._stack_entries_for_validation(["k001", "k004"])
    _patch_stack_validation_internals(monkeypatch, new_tput=115.0)

    result = await c._run_kernel_stack_validation_e2e(stack)

    assert result["decision"] == "NEEDS_REVIEW"
    assert "server_args" in seen
    # NEEDS_REVIEW is non-KEEP so reverts must have been called and succeeded.
    assert result["revert_result"]["status"] == "ok"


@pytest.mark.asyncio
async def test_integrate_handler_revert_partial_becomes_failed(
    tmp_path: Path,
    monkeypatch,
):
    """Non-KEEP + partial revert must set top-level status='failed'.

    A partial revert hides a multinode failure where the patch stayed live on a
    remote pod, so it is not a completed lifecycle.
    """
    import hyperloom.orchestrator.kernel.request_handlers as krh
    import hyperloom.orchestrator.actions.executors.baseline as baseline_mod
    import hyperloom.orchestrator.actions.executors.benchmark_result as br

    monkeypatch.setattr(
        krh,
        "_maybe_revert_kernel_patch",
        lambda apply_result: {"status": "partial", "reason": "mn_revert_failed"},
    )
    monkeypatch.setattr(
        krh,
        "_maybe_apply_kernel_patch",
        lambda payload, *, session_dir, kernel_id=None: {
            "status": "ok",
            "kernel_id": str(kernel_id or ""),
            "manifest_path": "/tmp/fake.manifest",
        },
    )

    class _FakeBaseline:
        default_timeout_sec = baseline_mod.BASELINE_DEFAULT_TIMEOUT_SEC

        def __init__(self, *, session_dir):
            self.session_dir = session_dir

        async def __call__(self, ctx):
            return {"output_throughput": 98.0}  # below base_tput -> REVERT

    monkeypatch.setattr(baseline_mod, "BaselineExecutor", _FakeBaseline)
    monkeypatch.setattr(br, "is_valid_measurement", lambda r: True)

    from hyperloom.orchestrator.kernel.request_handlers import integrate_handler

    result = await integrate_handler(
        {
            "task_id": "test-partial-revert",
            "kernel_id": "k-test",
            "patch_path": "/tmp/fake.patch",
            "target_file": "/tmp/fake.cu",
            "base_tput": 100.0,
            "config_path": str(tmp_path / "base.yaml"),
        },
        session_dir=tmp_path,
    )

    assert result["decision"] == "REVERT"
    assert result["status"] == "failed"
    assert result["patch_cleanup_status"] == "recovery_required"
    assert result["patch_cleanup_action"] == "revert"
    assert result.get("error_class") == "patch_revert_incomplete"
