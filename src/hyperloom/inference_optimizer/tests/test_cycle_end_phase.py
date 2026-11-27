# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CYCLE_END phase tests: the stack work its entry settles, and the exit it takes once settled."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from types import SimpleNamespace
from typing import Any

import pytest
from hyperloom.orchestrator.kernel import kernel_decisions as kd
from hyperloom.orchestrator.phases.machine import Transition

from hyperloom.common.perf_metric import GRADED_INTVTY, GRADED_INTVTY_P50
from hyperloom.inference_optimizer.breakdown.recorder.event_ids import INLINE_EVENT_PARAM
from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import kernel_event_id
from hyperloom.orchestrator.collaborator import OrchestrationPrompt
from hyperloom.orchestrator.tests.mock_backend import MockBackend, MockTurn, ScriptedPlan
from hyperloom.orchestrator.loop.coordinator import AdvisoryServices, Coordinator
from hyperloom.orchestrator.kernel.patch_lifecycle import lifecycle_complete
from hyperloom.orchestrator.phases import machine_state
from hyperloom.orchestrator.phases import phase_predicates
from hyperloom.orchestrator.phases import phase_transitions
from hyperloom.orchestrator.state.shared_state import SharedState

from .conftest import seed_kernel_keep
from hyperloom.orchestrator.kernel import integrate_accuracy as integrate_accuracy


_STACK_ORIGINAL_SOURCE = "def kernel():\n    return 1\n"

_STACK_PATCHED_SOURCE = "def kernel():\n    return 2\n"


class _StubTaskRegistry:
    """create_or_return_existing double, keyed by idempotency_key."""

    def __init__(self):
        self._tasks: dict[str, Any] = {}

    async def create_or_return_existing(self, spec):
        from hyperloom.orchestrator.state.task_registry import Task

        self.last_lease_ttl_sec = spec.lease_ttl_sec
        existing = self._tasks.get(spec.idempotency_key)
        if existing is not None:
            return existing, True
        import uuid as _uuid

        task = Task(
            task_id=spec.task_id or _uuid.uuid4().hex,
            kind=spec.kind,
            state="queued",
            params=dict(spec.params),
            idempotency_key=spec.idempotency_key,
            requires_lanes=list(spec.requires_lanes) if spec.requires_lanes else [],
        )
        self._tasks[spec.idempotency_key] = task
        return task, False


@pytest.mark.asyncio
async def test_drain_pending_keep_integrates_records_result_once(
    tmp_path: Path,
    monkeypatch,
):
    """CYCLE_END entry drain must record integrate results so the same KEEP is not retried until cap."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState(
        baseline_tput=100.0,
        current_best={"action": "baseline", "tput": 100.0},
    )
    seed_kernel_keep(c.shared_state, "k004", micro=4.21, source_file="/tmp/kernel.cu", artifact="/tmp/optimized.cu")
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
    c.phase_kernel.maybe_enqueue_watermark_roofline = _noop_roofline

    await c.phase_kernel_stack.drain_pending_keep_integrates()

    assert calls == ["k004"]
    assert c.shared_state.kernel_integrate_attempts
    assert kd.pending_keep_kernel_ids(c.shared_state) == []
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
    seed_kernel_keep(state, "k004", micro=4.21, source_file="/tmp/rmsnorm.cu", artifact="/tmp/k004_opt.cu")
    seed_kernel_keep(state, "k001", micro=1.51, source_file="/tmp/moe.cu", artifact="/tmp/k001_opt.cu")

    assert kd.pending_keep_kernel_ids(state) == ["k001", "k004"]


def test_pending_keep_kernel_ids_do_not_retry_needs_review():
    """A recorded NEEDS_REVIEW attempt should not auto-rerun the same patch."""
    state = SharedState()
    seed_kernel_keep(state, "k004", micro=4.21, source_file="/tmp/rmsnorm.cu", artifact="/tmp/k004_opt.cu")
    seed_kernel_keep(state, "k001", micro=1.51, source_file="/tmp/moe.cu", artifact="/tmp/k001_opt.cu")
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

    assert kd.pending_keep_kernel_ids(state) == ["k001"]


def _patch_stack_validation_internals(monkeypatch, *, new_tput: float, revert_status: str = "ok"):
    """Stub apply/revert/bench so the real stack-validation decision path runs."""
    import hyperloom.orchestrator.actions.executors._kernel_agent_tool as kernel_agent_tool
    import hyperloom.orchestrator.actions.executors.baseline as baseline_mod
    import hyperloom.orchestrator.actions.executors.benchmark_result as br

    def _fake_apply(payload, *, session_dir, kernel_id):
        return {"status": "ok", "kernel_id": kernel_id, "manifest_path": None}

    def _fake_revert(applied):
        return {"status": revert_status}

    class _FakeBaselineExecutor:
        default_timeout_sec = baseline_mod.resolve_benchmark_timeouts()[1]

        def __init__(self, *, session_dir):
            self.session_dir = session_dir

        async def __call__(self, ctx):
            return {
                "output_throughput": new_tput,
                "report_path": "/tmp/report",
                "workspace": "/tmp/workspace",
            }

    monkeypatch.setattr(kernel_agent_tool, "_maybe_apply_kernel_patch", _fake_apply)
    monkeypatch.setattr(kernel_agent_tool, "_maybe_revert_kernel_patch", _fake_revert)
    monkeypatch.setattr(baseline_mod, "BaselineExecutor", _FakeBaselineExecutor)
    monkeypatch.setattr(br, "is_valid_measurement", lambda result: True)


def _ledger_rows(c: Coordinator, kernel_ids: list[str]) -> list[dict[str, Any]]:
    """The live integrate-ledger rows the named kernels' latest attempts wrote, in the given order."""
    by_kernel = {row["kernel_id"]: row for row in c.shared_state.kernel_integrate_attempts.values()}
    return [by_kernel[kid] for kid in kernel_ids]


def _record_needs_review_pair(state: SharedState) -> None:
    """Integrate k001 (+0.6%) and k004 (+0.8%) as NEEDS_REVIEW, the pair a stack validation combines."""
    for kid, gain in (("k001", 0.6), ("k004", 0.8)):
        state.record_kernel_integrate_result(
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


def _baseline_coordinator_with_needs_review_pair(tmp_path: Path) -> Coordinator:
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState(
        baseline_tput=100.0,
        current_best={"action": "baseline", "tput": 100.0},
    )
    _record_needs_review_pair(c.shared_state)
    return c


def _stack_keep(entries: list[dict[str, Any]], tmp_path: Path) -> dict[str, Any]:
    """The KEEP a combined E2E validation of ``entries`` reports at +2% over a 100 tok/s anchor."""
    return {
        "status": "ok",
        "decision": "KEEP",
        "kernel_id": "+".join(e["kernel_id"] for e in entries),
        "patch_path": "+".join(e["patch_path"] for e in entries),
        "target_file": "+".join(e["target_file"] for e in entries),
        "base_tput": 100.0,
        "new_tput": 102.0,
        "gain_pct": 2.0,
        "workspace": str(tmp_path / "integrate-stack"),
        "apply_result": {"status": "ok"},
        "stack_kernel_ids": [e["kernel_id"] for e in entries],
        "stack_validation": True,
    }


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
    _record_needs_review_pair(c.shared_state)
    return c


@pytest.mark.asyncio
async def test_stack_validation_reverts_when_no_gain_over_current_best(
    tmp_path: Path,
    monkeypatch,
):
    """Stack worse than current_best (110) but above baseline (100) must REVERT."""
    c = _stack_validation_coordinator(tmp_path)
    stack = _ledger_rows(c, ["k001", "k004"])
    _patch_stack_validation_internals(monkeypatch, new_tput=109.0)

    result = await c.phase_kernel_stack._run_kernel_stack_validation_e2e(stack)

    assert result["decision"] == "REVERT"
    assert result["gain_pct"] == pytest.approx(9.0)
    assert result["stack_incremental_gain_pct"] == pytest.approx(-0.9090909, rel=1e-3)
    assert result["revert_result"]["status"] == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("revert_status", ["partial", "failed"], ids=["partial-revert", "failed-revert"])
async def test_stack_validation_incomplete_revert_becomes_failed(
    tmp_path: Path,
    monkeypatch,
    revert_status,
):
    """An incomplete inner revert means the patch may still be on a remote pod."""
    c = _stack_validation_coordinator(tmp_path)
    stack = _ledger_rows(c, ["k001", "k004"])
    _patch_stack_validation_internals(monkeypatch, new_tput=109.0, revert_status=revert_status)

    result = await c.phase_kernel_stack._run_kernel_stack_validation_e2e(stack)

    assert result["decision"] == "REVERT"
    assert result["status"] == "failed"
    assert result["patch_cleanup_action"] == "revert"
    assert result.get("error_class") == "patch_revert_incomplete"
    assert all(r["status"] == revert_status for r in result["revert_result"]["stack_reverts"])


@pytest.mark.asyncio
async def test_stack_validation_keeps_on_positive_increment_over_current_best(
    tmp_path: Path,
    monkeypatch,
):
    """A real increment over current_best (110 -> 112, +1.8%) must KEEP."""
    c = _stack_validation_coordinator(tmp_path)
    stack = _ledger_rows(c, ["k001", "k004"])
    _patch_stack_validation_internals(monkeypatch, new_tput=112.0)

    result = await c.phase_kernel_stack._run_kernel_stack_validation_e2e(stack)

    assert result["decision"] == "KEEP"
    assert result["gain_pct"] == pytest.approx(12.0)
    assert result["stack_incremental_gain_pct"] == pytest.approx(1.8181818, rel=1e-3)
    assert result["revert_result"]["status"] == "skipped"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "grading_mode",
        "output",
        "total",
        "intvty",
        "missing",
        "submission_valid",
        "decision",
        "objective",
        "increment",
        "verdict",
    ),
    [
        pytest.param(
            "agentx",
            130.0,
            18000.0,
            410.0,
            None,
            True,
            "REVERT",
            GRADED_INTVTY_P50,
            2.5,
            "REVERT",
            id="median-below-bar",
        ),
        pytest.param(
            "agentx",
            105.0,
            22000.0,
            415.0,
            None,
            True,
            "KEEP",
            GRADED_INTVTY_P50,
            3.75,
            "KEEP",
            id="intvty-wins-output-dips",
        ),
        pytest.param(
            "agentx",
            130.0,
            22000.0,
            300.0,
            None,
            True,
            "REVERT",
            GRADED_INTVTY_P50,
            -25.0,
            "REVERT",
            id="interactivity-regresses",
        ),
        pytest.param(
            "agentx",
            130.0,
            18000.0,
            300.0,
            None,
            True,
            "REVERT",
            GRADED_INTVTY_P50,
            -25.0,
            "REVERT",
            id="both-axes-regress",
        ),
        pytest.param(
            "agentx",
            130.0,
            22000.0,
            404.0,
            None,
            True,
            "REVERT",
            GRADED_INTVTY_P50,
            1.0,
            "REVERT",
            id="intvty-below-keep-floor",
        ),
        pytest.param(
            "agentx", 130.0, 22000.0, 410.0, None, False, "REVERT", None, -100.0, "REVERT", id="invalid-submission"
        ),
        pytest.param(
            "agentx", 130.0, 22000.0, 410.0, None, None, "REVERT", None, -100.0, "REVERT", id="unverified-submission"
        ),
        pytest.param(
            "synthetic",
            130.0,
            18000.0,
            300.0,
            None,
            True,
            "KEEP",
            "output_throughput",
            200.0 / 11.0,
            "KEEP",
            id="synthetic-output",
        ),
        *[
            pytest.param(
                "agentx",
                output,
                22000.0,
                410.0,
                (side, axis),
                True,
                "NEEDS_REVIEW",
                "output_throughput",
                (output - 110.0) / 110.0 * 100.0,
                # A degraded pair never carries a KEEP verdict, whichever way
                # the output figure moved: it is a diagnostic, not a decision.
                "REVERT",
                id=f"missing-{side}-{axis}-output-{direction}",
            )
            for side in ("candidate", "reference")
            for axis in ("total", "intvty")
            for direction, output in (("up", 130.0), ("down", 90.0))
        ],
        *[
            pytest.param(
                mode,
                130.0,
                None,
                None,
                ("reference", "total"),
                True,
                "KEEP",
                "output_throughput",
                200.0 / 11.0,
                "KEEP",
                id=f"{mode}-missing-axes",
            )
            for mode in ("synthetic", "explicit-output")
        ],
    ],
)
async def test_stack_validation_preserves_actual_measurement(
    tmp_path: Path,
    monkeypatch,
    grading_mode,
    output,
    total,
    intvty,
    missing,
    submission_valid,
    decision,
    objective,
    increment,
    verdict,
):
    """The real stack verdict and its writeback envelope share one E2E measurement."""
    import hyperloom.orchestrator.actions.executors.baseline as baseline_mod
    from hyperloom.orchestrator.kernel import apply_kernel_patch as akp, patch_restore

    agentx = grading_mode != "synthetic"
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1" if agentx else "0")
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    if grading_mode == "explicit-output":
        monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output_throughput")
    monkeypatch.delenv("HYPERLOOM_ALLOW_UNVERIFIED_SUBMISSION", raising=False)
    monkeypatch.setenv("HYPERLOOM_PERF_NOISE_PCT", "5")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "1")
    for module in (akp, patch_restore):
        monkeypatch.setattr(module, "clear_python_kernel_caches", lambda target: {"status": "skipped"})
    c = _stack_validation_coordinator(tmp_path)
    c.shared_state.framework = "vllm"
    c.shared_state.benchmark_mode = "agentx" if agentx else "synthetic"
    c.shared_state.baseline_accuracy = 0.9
    c.shared_state.current_best["workload_identity"] = "fixed-workload"
    c.shared_state.current_best.update(
        total_throughput=20000.0,
        e2e_norm_intvty_p90=400.0,
        e2e_norm_intvty_p50=400.0,
        duration_seconds=900.0,
        request_error_rate=0.0,
        extra_server_args="--max-model-len 8192",
    )
    stack = _ledger_rows(c, ["k001", "k004"])
    original_source = "def kernel():\n    return 1\n"
    optimized_source = "def kernel():\n    return 2\n"
    for entry in stack:
        target = tmp_path / f"{entry['kernel_id']}.py"
        patch = tmp_path / f"{entry['kernel_id']}_opt.py"
        target.write_text(original_source, encoding="utf-8")
        patch.write_text(optimized_source, encoding="utf-8")
        entry.update(target_file=str(target), patch_path=str(patch))

    bench_result = {
        "status": "succeeded",
        "output_throughput": output,
        "input_throughput": total - output if total is not None else None,
        "total_token_throughput": total,
        GRADED_INTVTY: intvty,
        GRADED_INTVTY_P50: intvty,
        "duration_seconds": 900.0,
        "request_error_rate": 0.0,
        "completed_requests": 64,
        "valid_measurement": not agentx or submission_valid is True,
        "diagnostic_only": False,
        "workload_identity": "fixed-workload",
        "submission_invalid_reasons": ["scenario_constraint"] if submission_valid is False else [],
        "accuracy": 0.9,
        "ttft_mean_ms": 20.0,
        "e2el_mean_ms": 1000.0,
        "tpot_mean_ms": 2.0,
        "launch_evidence": {
            "observed_server_launch_flags": "--max-model-len 8192",
            "observed_server_identity": {"model_path": "/models/test-model", "tp_size": 1},
        },
        "launch_evidence_path": str(tmp_path / "measured" / "launch_evidence.json"),
        "server_log_path": str(tmp_path / "measured" / "server.log"),
        "report_path": str(tmp_path / "measured" / "benchmark_report.json"),
        "workspace": str(tmp_path / "measured"),
    }
    if missing:
        side, axis = missing
        incomplete = bench_result if side == "candidate" else c.shared_state.current_best
        missing_keys = (
            ("total_token_throughput", "total_throughput", "input_throughput")
            if axis == "total"
            else (GRADED_INTVTY, GRADED_INTVTY_P50)
        )
        for key in missing_keys:
            incomplete.pop(key, None)
    original_measurement = dict(bench_result)
    original_best = dict(c.shared_state.current_best)
    calls = []

    async def _benchmark(self, ctx):
        calls.append(ctx)
        assert ctx.extra["shared_state"] is c.shared_state
        assert ctx.task.params["extra_server_args"] == "--max-model-len 8192"
        assert ctx.task.params["quality_ref_exempt"] is True
        assert ctx.task.params[INLINE_EVENT_PARAM] == kernel_event_id(0)
        assert all(Path(entry["target_file"]).read_text(encoding="utf-8") == optimized_source for entry in stack)
        return bench_result

    monkeypatch.setattr(baseline_mod.BaselineExecutor, "__call__", _benchmark)

    result = await c.phase_kernel_stack._run_kernel_stack_validation_e2e(stack)

    assert len(calls) == 1
    assert result["status"] == "ok", result
    assert result["decision"] == decision
    assert result["graded_verdict"] == verdict
    assert result["stack_incremental_gain_pct"] == pytest.approx(increment)
    assert result["base_tput"] == 100.0
    valid = not agentx or submission_valid is True
    assert result["new_tput"] == (output if valid else 0.0)
    assert result["gain_pct"] == pytest.approx(output - 100.0 if valid else -100.0)
    assert result["stack_kernel_ids"] == ["k001", "k004"]
    assert result["patch_cleanup_action"] == ""
    expected_source = optimized_source if decision == "KEEP" else original_source
    assert all(Path(entry["target_file"]).read_text(encoding="utf-8") == expected_source for entry in stack)
    assert result["bench_result"] == original_measurement
    assert bench_result == original_measurement
    assert c.shared_state.current_best == original_best
    assert result.get("graded_objective") == objective
    if missing and grading_mode == "agentx":
        reason = "candidate_axes_missing" if missing[0] == "candidate" else "current_best_axes_missing"
        assert reason in result["reason"]
        assert result["revert_result"]["status"] == "ok"
        assert result["finalize_results"] == []
    assert result["workspace"] == bench_result["workspace"]
    assert result["report_path"] == bench_result["report_path"]
    assert result["ttft_mean_ms"] == bench_result["ttft_mean_ms"]
    assert result["e2el_mean_ms"] == bench_result["e2el_mean_ms"]
    assert result["tpot_mean_ms"] == bench_result["tpot_mean_ms"]


@pytest.mark.asyncio
async def test_positive_needs_review_stack_validation_promotes_combo(tmp_path: Path):
    """Two positive sub-threshold kernel patches should get one combined E2E validation."""
    c = _baseline_coordinator_with_needs_review_pair(tmp_path)

    validation_calls = 0

    async def _fake_stack_validation(entries):
        nonlocal validation_calls
        validation_calls += 1
        assert {e["kernel_id"] for e in entries} == {"k001", "k004"}
        return {
            **_stack_keep(entries, tmp_path),
            "stack_member_identities": [{k: e[k] for k in ("kernel_id", "patch_path", "target_file")} for e in entries],
        }

    async def _noop_roofline(*, reason: str):
        return None

    c.phase_kernel_stack._run_kernel_stack_validation_e2e = _fake_stack_validation
    c.phase_kernel.maybe_enqueue_watermark_roofline = _noop_roofline

    await c.phase_kernel_stack.maybe_validate_positive_needs_review_stack()

    expected_members = ["k004", "k001"]
    expected_display_id = "+".join(expected_members)
    assert c.shared_state.current_best["action"] == "integrate"
    assert c.shared_state.current_best["variant_name"] == expected_display_id
    assert c.shared_state.cumulative_gain_validated == pytest.approx(2.0)
    assert validation_calls == 1
    resolved_entries = [
        entry
        for entry in c.shared_state.kernel_integrate_attempts.values()
        if entry.get("kernel_id") in {"k001", "k004"}
    ]
    assert all(entry["stack_resolved"] is True for entry in resolved_entries)

    # Re-invoking must be a no-op (idempotent): the call count must not advance.
    calls_before_recall = validation_calls
    await c.phase_kernel_stack.maybe_validate_positive_needs_review_stack()

    assert validation_calls == calls_before_recall
    stack_entries = [
        item
        for item in c.shared_state.optimization_stack
        if isinstance(item, dict) and item.get("kernel_id") == expected_display_id
    ]
    assert stack_entries
    assert stack_entries[0].get("stack_kernel_ids") == expected_members


@pytest.mark.asyncio
async def test_recovers_pending_stack_validation_after_crash(tmp_path: Path):
    """A saved pending stack result should finish promotion without re-applying."""
    c = _baseline_coordinator_with_needs_review_pair(tmp_path)
    stack = _ledger_rows(c, ["k001", "k004"])
    c.phase_kernel_stack._mark_stack_validation_in_progress(stack, "k001+k004")
    c.shared_state.pending_stack_validation_result = {
        **c.shared_state.pending_stack_validation_result,
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
    c.phase_kernel.maybe_enqueue_watermark_roofline = _noop_roofline

    await c.phase_kernel_stack.recover_interrupted_stack_validation()

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

    eligible = c.phase_kernel_stack._positive_needs_review_integrates()
    assert len(eligible) == 1
    assert eligible[0]["kernel_id"] == "k004"


@pytest.mark.asyncio
async def test_on_enter_cycle_end_triggers_stack_validation_without_pending_keeps(
    tmp_path: Path,
    monkeypatch,
):
    """Stack validation must run even when has_keep_pending_integrate is False."""
    c = _baseline_coordinator_with_needs_review_pair(tmp_path)
    c.tasks = _StubTaskRegistry()
    c.knowledge_plane = None
    # All KEEPs already integrated as NEEDS_REVIEW — no pending KEEP.
    assert not kd.has_keep_pending_integrate(c.shared_state)

    validation_calls = []

    async def _fake_stack_validation(entries):
        validation_calls.append([e["kernel_id"] for e in entries])
        return _stack_keep(entries, tmp_path)

    async def _noop_roofline(*, reason: str):
        return None

    c.phase_kernel_stack._run_kernel_stack_validation_e2e = _fake_stack_validation
    c.phase_kernel.maybe_enqueue_watermark_roofline = _noop_roofline
    c.orch_prompt = OrchestrationPrompt({}, is_user_supplied=True)

    await c.phase_cycle_end.on_enter_cycle_end(
        Transition(from_phase="KERNEL_AGENT", to_phase="CYCLE_END", reason="test", evidence={})
    )

    assert validation_calls == [["k004", "k001"]]
    assert c.shared_state.current_best["variant_name"] == "+".join(validation_calls[0])


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
    seed_kernel_keep(c.shared_state, "k_new", micro=2.0, source_file="/tmp/new.cu", artifact="/tmp/new_opt.cu")
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
    c.phase_kernel.maybe_enqueue_watermark_roofline = _noop_roofline

    await c.phase_kernel_stack.drain_pending_keep_integrates()

    assert len(captured_payloads) == 1
    # use current_best.tput (110.0), not baseline (100.0)
    assert captured_payloads[0]["base_tput"] == 110.0


# End-to-end via the real phase machine
def _real_coordinator(tmp_path: Path) -> Coordinator:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    coordinator = Coordinator(
        session_dir=session_dir,
        backends={"orchestration": MockBackend(idle_plan)},
        advisory=AdvisoryServices(knowledge_plane=None),
    )
    coordinator.orch_prompt.set("orchestration", "SYSTEM PROMPT")
    return coordinator


@pytest.mark.asyncio
async def test_a_transition_into_cycle_end_settles_the_cycle_and_dispatches_nothing(tmp_path: Path):
    coord = _real_coordinator(tmp_path)
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.macro_cycle = 2
    machine_state.record_phase_transition(
        coord.shared_state, to_phase="CYCLE_END", reason="kernel_no_more_leverage", evidence={}
    )

    await coord.phase_machine._on_phase_entered(from_phase="KERNEL_AGENT", to_phase="CYCLE_END")

    assert coord.shared_state.cycle_end_settled_cycle == 2
    assert SharedState.load_or_init(coord.session_dir).cycle_end_settled_cycle == 2
    assert await coord.tasks.queued() == []


# The exit CYCLE_END takes
_NOW = 1_800_000_000.0


def _cycle_end_state(*, settled_cycle: int, macro_cycle: int = 0, entered_min_ago: float = 1.0) -> SharedState:
    """A CYCLE_END entered ``entered_min_ago`` minutes ago, one hour into a ten-hour session with gain this cycle."""
    return SharedState(
        phase=machine_state.PHASE_CYCLE_END,
        phase_started_unix=_NOW - entered_min_ago * 60.0,
        start_ts=datetime.fromtimestamp(_NOW - 3600.0, tz=timezone.utc).isoformat(),
        max_minutes=600,
        phase_budget_pct=dict(machine_state.DEFAULT_PHASE_BUDGET_PCT),
        macro_cycle=macro_cycle,
        cumulative_gain_validated=5.0,
        gain_at_cycle_start=0.0,
        baseline_tput=100.0,
        cycle_end_settled_cycle=settled_cycle,
    )


def test_a_settled_cycle_end_loops_back_while_another_cycle_is_feasible():
    out = phase_transitions.compute_next_phase(_cycle_end_state(settled_cycle=0), now_unix=_NOW)

    assert out is not None
    target, reason, evidence = out
    assert (target, reason) == (machine_state.PHASE_FRAMEWORK_AGENT, "cycle_reloop")
    assert evidence["loopback"] is True
    assert evidence["predicate_inputs"]["cycle_end_result"]["settled"] is True
    assert phase_transitions.replay_next_phase(evidence["predicate_inputs"]) == out


def test_a_settled_cycle_end_with_no_further_cycle_closes_as_cycle_end_done():
    out = phase_transitions.compute_next_phase(
        _cycle_end_state(settled_cycle=0),
        flags=phase_predicates.PhaseFlags(optimize_enabled=False),
        now_unix=_NOW,
    )

    assert out is not None
    assert out[:2] == (machine_state.PHASE_CLOSE, "cycle_end_done")
    assert "terminal" not in out[2]


def test_a_settled_cycle_end_closes_on_the_rule_that_refuses_the_next_cycle():
    state = _cycle_end_state(settled_cycle=0)
    state.saturated_directions = {"kv_cache": {"saturated": True}}

    out = phase_transitions.compute_next_phase(state, now_unix=_NOW)

    assert out is not None
    assert out[:2] == (machine_state.PHASE_CLOSE, "global_converged")
    assert out[2]["reloop_blocked"] == "all_directions_saturated"


@pytest.mark.parametrize("settled_cycle", [-1, 0], ids=["never", "earlier_cycle"])
def test_an_unsettled_cycle_end_holds_until_its_entry_work_settles_the_cycle(settled_cycle):
    state = _cycle_end_state(settled_cycle=settled_cycle, macro_cycle=1, entered_min_ago=600.0)

    assert phase_transitions.compute_next_phase(state, now_unix=_NOW) is None


# ────────────────────────────────────────────────────────────────────────────── Patch lifecycle convergence — new
# tests (P1-19 fix verification) ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stack_validation_keep_calls_finalize(
    tmp_path: Path,
    monkeypatch,
):
    """A KEEP result must call _maybe_finalize_kernel_patch for each applied patch."""
    import hyperloom.orchestrator.actions.executors._kernel_agent_tool as kernel_agent_tool

    finalize_calls: list[dict] = []

    def _spy_finalize(apply_result):
        finalize_calls.append(apply_result)
        return {"status": "ok", "manifest_path": str(apply_result.get("manifest_path") or "")}

    monkeypatch.setattr(kernel_agent_tool, "_maybe_finalize_kernel_patch", _spy_finalize)

    def _fake_apply_with_manifest(payload, *, session_dir, kernel_id):
        return {
            "status": "ok",
            "kernel_id": kernel_id,
            "manifest_path": f"/tmp/{kernel_id}.manifest",
        }

    monkeypatch.setattr(kernel_agent_tool, "_maybe_apply_kernel_patch", _fake_apply_with_manifest)

    c = _stack_validation_coordinator(tmp_path)
    stack = _ledger_rows(c, ["k001", "k004"])
    _patch_stack_validation_internals(monkeypatch, new_tput=115.0)

    result = await c.phase_kernel_stack._run_kernel_stack_validation_e2e(stack)

    assert result["decision"] == "KEEP"
    assert result["status"] == "ok"
    assert result["patch_cleanup_action"] == ""
    # One finalize call per patch in the two-entry stack.
    assert len(finalize_calls) == 2


@pytest.mark.asyncio
async def test_stack_validation_keep_partial_finalize_requires_recovery(
    tmp_path: Path,
    monkeypatch,
):
    """KEEP + partial finalize must ask for recovery, not report cleanup complete."""
    import hyperloom.orchestrator.actions.executors._kernel_agent_tool as kernel_agent_tool

    monkeypatch.setattr(
        kernel_agent_tool,
        "_maybe_finalize_kernel_patch",
        lambda apply_result: {"status": "partial", "issues": [{"kind": "multinode_finalize"}]},
    )
    monkeypatch.setattr(
        kernel_agent_tool,
        "_maybe_apply_kernel_patch",
        lambda payload, *, session_dir, kernel_id: {
            "status": "ok",
            "kernel_id": kernel_id,
            "manifest_path": f"/tmp/{kernel_id}.manifest",
        },
    )

    c = _stack_validation_coordinator(tmp_path)
    stack = _ledger_rows(c, ["k001", "k004"])
    _patch_stack_validation_internals(monkeypatch, new_tput=115.0)

    result = await c.phase_kernel_stack._run_kernel_stack_validation_e2e(stack)

    assert result["decision"] == "KEEP"
    assert result["status"] == "ok"
    assert result["patch_cleanup_action"] == "finalize"


@pytest.mark.asyncio
async def test_stack_validation_accuracy_regression_downgrades_to_needs_review(
    tmp_path: Path,
    monkeypatch,
):
    """An accuracy regression on a stack KEEP must drop decision to NEEDS_REVIEW."""

    seen: dict[str, object] = {}

    def _fake_accuracy_gate(bench_result, *, reference, workspace, server_args=""):
        # The lane must hand the gate the args the bench server ran under, or a context too small to host an eval
        # reads as a broken eval.
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

    monkeypatch.setattr(integrate_accuracy, "grade_integrate_accuracy", _fake_accuracy_gate)

    c = _stack_validation_coordinator(tmp_path)
    stack = _ledger_rows(c, ["k001", "k004"])
    _patch_stack_validation_internals(monkeypatch, new_tput=115.0)

    result = await c.phase_kernel_stack._run_kernel_stack_validation_e2e(stack)

    assert result["decision"] == "NEEDS_REVIEW"
    assert "server_args" in seen
    # NEEDS_REVIEW is non-KEEP so reverts must have been called and succeeded.
    assert result["revert_result"]["status"] == "ok"


@pytest.mark.asyncio
async def test_integrate_handler_revert_partial_becomes_failed(
    tmp_path: Path,
    monkeypatch,
):
    """Non-KEEP + partial revert must set top-level status='failed'."""
    from hyperloom.orchestrator.kernel import _integrate_outcomes as integrate_outcomes
    import hyperloom.orchestrator.actions.executors.baseline as baseline_mod
    import hyperloom.orchestrator.actions.executors.benchmark_result as br

    monkeypatch.setattr(
        integrate_outcomes,
        "_maybe_revert_kernel_patch",
        lambda apply_result: {"status": "partial", "reason": "mn_revert_failed"},
    )
    monkeypatch.setattr(
        integrate_outcomes,
        "_maybe_apply_kernel_patch",
        lambda payload, *, session_dir, kernel_id=None: {
            "status": "ok",
            "kernel_id": str(kernel_id or ""),
            "manifest_path": "/tmp/fake.manifest",
        },
    )

    class _FakeBaseline:
        default_timeout_sec = baseline_mod.resolve_benchmark_timeouts()[1]

        def __init__(self, *, session_dir, shared_state=None):
            self.session_dir = session_dir
            self.shared_state = shared_state

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
    assert result["patch_cleanup_action"] == "revert"
    assert result.get("error_class") == "patch_revert_incomplete"


def _assert_halted_with_nothing_else_changed(after: dict[str, Any], before: dict[str, Any]) -> None:
    """The recovery refused: it recorded the halt and left every checkpoint and ledger row as it found them."""
    from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON

    assert after["stop_reason"] == PATCH_RECOVERY_INCOMPLETE_STOP_REASON
    halt_fields = ("stop_reason", "stop_ts")
    assert {k: v for k, v in after.items() if k not in halt_fields} == {
        k: v for k, v in before.items() if k not in halt_fields
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "members",
    [
        {},
        {"stack_kernel_ids": "a+b"},
        {"stack_kernel_ids": None},
        {"stack_kernel_ids": []},
        {"stack_kernel_ids": ["a"]},
        {"stack_kernel_ids": ["a", "a"]},
        {"stack_kernel_ids": ["a", "b", "missing"]},
        {"stack_kernel_ids": ["a", None]},
        {"stack_kernel_ids": {"a": True, "b": True}},
    ],
    ids=[
        "legacy_ambiguous",
        "string",
        "null",
        "empty",
        "short",
        "duplicate",
        "missing_member",
        "non_string",
        "mapping",
    ],
)
async def test_stack_members_invalid_recovery_preserves_pending_evidence(tmp_path, monkeypatch, members):
    from hyperloom.orchestrator.actions.executors import _kernel_agent_tool as kat
    from hyperloom.inference_optimizer.session.session_binding import session_scope

    monkeypatch.setenv("HYPERLOOM_LANGFUSE_ENABLE", "0")
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState(baseline_tput=100.0, current_best={"action": "baseline", "tput": 100.0})
    for kid in ("a", "b"):
        c.shared_state.kernel_integrate_attempts[kid] = {
            "kernel_id": kid,
            "patch_path": str(tmp_path / f"{kid}.patch"),
            "target_file": str(tmp_path / f"{kid}.py"),
            "stack_validation_in_progress": True,
        }
    c.shared_state.pending_stack_validation_result = {
        "status": "ok",
        "decision": "KEEP",
        "kernel_id": "a+b",
        "patch_path": "display-only-paths",
        "target_file": "display-only-targets",
        "new_tput": 110.0,
        "stack_validation": True,
        "stack_member_identities": [
            {
                "kernel_id": kid,
                "patch_path": str(tmp_path / f"{kid}.patch"),
                "target_file": str(tmp_path / f"{kid}.py"),
            }
            for kid in ("a", "b")
        ],
        **members,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"kernel_id":"a","status":"applied"}\n', encoding="utf-8")
    c.shared_state.pending_stack_validation_apply_results = [{"status": "ok", "manifest_path": str(manifest)}]
    before = deepcopy(c.shared_state.to_dict())
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(before), encoding="utf-8")
    original = state_path.read_bytes()
    original_manifest = manifest.read_bytes()
    revert = Mock(return_value={"status": "ok"})
    monkeypatch.setattr(kat, "_maybe_revert_kernel_patch", revert)
    c.phase_kernel.maybe_enqueue_watermark_roofline = AsyncMock()

    with session_scope(tmp_path), pytest.raises(ValueError, match="(?i)stack|member"):
        await c.phase_kernel_stack.recover_interrupted_stack_validation()

    revert.assert_not_called()
    c.phase_kernel.maybe_enqueue_watermark_roofline.assert_not_called()
    _assert_halted_with_nothing_else_changed(c.shared_state.to_dict(), before)
    _assert_halted_with_nothing_else_changed(json.loads(state_path.read_bytes()), json.loads(original))
    assert manifest.read_bytes() == original_manifest


@pytest.mark.parametrize("single", [{"stack_validation": False}, {"stack_kernel_ids": ["a+b"]}])
def test_stack_members_explicit_single_id_preserves_plus(single):
    from hyperloom.orchestrator.phases.kernel_stack import resolve_stack_members

    assert resolve_stack_members({"kernel_id": "a+b", **single}) == ("a+b",)


@pytest.mark.parametrize("flag", [None, 0, "", "false"])
def test_stack_members_non_boolean_flag_is_not_single_evidence(flag):
    from hyperloom.orchestrator.phases.kernel_stack import resolve_stack_members

    with pytest.raises(ValueError, match="stack_validation"):
        resolve_stack_members({"kernel_id": "a+b", "stack_validation": flag})


@pytest.fixture
def historical_stack_coord(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_LANGFUSE_ENABLE", "0")
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState(baseline_tput=100.0, current_best={"action": "baseline", "tput": 100.0})
    for kid, patch_name, decision, gain in (
        ("a", "old-a.patch", "REVERT", -1.0),
        ("a", "new-a.patch", "NEEDS_REVIEW", 0.8),
        ("b", "b.patch", "NEEDS_REVIEW", 0.6),
    ):
        c.shared_state.record_kernel_integrate_result(
            {
                "status": "ok",
                "decision": decision,
                "kernel_id": kid,
                "patch_path": str(tmp_path / patch_name),
                "target_file": str(tmp_path / f"{kid}.py"),
                "new_tput": 100.0 + gain,
                "gain_pct": gain,
            }
        )
    c.phase_kernel.maybe_enqueue_watermark_roofline = AsyncMock()
    return c


@pytest.mark.asyncio
async def test_stack_members_selected_patch_ignores_other_patch_history(historical_stack_coord):
    from hyperloom.inference_optimizer.session.session_binding import session_scope

    c = historical_stack_coord
    historical = next(e for e in c.shared_state.kernel_integrate_attempts.values() if e["last_decision"] == "REVERT")
    original_history = deepcopy(historical)
    selected = []

    async def validate(entries):
        selected.extend((entry["kernel_id"], Path(entry["patch_path"]).name) for entry in entries)
        return {
            **c.shared_state.pending_stack_validation_result,
            "status": "ok",
            "decision": "KEEP",
            "new_tput": 110.0,
            "patch_path": "+".join(entry["patch_path"] for entry in entries),
            "target_file": "+".join(entry["target_file"] for entry in entries),
        }

    c.phase_kernel_stack._run_kernel_stack_validation_e2e = validate
    with session_scope(c.session_dir):
        await c.phase_kernel_stack.maybe_validate_positive_needs_review_stack()

    assert selected == [("a", "new-a.patch"), ("b", "b.patch")]
    assert historical == original_history
    assert c.shared_state.current_best["variant_name"] == "a+b"
    assert not c.shared_state.pending_stack_validation_result


@pytest.mark.asyncio
async def test_stack_members_checkpoint_selects_exact_patch_among_history(historical_stack_coord, monkeypatch):
    from hyperloom.inference_optimizer.session.session_binding import session_scope
    from hyperloom.orchestrator.kernel import _integrate_outcomes as integrate_outcomes

    c = historical_stack_coord
    selected = c.phase_kernel_stack._positive_needs_review_integrates()
    started = "2026-01-01T00:00:00+00:00"
    for entry in selected:
        entry.update(
            stack_validation_in_progress=True,
            stack_validation_kernel_id="a+b",
            stack_validation_started_at=started,
        )
    c.shared_state.pending_stack_validation_result = {
        "status": "ok",
        "decision": "KEEP",
        "new_tput": 110.0,
        "kernel_id": "a+b",
        "stack_validation": True,
        "stack_validation_started_at": started,
        "stack_kernel_ids": [entry["kernel_id"] for entry in selected],
        "stack_member_identities": [
            {key: entry[key] for key in ("kernel_id", "patch_path", "target_file")} for entry in selected
        ],
        "patch_path": "+".join(entry["patch_path"] for entry in selected),
        "target_file": "+".join(entry["target_file"] for entry in selected),
    }
    historical = next(e for e in c.shared_state.kernel_integrate_attempts.values() if e["last_decision"] == "REVERT")
    original_history = deepcopy(historical)
    c.shared_state = SharedState.from_dict(c.shared_state.to_dict())
    apply = Mock(side_effect=AssertionError("recovery must not apply patches"))
    revert = Mock(side_effect=AssertionError("completed validation must not revert"))
    monkeypatch.setattr(integrate_outcomes, "_maybe_apply_kernel_patch", apply)
    monkeypatch.setattr(integrate_outcomes, "_maybe_revert_kernel_patch", revert)

    with session_scope(c.session_dir):
        assert await c.phase_kernel_stack.recover_interrupted_stack_validation() is True

    apply.assert_not_called()
    revert.assert_not_called()
    assert c.shared_state.kernel_integrate_attempts[original_history["key"]] == original_history
    assert c.shared_state.current_best["variant_name"] == "a+b"
    assert not c.shared_state.pending_stack_validation_result
    assert not c.shared_state.pending_stack_validation_apply_results


def test_stack_members_same_selected_identity_is_still_ambiguous(historical_stack_coord):
    c = historical_stack_coord
    selected = c.phase_kernel_stack._positive_needs_review_integrates()
    c.shared_state.kernel_integrate_attempts["duplicate"] = deepcopy(selected[0])
    before = deepcopy(c.shared_state.to_dict())

    with pytest.raises(ValueError, match="(?i)stack|member"):
        c.phase_kernel_stack._mark_stack_validation_in_progress(selected, "a+b")

    assert c.shared_state.to_dict() == before


@pytest.mark.asyncio
async def test_stack_members_recovery_rejects_changed_patch(tmp_path, monkeypatch):
    c = _stack_validation_coordinator(tmp_path)
    stack = _ledger_rows(c, ["k001", "k004"])
    c.phase_kernel_stack._mark_stack_validation_in_progress(stack, "k001+k004")
    c.shared_state.pending_stack_validation_result.update(
        status="ok",
        decision="KEEP",
        new_tput=120.0,
        patch_path="display-patches",
        target_file="display-targets",
    )
    stack[0]["patch_path"] = str(tmp_path / "different.patch")
    before = deepcopy(c.shared_state.to_dict())
    c.phase_kernel.maybe_enqueue_watermark_roofline = AsyncMock()
    monkeypatch.setenv("HYPERLOOM_LANGFUSE_ENABLE", "0")

    with pytest.raises(ValueError, match="(?i)stack|member"):
        await c.phase_kernel_stack.recover_interrupted_stack_validation()

    _assert_halted_with_nothing_else_changed(c.shared_state.to_dict(), before)
    c.phase_kernel.maybe_enqueue_watermark_roofline.assert_not_called()


@pytest.mark.parametrize("case", ["empty", "short", "duplicate", "non_string", "missing_patch"])
def test_stack_members_invalid_refused_before_marking(tmp_path, monkeypatch, case):
    c = _stack_validation_coordinator(tmp_path)
    entries = [
        {"kernel_id": kid, "patch_path": str(tmp_path / f"{kid}.patch"), "target_file": str(tmp_path / f"{kid}.py")}
        for kid in ("a", "b")
    ]
    if case == "empty":
        entries = []
    elif case == "short":
        entries = entries[:1]
    elif case == "duplicate":
        entries[1]["kernel_id"] = "a"
    elif case == "non_string":
        entries[1]["kernel_id"] = None
    else:
        entries[1]["patch_path"] = ""
    monkeypatch.setenv("HYPERLOOM_LANGFUSE_ENABLE", "0")
    before = deepcopy(c.shared_state.to_dict())

    with pytest.raises(ValueError, match="(?i)stack|member"):
        c.phase_kernel_stack._mark_stack_validation_in_progress(entries, "a+b")

    assert c.shared_state.to_dict() == before


@pytest.mark.asyncio
async def test_stack_members_legacy_display_id_is_not_split_during_selection(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import baseline as baseline_mod
    from hyperloom.orchestrator.kernel import _integrate_outcomes as integrate_outcomes

    monkeypatch.setenv("HYPERLOOM_LANGFUSE_ENABLE", "0")
    c = _stack_validation_coordinator(tmp_path)
    c.shared_state.optimization_stack = [{"action": "integrate", "kernel_id": "a+b", "tput": 110.0}]
    monkeypatch.setattr(integrate_outcomes, "_maybe_apply_kernel_patch", Mock(return_value={"status": "failed"}))
    monkeypatch.setattr(integrate_outcomes, "_maybe_revert_kernel_patch", Mock(return_value={"status": "ok"}))
    monkeypatch.setattr(baseline_mod, "BaselineExecutor", Mock(side_effect=AssertionError("unexpected benchmark")))
    before = deepcopy(c.shared_state.to_dict())
    with pytest.raises(ValueError, match="(?i)stack|member"):
        await c.phase_kernel_stack.maybe_validate_positive_needs_review_stack()
    assert c.shared_state.to_dict() == before


def _materialize_stack_sources(tmp_path: Path, stack: list[dict[str, Any]]) -> None:
    """Give each member a real target file and a whole-file replacement patch."""
    for entry in stack:
        target = tmp_path / f"{entry['kernel_id']}.py"
        patch = tmp_path / f"{entry['kernel_id']}_opt.py"
        target.write_text(_STACK_ORIGINAL_SOURCE, encoding="utf-8")
        patch.write_text(_STACK_PATCHED_SOURCE, encoding="utf-8")
        entry.update(target_file=str(target), patch_path=str(patch))


def _stub_python_cache_clear(monkeypatch) -> None:
    """Keep the real apply away from this machine's Triton / inductor cache directories."""
    from hyperloom.orchestrator.kernel import apply_kernel_patch as akp, patch_restore

    for module in (akp, patch_restore):
        monkeypatch.setattr(module, "clear_python_kernel_caches", lambda target: {"status": "skipped"})


def _stub_stack_benchmark(monkeypatch, *, new_tput: float) -> None:
    """Replace only the E2E measurement; apply, revert and manifests stay real."""
    import hyperloom.orchestrator.actions.executors.baseline as baseline_mod
    import hyperloom.orchestrator.actions.executors.benchmark_result as br

    async def _benchmark(self, ctx):
        return {"output_throughput": new_tput, "workspace": "/tmp/stack-bench"}

    monkeypatch.setattr(baseline_mod.BaselineExecutor, "__call__", _benchmark)
    monkeypatch.setattr(br, "is_valid_measurement", lambda result: True)


def _break_backup_restore(monkeypatch, *, target: Path) -> None:
    """Fail one member's backup->target copy; its apply (patch->target) still succeeds."""
    from hyperloom.orchestrator.kernel import apply_kernel_patch as akp

    # apply_kernel_patch resolves both paths, so the discriminator has to as well.
    patched = target.with_name(f"{target.stem}_opt{target.suffix}").resolve()
    restored = target.resolve()
    real_copy2 = akp.shutil.copy2

    def _copy2(src, dst, *args, **kwargs):
        if Path(dst).resolve() == restored and Path(src).resolve() != patched:
            raise OSError(5, "injected revert failure")
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(akp.shutil, "copy2", _copy2)


def _stack_member_guards(state: SharedState) -> dict[str, bool]:
    """Per-member ``stack_validation_in_progress`` guards, by kernel id."""
    return {
        entry["kernel_id"]: bool(entry.get("stack_validation_in_progress"))
        for entry in state.kernel_integrate_attempts.values()
        if entry.get("kernel_id") in {"k001", "k004"}
    }


def _checkpointed_manifest_statuses(state: SharedState) -> dict[str, str]:
    """Apply-manifest status per member, read off the persisted apply checkpoints."""
    return {
        applied["kernel_id"]: json.loads(Path(applied["manifest_path"]).read_text(encoding="utf-8"))["status"]
        for applied in state.pending_stack_validation_apply_results
    }


def _session_manifest_statuses(tmp_path: Path) -> list[str]:
    """Every apply manifest the session wrote, for the cases that clear their checkpoints."""
    return sorted(
        json.loads(path.read_text(encoding="utf-8"))["status"] for path in tmp_path.glob("patches/**/manifest.json")
    )


def _resumed_stack_coordinator(tmp_path: Path) -> Coordinator:
    """A Coordinator over the session as a later resume would load it back from disk."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState.load_or_init(tmp_path)
    c.phase_kernel_stack._run_kernel_stack_validation_e2e = Mock(
        side_effect=AssertionError("recovery must not re-run the stack benchmark")
    )
    return c


async def _halt_a_stack_revert(tmp_path: Path, monkeypatch) -> Path:
    """Run a stack validation whose REVERT half-fails; return the target left holding its patch."""
    from hyperloom.inference_optimizer.session.session_binding import session_scope

    monkeypatch.setenv("HYPERLOOM_LANGFUSE_ENABLE", "0")
    _stub_python_cache_clear(monkeypatch)
    c = _stack_validation_coordinator(tmp_path)
    stack = _ledger_rows(c, ["k001", "k004"])
    _materialize_stack_sources(tmp_path, stack)
    stuck = Path(next(entry for entry in stack if entry["kernel_id"] == "k001")["target_file"])
    with monkeypatch.context() as mp:
        # 105 clears the 100 baseline but not the 110 current_best, so the stack decides REVERT.
        _stub_stack_benchmark(mp, new_tput=105.0)
        _break_backup_restore(mp, target=stuck)
        with session_scope(tmp_path), pytest.raises(RuntimeError, match="revert incomplete"):
            await c.phase_kernel_stack.maybe_validate_positive_needs_review_stack()
    return stuck


@pytest.mark.asyncio
async def test_stack_revert_failure_retains_checkpoints_and_halts(tmp_path: Path, monkeypatch):
    """An unfinished stack revert halts the session and persists everything a retry needs."""
    from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON

    stuck = await _halt_a_stack_revert(tmp_path, monkeypatch)

    saved = SharedState.load_or_init(tmp_path)
    assert saved.stop_reason == PATCH_RECOVERY_INCOMPLETE_STOP_REASON
    assert saved.pending_stack_validation_result["decision"] == "REVERT"
    assert saved.pending_stack_validation_result["patch_cleanup_action"] == "revert"
    assert len(saved.pending_stack_validation_apply_results) == 2
    assert _stack_member_guards(saved) == {"k001": True, "k004": True}
    assert _checkpointed_manifest_statuses(saved) == {"k001": "applied", "k004": "reverted"}
    assert stuck.read_text(encoding="utf-8") == _STACK_PATCHED_SOURCE
    assert (tmp_path / "k004.py").read_text(encoding="utf-8") == _STACK_ORIGINAL_SOURCE


@pytest.mark.asyncio
async def test_stack_revert_recovery_retries_the_unwind_and_clears(tmp_path: Path, monkeypatch):
    """The next resume retries the teardown; a clean tree returns the members to selectable."""
    from hyperloom.inference_optimizer.session.session_binding import session_scope

    stuck = await _halt_a_stack_revert(tmp_path, monkeypatch)
    c = _resumed_stack_coordinator(tmp_path)

    with session_scope(tmp_path):
        assert await c.phase_kernel_stack.recover_interrupted_stack_validation() is True

    assert stuck.read_text(encoding="utf-8") == _STACK_ORIGINAL_SOURCE
    assert (tmp_path / "k004.py").read_text(encoding="utf-8") == _STACK_ORIGINAL_SOURCE
    assert _session_manifest_statuses(tmp_path) == ["reverted", "reverted"]
    reloaded = SharedState.load_or_init(tmp_path)
    assert not reloaded.pending_stack_validation_result
    assert not reloaded.pending_stack_validation_apply_results
    assert _stack_member_guards(reloaded) == {"k001": False, "k004": False}
    assert {entry["kernel_id"] for entry in c.phase_kernel_stack._positive_needs_review_integrates()} == {
        "k001",
        "k004",
    }


_APPLY_KERNEL_PATCH_FIELDS = (
    "status",
    "manifest_path",
    "target_file",
    "backup_dir",
    "compiled",
    "artifact_count",
    "cache_clear",
    "rebuild",
    "jit_build_backup",
    "cpp_itfs_cache_backup",
)


@pytest.mark.asyncio
async def test_a_record_only_checkpoint_unwinds_instead_of_halting(tmp_path: Path, monkeypatch):
    """Apply rows with no record still say what reached the tree, and that is enough to undo it."""
    from hyperloom.inference_optimizer.session.session_binding import session_scope

    stuck = await _halt_a_stack_revert(tmp_path, monkeypatch)
    state = SharedState.load_or_init(tmp_path)
    state.pending_stack_validation_result = {}
    # Only what apply_kernel_patch itself returns: the unwind must not lean on anything the stack layer adds.
    state.pending_stack_validation_apply_results = [
        {key: row[key] for key in _APPLY_KERNEL_PATCH_FIELDS if key in row}
        for row in state.pending_stack_validation_apply_results
    ]
    state.save(tmp_path)
    c = _resumed_stack_coordinator(tmp_path)
    c.shared_state.set_stop_reason("")
    report: dict[str, Any] = {"fixes": [], "warnings": []}

    with session_scope(tmp_path):
        await c.writeback._resume_recover_interrupted_stack(report)

    assert stuck.read_text(encoding="utf-8") == _STACK_ORIGINAL_SOURCE
    assert c.shared_state.stop_reason == ""
    assert [f["kind"] for f in report["fixes"]] == ["interrupted_stack_validation_recovered"]
    reloaded = SharedState.load_or_init(tmp_path)
    assert not reloaded.pending_stack_validation_apply_results
    assert _stack_member_guards(reloaded) == {"k001": False, "k004": False}


@pytest.mark.asyncio
async def test_a_cycle_end_entry_that_settles_an_owed_unwind_still_validates(tmp_path: Path, monkeypatch):
    """The members an unwind frees are validated in the same CYCLE_END entry.

    CYCLE_END can exit straight to CLOSE, which never validates, so a validation
    deferred to the next entry may never run.
    """
    from hyperloom.inference_optimizer.session.session_binding import session_scope

    stuck = await _halt_a_stack_revert(tmp_path, monkeypatch)
    c = _resumed_stack_coordinator(tmp_path)
    del c.phase_kernel_stack._run_kernel_stack_validation_e2e
    c.tasks = _StubTaskRegistry()
    c.knowledge_plane = None
    c.orch_prompt = OrchestrationPrompt({}, is_user_supplied=True)
    _stub_stack_benchmark(monkeypatch, new_tput=105.0)

    with session_scope(tmp_path):
        await c.phase_cycle_end.on_enter_cycle_end(
            Transition(from_phase="KERNEL_AGENT", to_phase="CYCLE_END", reason="test", evidence={})
        )

    assert stuck.read_text(encoding="utf-8") == _STACK_ORIGINAL_SOURCE
    stack_rows = [row for row in c.shared_state.kernel_integrate_attempts.values() if row["kernel_id"] == "k004+k001"]
    assert [row["attempt_count"] for row in stack_rows] == [2]
    assert {entry["kernel_id"] for entry in c.phase_kernel_stack._positive_needs_review_integrates()} == {
        "k001",
        "k004",
    }


@pytest.mark.asyncio
async def test_a_guard_only_attempt_recovers_and_frees_its_members(tmp_path: Path, monkeypatch):
    """A guard with no record and no apply row still names an interrupted attempt.

    An attempt that cleared the record when it marked the rows leaves only the
    guards behind if it crashes before its first apply checkpoint. Nothing
    reached the tree, so recovery releases the members.
    """
    monkeypatch.setenv("HYPERLOOM_LANGFUSE_ENABLE", "0")
    c = _stack_validation_coordinator(tmp_path)
    for row in _ledger_rows(c, ["k001", "k004"]):
        row["stack_validation_in_progress"] = True
    c.shared_state.save(tmp_path)
    c = _resumed_stack_coordinator(tmp_path)
    report: dict[str, Any] = {"fixes": [], "warnings": []}

    await c.writeback._resume_recover_interrupted_stack(report)

    assert [f["kind"] for f in report["fixes"]] == ["interrupted_stack_validation_recovered"]
    assert _stack_member_guards(SharedState.load_or_init(tmp_path)) == {"k001": False, "k004": False}
    assert {entry["kernel_id"] for entry in c.phase_kernel_stack._positive_needs_review_integrates()} == {
        "k001",
        "k004",
    }


def _drop_attempt_evidence(state: SharedState) -> None:
    """Strip the member identities that bind the checkpoint to its ledger rows, keeping the apply rows."""
    del state.pending_stack_validation_result["stack_member_identities"]


def _duplicate_a_member_row(state: SharedState) -> None:
    """Copy a member's ledger row whole, so the record's identities no longer single one row out."""
    rows = state.kernel_integrate_attempts
    rows["k001-second-attempt"] = deepcopy(next(row for row in rows.values() if row["kernel_id"] == "k001"))


async def _halt_a_stack_keep(tmp_path: Path, monkeypatch) -> list[Path]:
    """Decide KEEP, then die on the way to the stack; return the members' target files."""
    from hyperloom.inference_optimizer.session.session_binding import session_scope

    monkeypatch.setenv("HYPERLOOM_LANGFUSE_ENABLE", "0")
    _stub_python_cache_clear(monkeypatch)
    c = _stack_validation_coordinator(tmp_path)
    stack = _ledger_rows(c, ["k001", "k004"])
    _materialize_stack_sources(tmp_path, stack)
    with monkeypatch.context() as mp:
        # Well clear of the 110 current_best, so the stack decides KEEP.
        _stub_stack_benchmark(mp, new_tput=140.0)
        c.writeback.record_integrate_keep = AsyncMock(side_effect=RuntimeError("crashed before promoting"))
        with session_scope(tmp_path), pytest.raises(RuntimeError, match="crashed before promoting"):
            await c.phase_kernel_stack.maybe_validate_positive_needs_review_stack()
    return [Path(entry["target_file"]) for entry in stack]


async def _crash_after_a_stack_revert(tmp_path: Path, monkeypatch) -> None:
    """Decide REVERT and finish the revert, then die before the checkpoint is cleared."""
    from hyperloom.inference_optimizer.session.session_binding import session_scope

    monkeypatch.setenv("HYPERLOOM_LANGFUSE_ENABLE", "0")
    _stub_python_cache_clear(monkeypatch)
    c = _stack_validation_coordinator(tmp_path)
    stack = _ledger_rows(c, ["k001", "k004"])
    _materialize_stack_sources(tmp_path, stack)
    with monkeypatch.context() as mp:
        # 105 clears the 100 baseline but not the 110 current_best, so the stack decides REVERT.
        _stub_stack_benchmark(mp, new_tput=105.0)
        c.phase_kernel_stack._finalize_stack_validation_outcome = AsyncMock(side_effect=RuntimeError("crashed"))
        with session_scope(tmp_path), pytest.raises(RuntimeError, match="crashed"):
            await c.phase_kernel_stack.maybe_validate_positive_needs_review_stack()


@pytest.mark.asyncio
async def test_a_settled_revert_that_names_no_members_still_recovers(tmp_path: Path, monkeypatch):
    """A finished REVERT promotes nothing and left the tree clean, so it settles without binding members."""
    from hyperloom.inference_optimizer.session.session_binding import session_scope

    await _crash_after_a_stack_revert(tmp_path, monkeypatch)
    c = _resumed_stack_coordinator(tmp_path)
    c.shared_state.set_stop_reason("")
    _drop_attempt_evidence(c.shared_state)
    report: dict[str, Any] = {"fixes": [], "warnings": []}

    with session_scope(tmp_path):
        await c.writeback._resume_recover_interrupted_stack(report)

    assert c.shared_state.stop_reason == ""
    assert [f["kind"] for f in report["fixes"]] == ["interrupted_stack_validation_recovered"]
    assert all(
        (tmp_path / f"{kid}.py").read_text(encoding="utf-8") == _STACK_ORIGINAL_SOURCE for kid in ("k001", "k004")
    )
    reloaded = SharedState.load_or_init(tmp_path)
    assert not reloaded.pending_stack_validation_result
    assert not reloaded.pending_stack_validation_apply_results
    assert _stack_member_guards(reloaded) == {"k001": False, "k004": False}


async def _enter_at_resume(c: Coordinator, report: dict[str, Any]) -> None:
    await c.writeback._resume_recover_interrupted_stack(report)


async def _enter_at_cycle_end(c: Coordinator, report: dict[str, Any]) -> None:
    c.tasks = _StubTaskRegistry()
    c.knowledge_plane = None
    await c.phase_cycle_end.on_enter_cycle_end(
        Transition(from_phase="KERNEL_AGENT", to_phase="CYCLE_END", reason="test", evidence={})
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("enter", [_enter_at_resume, _enter_at_cycle_end], ids=["resume", "cycle_end_entry"])
@pytest.mark.parametrize(
    "corrupt", [_drop_attempt_evidence, _duplicate_a_member_row], ids=["no_identities", "duplicate_row"]
)
async def test_an_unbindable_keep_records_the_halt_then_raises(tmp_path: Path, monkeypatch, corrupt, enter):
    """A KEEP has finalized its patches with no backup left, so a record bound to no single row can only halt.

    The stop reason is durable before the raise, from either entry, and nothing is promoted or marked resolved.
    """
    from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON
    from hyperloom.inference_optimizer.session.session_binding import session_scope

    targets = await _halt_a_stack_keep(tmp_path, monkeypatch)
    c = _resumed_stack_coordinator(tmp_path)
    c.shared_state.set_stop_reason("")
    corrupt(c.shared_state)
    report: dict[str, Any] = {"fixes": [], "warnings": []}

    with session_scope(tmp_path), pytest.raises(ValueError, match="(?i)stack|member"):
        await enter(c, report)

    assert report["fixes"] == []
    assert all(path.read_text(encoding="utf-8") == _STACK_PATCHED_SOURCE for path in targets)
    reloaded = SharedState.load_or_init(tmp_path)
    assert reloaded.stop_reason == PATCH_RECOVERY_INCOMPLETE_STOP_REASON
    assert not any(item.get("stack_validation") for item in reloaded.optimization_stack)
    assert reloaded.pending_stack_validation_result["decision"] == "KEEP"
    assert not any(entry.get("stack_resolved") for entry in reloaded.kernel_integrate_attempts.values())


@pytest.mark.asyncio
async def test_a_stack_row_that_names_no_members_fails_the_resume_pass(tmp_path: Path, monkeypatch):
    """The persisted stack is bound to its members once, where the state file is loaded back.

    Every reader below derives kept kernel ids from those rows, so a row that
    cannot say what it integrated is a corrupt state file rather than a finding
    for whichever hot path reaches it first. The stop reason is durable before the raise.
    """
    from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON

    monkeypatch.setenv("HYPERLOOM_LANGFUSE_ENABLE", "0")
    c = _stack_validation_coordinator(tmp_path)
    c.shared_state.optimization_stack = [{"action": "integrate", "kernel_id": "k001+k004", "tput": 120.0}]
    c.writeback._resumed_from = {"is_resume": True, "rebuilt": True}

    with pytest.raises(ValueError, match="(?i)stack|member"):
        await c.writeback._resume_consistency_pass()

    assert SharedState.load_or_init(tmp_path).stop_reason == PATCH_RECOVERY_INCOMPLETE_STOP_REASON


@pytest.mark.asyncio
async def test_stack_revert_recovery_that_fails_again_halts_again(tmp_path: Path, monkeypatch):
    """A teardown that fails on the retry too keeps its checkpoints and asks for a human."""
    from hyperloom.inference_optimizer.session.session_binding import session_scope
    from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON

    stuck = await _halt_a_stack_revert(tmp_path, monkeypatch)
    c = _resumed_stack_coordinator(tmp_path)
    c.shared_state.set_stop_reason("")
    _break_backup_restore(monkeypatch, target=stuck)

    with session_scope(tmp_path), pytest.raises(RuntimeError, match="revert incomplete"):
        await c.phase_kernel_stack.recover_interrupted_stack_validation()

    assert stuck.read_text(encoding="utf-8") == _STACK_PATCHED_SOURCE
    reloaded = SharedState.load_or_init(tmp_path)
    assert reloaded.stop_reason == PATCH_RECOVERY_INCOMPLETE_STOP_REASON
    assert reloaded.pending_stack_validation_result["patch_cleanup_action"] == "revert"
    assert len(reloaded.pending_stack_validation_apply_results) == 2
    assert _stack_member_guards(reloaded) == {"k001": True, "k004": True}


@pytest.mark.asyncio
async def test_stack_revert_success_clears_checkpoints(tmp_path: Path, monkeypatch):
    """The same real path with nothing injected still clears the guards and the checkpoints."""
    from hyperloom.inference_optimizer.session.session_binding import session_scope

    monkeypatch.setenv("HYPERLOOM_LANGFUSE_ENABLE", "0")
    _stub_python_cache_clear(monkeypatch)
    _stub_stack_benchmark(monkeypatch, new_tput=105.0)
    c = _stack_validation_coordinator(tmp_path)
    stack = _ledger_rows(c, ["k001", "k004"])
    _materialize_stack_sources(tmp_path, stack)

    with session_scope(tmp_path):
        await c.phase_kernel_stack.maybe_validate_positive_needs_review_stack()

    assert all(
        (tmp_path / f"{kid}.py").read_text(encoding="utf-8") == _STACK_ORIGINAL_SOURCE for kid in ("k001", "k004")
    )
    assert _session_manifest_statuses(tmp_path) == ["reverted", "reverted"]
    saved = SharedState.load_or_init(tmp_path)
    assert saved.stop_reason == ""
    assert not saved.pending_stack_validation_result
    assert not saved.pending_stack_validation_apply_results
    assert _stack_member_guards(saved) == {"k001": False, "k004": False}


@pytest.mark.asyncio
async def test_the_resume_pass_retries_the_unwind_before_anything_benchmarks(tmp_path: Path, monkeypatch):
    """The halt promises the next resume retries the teardown.

    Stack recovery used to run only at CYCLE_END entry, so everything a resumed leg
    measured before reaching CYCLE_END measured the still-patched tree.
    """
    from hyperloom.inference_optimizer.session.session_binding import session_scope

    stuck = await _halt_a_stack_revert(tmp_path, monkeypatch)
    c = _resumed_stack_coordinator(tmp_path)
    c.writeback._resumed_from = {"is_resume": True}
    report: dict[str, Any] = {"skipped": False, "fixes": [], "warnings": []}

    with session_scope(tmp_path):
        await c.writeback._resume_recover_interrupted_stack(report)

    assert {"kind": "interrupted_stack_validation_recovered"} in report["fixes"]
    assert stuck.read_text(encoding="utf-8") == _STACK_ORIGINAL_SOURCE
    assert _session_manifest_statuses(tmp_path) == ["reverted", "reverted"]


@pytest.mark.asyncio
async def test_a_settled_session_does_not_pay_the_stack_recovery(tmp_path: Path, monkeypatch):
    """Nothing pending means nothing to unwind; the resume pass must not report a fix."""
    from hyperloom.inference_optimizer.session.session_binding import session_scope

    monkeypatch.setenv("HYPERLOOM_LANGFUSE_ENABLE", "0")
    c = _stack_validation_coordinator(tmp_path)
    c.writeback._resumed_from = {"is_resume": True}
    report: dict[str, Any] = {"skipped": False, "fixes": [], "warnings": []}

    with session_scope(tmp_path):
        await c.writeback._resume_recover_interrupted_stack(report)

    assert report["fixes"] == []


def test_close_neither_rebenches_nor_profiles_a_tree_it_refused_to_trust():
    """CLOSE re-benches an unvalidated stack and runs a post-opt roofline on the live tree.

    Under this stop reason that tree is the one the halt declared untrustworthy.
    """
    from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON
    from hyperloom.orchestrator.phases import close as close_phase

    assert PATCH_RECOVERY_INCOMPLETE_STOP_REASON in close_phase._NO_REVALIDATION_STOP_REASONS

    seen: list[str] = []

    async def _enqueue(**_kw):
        seen.append("enqueued")
        return

    phase = close_phase.ClosePhase.__new__(close_phase.ClosePhase)
    object.__setattr__(
        phase,
        "_coord",
        SimpleNamespace(
            shared_state=SimpleNamespace(
                closing_phase=False,
                stop_reason=PATCH_RECOVERY_INCOMPLETE_STOP_REASON,
                optimization_stack=[{"action": "integrate"}],
                phase_history=[],
                save=lambda _: None,
            ),
            session_dir=None,
            phase_prelude=SimpleNamespace(
                internal_analysis_kind=lambda: seen.append("analysis_kind") or "roofline",
                enqueue_internal_analysis_task=_enqueue,
            ),
        ),
    )

    asyncio.run(phase._maybe_run_close_post_opt_roofline())

    assert seen == []
    assert not any("roofline" in str(s) for s in seen)


def test_a_revert_that_already_completed_is_not_run_again(tmp_path: Path):
    """The first revert moves each backup back, so a second would fail on a clean tree.

    An apply that reverted itself and is then unwound by the stack hits exactly that.
    """
    from hyperloom.orchestrator.kernel.apply_kernel_patch import revert_kernel_patch

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "reverted",
                "reverted_at": "2026-01-01T00:00:00Z",
                "restored_paths": ["/framework/src.py"],
                # Present and "ok" is what drives the second restore attempt.
                "jit_build_backup": {"status": "ok", "backup_path": str(tmp_path / "gone")},
                "artifacts": [{"backup_path": str(tmp_path / "missing.bak"), "path": str(tmp_path / "missing.py")}],
            }
        ),
        encoding="utf-8",
    )

    result = revert_kernel_patch(manifest)

    assert result["status"] == "skipped"
    assert result["already_reverted"] is True
    assert result["restored_paths"] == ["/framework/src.py"]
    assert lifecycle_complete(result)
