# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK search termination must not be inferred from its performance verdict."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.geak_completion import search_termination
from hyperloom.orchestrator.phases import machine_state as ms
from hyperloom.orchestrator.phases.machine_state import PHASE_SWEEP, compute_next_phase
from hyperloom.orchestrator.state.shared_state import ESCALATE_HINT_SKIP_TO_SWEEP, SharedState


def _kernel_agent_state(**overrides) -> SharedState:
    state = SharedState(
        phase="KERNEL_AGENT",
        phase_started_unix=0.0,
        max_minutes=0,
        phase_budget_pct={},
        macro_cycle=0,
        attempts=[],
        specialist_rounds=[],
        params_no_promote_streak=0,
        rejected_kernel_ids=[],
        optimization_stack=[],
        pending_escalate_hint=ESCALATE_HINT_SKIP_TO_SWEEP,
        stop_reason="",
        plateau_overrides={},
        framework_agent_phase_done=False,
        kernel_optimizer="",
    )
    for name, value in overrides.items():
        setattr(state, name, value)
    return state


@pytest.mark.parametrize("status", ["ok", "no_gain"])
@pytest.mark.parametrize(
    ("termination", "reason"),
    [
        ({"reason": "dispatch_cutoff", "budget_s": 14282, "remaining_s": 5400}, "kernel_geak_dispatch_cutoff"),
        ({"reason": "completed"}, "kernel_geak_completed"),
        (None, "kernel_geak_completion_unknown"),
        ({"reason": "future_schema_value"}, "kernel_geak_completion_unknown"),
    ],
)
def test_geak_exit_does_not_claim_exhausted_search(status, termination, reason):
    state = _kernel_agent_state(
        kernel_optimizer="geak",
        geak_result={"status": status, "search_termination": termination},
        untried_hot_reusable_kernels=lambda: [],
    )
    out = compute_next_phase(state, kernel_enabled=True)
    assert out is not None
    target, actual_reason, evidence = out
    assert target == PHASE_SWEEP
    assert actual_reason == reason
    assert evidence["geak_status"] == status
    assert evidence["search_termination"] == search_termination(state.geak_result)


def test_cutoff_does_not_bypass_pending_revalidation(monkeypatch):
    # A GEAK terminal status (here "ok") otherwise short-circuits kernel_work_pending
    # (see _geak_phase_terminal), so this forces the pending branch directly to assert
    # the check order: pending work always wins over a GEAK dispatch_cutoff reason.
    monkeypatch.setattr(ms, "kernel_work_pending", lambda state: True)
    state = _kernel_agent_state(
        kernel_optimizer="geak",
        geak_result={"status": "ok", "search_termination": {"reason": "dispatch_cutoff"}},
    )
    assert compute_next_phase(state, kernel_enabled=True) is None


def test_non_geak_exit_does_not_use_old_geak_result():
    state = _kernel_agent_state(
        kernel_optimizer="forge",
        geak_result={"search_termination": {"reason": "dispatch_cutoff"}},
        untried_hot_reusable_kernels=lambda: [],
    )
    out = compute_next_phase(state, kernel_enabled=True)
    assert out is not None
    _target, reason, _evidence = out
    assert reason == "kernel_no_more_leverage"


def test_termination_does_not_infer_cutoff_from_short_runtime_or_report():
    assert search_termination(
        {"status": "ok", "elapsed_s": 1, "runner_timeout_s": 10000, "report": "wall-clock ended"}
    ) == {"reason": "unknown"}


def test_malformed_timing_is_not_published():
    assert search_termination(
        {
            "search_termination": {
                "reason": "dispatch_cutoff",
                "budget_s": float("nan"),
                "remaining_s": -1,
                "elapsed_s": True,
                "dispatch_cutoff_s": "unknown",
            }
        }
    ) == {"reason": "dispatch_cutoff"}
