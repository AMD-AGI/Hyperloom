# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK search termination must not be inferred from its performance verdict."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.geak_completion import search_termination
from hyperloom.orchestrator.phases import machine_state as ms


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
def test_geak_exit_does_not_claim_exhausted_search(monkeypatch, status, termination, reason):
    monkeypatch.setattr(ms, "kernel_work_pending", lambda state: False)
    monkeypatch.setattr(ms, "_pending_escalate_hint", lambda state: ms.ESCALATE_HINT_SKIP_TO_SWEEP)
    state = SimpleNamespace(kernel_optimizer="geak", geak_result={"status": status, "search_termination": termination})
    actual_reason, evidence = ms.exit_normal_kernel(state)
    assert actual_reason == reason
    assert evidence["geak_status"] == status
    assert evidence["search_termination"] == search_termination(state.geak_result)


def test_cutoff_does_not_bypass_pending_revalidation(monkeypatch):
    monkeypatch.setattr(ms, "kernel_work_pending", lambda state: True)
    monkeypatch.setattr(ms, "_pending_escalate_hint", lambda state: ms.ESCALATE_HINT_SKIP_TO_SWEEP)
    monkeypatch.setattr(ms, "phase_budget_remaining_seconds", lambda *args, **kwargs: 9999)
    monkeypatch.setattr(ms, "phase_cap_exceeded", lambda *args, **kwargs: False)
    state = SimpleNamespace(
        kernel_optimizer="geak",
        phase="KERNEL_AGENT",
        geak_result={"status": "ok", "search_termination": {"reason": "dispatch_cutoff"}},
    )
    assert ms.exit_normal_kernel(state) is None


def test_non_geak_exit_does_not_use_old_geak_result(monkeypatch):
    monkeypatch.setattr(ms, "kernel_work_pending", lambda state: False)
    monkeypatch.setattr(ms, "_pending_escalate_hint", lambda state: ms.ESCALATE_HINT_SKIP_TO_SWEEP)
    state = SimpleNamespace(kernel_optimizer="forge", geak_result={"search_termination": {"reason": "dispatch_cutoff"}})
    assert ms.exit_normal_kernel(state)[0] == "kernel_no_more_leverage"


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
