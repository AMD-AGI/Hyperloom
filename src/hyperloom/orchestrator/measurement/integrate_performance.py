# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared native integration performance decision, before accuracy and runtime gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from hyperloom.common.gain_math import gain_pct_or_zero, incremental_gain_pct
from hyperloom.common.perf_metric import GradedComparison
from ..state.shared_state import resolve_graded_comparison


@dataclass(frozen=True)
class IntegratePerformance:
    """The performance evidence used to schedule accuracy and evaluate a KEEP."""

    graded: GradedComparison
    gain_pct: float
    stack_incremental_gain_pct: float | None
    stack_positive_keep: bool
    decision: str


def assess_integrate_performance(
    state: Any,
    measurement: Mapping[str, Any],
    *,
    base_tput: float,
    keep_threshold_pct: float,
    stack_incremental_keep_threshold_pct: float,
) -> IntegratePerformance:
    """Apply native KEEP thresholds on the shared grader's selected measurement axis."""
    graded = resolve_graded_comparison(state, measurement)
    new_tput = float(measurement.get("output_throughput") or 0.0)
    gain_pct = (
        gain_pct_or_zero(graded.candidate, graded.reference)
        if graded.graded_on_total
        else gain_pct_or_zero(new_tput, base_tput)
    )
    current_best = getattr(state, "current_best", None) or {}
    current_best_tput = float(current_best.get("tput") or 0.0)
    stack_incremental_gain_pct = (
        incremental_gain_pct(graded.candidate, graded.reference)
        if graded.graded_on_total
        else incremental_gain_pct(new_tput, current_best_tput)
    )
    stack_positive_keep = (
        graded.comparable
        and not graded.vetoed
        and bool(getattr(state, "optimization_stack", None))
        and str(current_best.get("action") or "") == "integrate"
        and stack_incremental_gain_pct is not None
        and stack_incremental_gain_pct >= stack_incremental_keep_threshold_pct
    )
    if not graded.comparable:
        decision = "NEEDS_REVIEW"
    else:
        decision = (
            "KEEP"
            if not graded.vetoed and (gain_pct > keep_threshold_pct or stack_positive_keep)
            else ("REVERT" if gain_pct < -keep_threshold_pct else "NEEDS_REVIEW")
        )
    return IntegratePerformance(
        graded=graded,
        gain_pct=gain_pct,
        stack_incremental_gain_pct=stack_incremental_gain_pct,
        stack_positive_keep=stack_positive_keep,
        decision=decision,
    )
