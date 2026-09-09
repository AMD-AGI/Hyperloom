# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared native integration performance decision, before accuracy and runtime gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from hyperloom.common.gain_math import gain_pct_or_zero, incremental_gain_pct
from hyperloom.common.perf_metric import VERDICT_KEEP, VERDICT_REVERT, GradedComparison
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
    # The threshold goes into the chokepoint rather than being re-applied here: on the interactivity axis the
    # chokepoint raises it to the AgentX floor and pairs it with the throughput guard, and a lane that graded the
    # gain itself would promote points the 2-D rule only RECORDED.
    graded = resolve_graded_comparison(state, measurement, keep_threshold_pct=keep_threshold_pct)
    new_tput = float(measurement.get("output_throughput") or 0.0)
    gain_pct = (
        gain_pct_or_zero(graded.candidate, graded.reference)
        if graded.graded_on_intvty
        else gain_pct_or_zero(new_tput, base_tput)
    )
    current_best = getattr(state, "current_best", None) or {}
    current_best_tput = float(current_best.get("tput") or 0.0)
    stack_incremental_gain_pct = (
        incremental_gain_pct(graded.candidate, graded.reference)
        if graded.graded_on_intvty
        else incremental_gain_pct(new_tput, current_best_tput)
    )
    # The lowered stack threshold is an output-axis concession. On the interactivity axis the verdict already owns
    # the KEEP, so honouring it here would let a stacked layer clear a bar the frontier rule rejected.
    stack_positive_keep = (
        graded.comparable
        and not graded.graded_on_intvty
        and bool(getattr(state, "optimization_stack", None))
        and str(current_best.get("action") or "") == "integrate"
        and stack_incremental_gain_pct is not None
        and stack_incremental_gain_pct >= stack_incremental_keep_threshold_pct
    )
    if not graded.comparable:
        # Fail closed: the axis the session asked for did not apply, so the output figure is a diagnostic, not a
        # verdict, and promoting or discarding a native integration on it is a call for a human.
        decision = "NEEDS_REVIEW"
    elif graded.graded_on_intvty:
        # RECORDED is a different point on the frontier, not a dominated one: it neither promotes nor reverts.
        decision = (
            "KEEP"
            if graded.verdict == VERDICT_KEEP
            else ("REVERT" if graded.verdict == VERDICT_REVERT else "NEEDS_REVIEW")
        )
    else:
        decision = (
            "KEEP"
            if gain_pct > keep_threshold_pct or stack_positive_keep
            else ("REVERT" if gain_pct < -keep_threshold_pct else "NEEDS_REVIEW")
        )
    return IntegratePerformance(
        graded=graded,
        gain_pct=gain_pct,
        stack_incremental_gain_pct=stack_incremental_gain_pct,
        stack_positive_keep=stack_positive_keep,
        decision=decision,
    )
