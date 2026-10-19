# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-axis grading of a candidate measurement, behind ``shared_state.resolve_graded_comparison``."""

from __future__ import annotations

from typing import Any, NamedTuple

from hyperloom.common.gain_math import gain_pct
from hyperloom.common.perf_metric import (
    AGENTX_KEEP_P50_THRESHOLD_PCT,
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_OUTPUT,
    GradedComparison,
    VERDICT_KEEP,
    VERDICT_REVERT,
    axis_of,
    constraint_veto_reason,
    holds_within_band,
    incomparability_reason,
    output_tput_of,
    perf_snapshot_from_mapping,
    resolve_grading_anchor_perf,
    total_tput_of,
)

__all__ = ["InteractivityOutcome", "grade_interactivity", "grade_output_throughput"]


class InteractivityOutcome(NamedTuple):
    """What the interactivity axis resolved to: the graded pair, or why that pair could not be graded at all."""

    comparison: GradedComparison | None
    degrade_reason: str


def _interactivity_reference(state: Any, *, against_baseline: bool, anchor_perf: Any) -> tuple[Any, str]:
    """Resolve the snapshot a candidate is graded against, and which source failed to supply one."""
    if anchor_perf is not None:
        return anchor_perf, ""
    if against_baseline:
        ref_perf = perf_snapshot_from_mapping(getattr(state, "baseline_perf", None))
        return ref_perf, "" if ref_perf else "baseline_axes_missing"
    return resolve_grading_anchor_perf(state)


def _interactivity_refusals(candidate: Any, reference: Any, *, noise_pct: float | None) -> list[str]:
    """Name every gate the pair fails: median gain, slow tail, output throughput, and comparability."""
    gain = gain_pct(axis_of(candidate, GRADED_INTVTY_P50), axis_of(reference, GRADED_INTVTY_P50))
    # The output guard reads the raw aggregate: the chip count divides both sides of the ratio, so the band
    # holds identically per GPU.
    gates = (
        ("p50_gain_below_threshold", gain is not None and gain >= AGENTX_KEEP_P50_THRESHOLD_PCT),
        ("p90_regressed", holds_within_band(candidate, reference, GRADED_INTVTY, noise_pct=noise_pct)),
        ("output_regressed", holds_within_band(candidate, reference, GRADED_OUTPUT, noise_pct=noise_pct)),
    )
    refused = [gate for gate, held in gates if not held]
    incomparable = incomparability_reason(candidate, reference)
    if incomparable:
        refused.append(incomparable)
    return refused


def grade_interactivity(
    state: Any,
    measurement: Any,
    *,
    against_baseline: bool,
    anchor_perf: Any,
    noise_pct: float | None,
) -> InteractivityOutcome:
    """Grade a candidate on median interactivity, guarded by the slow tail, output throughput and the session's constraints.

    Returns:
        The graded pair, or a degrade reason when either side cannot supply the
        interactivity axes.
    """
    ref_perf, reason = _interactivity_reference(state, against_baseline=against_baseline, anchor_perf=anchor_perf)
    cand_perf = perf_snapshot_from_mapping(measurement)
    if not (ref_perf and cand_perf):
        return InteractivityOutcome(None, reason or "candidate_axes_missing")
    refused = _interactivity_refusals(cand_perf, ref_perf, noise_pct=noise_pct)
    # A candidate that already lost carries no veto, so the ledger names the gate that actually refused it.
    sla_veto = "" if refused else constraint_veto_reason(measurement, state)
    return InteractivityOutcome(
        GradedComparison(
            objective=GRADED_INTVTY_P50,
            candidate=axis_of(cand_perf, GRADED_INTVTY_P50),
            reference=axis_of(ref_perf, GRADED_INTVTY_P50),
            verdict=VERDICT_REVERT if refused or sla_veto else VERDICT_KEEP,
            tput_candidate=total_tput_of(cand_perf),
            tput_reference=total_tput_of(ref_perf),
            veto_reason=sla_veto,
            refused_by=",".join(refused),
        ),
        "",
    )


def grade_output_throughput(
    state: Any,
    measurement: Any,
    *,
    reference: float,
    keep_threshold_pct: float,
    degrade_reason: str,
) -> GradedComparison:
    """Grade a candidate on output throughput against *reference*, vetoed by the session's constraints.

    A non-empty ``degrade_reason`` comes from a pair that could not be graded on
    interactivity: its output figures are diagnostic only and must not read as a
    throughput KEEP at this chokepoint.

    Returns:
        The graded pair on the output axis.
    """
    candidate = output_tput_of(measurement)
    gain = gain_pct(candidate, reference) if reference > 0 else None
    cleared = gain is not None and gain >= keep_threshold_pct
    verdict = VERDICT_KEEP if cleared and not degrade_reason else VERDICT_REVERT
    sla_veto = constraint_veto_reason(measurement, state) if verdict == VERDICT_KEEP else ""
    return GradedComparison(
        objective=GRADED_OUTPUT,
        candidate=candidate,
        reference=reference,
        verdict=VERDICT_REVERT if sla_veto else verdict,
        degrade_reason=degrade_reason,
        veto_reason=sla_veto,
    )
