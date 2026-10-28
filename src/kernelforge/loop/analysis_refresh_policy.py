# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic refresh policy for commit-bound Analysis evidence."""

from __future__ import annotations

from dataclasses import dataclass

from kernelforge.loop.run_state import AnalysisRefreshState
from kernelforge.orchestrator.contracts import calculate_evidence_gain


ANALYSIS_REFRESH_THRESHOLD = 0.05
# (refresh, reason) for a commit already attempted, by how that attempt ended.
_ATTEMPTED_COMMIT_OUTCOMES = {
    "exhausted": (False, "ANALYSIS_ATTEMPTS_EXHAUSTED"),
    "failed": (True, "RETRY_FAILED_ANALYSIS"),
}


@dataclass(frozen=True)
class AnalysisRefreshDecision:
    """One auditable decision to refresh or reuse Analysis evidence."""

    refresh: bool
    reasons: tuple[str, ...]
    evidence_stale: bool
    gain_since_evidence: float | None


def decide_analysis_refresh(
    state: AnalysisRefreshState,
    *,
    canonical_commit: str,
    current_mean_case_speedup: float | None,
    supervisor_due: bool,
    current_iteration: int = 0,
) -> AnalysisRefreshDecision:
    """Decide whether the current canonical commit needs fresh Analysis."""
    canonical = canonical_commit.strip()
    evidence = str(state.evidence_commit or "").strip()
    stale = bool(evidence) and canonical != evidence
    gain = calculate_evidence_gain(state.evidence_mean_case_speedup, current_mean_case_speedup)

    if state.last_attempt_commit == canonical:
        outcome = _ATTEMPTED_COMMIT_OUTCOMES.get(state.last_attempt_status)
        if state.last_attempt_iteration == current_iteration:
            outcome = (False, "ALREADY_ATTEMPTED_THIS_ITERATION")
        if outcome is not None:
            refresh, reason = outcome
            return AnalysisRefreshDecision(
                refresh=refresh,
                reasons=(reason,),
                evidence_stale=stale or not evidence,
                gain_since_evidence=gain,
            )

    status = str(state.evidence_status or "").strip().lower()
    if evidence == canonical and status == "partial" and state.last_attempt_iteration < current_iteration:
        return AnalysisRefreshDecision(
            refresh=True,
            reasons=("PARTIAL_UPGRADE",),
            evidence_stale=False,
            gain_since_evidence=gain,
        )

    if not evidence or state.evidence_mean_case_speedup is None:
        return AnalysisRefreshDecision(
            refresh=True,
            reasons=("INITIAL_ANALYSIS",),
            evidence_stale=stale,
            gain_since_evidence=gain,
        )

    reasons: list[str] = []
    if stale and gain is not None and gain + 1e-12 >= ANALYSIS_REFRESH_THRESHOLD:
        reasons.append("CUMULATIVE_GAIN")
    if stale and supervisor_due:
        reasons.append("SUPERVISOR_STALE_EVIDENCE")
    if reasons:
        return AnalysisRefreshDecision(
            refresh=True,
            reasons=tuple(reasons),
            evidence_stale=True,
            gain_since_evidence=gain,
        )

    return AnalysisRefreshDecision(
        refresh=False,
        reasons=(("CURRENT_EVIDENCE",) if not stale else ("CUMULATIVE_GAIN_BELOW_THRESHOLD",)),
        evidence_stale=stale,
        gain_since_evidence=gain,
    )
