# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic refresh policy for commit-bound Analysis evidence."""

from __future__ import annotations

from dataclasses import dataclass

from kernelforge.orchestrator.contracts import calculate_evidence_gain


ANALYSIS_REFRESH_THRESHOLD = 0.05


@dataclass(frozen=True)
class AnalysisRefreshDecision:
    """One auditable decision to refresh or reuse Analysis evidence."""

    refresh: bool
    reasons: tuple[str, ...]
    evidence_stale: bool
    gain_since_evidence: float | None


def decide_analysis_refresh(
    *,
    canonical_commit: str,
    evidence_commit: str,
    evidence_mean_case_speedup: float | None,
    evidence_status: str,
    current_mean_case_speedup: float | None,
    supervisor_due: bool,
    last_attempt_commit: str = "",
    last_attempt_status: str = "",
    last_attempt_iteration: int = -1,
    current_iteration: int = 0,
) -> AnalysisRefreshDecision:
    """Decide whether the current canonical commit needs fresh Analysis.

    The performance gate is cumulative from the commit that produced the
    currently active evidence. A Supervisor intervention bypasses that gate,
    but only when the evidence is stale. The Analysis service owns its durable
    two-session attempt budget; this policy only prevents duplicate calls in
    one planning iteration.
    """

    canonical = str(canonical_commit or "").strip()
    evidence = str(evidence_commit or "").strip()
    stale = bool(canonical and evidence and canonical != evidence)
    gain = calculate_evidence_gain(
        evidence_mean_case_speedup,
        current_mean_case_speedup,
    )

    attempted_this_iteration = (
        canonical and last_attempt_commit == canonical and last_attempt_iteration == current_iteration
    )
    if attempted_this_iteration:
        return AnalysisRefreshDecision(
            refresh=False,
            reasons=("ALREADY_ATTEMPTED_THIS_ITERATION",),
            evidence_stale=stale or not evidence,
            gain_since_evidence=gain,
        )

    if canonical and last_attempt_commit == canonical and last_attempt_status == "exhausted":
        return AnalysisRefreshDecision(
            refresh=False,
            reasons=("ANALYSIS_ATTEMPTS_EXHAUSTED",),
            evidence_stale=stale or not evidence,
            gain_since_evidence=gain,
        )

    if canonical and last_attempt_commit == canonical and last_attempt_status == "failed":
        return AnalysisRefreshDecision(
            refresh=True,
            reasons=("RETRY_FAILED_ANALYSIS",),
            evidence_stale=stale or not evidence,
            gain_since_evidence=gain,
        )

    status = str(evidence_status or "").strip().lower()
    if canonical and evidence == canonical and status == "partial" and last_attempt_iteration < current_iteration:
        return AnalysisRefreshDecision(
            refresh=True,
            reasons=("PARTIAL_UPGRADE",),
            evidence_stale=False,
            gain_since_evidence=gain,
        )

    if not evidence or evidence_mean_case_speedup is None:
        return AnalysisRefreshDecision(
            refresh=True,
            reasons=("INITIAL_ANALYSIS",),
            evidence_stale=bool(evidence and canonical != evidence),
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
