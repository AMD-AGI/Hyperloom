"""Critic — synchronous review gate for patch acceptance.

Called between "agent proposes patch" and "patch is applied".
Evaluates performance impact, accuracy regression, and KB-based
risk priors to produce accept/reject verdicts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .plugins.base import AccuracyResult, BenchResult

log = logging.getLogger(__name__)


class Verdict(Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    ACCEPT_WITH_CONCERNS = "accept_with_concerns"


@dataclass
class ReviewResult:
    """Result of a critic review."""

    verdict: Verdict
    justification: str = ""
    risk_signals: list[str] | None = None
    performance_delta_pct: float = 0.0
    accuracy_delta: float = 0.0


# Known-bad patterns that should trigger rejection or concern
RISK_PATTERNS: list[str] = [
    "removes error handling",
    "disables safety check",
    "hardcodes",
    "skips validation",
    "infinite loop",
    "memory leak",
    "race condition",
]


def review_patch(
    patch_description: str,
    baseline_bench: BenchResult,
    post_patch_bench: BenchResult,
    baseline_accuracy: AccuracyResult | None = None,
    post_patch_accuracy: AccuracyResult | None = None,
    kb_priors: list[str] | None = None,
    regression_tolerance_pct: float = 2.0,
    accuracy_tolerance: float = 0.01,
) -> ReviewResult:
    """Synchronous review gate for a proposed optimization.

    Decision logic:
      1. Performance regression > tolerance -> REJECT
      2. Accuracy regression > tolerance -> REJECT
      3. KB risk patterns matched -> ACCEPT_WITH_CONCERNS
      4. Otherwise -> ACCEPT
    """
    perf_delta = 0.0
    if baseline_bench.throughput > 0:
        perf_delta = (post_patch_bench.throughput / baseline_bench.throughput - 1) * 100

    acc_delta = 0.0
    if baseline_accuracy and post_patch_accuracy:
        acc_delta = post_patch_accuracy.score - baseline_accuracy.score

    risk_signals = _check_risk_signals(patch_description, kb_priors)

    if perf_delta < -regression_tolerance_pct:
        return ReviewResult(
            verdict=Verdict.REJECT,
            justification=f"Performance regression: {perf_delta:.1f}% (tolerance: -{regression_tolerance_pct}%)",
            risk_signals=risk_signals,
            performance_delta_pct=perf_delta,
            accuracy_delta=acc_delta,
        )

    if baseline_accuracy and post_patch_accuracy:
        if not post_patch_accuracy.passed:
            return ReviewResult(
                verdict=Verdict.REJECT,
                justification=f"Accuracy gate failed: {post_patch_accuracy.score:.4f} < {post_patch_accuracy.threshold:.4f}",
                risk_signals=risk_signals,
                performance_delta_pct=perf_delta,
                accuracy_delta=acc_delta,
            )
        if acc_delta < -accuracy_tolerance:
            return ReviewResult(
                verdict=Verdict.REJECT,
                justification=f"Accuracy regression: {acc_delta:.4f} (tolerance: -{accuracy_tolerance})",
                risk_signals=risk_signals,
                performance_delta_pct=perf_delta,
                accuracy_delta=acc_delta,
            )

    if risk_signals:
        return ReviewResult(
            verdict=Verdict.ACCEPT_WITH_CONCERNS,
            justification=f"Accepted with {len(risk_signals)} risk signal(s): {', '.join(risk_signals[:3])}",
            risk_signals=risk_signals,
            performance_delta_pct=perf_delta,
            accuracy_delta=acc_delta,
        )

    return ReviewResult(
        verdict=Verdict.ACCEPT,
        justification=f"Accepted: +{perf_delta:.1f}% throughput, no regressions",
        risk_signals=[],
        performance_delta_pct=perf_delta,
        accuracy_delta=acc_delta,
    )


def _check_risk_signals(patch_description: str, kb_priors: list[str] | None) -> list[str]:
    """Check patch against known risk patterns and KB priors."""
    signals = []
    desc_lower = patch_description.lower()

    for pattern in RISK_PATTERNS:
        if pattern in desc_lower:
            signals.append(f"risk_pattern: {pattern}")

    if kb_priors:
        for prior in kb_priors:
            prior_lower = prior.lower()
            if "known regression" in prior_lower or "revert" in prior_lower:
                signals.append(f"kb_prior: {prior[:80]}")

    return signals
