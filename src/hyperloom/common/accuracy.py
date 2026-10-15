# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The accuracy bars a baseline and a candidate are judged against, and the two checks that apply them."""

from __future__ import annotations

import math
from typing import Any

ACCURACY_THRESHOLD = 0.05  # allowed deviation

# Upstream rejects an AgentX submission whose error rate over completed requests exceeds 10% (InferenceX
# ``validate_agentic_result.py``). A percentage because that is aiperf's unit: ``RequestErrorRateMetric`` is declared
# PERCENT and derives ``100.0 * errors / total``.
AGENTX_ERROR_RATE_THRESHOLD_PCT = 10.0

# Shared accuracy floor, used by BOTH the baseline eval-failure trigger and the enablement KEEP gate so the two never
# diverge.
DEFAULT_ENABLEMENT_ACCURACY_FLOOR = 0.5

# The lm-eval metrics an eval score is read from, most preferred first; the thresholds above were set on these scores,
# and the first task that reports one of them gives the score.
ACCURACY_METRICS = ("exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none", "acc,none")


def finite_score(score: Any) -> float | None:
    """Return ``score`` as a finite float, or ``None`` when unusable."""
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    val = float(score)
    return val if math.isfinite(val) else None


def accuracy_meets_floor(score: Any, floor: float) -> bool:
    """True only when ``score`` is finite, strictly positive and ``>= floor``."""
    val = finite_score(score)
    if val is None or val <= 0.0:
        return False
    return val >= floor


def accuracy_passed(
    baseline_accuracy: float,
    new_accuracy: float,
    threshold: float = ACCURACY_THRESHOLD,
) -> bool:
    """Return True if accuracy drop is within tolerance."""
    if baseline_accuracy <= 0:
        # No baseline recorded; skip gate.
        return True
    drop = baseline_accuracy - new_accuracy
    return drop <= threshold


__all__ = [
    "ACCURACY_METRICS",
    "ACCURACY_THRESHOLD",
    "AGENTX_ERROR_RATE_THRESHOLD_PCT",
    "DEFAULT_ENABLEMENT_ACCURACY_FLOOR",
    "accuracy_meets_floor",
    "accuracy_passed",
    "finite_score",
]
