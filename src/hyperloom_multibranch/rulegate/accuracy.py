# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a candidate is still correct enough to commit: its eval score on InferenceX, its error rate on AgentX."""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from hyperloom.common.gain_math import (
    AGENTX_ERROR_RATE_THRESHOLD_PCT,
    DEFAULT_ENABLEMENT_ACCURACY_FLOOR,
    accuracy_meets_floor,
    accuracy_passed,
    finite_score,
)


class BaselineAccuracyError(ValueError):
    """PRELUDE fails the session on this: no candidate can be judged against the baseline it measured."""


def baseline_accuracy(scores: Sequence[float | None]) -> float:
    """The reference every InferenceX candidate's score is held to: the mean of the PRELUDE baselines' scores."""
    usable = [finite_score(score) for score in scores]
    if not usable or None in usable:
        raise BaselineAccuracyError(f"every baseline must report an accuracy score, got {list(scores)!r}")
    mean = statistics.fmean(usable)
    if not accuracy_meets_floor(mean, DEFAULT_ENABLEMENT_ACCURACY_FLOOR):
        raise BaselineAccuracyError(
            f"baseline accuracy {mean:.3f} is below the floor {DEFAULT_ENABLEMENT_ACCURACY_FLOOR}"
        )
    return mean


def inferencex_accuracy_ok(baseline: float, candidate: float | None) -> bool:
    """Above the floor and within the allowed drop of ``baseline``; a missing score fails."""
    if not accuracy_meets_floor(candidate, DEFAULT_ENABLEMENT_ACCURACY_FLOOR):
        return False
    return accuracy_passed(baseline, candidate)


def agentx_accuracy_ok(error_rate_pct: float | None) -> bool:
    """Upstream's submission bar on the request error rate; a run that reported no rate fails."""
    rate = finite_score(error_rate_pct)
    return rate is not None and rate <= AGENTX_ERROR_RATE_THRESHOLD_PCT


__all__ = ["BaselineAccuracyError", "agentx_accuracy_ok", "baseline_accuracy", "inferencex_accuracy_ok"]
