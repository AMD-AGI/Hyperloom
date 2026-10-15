# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the rule gate concluded about a measured candidate against its parent commit's recorded value."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

#: The ``recorded_value`` key of an InferenceX measurement's eval score, kept beside its throughput.
ACCURACY_SCORE = "accuracy_score"


class Verdict(str, Enum):
    """``NEEDS_CONFIRM`` is the only provisional verdict: a retest on another sandbox settles it.

    ``BLOCKED`` is a gain both measurements showed, held back by a defect both showed too: a failed accuracy gate, or on
    AgentX a broken guard or more failed requests than the parent. ``INCONCLUSIVE`` is a retest that disagreed with the
    first measurement on the gain or the defect.
    """

    IMPROVED = "improved"
    FLAT = "flat"
    REGRESSED = "regressed"
    NEEDS_CONFIRM = "needs_confirm"
    BLOCKED = "blocked"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class Judgement:
    """One comparison: its verdict, the gain and the bar it turned on, and the tests behind it.

    ``recorded_value`` is the candidate's value per metric, in the units the PRELUDE baseline records: the mean of both
    measurements once retested, otherwise the single one; it becomes the commit's recorded value when the candidate is
    committed. On InferenceX it holds the eval score under ``ACCURACY_SCORE`` too, on AgentX the request error rate.
    ``accuracy_ok`` holds only when every measurement behind the judgement passed the accuracy gate, and ``refused_by``
    names the performance gates that kept it from ``IMPROVED``.
    """

    verdict: Verdict
    gain_pct: float
    threshold_pct: float
    request_ids: tuple[str, ...]
    recorded_value: Mapping[str, float]
    accuracy_ok: bool
    refused_by: tuple[str, ...] = ()
    fixes_correctness: bool = False
    foundation_fix: bool = False


__all__ = ["ACCURACY_SCORE", "Judgement", "Verdict"]
