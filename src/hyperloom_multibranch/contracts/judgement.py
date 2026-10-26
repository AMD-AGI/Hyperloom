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

    ``BLOCKED`` is a gain both measurements showed, held back by a defect both showed too: a failed accuracy gate, on
    InferenceX requests the graded rounds did not complete, or on AgentX a broken guard or more failed requests than the
    parent. ``INCONCLUSIVE`` is a retest that disagreed with the
    first measurement on the gain or the defect.
    """

    IMPROVED = "improved"
    FLAT = "flat"
    REGRESSED = "regressed"
    NEEDS_CONFIRM = "needs_confirm"
    BLOCKED = "blocked"
    INCONCLUSIVE = "inconclusive"


class AccuracyVerdict(str, Enum):
    """What the accuracy gate made of the measurements behind a judgement.

    ``NOT_MEASURED`` is a judgement on throughput alone, from a measurement that skipped its eval. It passes nothing
    that requires accuracy: the candidate commits, merges or counts as a fix only on a measurement that ran its eval.
    """

    PASSED = "passed"
    FAILED = "failed"
    NOT_MEASURED = "not_measured"

    @classmethod
    def measured(cls, passed: bool) -> AccuracyVerdict:
        return cls.PASSED if passed else cls.FAILED


@dataclass(frozen=True)
class Judgement:
    """One comparison: its verdict, the gain and the bar it turned on, and the tests behind it.

    ``recorded_value`` is the candidate's value per metric, in the units the PRELUDE baseline records: the mean of both
    measurements once retested, otherwise the single one; it becomes the commit's recorded value when the candidate is
    committed. On InferenceX it holds the eval score under ``ACCURACY_SCORE`` too when the eval ran, on AgentX the request
    error rate. ``accuracy`` is ``FAILED`` when a measurement behind the judgement failed the accuracy gate, ``PASSED``
    when every one that ran its eval passed it, and ``NOT_MEASURED`` when none ran it; ``refused_by`` names the
    performance gates that kept it from ``IMPROVED``.
    """

    verdict: Verdict
    gain_pct: float
    threshold_pct: float
    request_ids: tuple[str, ...]
    recorded_value: Mapping[str, float]
    accuracy: AccuracyVerdict
    refused_by: tuple[str, ...] = ()
    fixes_correctness: bool = False
    foundation_fix: bool = False


__all__ = ["ACCURACY_SCORE", "AccuracyVerdict", "Judgement", "Verdict"]
