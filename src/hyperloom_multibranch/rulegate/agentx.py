# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX rule: the optimizer's P50 interactivity bar and its two guards, retested only when they block a gain."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from hyperloom.common.gain_math import gain_pct
from hyperloom.common.perf_metric import (
    AGENTX_KEEP_P50_THRESHOLD_PCT,
    GRADED_DURATION,
    GRADED_ERROR_RATE,
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_OUTPUT,
    axis_of,
    holds_within_band,
    incomparability_reason,
)
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.rulegate.accuracy import agentx_accuracy_ok
from hyperloom_multibranch.rulegate.readings import (
    GRADED_REQUESTS_TOTAL,
    GRADED_SHORT_REQUESTS,
    GRADED_UNSERVED_REQUESTS,
)

P50_BELOW_THRESHOLD = "p50_gain_below_threshold"
# Failing more requests than the parent is a defect of the candidate a repair can look at; the other reasons
# ``incomparability_reason`` gives mean the pair did not measure the same work. Only unserved requests count: a short
# answer is the client's quality call, recorded for the specialist and never a refusal.
_EXTRA_FAILED_REQUESTS = "extra_failed_requests"
_GUARDS = (("p90_regressed", GRADED_INTVTY), ("output_regressed", GRADED_OUTPUT))
_RECORDED_AXES = (
    GRADED_INTVTY_P50,
    GRADED_INTVTY,
    GRADED_OUTPUT,
    GRADED_DURATION,
    GRADED_ERROR_RATE,
    GRADED_UNSERVED_REQUESTS,
    GRADED_SHORT_REQUESTS,
    GRADED_REQUESTS_TOTAL,
)


@dataclass(frozen=True)
class _Reading:
    """One measurement against the parent: its P50 gain, the defects it showed, and why it is not comparable."""

    gain: float
    defects: tuple[str, ...]
    incomparable: str
    accuracy_ok: bool

    @property
    def cleared(self) -> bool:
        return self.gain >= AGENTX_KEEP_P50_THRESHOLD_PCT

    @property
    def blocked(self) -> bool:
        return self.cleared and not self.incomparable and (bool(self.defects) or not self.accuracy_ok)

    @property
    def refused_by(self) -> tuple[str, ...]:
        reasons = [*self.defects, *([self.incomparable] if self.incomparable else [])]
        return tuple(reasons) if self.cleared else (*reasons, P50_BELOW_THRESHOLD)


def _read(candidate: Mapping[str, float], parent: Mapping[str, float]) -> _Reading:
    p50, parent_p50 = axis_of(candidate, GRADED_INTVTY_P50), axis_of(parent, GRADED_INTVTY_P50)
    gain = gain_pct(p50, parent_p50)
    if gain is None:
        raise ValueError(f"cannot compare a P50 of {p50!r} against a parent P50 of {parent_p50!r}")
    defects = [name for name, key in _GUARDS if not holds_within_band(candidate, parent, key)]
    incomparable = incomparability_reason(candidate, parent)
    if incomparable == _EXTRA_FAILED_REQUESTS:
        incomparable = ""
        if _more_unserved(candidate, parent):
            defects.append(_EXTRA_FAILED_REQUESTS)
    return _Reading(gain, tuple(defects), incomparable, agentx_accuracy_ok(candidate.get(GRADED_ERROR_RATE)))


def _more_unserved(candidate: Mapping[str, float], parent: Mapping[str, float]) -> bool:
    """Whether the candidate left more requests unserved; a side that did not count them may have, so it did."""
    if GRADED_UNSERVED_REQUESTS not in candidate or GRADED_UNSERVED_REQUESTS not in parent:
        return True
    return candidate[GRADED_UNSERVED_REQUESTS] > parent[GRADED_UNSERVED_REQUESTS]


def _recorded(candidate: Mapping[str, float]) -> dict[str, float]:
    return {key: float(candidate[key]) for key in _RECORDED_AXES if key in candidate}


def compare(candidate: Mapping[str, float], parent: Mapping[str, float], request_id: str) -> Judgement:
    """One measurement against the parent's record.

    A P50 gain held back by a broken guard, extra failed requests or the error-rate bar waits for a retest. Otherwise
    the single measurement is final: improved at the P50 bar with nothing broken; regressed when the P50 falls as far,
    a guard breaks, the candidate fails more requests than the parent or the pair did not measure the same work, so a
    correctness fix measured on it forks rather than landing on the branch; flat in between. More failed requests count
    only when more of them went unserved: more short answers alone are recorded, not refused.
    """
    reading = _read(candidate, parent)
    if reading.blocked:
        verdict = Verdict.NEEDS_CONFIRM
    elif reading.incomparable or reading.defects or reading.gain <= -AGENTX_KEEP_P50_THRESHOLD_PCT:
        verdict = Verdict.REGRESSED
    elif reading.cleared:
        verdict = Verdict.IMPROVED
    else:
        verdict = Verdict.FLAT
    return Judgement(
        verdict,
        reading.gain,
        AGENTX_KEEP_P50_THRESHOLD_PCT,
        (request_id,),
        _recorded(candidate),
        reading.accuracy_ok,
        reading.refused_by,
    )


def confirm(first: Judgement, retest: Mapping[str, float], parent: Mapping[str, float], request_id: str) -> Judgement:
    """Settles a P50 gain held back by a defect: blocked when the retest shows the gain and a defect again, so the
    specialist diagnoses it for a repair; inconclusive, and only recorded, when it does not."""
    if first.verdict is not Verdict.NEEDS_CONFIRM:
        raise ValueError(f"only a provisional judgement is retested, not a {first.verdict.value} one")
    reading = _read(retest, parent)
    recorded = {
        key: (value + first.recorded_value[key]) / 2.0
        for key, value in _recorded(retest).items()
        if key in first.recorded_value
    }
    return Judgement(
        Verdict.BLOCKED if reading.blocked else Verdict.INCONCLUSIVE,
        (first.gain_pct + reading.gain) / 2.0,
        AGENTX_KEEP_P50_THRESHOLD_PCT,
        (*first.request_ids, request_id),
        recorded,
        first.accuracy_ok and reading.accuracy_ok,
        tuple(dict.fromkeys((*first.refused_by, *reading.refused_by))),
    )


__all__ = ["P50_BELOW_THRESHOLD", "compare", "confirm"]
