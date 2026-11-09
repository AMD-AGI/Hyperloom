# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX rule: a bar on the session's objective, P50 interactivity unless the session names total token
throughput, and the objective's two guards, retested only when they block a gain."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from hyperloom.common.agentx_objective import (
    BELOW_THRESHOLD,
    P50_BELOW_THRESHOLD,
    TOTAL_BELOW_THRESHOLD,
    AgentXObjective,
    Guard,
)
from hyperloom.common.gain_math import gain_pct
from hyperloom.common.perf_metric import (
    GRADED_DURATION,
    GRADED_ERROR_RATE,
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_INTVTY_P50_CHECKPOINT,
    GRADED_OUTPUT,
    GRADED_REQUESTS_TOTAL,
    GRADED_UNSERVED_REQUESTS,
    TOTAL_TOKEN_THROUGHPUT,
    axis_of,
    holds_within_band,
    incomparability_reason,
)
from hyperloom_multibranch.contracts.judgement import AccuracyVerdict, Judgement, Verdict
from hyperloom_multibranch.rulegate.accuracy import agentx_accuracy_ok
from hyperloom_multibranch.rulegate.readings import GRADED_SHORT_REQUESTS

# Failing more requests than the parent is a defect of the candidate a repair can look at; the other reasons
# ``incomparability_reason`` gives mean the pair did not measure the same work. Only unserved requests count: a short
# answer is the client's quality call, recorded for the specialist and never a refusal.
_EXTRA_FAILED_REQUESTS = "extra_failed_requests"
#: The refusals that say only that the candidate is slower than its parent: any objective's bar and guards.
SPEED_GATES = frozenset(
    {*BELOW_THRESHOLD, *(guard.refusal for objective in AgentXObjective for guard in objective.guards)}
)
#: Every axis any objective's bar or guards read is among them, so a judgement records what its session is graded on.
#: The checkpoint median is read by no bar or guard: a committed candidate records it as the reference its children's
#: AgentX checkpoint compares against.
_RECORDED_AXES = (
    GRADED_INTVTY_P50,
    GRADED_INTVTY_P50_CHECKPOINT,
    TOTAL_TOKEN_THROUGHPUT,
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
    """One measurement against the parent: its gain on the objective, the defects it showed, and why it is not
    comparable."""

    objective: AgentXObjective
    gain: float
    defects: tuple[str, ...]
    incomparable: str
    accuracy_ok: bool

    @property
    def cleared(self) -> bool:
        return self.gain >= self.objective.bar_pct

    @property
    def blocked(self) -> bool:
        return self.cleared and not self.incomparable and (bool(self.defects) or not self.accuracy_ok)

    @property
    def refused_by(self) -> tuple[str, ...]:
        reasons = [*self.defects, *([self.incomparable] if self.incomparable else [])]
        return tuple(reasons) if self.cleared else (*reasons, self.objective.below_threshold)


def _read(candidate: Mapping[str, float], parent: Mapping[str, float], objective: AgentXObjective) -> _Reading:
    metric = objective.metric
    value, parent_value = axis_of(candidate, metric), axis_of(parent, metric)
    gain = gain_pct(value, parent_value)
    if gain is None:
        raise ValueError(f"cannot compare a {metric} of {value!r} against a parent {metric} of {parent_value!r}")
    band = objective.guard_band_pct
    defects = [guard.refusal for guard in objective.guards if not _guard_holds(candidate, parent, guard.metric, band)]
    incomparable = incomparability_reason(candidate, parent)
    if incomparable == _EXTRA_FAILED_REQUESTS:
        incomparable = ""
        if _more_unserved(candidate, parent):
            defects.append(_EXTRA_FAILED_REQUESTS)
    return _Reading(objective, gain, tuple(defects), incomparable, agentx_accuracy_ok(candidate.get(GRADED_ERROR_RATE)))


def _guard_holds(candidate: Mapping[str, float], parent: Mapping[str, float], key: str, band_pct: float) -> bool:
    """A side that did not report the guard's axis breaks it: ``holds_within_band`` reads it as 0.0, which holds."""
    return key in candidate and key in parent and holds_within_band(candidate, parent, key, noise_pct=band_pct)


def _more_unserved(candidate: Mapping[str, float], parent: Mapping[str, float]) -> bool:
    """Whether the candidate left more requests unserved; a side that did not count them may have, so it did."""
    if GRADED_UNSERVED_REQUESTS not in candidate or GRADED_UNSERVED_REQUESTS not in parent:
        return True
    return candidate[GRADED_UNSERVED_REQUESTS] > parent[GRADED_UNSERVED_REQUESTS]


def _recorded(candidate: Mapping[str, float]) -> dict[str, float]:
    return {key: float(candidate[key]) for key in _RECORDED_AXES if key in candidate}


def compare(
    candidate: Mapping[str, float], parent: Mapping[str, float], request_id: str, objective: AgentXObjective
) -> Judgement:
    """One measurement against the parent's record, on ``objective``.

    A gain held back by a broken guard, extra failed requests or the error-rate bar waits for a retest. Otherwise the
    single measurement is final: improved at the bar with nothing broken; regressed when the objective falls as far,
    a guard breaks, the candidate fails more requests than the parent or the pair did not measure the same work, so a
    correctness fix measured on it forks rather than landing on the branch; flat in between. More failed requests count
    only when more of them went unserved: more short answers alone are recorded, not refused.
    """
    reading = _read(candidate, parent, objective)
    if reading.blocked:
        verdict = Verdict.NEEDS_CONFIRM
    elif reading.incomparable or reading.defects or reading.gain <= -objective.bar_pct:
        verdict = Verdict.REGRESSED
    elif reading.cleared:
        verdict = Verdict.IMPROVED
    else:
        verdict = Verdict.FLAT
    return Judgement(
        verdict,
        reading.gain,
        objective.bar_pct,
        (request_id,),
        _recorded(candidate),
        AccuracyVerdict.measured(reading.accuracy_ok),
        reading.refused_by,
    )


def confirm(
    first: Judgement,
    retest: Mapping[str, float],
    parent: Mapping[str, float],
    request_id: str,
    objective: AgentXObjective,
) -> Judgement:
    """Settles a gain on ``objective`` held back by a defect: blocked when the retest shows the gain and a defect again,
    so the specialist diagnoses it for a repair; inconclusive, and only recorded, when it does not."""
    if first.verdict is not Verdict.NEEDS_CONFIRM:
        raise ValueError(f"only a provisional judgement is retested, not a {first.verdict.value} one")
    reading = _read(retest, parent, objective)
    recorded = {
        key: (value + first.recorded_value[key]) / 2.0
        for key, value in _recorded(retest).items()
        if key in first.recorded_value
    }
    return Judgement(
        Verdict.BLOCKED if reading.blocked else Verdict.INCONCLUSIVE,
        (first.gain_pct + reading.gain) / 2.0,
        objective.bar_pct,
        (*first.request_ids, request_id),
        recorded,
        AccuracyVerdict.measured(first.accuracy is not AccuracyVerdict.FAILED and reading.accuracy_ok),
        tuple(dict.fromkeys((*first.refused_by, *reading.refused_by))),
    )


__all__ = [
    "BELOW_THRESHOLD",
    "P50_BELOW_THRESHOLD",
    "SPEED_GATES",
    "TOTAL_BELOW_THRESHOLD",
    "AgentXObjective",
    "Guard",
    "compare",
    "confirm",
]
