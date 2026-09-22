# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How much of the estimated ceiling a kernel has actually reached.

Attainment is ``t_ideal / t_current`` per case: the fraction of the estimated
best achievable latency the current kernel is delivering. One at the ceiling,
one half at twice the ceiling's latency.

It is derived here rather than stored on the report because the divisor is not
the ceiling module's to choose. The ceiling run times the driver under a
profiler, and that figure is inflated by the profiler; dividing by it would
report a kernel as closer to its limit than it is. The campaign's own per-case
medians are the only divisor that makes the ratio mean anything, so attainment
is computed where those live and cached nowhere.

Aggregation is the equal-weight mean over cases, matching
``calculate_mean_case_speedup`` -- the objective a KEEP is already judged by. A
stopping rule and a KEEP rule that optimize different aggregates would pull the
campaign in two directions.

Coverage is the load-bearing part. A case whose ceiling cannot be scored is
excluded from the mean, which means the mean no longer describes the objective:
drop the three hard shapes and the four easy ones average well above target.
:meth:`Attainment.covers` exists so a caller that acts on the number can insist
on the whole case set first, the same way ``CaseCoverageError`` guards scoring a
candidate against the baseline.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

#: An attainment above this is arithmetically impossible: the ceiling claims a
#: latency the kernel already beat, so the work model understates the minimum
#: legal work. Reported, never clamped -- clamping turns a broken estimate into
#: a perfect score, which is the one failure that must not look like success.
_IMPOSSIBLE_ABOVE = 1.0


@dataclass(frozen=True)
class CaseAttainment:
    """What one shape has reached, against the ceiling estimated for it."""

    case_id: str
    t_ideal_ms: float
    t_current_ms: float
    attainment: float
    bound: str

    @property
    def remaining_speedup(self) -> float:
        """How much faster this case would have to get to sit at its ceiling."""
        return self.t_current_ms / self.t_ideal_ms if self.t_ideal_ms > 0 else 0.0


@dataclass(frozen=True)
class Attainment:
    """Per-case and campaign-level attainment of one ceiling report."""

    mean: float | None
    cases: tuple[CaseAttainment, ...]
    #: ``case_id -> why it has no attainment``. Never silently dropped: a case
    #: missing from the mean is the reason the mean can be wrong.
    excluded: dict[str, str]

    @property
    def usable(self) -> bool:
        """Whether any case produced a figure at all."""
        return self.mean is not None

    def covers(self, case_ids: Iterable[str]) -> bool:
        """Whether every one of ``case_ids`` contributed to the mean.

        The mean is only the campaign's objective when it is taken over the
        campaign's whole scored set. A caller deciding anything on the figure
        asks this first.
        """
        scored = {str(case_id) for case_id in case_ids}
        return bool(scored) and scored <= {entry.case_id for entry in self.cases}

    def below(self, target: float) -> tuple[CaseAttainment, ...]:
        """Cases short of ``target``, worst first -- where the headroom is."""
        return tuple(sorted((c for c in self.cases if c.attainment < target), key=lambda c: c.attainment))


def _positive(value: object) -> float | None:
    """Coerce to a finite positive float, or ``None``."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def measure_attainment(
    report: object,
    current_ms: Mapping[str, float] | None,
    *,
    unscored_cases: Sequence[str] | set[str] | None = None,
) -> Attainment:
    """Score one ceiling report against the latencies a campaign is measuring.

    ``report`` is a :class:`~kernelforge.roofline_ceiling.contract.CeilingReport`;
    ``current_ms`` is the campaign's own per-case medians, never the ceiling
    run's profiled observation.

    A case is excluded, with a reason, when it was not measured, when either
    latency is not a positive number, when the objective does not score it, or
    when its ceiling sits below the measured latency. That last one is the
    estimate contradicting itself, and it is the exclusion that matters: left in,
    it contributes an attainment above one and pulls the mean up, so a broken
    work model reads as a finished kernel.
    """
    measured = dict(current_ms or {})
    excluded_ids = {str(case_id) for case_id in (unscored_cases or ())}

    scored: list[CaseAttainment] = []
    excluded: dict[str, str] = {}

    for case in getattr(report, "cases", ()) or ():
        case_id = str(getattr(case, "case_id", "") or "")
        if not case_id:
            continue
        if case_id in excluded_ids:
            excluded[case_id] = "correctness-only case; the objective does not score it"
            continue

        t_ideal = _positive(getattr(case, "t_ideal_ms", None))
        if t_ideal is None:
            excluded[case_id] = "the ceiling report carries no usable ideal latency"
            continue
        t_current = _positive(measured.get(case_id))
        if t_current is None:
            excluded[case_id] = "the campaign has no measured latency for this case"
            continue

        ratio = t_ideal / t_current
        if ratio > _IMPOSSIBLE_ABOVE:
            excluded[case_id] = (
                f"ceiling {t_ideal:.6g} ms is above the measured {t_current:.6g} ms, so the estimate "
                "understates the minimum legal work; it cannot say how much of the ceiling is reached"
            )
            continue

        scored.append(
            CaseAttainment(
                case_id=case_id,
                t_ideal_ms=t_ideal,
                t_current_ms=t_current,
                attainment=ratio,
                bound=str(getattr(case, "bound", "") or ""),
            )
        )

    mean = (sum(entry.attainment for entry in scored) / len(scored)) if scored else None
    return Attainment(mean=mean, cases=tuple(scored), excluded=excluded)


__all__ = [
    "Attainment",
    "CaseAttainment",
    "measure_attainment",
]
