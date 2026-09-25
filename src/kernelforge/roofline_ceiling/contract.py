# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one file a ceiling run has to produce, and what makes it readable.

The analyst writes ``performance_ceiling.json`` itself, along with the
derivation beside it. This module reads that file back and decides whether it
can be used at all -- nothing more. A latency it cannot parse is sent back to
be rewritten; a latency it can parse is taken as given.

There is no check on the arithmetic, and deliberately none: the analyst owns
the roofs, the work model and the composition, and a framework that re-derived
any of them would be asserting a model it had already found too narrow to
express real operators. What that costs is worth naming. A ceiling that is too
loose -- because a roof read low, or a byte count came out high -- raises
nothing here. It reads as a kernel closer to done than it is, and a campaign
with an attainment target stops early on it. The derivation published beside
the file is what a reader has to catch that with.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class CeilingContractError(ValueError):
    """Raised when the published ceiling file cannot be read as an answer."""


@dataclass(frozen=True)
class CaseCeiling:
    """The deliverable for one test shape: its theoretical achievable latency."""

    case_id: str
    t_ideal_ms: float


@dataclass(frozen=True)
class CeilingReport:
    """Per-shape theoretical achievable latency for one kernel on one box."""

    cases: tuple[CaseCeiling, ...]

    def case(self, case_id: str) -> CaseCeiling | None:
        """Look one case up by id."""
        for entry in self.cases:
            if entry.case_id == case_id:
                return entry
        return None

    def ideal_ms(self) -> dict[str, float]:
        """The headline answer: ``case_id -> theoretical achievable latency``."""
        return {entry.case_id: entry.t_ideal_ms for entry in self.cases}

    def mean_ideal_ms(self) -> float | None:
        """The equal-weight mean across cases, or ``None`` when there are none.

        Equal weight because that is how a campaign scores its suite. Weighting
        by latency would let the largest shape speak for all of them, and the
        aggregate would then describe something the objective does not.
        """
        if not self.cases:
            return None
        return sum(entry.t_ideal_ms for entry in self.cases) / len(self.cases)


def _positive(value: Any) -> float | None:
    """Coerce to a finite positive float, or ``None`` when the value is not one."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def load_report(payload: Mapping[str, Any]) -> CeilingReport:
    """Read the published ceiling file, refusing one that cannot be used.

    The checks are on shape alone: a ``cases`` mapping of case id to a finite
    positive latency in milliseconds. ``mean_ideal_ms`` is recomputed rather
    than read, so the file cannot disagree with itself about its own aggregate.
    """
    if not isinstance(payload, Mapping):
        raise CeilingContractError("the ceiling file must contain a JSON object")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, Mapping) or not raw_cases:
        raise CeilingContractError(
            "the ceiling file needs a non-empty 'cases' object mapping each scored case id "
            "to its ideal latency in milliseconds"
        )

    cases: list[CaseCeiling] = []
    for case_id, raw in raw_cases.items():
        latency = _positive(raw)
        if latency is None:
            raise CeilingContractError(
                f"case {str(case_id)!r} has {raw!r} for its ideal latency; it must be a finite "
                "positive number of milliseconds"
            )
        cases.append(CaseCeiling(case_id=str(case_id), t_ideal_ms=latency))

    return CeilingReport(cases=tuple(cases))


__all__ = [
    "CaseCeiling",
    "CeilingContractError",
    "CeilingReport",
    "load_report",
]
