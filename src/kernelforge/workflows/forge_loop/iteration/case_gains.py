# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which scored cases one candidate ran faster than another by more than its own run-to-run noise."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence


def case_spreads(
    measurements: Sequence[dict] | None,
) -> dict[str, float]:
    """Each case's run-to-run spread across one candidate's own measurements."""
    per_case: dict[str, list[float]] = {}
    for measurement in measurements or ():
        if not isinstance(measurement, dict):
            continue
        for case_id, value in (measurement.get("case_times") or {}).items():
            if not isinstance(value, (int, float)) or float(value) <= 0.0:
                continue
            per_case.setdefault(str(case_id), []).append(float(value))
    return {case_id: statistics.stdev(times) for case_id, times in per_case.items() if len(times) >= 2}


def cases_beating_reference(
    case_times: Mapping[str, float],
    reference_case_times: Mapping[str, float],
    spreads: Mapping[str, float],
) -> frozenset[str]:
    """The cases this candidate ran faster than *reference* by more than noise."""
    owned: set[str] = set()
    for case_id, reference in reference_case_times.items():
        measured = case_times.get(case_id)
        spread = spreads.get(str(case_id))
        if not isinstance(reference, (int, float)) or float(reference) <= 0.0:
            continue
        if not isinstance(measured, (int, float)) or float(measured) <= 0.0:
            continue
        if spread is None or float(spread) <= 0.0:
            continue
        if float(measured) < float(reference) - float(spread):
            owned.add(str(case_id))
    return frozenset(owned)
