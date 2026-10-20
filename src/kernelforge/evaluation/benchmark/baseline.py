# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cross-check the in-run pristine baseline against the task's reference file."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

# Task workspaces that carry independently measured pristine timings ship them here.
BASELINE_REFERENCE_FILENAME = "baseline_perf.yaml"

# On the run above, the other ten kernels measured that day were all within 1% of their historical medians, so this is
# far wider than legitimate run-to-run spread and far tighter than any drift worth acting on.
BASELINE_DRIFT_TOLERANCE = 0.25

# The reference times were measured on one machine and one image, so the same task on another GPU SKU can exceed the
# default deviation with nothing wrong.
BASELINE_DRIFT_TOLERANCE_ENV = "FORGE_BASELINE_DRIFT_TOLERANCE"


class BaselineReferenceError(RuntimeError):
    """The pristine baseline cannot be reconciled with the shipped reference."""


@dataclass(frozen=True)
class ReferenceCases:
    """One reference file's usable case times and the entries it lost."""

    case_times: dict[str, float]
    unusable_entries: tuple[str, ...]
    unreadable_reason: str = ""


@dataclass(frozen=True)
class BaselineReferenceCheck:
    """How much of this run's pristine anchor the reference actually backed."""

    compared_case_count: int = 0
    measured_case_count: int = 0
    reference_case_count: int = 0
    unusable_entries: tuple[str, ...] = ()
    drift_tolerance: float = BASELINE_DRIFT_TOLERANCE
    tolerance_overridden: bool = False
    unverified_reason: str = ""


def resolve_drift_tolerance() -> tuple[float, bool]:
    """Return the drift tolerance in force and whether an operator set it."""
    raw = os.environ.get(BASELINE_DRIFT_TOLERANCE_ENV, "").strip()
    if not raw:
        return BASELINE_DRIFT_TOLERANCE, False
    try:
        tolerance = float(raw)
    except ValueError as error:
        raise BaselineReferenceError(
            f"{BASELINE_DRIFT_TOLERANCE_ENV}={raw!r} is not a number; it is the "
            "permitted deviation as a fraction of the reference time, so 0.5 "
            f"means 50% (the default is {BASELINE_DRIFT_TOLERANCE})"
        ) from error
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise BaselineReferenceError(
            f"{BASELINE_DRIFT_TOLERANCE_ENV}={raw!r} is not a finite, non-negative fraction of the reference time"
        )
    return tolerance, True


def load_reference_case_times(workspace_dir: str) -> ReferenceCases | None:
    """Read the reference pristine per-case times, or None when none is shipped."""
    path = Path(workspace_dir) / BASELINE_REFERENCE_FILENAME
    if not path.is_file():
        return None
    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        return ReferenceCases(
            case_times={},
            unusable_entries=(),
            unreadable_reason=f"{path} exists but could not be read: {error}",
        )

    entries = document.get("test_cases") if isinstance(document, dict) else None
    cases: dict[str, float] = {}
    unusable: list[str] = []
    for position, entry in enumerate(entries or (), start=1):
        if not isinstance(entry, dict):
            unusable.append(f"entry {position} is not a mapping")
            continue
        case_id = str(entry.get("test_case_id") or "").strip().replace(" ", "_")
        if not case_id:
            unusable.append(f"entry {position} declares no 'test_case_id'")
            continue
        raw_time = entry.get("execution_time_ms")
        try:
            execution_time_ms = float(raw_time)
        except (TypeError, ValueError):
            unusable.append(f"entry {position} ({case_id}) declares no numeric 'execution_time_ms': {raw_time!r}")
            continue
        if not math.isfinite(execution_time_ms) or execution_time_ms <= 0.0:
            unusable.append(f"entry {position} ({case_id}) declares a non-positive 'execution_time_ms': {raw_time!r}")
            continue
        cases[case_id] = execution_time_ms

    if not cases:
        return ReferenceCases(
            case_times={},
            unusable_entries=tuple(unusable),
            unreadable_reason=(
                f"{path} declares no usable test case ('test_cases:' entries "
                "carrying 'test_case_id' and a positive 'execution_time_ms')"
                + (f": {'; '.join(unusable)}" if unusable else "")
            ),
        )
    return ReferenceCases(case_times=cases, unusable_entries=tuple(unusable))


def check_baseline_against_reference(
    workspace_dir: str,
    measured_case_times: dict[str, float],
) -> BaselineReferenceCheck:
    """Raise when a measured pristine case time drifts from the reference."""
    drift_tolerance, tolerance_overridden = resolve_drift_tolerance()
    loaded = load_reference_case_times(workspace_dir)
    if loaded is None:
        return BaselineReferenceCheck(
            drift_tolerance=drift_tolerance,
            tolerance_overridden=tolerance_overridden,
            unverified_reason=(f"this task ships no {BASELINE_REFERENCE_FILENAME}"),
        )
    if loaded.unreadable_reason:
        return BaselineReferenceCheck(
            unusable_entries=loaded.unusable_entries,
            drift_tolerance=drift_tolerance,
            tolerance_overridden=tolerance_overridden,
            unverified_reason=loaded.unreadable_reason,
        )
    reference = loaded.case_times

    measured = {
        str(case_id): float(value)
        for case_id, value in (measured_case_times or {}).items()
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0
    }
    shared = sorted(set(reference) & set(measured))
    if not shared:
        return BaselineReferenceCheck(
            measured_case_count=len(measured),
            reference_case_count=len(reference),
            unusable_entries=loaded.unusable_entries,
            drift_tolerance=drift_tolerance,
            tolerance_overridden=tolerance_overridden,
            unverified_reason=(
                f"{BASELINE_REFERENCE_FILENAME} names no case this run "
                f"measured: reference cases={sorted(reference)}, measured "
                f"cases={sorted(measured)}"
            ),
        )

    drifted = [
        f"{case_id}: measured {measured[case_id]:g} ms vs reference "
        f"{reference[case_id]:g} ms "
        f"({abs(measured[case_id] - reference[case_id]) / reference[case_id] * 100:.1f}%"
        " drift)"
        for case_id in shared
        if abs(measured[case_id] - reference[case_id]) / reference[case_id] > drift_tolerance
    ]
    if drifted:
        raise BaselineReferenceError(
            "the pristine baseline disagrees with "
            f"{BASELINE_REFERENCE_FILENAME} by more than "
            f"{drift_tolerance * 100:.0f}%, so every speedup measured "
            "against it would be wrong: " + "; ".join(drifted) + ". The "
            "reference was measured on one machine and image: if this run is "
            "on different hardware, re-measure it, or export "
            f"{BASELINE_DRIFT_TOLERANCE_ENV}=<fraction, e.g. 0.5 for 50%> to "
            "widen the tolerance for this run, which the console then reports"
        )
    return BaselineReferenceCheck(
        compared_case_count=len(shared),
        measured_case_count=len(measured),
        reference_case_count=len(reference),
        unusable_entries=loaded.unusable_entries,
        drift_tolerance=drift_tolerance,
        tolerance_overridden=tolerance_overridden,
    )
