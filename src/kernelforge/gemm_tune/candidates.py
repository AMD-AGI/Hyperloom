# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Project a gemm tuning session into independent per-tuner candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .tuners.base import TuneResult


@dataclass(frozen=True)
class TunerCandidate:
    """One tuner's deployable result, landable on its own."""

    tuner: str
    env: dict[str, str]
    artifact_path: str
    best_micro_speedup: float
    improved_shapes: int
    #: True when the artifact must still be confirmed at e2e before final deploy
    #: (always true today: micro is a screen, not the verdict).
    requires_e2e_validation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "tuner": self.tuner,
            "env": dict(self.env),
            "artifact_path": self.artifact_path,
            "best_micro_speedup": round(self.best_micro_speedup, 4),
            "improved_shapes": self.improved_shapes,
            "requires_e2e_validation": self.requires_e2e_validation,
        }


def _tuner_env(result: TuneResult) -> dict[str, str]:
    """This tuner's environment, single var and the extra map merged."""
    env: dict[str, str] = {}
    if result.env_var and result.env_value:
        env[result.env_var] = result.env_value
    if result.env_vars:
        env.update({str(k): str(v) for k, v in result.env_vars.items()})
    return env


def is_candidate(result: TuneResult) -> bool:
    """Whether a tuner result is a deployable candidate."""
    if result.status in ("ok", "partial_output") and result.has_improvement:
        return True
    return bool(result.candidate) and result.status != "failed"


def per_tuner_candidates(results: Iterable[TuneResult]) -> list[TunerCandidate]:
    """Every tuner that produced a deployable artifact, as its own candidate."""
    candidates: list[TunerCandidate] = []
    for result in results:
        if not isinstance(result, TuneResult):
            continue
        if not is_candidate(result):
            continue
        artifact = str(result.artifact_path or "").strip()
        env = _tuner_env(result)
        if not artifact and not env:
            # Nothing to apply: a candidate the integrate lane cannot land.
            continue
        candidates.append(
            TunerCandidate(
                tuner=result.tuner_name,
                env=env,
                artifact_path=artifact,
                best_micro_speedup=float(result.best_micro_speedup or 1.0),
                improved_shapes=int(result.improved_shapes or 0),
            )
        )
    return candidates


def failed_tuner_records(results: Iterable[TuneResult]) -> list[dict[str, Any]]:
    """Every crashed tuner, listed regardless of whether a sibling won."""
    records: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, TuneResult) or result.status != "failed":
            continue
        records.append(
            {
                "tuner": result.tuner_name,
                "error_class": result.error_class,
                "error": result.error,
            }
        )
    return records
