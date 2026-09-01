# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Project a gemm tuning session into independent per-tuner candidates.

A gemm session runs at most one MoE tuner and one dense tuner (the router is a
set of mutually exclusive precision/quant branches), and the two write disjoint
config tables read through disjoint environment variables. Yet the report used
to collapse both into one ``recommended_env`` blob under one ``micro_decision``:
the MoE result and the dense result could only be accepted or rejected together.
When one tuner wins and the other regresses, all-or-nothing forces a choice
between deploying a regression and discarding a win.

The nomination contract lands patches as independent siblings -- apply, re-bench,
KEEP or REVERT each on its own. This module produces the per-tuner projection
that makes that possible for gemm: one :class:`TunerCandidate` per tuner that
actually produced a deployable artifact, each carrying only its own environment
and artifact so Hyperloom can KEEP the winner and REVERT the loser.

Everything here is a pure function over :class:`TuneResult`; the report builder
forwards to it. The single-blob ``recommended_env`` / ``artifacts`` fields are
still emitted alongside for the legacy consumer, so this is additive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .tuners.base import TuneResult


@dataclass(frozen=True)
class TunerCandidate:
    """One tuner's deployable result, landable on its own.

    A candidate exists only when the tuner produced something worth validating at
    e2e: a real micro improvement, or an explicitly forced candidate (split-K
    tuning whose benefit is e2e-only and reports ``no_improvement`` at the micro
    level). ``env`` is exactly this tuner's variables and no sibling's, so
    applying one candidate cannot drag another's table into the run.
    """

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
    """This tuner's environment, single var and the extra map merged.

    A tuner may report a primary ``env_var``/``env_value`` pair, a bag of
    ``env_vars``, or both. Merged into one map keyed by variable so the candidate
    carries a self-contained apply set. The primary pair is written first so an
    ``env_vars`` entry that repeats the same key (should not happen, but is cheap
    to be right about) reflects the tuner's own last word.
    """
    env: dict[str, str] = {}
    if result.env_var and result.env_value:
        env[result.env_var] = result.env_value
    if result.env_vars:
        env.update({str(k): str(v) for k, v in result.env_vars.items()})
    return env


def is_candidate(result: TuneResult) -> bool:
    """Whether a tuner result is a deployable candidate.

    Mirrors the report builder's promotion rule exactly so the per-tuner view and
    the collapsed view never disagree about what counts:

    * ``ok`` / ``partial_output`` with a real improvement -- the rows it wrote
      are a valid artifact even when some shapes were lost; the shortfall is
      reported separately rather than by discarding the result;
    * an explicitly forced ``candidate`` in any non-failed status -- split-K
      tuning delivers e2e-only benefit and reports ``no_improvement`` at micro.
    """
    if result.status in ("ok", "partial_output") and result.has_improvement:
        return True
    return bool(result.candidate) and result.status != "failed"


def per_tuner_candidates(results: Iterable[TuneResult]) -> list[TunerCandidate]:
    """Every tuner that produced a deployable artifact, as its own candidate.

    A tuner with no artifact path is not landable no matter its status, so it is
    dropped here rather than emitted as an empty candidate the integrate lane
    would fail on. Order follows the input (the router's priority order).

    Args:
        results: Results from tuners that actually ran.

    Returns:
        One :class:`TunerCandidate` per deployable tuner; empty when none won.
    """
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
    """Every crashed tuner, listed regardless of whether a sibling won.

    A sibling tuner succeeding must not make a crash invisible: a single winning
    dense tuner used to mask fourteen MoE failures as "no headroom". Each record
    names the tuner and its error so the failure survives into the report even
    when the session's overall decision is a KEEP.

    Args:
        results: Results from tuners that actually ran.

    Returns:
        One record per failed tuner; empty when none failed.
    """
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
