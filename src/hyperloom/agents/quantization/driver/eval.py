"""Eval-gap policy: threshold resolution + acceptance check."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ACCEPTABLE_GAP = 0.03

_REQUIRED_EVAL_KEYS = ("source_score", "quantized_score", "relative_gap")

# How far the reported relative_gap may drift from the value recomputed out of the two scores before the report is
# treated as self-contradictory.
_GAP_CONSISTENCY_TOLERANCE = 0.02


@dataclass(frozen=True)
class EvalDecision:
    """Outcome of comparing ``relative_gap`` to the acceptance threshold."""

    status: str  # "within" | "exceeded" | "missing"
    relative_gap: float | None
    threshold: float
    threshold_source: str  # "arg" | "file" | "default"


def resolve_threshold(
    workspace: Path,
    *,
    acceptable_eval_gap: float | None,
) -> tuple[float, str]:
    """Resolve the eval-gap threshold per the SKILL.md §5.4 priority chain."""

    if acceptable_eval_gap is not None:
        return float(acceptable_eval_gap), "arg"
    threshold_file = workspace / "eval_gap_threshold.txt"
    if threshold_file.is_file():
        raw = threshold_file.read_text(encoding="utf-8").strip()
        if raw:
            try:
                return float(raw), "file"
            except ValueError:
                # Malformed file falls through to the safe default.
                pass
    return DEFAULT_ACCEPTABLE_GAP, "default"


def decide(
    eval_report: dict | None,
    *,
    workspace: Path,
    acceptable_eval_gap: float | None,
) -> EvalDecision:
    """Decide whether an evaluation report passes the quality gap threshold."""
    threshold, source = resolve_threshold(workspace, acceptable_eval_gap=acceptable_eval_gap)
    missing = EvalDecision(
        status="missing",
        relative_gap=None,
        threshold=threshold,
        threshold_source=source,
    )

    if not isinstance(eval_report, dict):
        return missing
    if any(k not in eval_report for k in _REQUIRED_EVAL_KEYS):
        return missing

    try:
        gap = float(eval_report["relative_gap"])
        src = float(eval_report["source_score"])
        qtd = float(eval_report["quantized_score"])
    except (TypeError, ValueError):
        return missing
    if not all(math.isfinite(v) for v in (gap, src, qtd)):
        return missing

    # relative_gap is LLM-authored, so recompute it from the scores it claims to summarise (SKILL.md §5.3) and reject
    # a report that contradicts itself.
    if src > 0 and abs(gap - max(0.0, (src - qtd) / src)) > _GAP_CONSISTENCY_TOLERANCE:
        return missing

    return EvalDecision(
        status="within" if gap <= threshold else "exceeded",
        relative_gap=gap,
        threshold=threshold,
        threshold_source=source,
    )


__all__ = [
    "DEFAULT_ACCEPTABLE_GAP",
    "EvalDecision",
    "decide",
    "resolve_threshold",
]
