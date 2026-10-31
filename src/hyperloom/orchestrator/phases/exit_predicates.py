# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Budget and advisory exit decisions from recorded phase facts."""

from __future__ import annotations

from typing import Any, NamedTuple

from hyperloom_common.coerce import to_float

from ..state.shared_state import ESCALATE_HINT_SKIP_TO_KERNEL, ESCALATE_HINT_SKIP_TO_SWEEP


def budget_exit(
    budget: dict[str, Any],
    *,
    exhausted_reason: str,
    evidence: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    remaining = to_float(budget.get("remaining_sec"))
    elapsed = to_float(budget.get("entry_elapsed_sec")) or 0.0
    cumulative = to_float(budget.get("cumulative_elapsed_sec")) or 0.0
    timing = {
        **evidence,
        "entry_elapsed_seconds": elapsed,
        "cumulative_elapsed_seconds": cumulative,
    }
    if remaining is not None and remaining <= 0.0:
        return exhausted_reason, timing
    return None


class FrameworkPlateau(NamedTuple):
    """Whether each FRAMEWORK_AGENT arm has stopped paying, and the arm evidence that says so."""

    config_dry: bool
    source_dry: bool
    arms: dict[str, Any]
    did_work: bool


def framework_plateau(plateau: dict[str, Any]) -> FrameworkPlateau:
    """Read the per-arm plateau verdicts off the FRAMEWORK_AGENT predicate inputs."""
    recent_gain = to_float(plateau.get("recent_keep_gain_pct")) or 0.0
    keep_threshold = to_float(plateau.get("keep_gain_threshold_pct")) or 0.0
    empty_streak = int(plateau.get("empty_streak") or 0)
    empty_threshold = int(plateau.get("empty_streak_threshold") or 0)
    source_streak = int(plateau.get("source_consecutive_no_keep") or 0)
    source_threshold = int(plateau.get("source_threshold") or 0)
    config_dry = recent_gain < keep_threshold and empty_streak >= empty_threshold
    source_dry = source_streak >= source_threshold or bool(plateau.get("source_candidates_exhausted"))
    return FrameworkPlateau(
        config_dry=config_dry,
        source_dry=source_dry,
        arms={
            **{key: value for key, value in plateau.items() if key not in {"attempt_count", "specialist_round_count"}},
            "config_arm_plateaued": config_dry,
            "source_arm_plateaued": source_dry,
            "switch_bottleneck": bool(config_dry or source_dry),
        },
        did_work=int(plateau.get("attempt_count") or 0) > 0 or int(plateau.get("specialist_round_count") or 0) > 0,
    )


def framework_exit(inputs: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    plateau = framework_plateau(dict(inputs.get("plateau") or {}))
    arms = plateau.arms
    budget = budget_exit(
        dict(inputs.get("budget") or {}),
        exhausted_reason="optimize_phase_budget_exhausted",
        evidence=arms,
    )
    if budget is not None:
        return budget
    hint = str(inputs.get("hint") or "")
    if hint == ESCALATE_HINT_SKIP_TO_KERNEL and plateau.did_work:
        return "optimize_no_more_leverage", {**arms, "evidence": "llm_escalation", "hint": hint}
    if hint == ESCALATE_HINT_SKIP_TO_SWEEP:
        return "optimize_no_more_leverage", {**arms, "evidence": "skip_to_sweep", "hint": hint}
    if plateau.config_dry and plateau.source_dry:
        return "optimize_no_more_leverage", {**arms, "evidence": "both_arms_plateaued", "plateau": True}
    return None


__all__ = ["FrameworkPlateau", "budget_exit", "framework_exit", "framework_plateau"]
