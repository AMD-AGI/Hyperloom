# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE's hold for the initial roofline it enqueued."""

from __future__ import annotations

from typing import Any, Mapping


def holds_for_initial_analysis(baseline: Mapping[str, Any]) -> bool:
    """Whether the phase must stay open for an initial roofline it enqueued but has not booked yet.

    Leaving now would hand that analysis to the next phase instead of seeding this one with it.
    The hold is bounded by the optimization reserve, which it must never eat into.
    """
    if not bool(baseline.get("initial_analysis_pending")):
        return False
    affordable = baseline.get("analysis_affordable_sec")
    try:
        return affordable is None or float(affordable) > 0.0
    except (TypeError, ValueError):
        return True


def initial_analysis_inputs(state: Any, affordable_sec: float | None) -> dict[str, Any]:
    """The hold's two facts, as the phase predicates and the status block read them."""
    return {
        "initial_analysis_pending": bool((state.auto_roofline_pending_task_id or "").strip()),
        "analysis_affordable_sec": affordable_sec,
    }


def initial_analysis_hold_lines(baseline: Mapping[str, Any]) -> list[str]:
    """Status line for a held phase, so its spent share does not read as a spent session."""
    if not holds_for_initial_analysis(baseline):
        return []
    affordable = baseline.get("analysis_affordable_sec")
    bound = "" if affordable is None else f" for up to {int(float(affordable))}s"
    return [
        f"hold      : waiting{bound} for the initial roofline; remaining_sec above is this phase's share, "
        "not the session's, and the optimization phases follow"
    ]
