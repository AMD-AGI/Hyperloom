# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the settled baseline facts say about PRELUDE: the evidence each of its exits is built from."""

from __future__ import annotations

from typing import Any

from hyperloom.common.coerce import optional_positive_float, to_float


#: The ``warm_replay_status`` of a baseline round that has not finished yet.
_IN_FLIGHT = "in_flight"


def _prelude_viability(baseline: dict[str, Any]) -> dict[str, Any]:
    usable = to_float(baseline.get("session_usable_sec"))
    runtime = optional_positive_float(baseline.get("runtime_sec"))
    post_ready = optional_positive_float(baseline.get("post_ready_runtime_sec"))
    warm = optional_positive_float(baseline.get("warm_runtime_sec"))
    benchmark = warm or post_ready
    round_sec = None
    if runtime is not None and post_ready is not None and benchmark is not None:
        round_sec = max(0.0, runtime - post_ready) + benchmark
        priced_by = "boot_plus_benchmark"
    else:
        round_sec = runtime
        priced_by = "cold_round"
    if usable is None or round_sec is None:
        return {}
    return {
        "session_usable_sec": round(usable, 1),
        "measured_round_sec": round(round_sec, 1),
        "priced_by": priced_by,
        "affordable_rounds": round(usable / round_sec, 2),
        "fits_one_optimization_round": usable >= round_sec,
    }


def _cold_anchor_shortfall(baseline: dict[str, Any]) -> dict[str, Any] | None:
    """The evidence that retrying a dropped measure round no longer fits the session's budget."""
    if not bool(baseline.get("measure_round_dropped")):
        return None
    usable = to_float(baseline.get("session_usable_sec"))
    runtime = optional_positive_float(baseline.get("runtime_sec"))
    benchmark = optional_positive_float(baseline.get("warm_runtime_sec")) or optional_positive_float(
        baseline.get("post_ready_runtime_sec")
    )
    retry = runtime
    if runtime is not None and benchmark is not None:
        retry += benchmark
    viability = _prelude_viability(baseline)
    use_sec = to_float(viability.get("measured_round_sec")) or runtime
    if usable is None or retry is None or use_sec is None or usable >= retry + use_sec:
        return None
    return {"terminal": True, "baseline_anchor": "cold", "retry_round_sec": round(retry, 1), **viability}


def _prelude_budget_line(baseline: dict[str, Any]) -> dict[str, Any] | None:
    """The evidence that PRELUDE is out of its own budget with no baseline and nothing left to dispatch.

    A failed baseline is retried only on a proposal from the Orchestration agent, so once the phase
    has spent everything outside the optimization reserve it has no arm it can buy by itself: the
    affordability figure refuses every optional arm and the initial-analysis hold has run out. Holding
    the phase open there spends the reserve on an idle preparation instead of on the optimization the
    reserve is held for, and leaves the exit to an LLM escalation. A round still in flight is not idle,
    so the line is read only once the phase is waiting on nothing of its own.
    """
    streak = int(baseline.get("failure_streak") or 0)
    if streak < 1 or str(baseline.get("warm_replay_status") or "") == _IN_FLIGHT:
        return None
    affordable = to_float(baseline.get("analysis_affordable_sec"))
    if affordable is None or affordable > 0.0:
        return None
    return {
        "terminal": True,
        "baseline_failure_streak": streak,
        "prelude_affordable_sec": round(affordable, 1),
        "session_usable_sec": round(to_float(baseline.get("session_usable_sec")) or 0.0, 1),
        "prelude_spent_sec": round(to_float(baseline.get("phase_spent_sec")) or 0.0, 1),
    }


def _baseline_is_anchored(baseline: dict[str, Any]) -> bool:
    """Whether a measured baseline the rest of the session can be compared against is in hand."""
    return (
        str(baseline.get("warm_replay_status") or "") != _IN_FLIGHT
        and not bool(baseline.get("measure_round_dropped"))
        and (to_float(baseline.get("tput")) or 0.0) > 0.0
    )
