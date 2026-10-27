# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Project the session's kernel candidates into ``reports/kernel_optimization_summary.json``.

The candidates are the ``attempts`` of every KERNEL timeline event: one row per candidate from either route, each
already settled by the instrument that ruled on it (the GEAK rebench, the integrate gate, or the lane's own timing).
This module only groups and counts them; it states no verdict of its own.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kernelforge.knowledge import normalize_operator_name

from hyperloom.common.coerce import to_float
from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import (
    EVENT_TYPE as KERNEL_EVENT_TYPE,
    OUTCOME_ADOPTED,
    OUTCOME_FAILED,
    OUTCOME_NEEDS_REVIEW,
    OUTCOME_REJECTED,
    ROUTE_GEAK,
)
from hyperloom.inference_optimizer.breakdown.recorder.kernel_records import SOURCE_GEMM_TUNING
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events

from .kernel_decisions import _canonical_kernel_recipe_operator

if TYPE_CHECKING:
    from ..state.shared_state import SharedState

SCHEMA_VERSION = 3

#: Terminal lane outcome vocabulary the dashboard reads directly.
OUTCOME_SUCCESS = "success"
OUTCOME_UNVALIDATED = "unvalidated"
OUTCOME_FAIL = "fail"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_SKIP = "skip"

LANE_SOURCE_LEVEL = "source_level"
LANE_GEAK = "geak"
LANE_GEMM_TUNING = "gemm_tuning"
_LANES = (LANE_SOURCE_LEVEL, LANE_GEAK, LANE_GEMM_TUNING)

FIELD_GLOSSARY: dict[str, str] = {
    "claimed": (
        "The producer's own verdict on its candidate: the lane kept it, or GEAK named it among its acceptances. "
        "A claim is not a result; ``outcome`` is."
    ),
    "outcome": (
        "What Hyperloom ruled: adopted, rejected, failed (the lane never produced a candidate to judge) or "
        "needs_review (no instrument concluded)."
    ),
    "settled_by": (
        "Which instrument ruled: ``rebench`` (GEAK candidate re-benched end to end), ``integrate`` (the integrate "
        "gate measured the patch end to end) or ``lane`` (the producing lane's own timing)."
    ),
    "gain_pct": "The settling instrument's measurement, as a percentage; null when nothing measured it.",
    "gpu_pct": (
        "Share of total GPU time spent in this kernel in the latest trace, when the candidate names a kernel the "
        "trace ranked."
    ),
    "efficiency_pct": "Achieved throughput as a percentage of the kernel's roofline peak for its bound_type.",
}


def _session_kernel_attempts(session_dir: Path) -> list[dict[str, Any]]:
    """Every candidate the session's KERNEL visits recorded, oldest visit first, tagged with its macro cycle."""
    # An event still open carries only its shell ``ext``; ``attempts`` arrives when it is assembled.
    return [
        {**attempt, "macro_cycle": event["ext"]["macro_cycle"]}
        for event in read_timeline_events(session_dir)
        if event["type"] == KERNEL_EVENT_TYPE
        for attempt in event["ext"].get("attempts", [])
    ]


def _lane_of(attempt: dict[str, Any]) -> str:
    """The lane a candidate is counted in."""
    if attempt["route"] == ROUTE_GEAK:
        return LANE_GEAK
    if attempt["source_kind"] == SOURCE_GEMM_TUNING:
        return LANE_GEMM_TUNING
    return LANE_SOURCE_LEVEL


def _operator_of(name: str) -> str:
    """The normalized operator a kernel name or kernel-recipe id refers to, ``\"\"`` when there is none."""
    text = name.strip()
    return normalize_operator_name(_canonical_kernel_recipe_operator(text) or text) if text else ""


class _TraceIndex:
    """The latest trace's ranked kernels, looked up by trace ordinal or by operator name."""

    def __init__(self, top: list[dict[str, Any]]) -> None:
        self.rows = top
        self._by_id = {str(row.get("kernel_id") or ""): row for row in self.rows if row.get("kernel_id")}
        self._by_operator: dict[str, dict[str, Any]] = {}
        for row in self.rows:
            operator = _operator_of(str(row.get("name") or ""))
            if operator:
                self._by_operator.setdefault(operator, row)

    def lookup(self, kernel_id: str, name: str) -> dict[str, Any]:
        """The trace row a candidate names, or ``{}``."""
        if kernel_id in self._by_id:
            return self._by_id[kernel_id]
        for candidate in (kernel_id, name):
            operator = _operator_of(candidate)
            if operator in self._by_operator:
                return self._by_operator[operator]
        return {}


def _to_float(value: Any) -> float | None:
    """Coerce a value to a 4-decimal float, or ``None``."""
    parsed = to_float(value)
    return round(parsed, 4) if parsed is not None else None


#: Text fields a ``by_kernel`` row carries from the attempt; the timeline states an unset one as null.
_ATTEMPT_TEXT_FIELDS = (
    "kernel_id",
    "attempt_id",
    "route",
    "source_kind",
    "status",
    "backend",
    "micro_decision",
    "outcome",
    "settled_by",
    "unsettled_reason",
    "error_class",
    "failure_reason",
    "skip_reason",
    "artifact_path",
)


def _attempt_row(attempt: dict[str, Any], trace: _TraceIndex) -> dict[str, Any]:
    """One ``by_kernel`` row: the candidate as the timeline settled it, joined to the trace row it names."""
    trace_row = trace.lookup(attempt["kernel_id"] or "", attempt["name"] or "")
    return {
        **{field: attempt[field] or "" for field in _ATTEMPT_TEXT_FIELDS},
        "kernel_name": attempt["name"] or str(trace_row.get("name") or ""),
        "macro_cycle": attempt["macro_cycle"],
        "dispatched": attempt["dispatched"],
        "backends_tried": attempt["backends_tried"],
        "lane": _lane_of(attempt),
        "claimed": attempt["accepted"],
        "gain_pct": _to_float(attempt["gain_pct"]),
        "speedup": _to_float(attempt["speedup"]),
        "trace_kernel_id": str(trace_row.get("kernel_id") or ""),
        "source_file": str(trace_row.get("source_file") or ""),
        "gpu_pct": _to_float(trace_row.get("gpu_pct")),
        "efficiency_pct": _to_float(trace_row.get("efficiency_percent")),
        "bound_type": str(trace_row.get("bound_type") or ""),
    }


def _is_timeout(row: dict[str, Any]) -> bool:
    """Whether a candidate ended because it ran out of time."""
    return "timeout" in (row["status"].lower(), row["error_class"].lower())


def _lane_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One lane's counters and terminal outcome, counted over the candidates it dispatched."""
    dispatched = [row for row in rows if row["dispatched"]]
    outcomes = Counter(row["outcome"] for row in dispatched)
    success = outcomes[OUTCOME_ADOPTED]
    unvalidated = outcomes[OUTCOME_NEEDS_REVIEW]
    failed = len(dispatched) - success - unvalidated
    if not dispatched:
        outcome = OUTCOME_SKIP
    elif success:
        outcome = OUTCOME_SUCCESS
    elif unvalidated:
        outcome = OUTCOME_UNVALIDATED
    elif all(_is_timeout(row) for row in dispatched):
        outcome = OUTCOME_TIMEOUT
    else:
        outcome = OUTCOME_FAIL
    return {
        "attempted": len(dispatched),
        "claimed": sum(1 for row in dispatched if row["claimed"]),
        "success": success,
        "unvalidated": unvalidated,
        "failed": failed,
        "outcome": outcome,
    }


def _overall_lane_outcome(lanes: dict[str, dict[str, Any]]) -> str:
    """Roll lane outcomes up without calling unvalidated work successful."""
    outcomes = {row["outcome"] for row in lanes.values()}
    for outcome in (OUTCOME_SUCCESS, OUTCOME_UNVALIDATED, OUTCOME_TIMEOUT, OUTCOME_FAIL):
        if outcome in outcomes:
            return outcome
    return OUTCOME_SKIP


def _totals(rows: list[dict[str, Any]], *, top_candidates: int) -> dict[str, int]:
    """Session-wide counts, in the timeline's outcome vocabulary, over dispatched candidates."""
    dispatched = [row for row in rows if row["dispatched"]]
    outcomes = Counter(row["outcome"] for row in dispatched)
    return {
        "top_candidates": top_candidates,
        "attempted": len(dispatched),
        "claimed": sum(1 for row in dispatched if row["claimed"]),
        "adopted": outcomes[OUTCOME_ADOPTED],
        "rejected": outcomes[OUTCOME_REJECTED],
        "failed": outcomes[OUTCOME_FAILED],
        "needs_review": outcomes[OUTCOME_NEEDS_REVIEW],
        "not_dispatched": len(rows) - len(dispatched),
    }


def _failure_reason_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    """How the dispatched candidates that were not adopted ended, keyed by their error class."""
    return dict(
        Counter(
            row["error_class"] or "unclassified"
            for row in rows
            if row["dispatched"] and row["outcome"] in (OUTCOME_REJECTED, OUTCOME_FAILED)
        )
    )


def _highest_impact_missed(trace: _TraceIndex, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The hottest ranked trace kernel no candidate was adopted for."""
    adopted = {row["trace_kernel_id"] for row in rows if row["outcome"] == OUTCOME_ADOPTED}
    missed = [
        row
        for row in trace.rows
        if str(row.get("kernel_id") or "") not in adopted and to_float(row.get("gpu_pct")) is not None
    ]
    return max(missed, key=lambda row: float(row["gpu_pct"]), default=None)


def _top_takeaways(
    totals: dict[str, int],
    lane_totals: dict[str, dict[str, Any]],
    missed: dict[str, Any] | None,
    geak_result: dict[str, Any],
) -> list[str]:
    """Deterministic summary sentences, no LLM."""
    out: list[str] = []
    attempted = totals["attempted"]
    if not attempted:
        out.append("No kernel candidates were dispatched by any lane.")
    elif lane_totals[LANE_SOURCE_LEVEL]["attempted"] == 0:
        out.append("No source-level kernel rewrites were attempted; other kernel lanes did run.")
    if attempted:
        out.append(
            f"{totals['adopted']} of {attempted} dispatched kernel candidates were adopted; "
            f"{totals['rejected']} rejected, {totals['failed']} failed, {totals['needs_review']} unsettled."
        )
    if missed is not None:
        name = missed.get("name") or missed.get("kernel_id")
        gpu = float(missed["gpu_pct"])
        eff = to_float(missed.get("efficiency_percent")) or 0.0
        out.append(
            f"Highest-impact missed opportunity: {name} at {gpu:.1f}% GPU time, {eff:.1f}% efficiency — "
            "no candidate for it was adopted."
        )
    absent_backends = geak_result.get("absent_backends")
    if isinstance(absent_backends, dict) and absent_backends:
        out.append(
            f"GEAK ran without these backends (unavailable): {', '.join(sorted(absent_backends))}; "
            "any no-win it reports did not try them."
        )
    return out


def build_kernel_optimization_summary(state: SharedState, session_dir: Path | str) -> dict[str, Any]:
    """Build the kernel-optimization summary for one session from its KERNEL timeline events."""
    trace = _TraceIndex(state.last_trace_analyze.get("kernel_roofline_top15", []))
    rows = [_attempt_row(attempt, trace) for attempt in _session_kernel_attempts(Path(session_dir))]
    lane_totals = {lane: _lane_totals([row for row in rows if row["lane"] == lane]) for lane in _LANES}
    totals = _totals(rows, top_candidates=len(trace.rows))
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": state.session_id,
        "model_name": state.model_name,
        "cumulative_gain_validated_pct": state.cumulative_gain_validated,
        "kernel_opt_outcome": _overall_lane_outcome(lane_totals),
        "totals": totals,
        "failure_reason_breakdown": _failure_reason_breakdown(rows),
        "field_glossary": FIELD_GLOSSARY,
        "by_kernel": rows,
        "top_takeaways": _top_takeaways(totals, lane_totals, _highest_impact_missed(trace, rows), state.geak_result),
        "lane_totals": lane_totals,
    }


__all__ = [
    "FIELD_GLOSSARY",
    "OUTCOME_FAIL",
    "OUTCOME_SKIP",
    "OUTCOME_SUCCESS",
    "OUTCOME_TIMEOUT",
    "OUTCOME_UNVALIDATED",
    "SCHEMA_VERSION",
    "build_kernel_optimization_summary",
]
