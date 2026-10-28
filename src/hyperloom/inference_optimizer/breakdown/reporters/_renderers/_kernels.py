# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared per-kernel rollup over the ``kernel`` timeline events.

Three consumers need the same view and had three projections of it: the
lifecycle table, the capability summary's kernel routes, and the executive
summary's funnel. Building it once here is what keeps them from disagreeing
about how many kernels a session adopted.

A session visits the kernel phase more than once, and a kernel discovered in
one visit can be gated in a later one. So the rollup is keyed by kernel id
across every visit rather than per event, and the verdict a kernel ends on is
the last one the integrate gate reached about it.

The ``_`` prefix marks this as a helper: it registers no renderer.
"""

from __future__ import annotations

from typing import Any

from ave import AVE_BACKEND

from hyperloom_common.coerce import dict_rows, to_mapping

from ...recorder.stack_event import BACKENDS
from ..base import events_of

__all__ = [
    "AVE_SOURCES",
    "FORGE_SOURCES",
    "GEAK_SOURCES",
    "kernel_rows",
    "source_counters",
]

#: The source kinds each report-level capability owns. The recorder tallies at
#: the granularity of the producer that made the candidate; the report speaks
#: of the routes those producers belong to.
FORGE_SOURCES = ("kernel_rewrite", "fusion", "gemm_tuning")
GEAK_SOURCES = ("geak_authored_kernel", "geak_env_selection")
AVE_SOURCES = ("ave_kernel",)

#: Attempt outcomes that say the candidate was ruled against rather than left
#: unsettled. ``needs_review`` is deliberately absent: nothing concluded.
_REJECTED_OUTCOMES = frozenset({"rejected", "failed"})

#: Kernel-row verdicts the integrate gate or an attempt's own end-to-end ruling can set.
_ATTEMPTED = "attempted"
_REVERTED = "reverted"

#: The counters ``source_counters`` reports. ``e2e_gain_pct`` is a maximum
#: rather than a sum and is handled separately.
_SUMMED = (
    "attempted",
    "needs_review",
    "rejected",
    "failed",
    "keeps",
    "reverts",
    "micro_only_keeps",
)


def _visits(breakdown: dict[str, Any]) -> list[dict[str, Any]]:
    """The ``ext`` of every kernel visit, in timeline order.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        list[dict[str, Any]]: One entry per ``kernel`` event.
    """
    return [to_mapping(event.get("ext")) for event in events_of(breakdown, "kernel")]


def _best(current: Any, candidate: Any) -> float | None:
    """The larger of two speedups, ignoring the non-numeric.

    Returns:
        float | None: The larger, or ``None`` when neither is a number.
    """
    values = [float(v) for v in (current, candidate) if isinstance(v, (int, float))]
    return max(values) if values else None


def _measured_speedup(value: Any) -> Any:
    """``value`` when it is a speedup a run measured; an unmeasured attempt carries ``0.0``."""
    return value if isinstance(value, (int, float)) and value > 0 else None


def _attempts(breakdown: dict[str, Any]) -> list[dict[str, Any]]:
    """Every candidate across all routes and visits."""
    return [row for ext in _visits(breakdown) for row in dict_rows(ext.get("attempts"))]


def source_counters(breakdown: dict[str, Any], sources: tuple[str, ...]) -> dict[str, Any]:
    """Tally one route's candidates across every visit.

    Counted off the attempts themselves rather than off a tally the recorder
    also keeps: two counts of one population disagree the moment either side
    is edited, and the attempts are the side that carries the evidence.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.
        sources (tuple[str, ...]): The source kinds the route owns.

    Returns:
        dict[str, Any]: The counts, plus ``e2e_gain_pct`` as the best
            end-to-end gain the gate measured for any of them -- a maximum,
            because the gains are measured against a moving stack and adding
            them would claim a total no measurement supports.
    """
    totals: dict[str, Any] = dict.fromkeys(_SUMMED, 0)
    totals["e2e_gain_pct"] = None
    for row in _attempts(breakdown):
        if str(row.get("source_kind") or "") not in sources:
            continue
        totals["attempted"] += 1
        outcome = str(row.get("outcome") or "")
        e2e = to_mapping(row.get("e2e"))
        decision = str(e2e.get("decision") or "").upper()
        if decision == "REVERT":
            # A revoked keep is a revert whatever outcome the rebench settled on.
            totals["reverts"] += 1
        elif outcome == "adopted":
            # Without an end-to-end ruling the keep is micro-only: a rebench
            # promoted it, or a forge lane kept it ahead of the integrate gate.
            totals["keeps" if decision else "micro_only_keeps"] += 1
            totals["e2e_gain_pct"] = _best(totals["e2e_gain_pct"], e2e.get("e2e_gain_pct"))
        elif outcome in totals:
            totals[outcome] += 1
    return totals


def _seed(rows: dict[str, dict[str, Any]], kernel_id: str) -> dict[str, Any]:
    """The row for ``kernel_id``, created on first mention.

    Args:
        rows (dict[str, dict[str, Any]]): The rollup, keyed by kernel id.
        kernel_id (str): The kernel to fetch or create.

    Returns:
        dict[str, Any]: The row, ready to be updated in place.
    """
    row = rows.get(kernel_id)
    if row is None:
        row = {
            "kernel_id": kernel_id,
            "name": "",
            "gpu_pct": None,
            "duration_us": None,
            "call_count": None,
            "bandwidth_util_pct": None,
            "compute_util_pct": None,
            "selected_for_optimization": False,
            "geak": None,
            "forge": None,
            AVE_BACKEND: None,
            "adopted_by": None,
            "final_decision": "not_optimized",
        }
        rows[kernel_id] = row
    return row


def _fold_discovered(rows: dict[str, dict[str, Any]], ext: dict[str, Any]) -> None:
    """Fold one visit's discovered-kernel table into the rollup.

    Args:
        rows (dict[str, dict[str, Any]]): The rollup, updated in place.
        ext (dict[str, Any]): One kernel visit's ``ext``.
    """
    for entry in dict_rows(to_mapping(ext.get("forge")).get("discovered_kernels")):
        kernel_id = str(entry.get("kernel_id") or "")
        if not kernel_id:
            continue
        row = _seed(rows, kernel_id)
        row["name"] = str(entry.get("name") or "") or row["name"]
        # A re-profile in a later visit measures the same kernel again, and the
        # later measurement is the current one.
        for field in ("gpu_pct", "duration_us", "call_count", "bandwidth_util_pct", "compute_util_pct"):
            if entry.get(field) is not None:
                row[field] = entry.get(field)
        if entry.get("selected"):
            row["selected_for_optimization"] = True


def _fold_attempts(rows: dict[str, dict[str, Any]], ext: dict[str, Any]) -> None:
    """Fold one visit's candidate attempts into the rollup.

    All routes are read from the same array under the same field names, so
    the lane summaries are built by one pass rather than several that have to
    agree about what a speedup is called.

    Args:
        rows (dict[str, dict[str, Any]]): The rollup, updated in place.
        ext (dict[str, Any]): One kernel visit's ``ext``.
    """
    for entry in dict_rows(ext.get("attempts")):
        kernel_id = str(entry.get("kernel_id") or "")
        lane = str(entry.get("route") or "")
        if not kernel_id or lane not in BACKENDS:
            continue
        row = _seed(rows, kernel_id)
        if not row["name"]:
            row["name"] = str(entry.get("name") or "")
        # A route that dispatched against a kernel is a route that considered
        # it a target, whatever the discovery table said.
        row["selected_for_optimization"] = True
        summary = row.get(lane) or {"attempts": 0, "best_speedup": None, "decision": ""}
        summary["attempts"] = int(summary["attempts"]) + 1
        summary["best_speedup"] = _best(summary["best_speedup"], _measured_speedup(entry.get("speedup")))
        decision = str(to_mapping(entry.get("e2e")).get("decision") or "") or str(entry.get("micro_decision") or "")
        if decision:
            summary["decision"] = decision
        row[lane] = summary
        revoked = str(to_mapping(entry.get("e2e")).get("decision") or "").upper() == "REVERT"
        if revoked and row["final_decision"] in {"not_optimized", _ATTEMPTED, "rejected"}:
            row["final_decision"] = _REVERTED
        elif str(entry.get("outcome") or "") in _REJECTED_OUTCOMES and row["final_decision"] == "not_optimized":
            row["final_decision"] = "rejected"
        elif row["final_decision"] == "not_optimized":
            row["final_decision"] = _ATTEMPTED


def _fold_integrate(rows: dict[str, dict[str, Any]], ext: dict[str, Any]) -> None:
    """Fold one visit's integrate-gate verdicts into the rollup.

    The gate is the only thing that rules on a patch end to end, so its verdict
    overrides whatever the lane concluded about its own output.

    Args:
        rows (dict[str, dict[str, Any]]): The rollup, updated in place.
        ext (dict[str, Any]): One kernel visit's ``ext``.
    """
    for entry in dict_rows(ext.get("integrate")):
        kernel_id = str(entry.get("kernel_id") or "")
        decision = str(entry.get("decision") or "").upper()
        if not kernel_id or not decision:
            continue
        row = _seed(rows, kernel_id)
        row["final_decision"] = "kept" if decision == "KEEP" else _REVERTED


def _fold_adopted_by(rows: dict[str, dict[str, Any]]) -> None:
    """Name the route each kept kernel was adopted from.

    The gate rules on a patch and is handed no producer, so the route is the
    one that produced a candidate for that kernel. A kernel multiple routes touched
    names all of them rather than picking one, since nothing recorded says which
    patch the gate measured.

    Args:
        rows (dict[str, dict[str, Any]]): The rollup, updated in place.
    """
    for row in rows.values():
        if row["final_decision"] not in {"kept", _REVERTED}:
            continue
        touched = [lane for lane in BACKENDS if row.get(lane)]
        if touched:
            row["adopted_by"] = "+".join(touched)


def kernel_rows(breakdown: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per kernel the session touched, merged across every visit.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        list[dict[str, Any]]: The rows, by descending GPU share so the costly
            kernels lead. Each carries the profiling fields, whether it was
            selected, a per-route lane summary, the route it was adopted from
            and the verdict it ended on.
    """
    rows: dict[str, dict[str, Any]] = {}
    for ext in _visits(breakdown):
        _fold_discovered(rows, ext)
        _fold_attempts(rows, ext)
        _fold_integrate(rows, ext)
    _fold_adopted_by(rows)
    return sorted(rows.values(), key=lambda row: -(row.get("gpu_pct") or 0.0))
