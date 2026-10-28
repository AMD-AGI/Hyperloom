# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel lifecycle renderer — one row per detected kernel with its full optimization lifecycle.

Built from the per-kernel rollup over the ``kernel`` timeline events, so a
kernel discovered in one visit and gated in a later one reads as one kernel
rather than as two half-populated rows.

Columns are built locally in ``render`` and are dynamic: ``bw%`` /
``compute%`` are omitted when no detected kernel reports them. Skipped when no
kernels were detected (implies the profile phase never ran).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ave import AVE_BACKEND

from ..base import (
    Decision,
    RenderedSection,
    md_table,
    register_renderer,
)
from ._kernels import kernel_rows

_MAX_NAME_LEN = 70

#: The decision each settled verdict reads as: its kind, its subject, and what
#: the count of kernels that reached it says.
_VERDICT_DECISIONS = (
    ("kept", "kernel_lifecycle:adopted", "kernel patches adopted"),
    ("reverted", "kernel_lifecycle:reverted", "kernel patches reverted after integrate"),
    ("rejected", "kernel_lifecycle:rejected", "kernel patches rejected outright"),
)
#: The utilisation columns, each dropped when no detected kernel reports it.
_UTIL_COLUMNS = {"bw%": "bandwidth_util_pct", "compute%": "compute_util_pct"}


def _short_name(name: str) -> str:
    """Shorten a kernel name keeping head + tail (CK/ROCBLAS variants differ at the tail)."""
    if not name:
        return ""
    if len(name) <= _MAX_NAME_LEN:
        return name
    head = _MAX_NAME_LEN // 2 - 1
    tail = _MAX_NAME_LEN - head - 3
    return name[:head] + "..." + name[-tail:]


def _fmt_speedup(v: Any) -> str:
    """Format a speedup multiplier for a table cell."""
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{x:.2f}x"


def _lane_summary(lane: dict[str, Any] | None) -> str:
    """Format a per-lane summary cell: best-speedup + attempts + last decision."""
    if not lane:
        return "—"
    spd = _fmt_speedup(lane.get("best_speedup"))
    att = int(lane.get("attempts") or 0)
    dec = lane.get("decision") or ""
    parts = [spd]
    if att:
        parts.append(f"({att} att)")
    if dec:
        parts.append(f"[{dec}]")
    return " ".join(parts)


def _kernel_tables(detected: list[dict[str, Any]]) -> str:
    """The table of the kernels the pipeline acted on, then the residual long tail folded into ``<details>``."""
    util = {header: field for header, field in _UTIL_COLUMNS.items() if any(d.get(field) for d in detected)}
    headers = ["kernel_id", "name", "gpu%", "duration_us", "calls", *util]
    headers += ["selected", "GEAK", "Forge", "Ave", "adopted_by", "final"]

    def _row_for(d: dict[str, Any]) -> list[Any]:
        """Build one table row for a detected kernel record."""
        return [
            d.get("kernel_id"),
            _short_name(d.get("name")),
            d.get("gpu_pct"),
            d.get("duration_us"),
            d.get("call_count"),
            *(d.get(field) for field in util.values()),
            "yes" if d.get("selected_for_optimization") else "no",
            _lane_summary(d.get("geak")),
            _lane_summary(d.get("forge")),
            _lane_summary(d.get(AVE_BACKEND)),
            d.get("adopted_by") or "—",
            d.get("final_decision"),
        ]

    # The rollup marks every kernel a route dispatched against as selected, so
    # an unselected kernel is actionable only through a gate verdict.
    actionable: list[dict[str, Any]] = []
    residual: list[dict[str, Any]] = []
    for d in detected:
        if d.get("selected_for_optimization") or d.get("final_decision") != "not_optimized":
            actionable.append(d)
        else:
            residual.append(d)

    parts: list[str] = []
    if actionable:
        parts.append(md_table(headers, [_row_for(d) for d in actionable]))
    else:
        parts.append(
            "_No kernel was selected for optimization, and neither "
            "GEAK nor Forge nor Ave recorded an attempt. The kernel optimization "
            "pipeline did not run on this session._"
        )
    if residual:
        total_dur = sum((d.get("duration_us") or 0.0) for d in residual)
        total_gpu = sum((d.get("gpu_pct") or 0.0) for d in residual)
        # Inner blank lines needed so the table inside <details> parses as markdown.
        parts += [
            "",
            f"<details><summary>Show {len(residual)} residual kernels "
            f"not selected for optimization "
            f"(Σ duration_us={total_dur:.0f}, Σ gpu%={total_gpu:.2f}%)"
            "</summary>",
            "",
            md_table(headers, [_row_for(d) for d in residual]),
            "",
            "</details>",
        ]
    return "\n".join(parts)


@register_renderer("kernel_lifecycle")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the kernel-lifecycle section as a single per-kernel table.

    Emits one row per detected kernel with its full optimization
    lifecycle (selection, backend attempts, adoption, final decision),
    splitting long-tail unselected kernels into a collapsible block. The
    section is marked skipped when no kernels were detected.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered section with key facts, decisions and
            the markdown table.
    """
    detected = kernel_rows(breakdown)

    if not detected:
        return RenderedSection(
            section_id="kernel_lifecycle",
            title="Kernel Lifecycle",
            key_facts=["No kernels detected this session."],
            skipped=True,
        )

    selected = sum(1 for d in detected if d.get("selected_for_optimization"))
    geak_touched = sum(1 for d in detected if d.get("geak"))
    forge_touched = sum(1 for d in detected if d.get("forge"))
    ave_touched = sum(1 for d in detected if d.get(AVE_BACKEND))
    verdicts = Counter(d.get("final_decision") for d in detected)
    adopted = [d for d in detected if d.get("final_decision") == "kept"]

    facts = [
        f"{len(detected)} kernel(s) detected, {selected} selected for "
        f"optimization, GEAK touched {geak_touched}, Forge touched "
        f"{forge_touched}, Ave touched {ave_touched}, adopted={verdicts['kept']}, reverted={verdicts['reverted']}, "
        f"rejected={verdicts['rejected']}, attempted-no-decision={verdicts['attempted']}."
    ]
    if selected and not (geak_touched or forge_touched or ave_touched):
        facts.append(
            "Kernels were selected for optimization but neither GEAK nor "
            "Forge nor Ave recorded an attempt — kernel optimization pipeline "
            "stalled before launch."
        )
    if adopted:
        names = ", ".join(f"`{d.get('kernel_id')}`→{d.get('adopted_by') or '?'}" for d in adopted[:5])
        facts.append(f"Adopted kernels: {names}.")

    decisions = [
        Decision(kind=kind, subject=subject, rationale=f"{verdicts[kind]} {said}")
        for kind, subject, said in _VERDICT_DECISIONS
        if verdicts[kind]
    ]
    if verdicts["not_optimized"] == len(detected):
        decisions.append(
            Decision(
                kind="not_attempted",
                subject="kernel_lifecycle",
                rationale=f"all {len(detected)} detected kernels left un-optimized (neither GEAK nor Forge nor Ave attempted)",
            )
        )

    return RenderedSection(
        section_id="kernel_lifecycle",
        title="Kernel Lifecycle",
        key_facts=facts,
        markdown_block=_kernel_tables(detected),
        decisions=decisions,
        warnings=[],
        skipped=False,
    )
