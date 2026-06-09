# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Kernel lifecycle renderer — one row per detected kernel with its full optimization lifecycle.

Columns mirror the MAE dashboard contract. Skipped when no kernels were
detected (implies the profile phase never ran).
"""

from __future__ import annotations

from typing import Any

from ..base import (
    Decision,
    RenderedSection,
    md_table,
    register_renderer,
)

_MAX_NAME_LEN = 70


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


@register_renderer("kernel_lifecycle")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    kl = breakdown.get("kernel_lifecycle") or {}
    raw_detected = kl.get("detected") or []
    detected: list[dict[str, Any]] = []
    for d in raw_detected:
        if not isinstance(d, dict):
            detected.append({"kernel_id": str(d)})
            continue
        if not d.get("kernel_id"):
            # Filter out anonymous placeholder rows from older collectors.
            continue
        detected.append(d)

    if not detected:
        return RenderedSection(
            section_id="kernel_lifecycle",
            title="Kernel Lifecycle",
            key_facts=["No kernels detected this session."],
            markdown_block="",
            decisions=[Decision(
                kind="not_attempted",
                subject="kernel_lifecycle",
                rationale="no detected kernels",
            )],
            warnings=[],
            skipped=True,
        )

    selected = sum(1 for d in detected if d.get("selected_for_optimization"))
    geak_touched = sum(1 for d in detected if d.get("geak"))
    oob_touched = sum(1 for d in detected if d.get("oob"))
    adopted = [d for d in detected if d.get("final_decision") == "kept"]
    reverted = [d for d in detected if d.get("final_decision") == "reverted"]
    rejected = [d for d in detected if d.get("final_decision") == "rejected"]
    attempted = [d for d in detected if d.get("final_decision") == "attempted"]

    facts: list[str] = []
    facts.append(
        f"{len(detected)} kernel(s) detected, {selected} selected for "
        f"optimization, GEAK touched {geak_touched}, OOB touched "
        f"{oob_touched}, adopted={len(adopted)}, reverted={len(reverted)}, "
        f"rejected={len(rejected)}, attempted-no-decision={len(attempted)}."
    )
    if selected and not (geak_touched or oob_touched):
        facts.append(
            "Kernels were selected for optimization but neither GEAK nor "
            "OOB lane recorded an attempt — kernel optimization pipeline "
            "stalled before launch."
        )
    if adopted:
        names = ", ".join(
            f"`{d.get('kernel_id')}`→{d.get('adopted_by') or '?'}"
            for d in adopted[:5]
        )
        facts.append(f"Adopted kernels: {names}.")

    decisions: list[Decision] = []
    if adopted:
        decisions.append(Decision(
            kind="kept", subject="kernel_lifecycle:adopted",
            rationale=f"{len(adopted)} kernel patches adopted",
        ))
    if reverted:
        decisions.append(Decision(
            kind="reverted", subject="kernel_lifecycle:reverted",
            rationale=f"{len(reverted)} kernel patches reverted after integrate",
        ))
    if rejected:
        decisions.append(Decision(
            kind="rejected", subject="kernel_lifecycle:rejected",
            rationale=f"{len(rejected)} kernel patches rejected outright",
        ))
    if not adopted and not reverted and not rejected and not attempted:
        decisions.append(Decision(
            kind="not_attempted",
            subject="kernel_lifecycle",
            rationale=(
                f"all {len(detected)} detected kernels left un-optimized "
                "(neither GEAK nor OOB attempted)"
            ),
        ))

    # Drop bw% / compute% columns when every entry is None/0 (no roofline data).
    show_bw = any(d.get("bandwidth_util_pct") for d in detected)
    show_compute = any(d.get("compute_util_pct") for d in detected)

    headers: list[str] = ["kernel_id", "name", "gpu%", "duration_us", "calls"]
    if show_bw:
        headers.append("bw%")
    if show_compute:
        headers.append("compute%")
    headers += ["selected", "GEAK", "OOB", "adopted_by", "final"]

    def _row_for(d: dict[str, Any]) -> list[Any]:
        row: list[Any] = [
            d.get("kernel_id") or "—",
            _short_name(d.get("name") or ""),
            d.get("gpu_pct"),
            d.get("duration_us"),
            d.get("call_count"),
        ]
        if show_bw:
            row.append(d.get("bandwidth_util_pct"))
        if show_compute:
            row.append(d.get("compute_util_pct"))
        row += [
            "yes" if d.get("selected_for_optimization") else "no",
            _lane_summary(d.get("geak")),
            _lane_summary(d.get("oob")),
            d.get("adopted_by") or "—",
            d.get("final_decision") or "—",
        ]
        return row

    # Partition into "actionable" (selected/touched/decided) vs "residual"
    # long-tail kernels; residual rows collapse into a <details> block.
    actionable: list[dict[str, Any]] = []
    residual: list[dict[str, Any]] = []
    for d in detected:
        if (
            d.get("selected_for_optimization")
            or d.get("geak")
            or d.get("oob")
            or (d.get("final_decision") and d["final_decision"] != "not_optimized")
        ):
            actionable.append(d)
        else:
            residual.append(d)

    parts: list[str] = []
    if actionable:
        parts.append(md_table(headers, [_row_for(d) for d in actionable]))
    else:
        parts.append(
            "_No kernel was selected for optimization, and neither "
            "GEAK nor OOB recorded an attempt. The kernel optimization "
            "pipeline did not run on this session._"
        )
    if residual:
        total_dur = sum((d.get("duration_us") or 0.0) for d in residual)
        total_gpu = sum((d.get("gpu_pct") or 0.0) for d in residual)
        # Inner blank lines are required so the table inside <details> still parses as markdown.
        parts.append("")
        parts.append(
            f"<details><summary>Show {len(residual)} residual kernels "
            f"not selected for optimization "
            f"(Σ duration_us={total_dur:.0f}, Σ gpu%={total_gpu:.2f}%)"
            "</summary>"
        )
        parts.append("")
        parts.append(md_table(headers, [_row_for(d) for d in residual]))
        parts.append("")
        parts.append("</details>")
    md = "\n".join(parts)

    return RenderedSection(
        section_id="kernel_lifecycle",
        title="Kernel Lifecycle",
        key_facts=facts,
        markdown_block=md,
        decisions=decisions,
        warnings=[],
        skipped=False,
    )
