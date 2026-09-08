"""Canonical ``analysis.md`` renderer for the trace-analysis routes that build
the report themselves rather than having a model write it (today: bypass).

The report is human/downstream-readable and is NOT the LLM-agent parser
contract. This module is the single source of truth for its section structure
and table schemas: a route normalizes its own data into the inputs below and
unmodeled cells render as an em dash rather than a fabricated value.
Route-specific detail is appended verbatim via ``extra_sections`` after the
shared sections, under a divider.
"""

from __future__ import annotations

from typing import Any

from _kernel_category import canonical_category

#: Placeholder for a cell a route does not model (keeps every table shape aligned).
DASH = "\u2014"

#: Canonical section headings, in order.
EXEC_SUMMARY_HEADING = "## Executive Summary"
SYSTEM_SIGNALS_HEADING = "## System-Level Signals"
TOP_HOT_KERNELS_HEADING = "## Top Hot Kernels"

#: Column header row for the shared Top Hot Kernels table.
TOP_HOT_KERNELS_COLUMNS = "| Rank | Operation | Time (us) | GPU% | Eff% | AI | Bound | Category | Source File |"
_TOP_HOT_KERNELS_SEP = "|------|-----------|-----------|------|------|----|-------|----------|-------------|"

#: Column header row for the shared per-P-item Data table.
P_ITEM_COLUMNS = (
    "| Operation | Time (us) | GPU% | %E2E | Count | FLOPS/Byte | "
    "Efficiency | Bound | Args | Source File | Kernel Path (launcher) |"
)
_P_ITEM_SEP = (
    "|-----------|-----------|------|------|-------|------------|"
    "------------|-------|------|-------------|------------------------|"
)


def _pct(value: Any) -> str:
    """Render a percentage cell (``DASH`` when null/non-numeric)."""
    if isinstance(value, bool) or value is None:
        return DASH
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return DASH


def _num(value: Any, *, nd: int = 1) -> str:
    """Render a numeric cell to ``nd`` decimals (``DASH`` when null/non-numeric)."""
    if isinstance(value, bool) or value is None:
        return DASH
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return DASH


def _text(value: Any) -> str:
    """Render a text cell (``DASH`` when empty/null)."""
    if value is None:
        return DASH
    s = str(value).strip()
    return s or DASH


def render_report(
    *,
    route: str,
    model_name: str,
    provenance_detail: str,
    exec_summary: dict[str, Any],
    system_signals: dict[str, Any],
    idle_threshold: float,
    hot_kernels: list[dict[str, Any]],
    p_items: list[dict[str, Any]],
    extra_sections: str = "",
) -> str:
    """Render the canonical ``analysis.md`` body.

    Args:
        route: Route id for the provenance line (e.g. ``bypass``).
        model_name: Model identifier for the title (blank -> ``Workload``).
        provenance_detail: Route-specific trailing sentence for the provenance line.
        exec_summary: ``{total_gpu_time_ms, gpu_busy_pct, gpu_idle_pct,
            gpu_memcpy_ms, top_bottleneck_category, attribution_pct}`` (any may be
            ``None`` -> em dash).
        system_signals: ``{idle_pct, exposed_comm_pct, exposed_memcpy_pct}`` (any
            may be ``None`` -> row shows an em dash / is still emitted).
        idle_threshold: Idle-gate threshold for the idle-signal note.
        hot_kernels: Rows with ``name, time_us, gpu_pct, efficiency_percent,
            arithmetic_intensity, bound_type, category, source_file`` (ranked as
            given).
        p_items: ``{rank, category, rows[...]}`` groups; each row carries
            ``name, time_us, gpu_pct, e2e_pct, call_count, flops_per_byte,
            efficiency_percent, bound_type, args, source_file, kernel_path``.
        extra_sections: Pre-rendered markdown appended after the shared sections
            (route-specific detail), under a divider.

    Returns:
        The full canonical markdown report text.
    """
    lines: list[str] = []
    title = f"# Performance Analysis Report \u2014 {model_name}" if model_name else "# Performance Analysis Report"
    lines.append(title)
    lines.append("")
    prov = f"> Generated via {route} route (HYPERLOOM_TRACE_ANALYSIS_ROUTE={route})."
    if provenance_detail:
        prov += f" {provenance_detail.strip()}"
    lines.append(prov)
    lines.append("")

    # Executive Summary.
    lines.append(EXEC_SUMMARY_HEADING)
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    total_ms = exec_summary.get("total_gpu_time_ms")
    memcpy_ms = exec_summary.get("gpu_memcpy_ms")
    lines.append(f"| Total GPU Time | {_num(total_ms, nd=3)} ms |")
    lines.append(f"| GPU Busy % | {_pct(exec_summary.get('gpu_busy_pct'))} |")
    lines.append(f"| GPU Idle % | {_pct(exec_summary.get('gpu_idle_pct'))} |")
    lines.append(f"| GPU MemCpy | {_num(memcpy_ms, nd=3)} ms |")
    lines.append(
        f"| Top Bottleneck Category | {_text(canonical_category(exec_summary.get('top_bottleneck_category')))} |"
    )
    lines.append(f"| Op-attribution Coverage | {_pct(exec_summary.get('attribution_pct'))} |")
    lines.append("")

    # System-Level Signals.
    lines.append(SYSTEM_SIGNALS_HEADING)
    lines.append("")
    lines.append("| Signal | % of total GPU time | Note |")
    lines.append("|--------|---------------------|------|")
    idle_share = system_signals.get("idle_pct")
    if isinstance(idle_share, (int, float)) and not isinstance(idle_share, bool):
        note = (
            f"above {idle_threshold:.0f}% idle gate"
            if float(idle_share) > idle_threshold
            else f"within {idle_threshold:.0f}% idle gate"
        )
    else:
        note = "-"
    lines.append(f"| GPU idle | {_pct(idle_share)} | {note} |")
    lines.append(f"| Exposed communication | {_pct(system_signals.get('exposed_comm_pct'))} | - |")
    lines.append(f"| Exposed memcpy (device copy) | {_pct(system_signals.get('exposed_memcpy_pct'))} | - |")
    lines.append("")

    # Top Hot Kernels.
    lines.append(TOP_HOT_KERNELS_HEADING)
    lines.append("")
    if hot_kernels:
        lines.append(TOP_HOT_KERNELS_COLUMNS)
        lines.append(_TOP_HOT_KERNELS_SEP)
        for i, k in enumerate(hot_kernels, start=1):
            lines.append(
                f"| {i} | {_text(k.get('name'))} | {_num(k.get('time_us'))} | {_pct(k.get('gpu_pct'))} "
                f"| {_pct(k.get('efficiency_percent'))} | {_num(k.get('arithmetic_intensity'), nd=3)} "
                f"| {_text(k.get('bound_type'))} | {_text(canonical_category(k.get('category')))} | {_text(k.get('source_file'))} |"
            )
    else:
        lines.append("_No GPU kernels found in trace._")
    lines.append("")

    # Per-P-item Data tables.
    for item in p_items:
        rank = item.get("rank", 0)
        lines.append(f"### P{rank}: {_text(canonical_category(item.get('category')))} kernels")
        lines.append("")
        lines.append(f"<!-- reasoning-candidate tier=compute rank={rank} -->")
        lines.append("")
        lines.append("**Data:**")
        lines.append("")
        lines.append(P_ITEM_COLUMNS)
        lines.append(_P_ITEM_SEP)
        for r in item.get("rows") or []:
            args = r.get("args")
            args_str = " ".join(args) if isinstance(args, list) else _text(args)
            lines.append(
                f"| {_text(r.get('name'))} | {_num(r.get('time_us'))} | {_pct(r.get('gpu_pct'))} "
                f"| {_num(r.get('e2e_pct'), nd=2)} | {_text(r.get('call_count'))} "
                f"| {_num(r.get('flops_per_byte'), nd=3)} | {_pct(r.get('efficiency_percent'))} "
                f"| {_text(r.get('bound_type'))} | {args_str} | {_text(r.get('source_file'))} "
                f"| {_text(r.get('kernel_path'))} |"
            )
        lines.append("")

    body = "\n".join(lines)
    if extra_sections and extra_sections.strip():
        body += (
            "\n---\n\n"
            "_Additional route-specific detail below (not part of the shared "
            "cross-route sections above)._\n\n" + extra_sections.strip() + "\n"
        )
    return body
