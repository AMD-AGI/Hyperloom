# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Roofline comparison renderer — the baseline-vs-latest comparison built from
the snapshots the ``roofline`` timeline events recorded.

Each roofline run appends one snapshot as its own quantitative conclusion, so
the session's history is the snapshots in event order and the comparison is
its first against its last. The flat projection this replaces read the same
history out of a capped session-state list that later runs evict entries from.

Silently skipped when no roofline run recorded a snapshot, i.e. a session that
never ran the roofline pipeline.
"""

from __future__ import annotations

from typing import Any

from ..base import (
    RenderedSection,
    as_dict,
    dict_rows,
    events_of,
    fmt_pct,
    md_kv_list,
    md_table,
    register_renderer,
)


def _snapshots(breakdown: dict[str, Any]) -> list[dict[str, Any]]:
    """Every snapshot the session's roofline runs recorded, oldest first.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        list[dict[str, Any]]: The snapshots, in the order the runs that
            produced them appear on the timeline. A failed run recorded none
            and contributes nothing.
    """
    out: list[dict[str, Any]] = []
    for event in events_of(breakdown, "roofline"):
        for action in dict_rows(as_dict(event.get("ext")).get("actions")):
            snapshot = as_dict(action.get("outcome")).get("snapshot")
            if isinstance(snapshot, dict) and snapshot:
                out.append(snapshot)
    return out


def _snapshot_kv(label: str, snap: dict[str, Any] | None) -> str:
    """Render one roofline snapshot as a labelled key-value block."""
    if not isinstance(snap, dict) or not snap:
        return ""
    tk = as_dict(snap.get("top_kernel"))
    items = [
        ("snapshot_id", snap.get("snapshot_id")),
        ("ts", snap.get("ts")),
        ("compute_pct", snap.get("compute_pct")),
        ("idle_pct", snap.get("idle_pct")),
        ("comm_pct", snap.get("comm_pct")),
        ("top_bottleneck", snap.get("top_bottleneck")),
        ("top_kernel.name", tk.get("name")),
        ("top_kernel.gpu_pct", tk.get("gpu_pct")),
        ("top_kernel.efficiency_pct", tk.get("efficiency_pct")),
        ("top_kernel.bound_type", tk.get("bound_type")),
    ]
    body = md_kv_list(items)
    if not body:
        return ""
    return f"**{label}**\n\n{body}"


def _delta_block(delta: dict[str, Any] | None) -> str:
    """Render the roofline ``delta`` mapping as a two-column table."""
    if not isinstance(delta, dict) or not delta:
        return ""
    rows = [[key, value] for key, value in delta.items()]
    if not rows:
        return ""
    return "**Delta**\n\n" + md_table(["field", "value"], rows)


@register_renderer("roofline")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the roofline-comparison section.

    Shows the session's first and last roofline snapshots and the delta
    between them. Skipped when no roofline run recorded a snapshot.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered roofline section, or a skipped
            placeholder when the pipeline never ran.
    """
    snapshots = _snapshots(breakdown)
    if not snapshots:
        return RenderedSection(
            section_id="roofline",
            title="Roofline",
            skipped=True,
        )

    # The one definition of which ceilings may be compared and which deltas
    # survive a moved one. Re-deriving it here is how the report would come to
    # disagree with the analysis it is reporting on.
    from hyperloom.orchestrator.kernel.roofline_snapshot import build_roofline_comparison_from_history

    comparison = as_dict(build_roofline_comparison_from_history(snapshots))
    baseline = as_dict(comparison.get("baseline"))
    latest = as_dict(comparison.get("latest"))
    mode = str(comparison.get("mode") or "single_snapshot")

    facts: list[str] = [f"Roofline snapshots recorded: {len(snapshots)} (comparison mode: {mode})."]
    parts: list[str] = []
    for label, snap in (("Baseline", baseline), ("Latest", latest)):
        block = _snapshot_kv(label, snap)
        if block:
            parts.append(block)
            parts.append("")
    delta_md = _delta_block(as_dict(comparison.get("delta")))
    if delta_md:
        parts.append(delta_md)
        parts.append("")
    if comparison.get("ceilings_comparable") is False:
        facts.append(
            "The two snapshots were taken against different ceilings, so the "
            "saturation deltas are withheld: across a moved ceiling they would "
            "report a denominator change as a saturation change."
        )
    tk = as_dict(baseline.get("top_kernel"))
    facts.append(
        f"Baseline top kernel: {tk.get('name') or '(none)'} @ "
        f"{fmt_pct(tk.get('gpu_pct'))} GPU, efficiency "
        f"{fmt_pct(tk.get('efficiency_pct'))} "
        f"(bound: {tk.get('bound_type') or 'unknown'})."
    )
    return RenderedSection(
        section_id="roofline",
        title="Roofline",
        key_facts=facts,
        markdown_block="\n".join(parts).strip(),
        skipped=False,
    )
