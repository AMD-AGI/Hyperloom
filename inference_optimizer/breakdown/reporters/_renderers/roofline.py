# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Roofline comparison renderer — one block per discovered ``final.json``.

Silently skipped when ``roofline`` is absent/empty (pre-P3-roofline JSONs).
"""

from __future__ import annotations

from typing import Any

from ..base import (
    RenderedSection,
    fmt_pct,
    md_kv_list,
    md_table,
    register_renderer,
)


def _snapshot_kv(label: str, snap: dict[str, Any] | None) -> str:
    """Render one roofline snapshot as a labelled key-value block.

    Args:
        label (str): The block heading (e.g. ``"Baseline"`` or ``"Latest"``).
        snap (dict[str, Any] | None): The snapshot record, including an
            optional ``top_kernel`` sub-dict.

    Returns:
        str: The markdown block, or an empty string when the snapshot is
            missing or has no displayable fields.
    """
    if not isinstance(snap, dict) or not snap:
        return ""
    tk = snap.get("top_kernel") or {}
    items = [
        ("snapshot_id",    snap.get("snapshot_id")),
        ("ts",             snap.get("ts")),
        ("compute_pct",    snap.get("compute_pct")),
        ("idle_pct",       snap.get("idle_pct")),
        ("comm_pct",       snap.get("comm_pct")),
        ("top_bottleneck", snap.get("top_bottleneck")),
        ("top_kernel.name",          tk.get("name") if isinstance(tk, dict) else None),
        ("top_kernel.gpu_pct",       tk.get("gpu_pct") if isinstance(tk, dict) else None),
        ("top_kernel.efficiency_pct", tk.get("efficiency_pct") if isinstance(tk, dict) else None),
        ("top_kernel.bound_type",     tk.get("bound_type") if isinstance(tk, dict) else None),
    ]
    body = md_kv_list(items)
    if not body:
        return ""
    return f"**{label}**\n\n{body}"


def _delta_block(delta: dict[str, Any] | None) -> str:
    """Render the roofline ``delta`` mapping as a two-column table.

    Args:
        delta (dict[str, Any] | None): Field-to-value delta mapping.

    Returns:
        str: A markdown ``field``/``value`` table, or an empty string when the
            delta is missing or empty.
    """
    if not isinstance(delta, dict) or not delta:
        return ""
    rows = []
    for k, v in delta.items():
        rows.append([k, v])
    if not rows:
        return ""
    return "**Delta**\n\n" + md_table(["field", "value"], rows)


@register_renderer("roofline")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the roofline-comparison section, one block per ``final.json``.

    Each block shows the source path, comparison mode, baseline and latest
    snapshots, and any emitted delta values. Skipped when the breakdown has
    no ``roofline`` list.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered roofline section, or a skipped
            placeholder when there are no entries.
    """
    entries_raw = breakdown.get("roofline")
    entries: list[dict[str, Any]] = entries_raw if isinstance(entries_raw, list) else []
    if not entries:
        return RenderedSection(
            section_id="roofline",
            title="Roofline",
            skipped=True,
        )

    facts: list[str] = [f"Roofline final.json files surfaced: {len(entries)}."]
    parts: list[str] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        src = entry.get("source_path") or "(unknown path)"
        mode = entry.get("mode") or "(unspecified)"
        parts.append(f"#### Roofline #{idx + 1} — `{src}` (mode: `{mode}`)")
        parts.append("")
        baseline = entry.get("baseline")
        latest = entry.get("latest")
        delta = entry.get("delta")
        for label, snap in (("Baseline", baseline), ("Latest", latest)):
            block = _snapshot_kv(label, snap if isinstance(snap, dict) else None)
            if block:
                parts.append(block)
                parts.append("")
        delta_md = _delta_block(delta if isinstance(delta, dict) else None)
        if delta_md:
            parts.append(delta_md)
            parts.append("")
        # One-liner fact so the summary can cite the signal without the table.
        if isinstance(baseline, dict):
            tk = baseline.get("top_kernel") or {}
            facts.append(
                f"Roofline #{idx + 1} baseline top kernel: "
                f"{tk.get('name') or '(none)'} @ {fmt_pct(tk.get('gpu_pct'))} GPU, "
                f"efficiency {fmt_pct(tk.get('efficiency_pct'))} "
                f"(bound: {tk.get('bound_type') or 'unknown'})."
            )
    return RenderedSection(
        section_id="roofline",
        title="Roofline",
        key_facts=facts,
        markdown_block="\n".join(parts).strip(),
        skipped=False,
    )
