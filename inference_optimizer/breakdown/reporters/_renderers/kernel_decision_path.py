# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Kernel decision-path renderer — per-kid causal chain.

Surfaces ``trace_analyze → kernel_opt(GEAK|OOB) → integrate →
validate_stack`` as a single grouped section per kernel id, so the
reader can follow what the orchestrator did to one kernel without
cross-referencing the global phase timeline + kernel_lifecycle tables.

The renderer is fail-soft: when the breakdown lacks the
``kernel_decision_path`` field (older JSONs) it skips silently.
"""

from __future__ import annotations

from typing import Any

from ..base import RenderedSection, fmt_pct, md_table, register_renderer

_MAX_KIDS = 8
_MAX_STEPS_PER_KID = 12


def _fmt_duration(v: Any) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    if x < 60:
        return f"{x:.1f}s"
    return f"{x/60:.1f}min"


@register_renderer("kernel_decision_path")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    raw = breakdown.get("kernel_decision_path")
    if raw is None:
        # Field not present at all (old JSON) — render as skipped without
        # a warning so the section silently disappears on legacy data.
        return RenderedSection(
            section_id="kernel_decision_path",
            title="Kernel Decision Path",
            skipped=True,
        )
    entries: list[dict[str, Any]] = [e for e in raw if isinstance(e, dict)]
    if not entries:
        return RenderedSection(
            section_id="kernel_decision_path",
            title="Kernel Decision Path",
            key_facts=["No kernel selection / optimization / integration recorded this session."],
            markdown_block="",
            skipped=True,
        )

    facts: list[str] = [
        f"Tracked {len(entries)} kernel(s) through the decision pipeline.",
    ]
    # Funnel-style fact: how many of these reached integrate / validate.
    n_with_kopt = sum(1 for e in entries
                       if any((s or {}).get("step") == "kernel_opt" for s in e.get("steps") or []))
    n_with_integ = sum(1 for e in entries
                        if any((s or {}).get("step") == "integrate" for s in e.get("steps") or []))
    facts.append(
        f"Funnel: selected={len(entries)} → kernel_opt={n_with_kopt} → integrate={n_with_integ}."
    )

    blocks: list[str] = []
    head = entries[:_MAX_KIDS]
    for e in head:
        kid = str(e.get("kid") or "")
        kname = str(e.get("kernel_name") or "")
        summary = e.get("summary") or {}
        header = f"**`{kid}`**" + (f" — `{kname}`" if kname else "")
        meta_line = (
            f"  - steps={summary.get('total_steps') or 0} · "
            f"backends={', '.join(f'`{b}`' for b in (summary.get('backends_attempted') or [])) or '—'} · "
            f"final=`{summary.get('final_outcome') or '—'}` · "
            f"total_duration={_fmt_duration(summary.get('total_duration_seconds'))}"
        )
        steps = e.get("steps") or []
        if not isinstance(steps, list):
            steps = []
        shown = steps[:_MAX_STEPS_PER_KID]
        rows = []
        for s in shown:
            if not isinstance(s, dict):
                continue
            rows.append([
                s.get("ts") or "",
                s.get("step") or "",
                s.get("backend") or "",
                s.get("outcome") or "",
                fmt_pct(s.get("gain_pct"), plus=True),
                _fmt_duration(s.get("duration_seconds")),
                s.get("decision_note") or "",
            ])
        table = md_table(
            ["ts", "step", "backend", "outcome", "gain", "dur", "note"], rows,
        )
        block_parts = [header, meta_line]
        if len(steps) > _MAX_STEPS_PER_KID:
            block_parts.append(
                f"  _Showing first {_MAX_STEPS_PER_KID} of {len(steps)} step(s)._"
            )
        if table:
            block_parts.append("")
            block_parts.append(table)
        blocks.append("\n".join(block_parts))

    md = "\n\n".join(blocks)
    if len(entries) > _MAX_KIDS:
        md = f"_Showing first {_MAX_KIDS} of {len(entries)} kernel(s)._\n\n" + md

    return RenderedSection(
        section_id="kernel_decision_path",
        title="Kernel Decision Path",
        key_facts=facts,
        markdown_block=md,
        skipped=False,
    )
