"""Phase timeline renderer — chronological action events.

We cap the table at 30 entries (newest last) to keep the report
readable. The deterministic markdown block still shows total count so
operators know the table is truncated.
"""

from __future__ import annotations

from typing import Any

from ..base import RenderedSection, md_table, register_renderer

_MAX_ROWS = 30


@register_renderer("phase_timeline")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the chronological phase-timeline section.

    Shows the most recent action events (capped at ``_MAX_ROWS``) as a
    table plus a per-decision histogram fact. Skipped (with a warning)
    when no per-tick events were captured.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered phase-timeline section.
    """
    raw_pt = breakdown.get("phase_timeline") or []
    pt: list[dict[str, Any]] = [
        ev if isinstance(ev, dict) else {"action": str(ev)} for ev in raw_pt
    ]
    if not pt:
        return RenderedSection(
            section_id="phase_timeline",
            title="Phase Timeline",
            key_facts=[],
            markdown_block="",
            warnings=[
                "phase_timeline empty — no per-tick action events captured. "
                "Without this, process reconstruction and timing analysis are "
                "unavailable; the LLM cannot narrate how the run unfolded "
                "tick-by-tick."
            ],
            skipped=True,
        )

    facts: list[str] = [
        f"Recorded {len(pt)} phase event(s); newest = `{pt[-1].get('action')}` "
        f"({pt[-1].get('decision') or 'no-decision'})."
    ]
    head = pt[-_MAX_ROWS:] if len(pt) > _MAX_ROWS else pt
    rows = []
    for ev in head:
        rows.append([
            ev.get("ts") or ev.get("timestamp") or "",
            ev.get("action") or "",
            ev.get("decision") or "",
            ev.get("task_id") or ev.get("variant_name") or "",
            ev.get("error_class") or "",
        ])
    md = md_table(["ts", "action", "decision", "task / variant", "error_class"], rows)
    if len(pt) > _MAX_ROWS:
        md = f"_Showing last {_MAX_ROWS} of {len(pt)} events._\n\n" + md

    # Per-decision histogram, useful at a glance.
    histo: dict[str, int] = {}
    for ev in pt:
        d = str(ev.get("decision") or "(none)")
        histo[d] = histo.get(d, 0) + 1
    facts.append("Decision histogram: " + ", ".join(
        f"{k}={v}" for k, v in sorted(histo.items(), key=lambda kv: -kv[1])
    ))

    return RenderedSection(
        section_id="phase_timeline",
        title="Phase Timeline",
        key_facts=facts,
        markdown_block=md,
        warnings=[],
        skipped=False,
    )
