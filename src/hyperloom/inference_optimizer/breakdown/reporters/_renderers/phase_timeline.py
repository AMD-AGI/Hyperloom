# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase timeline renderer — chronological action events (table capped at 30, newest last).

Read off the ``phase`` events on the V6 timeline, whose action rows the
dispatch and the settle each recorded as they happened. The flat projection
this replaces held only settled rows, so an action killed mid-flight read as
one that never happened; here the gap between an event's ``count`` and its
``settled`` is what was still running when the run stopped.
"""

from __future__ import annotations

from typing import Any

from ..base import RenderedSection, as_dict, md_table, register_renderer

_MAX_ROWS = 30


def _action_rows(breakdown: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten every phase event's action rows into one ordered list.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        list[dict[str, Any]]: Action rows, oldest first, each carrying the
            phase and macro-cycle of the event that dispatched it. The timeline
            is already ordered by event start, and the rows inside an event by
            dispatch, so concatenating in order is chronological.
    """
    rows: list[dict[str, Any]] = []
    for event in as_dict(breakdown).get("timeline") or []:
        if not isinstance(event, dict) or str(event.get("type") or "") != "phase":
            continue
        ext = as_dict(event.get("ext"))
        for row in as_dict(ext.get("actions")).get("rows") or []:
            if isinstance(row, dict):
                rows.append(dict(row, phase=ext.get("phase"), macro_cycle=ext.get("macro_cycle")))
    return rows


@register_renderer("phase_timeline")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the chronological phase-timeline section.

    Shows the most recent action events (capped at ``_MAX_ROWS``) as a
    table plus a per-decision histogram fact. Skipped (with a warning)
    when no action events were recorded.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered phase-timeline section.
    """
    pt = _action_rows(breakdown)
    if not pt:
        return RenderedSection(
            section_id="phase_timeline",
            title="Phase Timeline",
            key_facts=[],
            markdown_block="",
            warnings=[
                "No phase event recorded an action — no per-tick dispatch was "
                "captured. Without this, process reconstruction and timing "
                "analysis are unavailable; the LLM cannot narrate how the run "
                "unfolded tick-by-tick."
            ],
            skipped=True,
        )

    facts: list[str] = [
        f"Recorded {len(pt)} action(s); newest = `{pt[-1].get('action')}` ({pt[-1].get('decision') or 'no-decision'})."
    ]
    in_flight = sum(1 for row in pt if not row.get("status"))
    if in_flight:
        # A dispatch with no settle is an action the run was still inside when
        # it stopped, which is a different fact from one that decided nothing.
        facts.append(f"{in_flight} action(s) were dispatched and never settled.")
    head = pt[-_MAX_ROWS:] if len(pt) > _MAX_ROWS else pt
    rows = [
        [
            ev.get("settled_at") or ev.get("dispatched_at") or "",
            f"{ev.get('phase') or ''}/{ev.get('macro_cycle') or 0}",
            ev.get("action") or "",
            ev.get("decision") or "",
            ev.get("task_id") or "",
            ev.get("error_class") or "",
        ]
        for ev in head
    ]
    md = md_table(["ts", "phase/cycle", "action", "decision", "task_id", "error_class"], rows)
    if len(pt) > _MAX_ROWS:
        md = f"_Showing last {_MAX_ROWS} of {len(pt)} actions._\n\n" + md

    histo: dict[str, int] = {}
    for ev in pt:
        d = str(ev.get("decision") or "(none)")
        histo[d] = histo.get(d, 0) + 1
    facts.append(
        "Decision histogram: " + ", ".join(f"{k}={v}" for k, v in sorted(histo.items(), key=lambda kv: -kv[1]))
    )

    return RenderedSection(
        section_id="phase_timeline",
        title="Phase Timeline",
        key_facts=facts,
        markdown_block=md,
        warnings=[],
        skipped=False,
    )
