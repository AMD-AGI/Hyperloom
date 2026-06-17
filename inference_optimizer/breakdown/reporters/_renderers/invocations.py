# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""GEAK + OOB invocation renderer — one ``_render_pair`` feeding two section ids (geak/oob share the shape)."""

from __future__ import annotations

from typing import Any

from ..base import Decision, RenderedSection, md_table, register_renderer

_MAX_ATTEMPT_ROWS = 25


def _render_pair(
    breakdown: dict[str, Any],
    *,
    section_id: str,
    title: str,
    invocations_key: str,
    legacy_key: str,
) -> RenderedSection:
    """Render either GEAK or OOB invocations (new ``invocations`` key, legacy fallback).

    Args:
        breakdown: The full ``session_breakdown.json`` dict.
        section_id: Section identifier for the rendered block.
        title: Human-readable section title.
        invocations_key: Key under ``invocations`` to read records from.
        legacy_key: Top-level fallback key for older breakdowns.

    Returns:
        The rendered section, or a skipped placeholder when no invocations
        are present.
    """
    raw = (
        (breakdown.get("invocations") or {}).get(invocations_key)
        or breakdown.get(legacy_key)
        or []
    )
    # Normalize stray string entries (kernel ids) into dicts.
    invs: list[dict[str, Any]] = []
    for v in raw:
        if isinstance(v, dict):
            invs.append(v)
        else:
            invs.append({"kernel_id": str(v)})
    if not invs:
        return RenderedSection(
            section_id=section_id,
            title=title,
            key_facts=[
                f"`{section_id}` never invoked this session — no attempts on disk."
            ],
            markdown_block="",
            decisions=[Decision(
                kind="not_attempted",
                subject=section_id,
                rationale="no attempts found in session_breakdown",
            )],
            warnings=[],
            skipped=True,
        )

    keeps = sum(1 for v in invs if v.get("decision") == "KEEP")
    failed = sum(1 for v in invs if v.get("decision") in ("FAILED", "ERROR"))
    decisions = [Decision(
        kind="kept" if keeps else ("attempted" if invs else "not_attempted"),
        subject=section_id,
        metric_pct=None,
        rationale=f"{keeps} KEEP / {failed} FAILED across {len(invs)} attempts",
    )]
    facts = [
        f"{section_id}: {len(invs)} invocation(s), {keeps} KEEP, {failed} FAILED.",
    ]
    head = invs[-_MAX_ATTEMPT_ROWS:] if len(invs) > _MAX_ATTEMPT_ROWS else invs
    rows = []
    for v in head:
        rows.append([
            v.get("ts") or "",
            v.get("kernel_id") or v.get("kernel_name") or "",
            v.get("decision") or "",
            v.get("micro_speedup") or "",
            v.get("workspace") or v.get("workspace_path") or "",
            (v.get("error") or "")[:80],
        ])
    md = md_table(
        ["ts", "kernel_id", "decision", "micro_speedup", "workspace", "error"],
        rows,
    )
    if len(invs) > _MAX_ATTEMPT_ROWS:
        md = f"_Showing last {_MAX_ATTEMPT_ROWS} of {len(invs)} attempts._\n\n" + md

    return RenderedSection(
        section_id=section_id,
        title=title,
        key_facts=facts,
        markdown_block=md,
        decisions=decisions,
        warnings=[],
        skipped=False,
    )


@register_renderer("geak_invocations")
def render_geak(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the GEAK invocations section.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered GEAK invocations section.
    """
    return _render_pair(
        breakdown,
        section_id="geak_invocations",
        title="GEAK Invocations",
        invocations_key="geak",
        legacy_key="geak_invocations",
    )


@register_renderer("oob_invocations")
def render_oob(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the OOB (out-of-box) invocations section.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered OOB invocations section.
    """
    return _render_pair(
        breakdown,
        section_id="oob_invocations",
        title="OOB Invocations",
        invocations_key="oob",
        legacy_key="oob_invocations",
    )
