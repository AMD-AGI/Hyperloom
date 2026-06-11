# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Data-provenance renderer — per-section source-artifact probes explaining why a section is empty/partial.

Backwards-compatible: older breakdowns without ``data_provenance`` skip
the section.
"""

from __future__ import annotations

from typing import Any

from ..base import RenderedSection, md_table, register_renderer


def _sources_summary(sources: list[dict[str, Any]]) -> str:
    """Render a ``<found>/<total> probed`` summary, distinguishing required hits from optional."""
    if not sources:
        return "—"
    total = len(sources)
    found = sum(1 for p in sources if p.get("found"))
    req_total = sum(1 for p in sources if p.get("required"))
    req_found = sum(1 for p in sources if p.get("required") and p.get("found"))
    return f"{found}/{total} found (required: {req_found}/{req_total})"


@register_renderer("data_provenance")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the data-provenance section as a per-section probe table.

    Shows, for each tracked section, whether it was populated and which
    required source artifacts were missing, so empty/partial sections are
    explainable. Skipped on breakdowns built before provenance shipped.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered section, or a skipped placeholder when
            no provenance entries exist.
    """
    entries_raw = breakdown.get("data_provenance")
    entries: list[dict[str, Any]] = (
        entries_raw if isinstance(entries_raw, list) else []
    )
    if not entries:
        return RenderedSection(
            section_id="data_provenance",
            title="Data Provenance",
            skipped=True,
        )

    rows: list[list[Any]] = []
    empty_sections: list[str] = []
    partial_sections: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        section = entry.get("section") or "(unknown)"
        status = entry.get("status") or "(unknown)"
        populated = entry.get("populated")
        sources = entry.get("sources") or []
        missing = entry.get("missing_required") or []
        rows.append([
            section,
            status,
            "yes" if populated else "no",
            ", ".join(str(m) for m in missing) if missing else "—",
            _sources_summary(sources),
        ])
        if status == "empty":
            empty_sections.append(section)
        elif status == "partial":
            partial_sections.append(section)

    md = md_table(
        ["section", "status", "populated", "missing_required", "sources"],
        rows,
    )

    facts: list[str] = [
        f"Data provenance tracked across {len(entries)} section(s).",
    ]
    if empty_sections:
        facts.append(
            "Sections empty due to missing required sources: "
            + ", ".join(f"`{s}`" for s in empty_sections)
        )
    if partial_sections:
        facts.append(
            "Sections partial (some required sources missing): "
            + ", ".join(f"`{s}`" for s in partial_sections)
        )

    return RenderedSection(
        section_id="data_provenance",
        title="Data Provenance",
        key_facts=facts,
        markdown_block=md,
        skipped=False,
    )
