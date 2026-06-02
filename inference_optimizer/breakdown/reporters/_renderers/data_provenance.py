"""Data-provenance renderer.

Surfaces the ``data_provenance`` section of the breakdown — a
per-section list of source-artifact probes that explains, in one
table, why any particular section is empty (or partial). Older
breakdowns built before ``data_provenance`` shipped silently skip the
section so the renderer is backwards-compatible.

The table intentionally stays compact: section / status / populated /
missing_required / sources summary (e.g. ``5 found / 7 probed``). The
full per-probe detail lives in the JSON; operators who need it can
inspect ``session_breakdown.json`` directly.
"""

from __future__ import annotations

from typing import Any

from ..base import RenderedSection, md_table, register_renderer


def _sources_summary(sources: list[dict[str, Any]]) -> str:
    """Render a ``<found>/<total> probed`` summary, with required hits
    distinguished from optional ones so an operator can tell at a
    glance whether the missing artifacts were required or optional.
    """
    if not sources:
        return "—"
    total = len(sources)
    found = sum(1 for p in sources if p.get("found"))
    req_total = sum(1 for p in sources if p.get("required"))
    req_found = sum(1 for p in sources if p.get("required") and p.get("found"))
    return f"{found}/{total} found (required: {req_found}/{req_total})"


@register_renderer("data_provenance")
def render(breakdown: dict[str, Any]) -> RenderedSection:
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
