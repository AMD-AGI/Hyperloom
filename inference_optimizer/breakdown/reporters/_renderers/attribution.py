"""Attribution renderer — gain split across optimization sources.

Splits gain among backends / params / sweep / geak / oob. Surfaces:

* the source_breakdown table (kept from the collector verbatim),
* explicit "validated vs. reconstructed" call-out so report consumers
  know how much to trust the split,
* any collector ``notes`` (assumption breadcrumbs).
"""

from __future__ import annotations

from typing import Any

from ..base import Decision, RenderedSection, fmt_pct, md_table, register_renderer


@register_renderer("attribution")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the source-attribution section: gain split across sources.

    Surfaces the per-source breakdown table, the collector's authoritative
    attribution method label, and any assumption notes. Skipped when no
    per-source split is available (single-source or unmined attribution).

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered section, marked skipped when there is no
            per-source split to show.
    """
    a = breakdown.get("attribution") or {}
    sb = a.get("source_breakdown") or {}
    notes = a.get("notes") or []
    # ``attribution.method`` is the collector's authoritative provenance
    # label (validated / single_source / reconstructed / missing). Render
    # it verbatim — never substitute a more-confident-sounding string,
    # otherwise we re-introduce the very hallucination this field exists
    # to prevent. Empty / "missing" surfaces as "unknown attribution
    # method" so the audit trail is explicit.
    method_raw = a.get("method")
    method = str(method_raw) if isinstance(method_raw, str) else ""
    if method in ("", "missing"):
        method_display = "unknown attribution method"
    else:
        method_display = method

    total_v = sb.get("validated_total_pct")
    rows = [
        # explore subsumes backends+params on sessions;
        # legacy rows kept for legacy resume reports.
        ["explore",  sb.get("explore_pct_of_total"),  sb.get("explore_share_pct")],
        ["backends", sb.get("backends_pct_of_total"), sb.get("backends_share_pct")],
        ["params",   sb.get("params_pct_of_total"),   sb.get("params_share_pct")],
        ["sweep",    sb.get("sweep_pct_of_total"),    sb.get("sweep_share_pct")],
        ["geak",     sb.get("geak_pct_of_total"),     sb.get("geak_share_pct")],
        ["oob",      sb.get("oob_pct_of_total"),      sb.get("oob_share_pct")],
    ]

    facts: list[str] = []
    if total_v is not None:
        facts.append(f"Validated total gain attributed: {fmt_pct(total_v, plus=True)}.")
    facts.append(f"Attribution method: `{method_display}`.")
    has_any_split = any(r[1] not in (None, 0, 0.0) for r in rows)
    if not has_any_split:
        facts.append(
            "No per-source split available — either the session ran a "
            "single capability (single-source) or attribution mining was "
            "not executed."
        )
    for src, pct, share in rows:
        if pct in (None, 0, 0.0):
            continue
        facts.append(
            f"{src}: {fmt_pct(pct)} of total"
            + (f" ({fmt_pct(share)} share)" if share is not None else "")
        )
    for n in notes:
        facts.append(f"Note: {n}")

    decisions: list[Decision] = []
    for src, pct, _share in rows:
        if pct and pct > 0:
            decisions.append(Decision(
                kind="kept",
                subject=f"attribution:{src}",
                metric_pct=float(pct),
                rationale="contributed positive validated gain",
            ))

    # Only emit the per-source table when at least one source has a
    # non-zero attribution. Otherwise the table is five rows of zeros
    # which is misleading: callers can't tell "no attribution mined yet"
    # from "every source contributed zero".
    md = md_table(["source", "pct_of_total", "share_pct"], rows) if has_any_split else ""
    # If we have no per-source breakdown and only the headline number
    # was recorded, skip the section entirely — the headline is already
    # in the executive summary, so a one-row "validated_total_pct = ..."
    # block would be redundant noise.
    skipped = not has_any_split
    return RenderedSection(
        section_id="attribution",
        title="Source Attribution",
        key_facts=facts,
        markdown_block=md,
        decisions=decisions,
        warnings=[],
        skipped=skipped,
    )
