# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical optimization renderer."""

from __future__ import annotations

from typing import Any

from ..base import Decision, RenderedSection, fmt_pct, md_table, register_renderer


@register_renderer("optimizations")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render adopted optimizations from the single canonical read model."""

    optimizations = breakdown.get("optimizations") or {}
    entries = [
        entry
        for entry in optimizations.get("entries") or []
        if isinstance(entry, dict)
    ]
    summary = optimizations.get("summary_by_source") or {}
    rows: list[list[Any]] = []
    decisions: list[Decision] = []
    for source, bucket in summary.items():
        if not isinstance(bucket, dict):
            continue
        keeps = int(bucket.get("keeps") or 0)
        gain = bucket.get("total_gain_pct")
        rows.append([source, keeps, gain])
        if keeps > 0:
            decisions.append(
                Decision(
                    kind="kept",
                    subject=f"optimizations:{source}",
                    metric_pct=float(gain or 0.0),
                    rationale=f"{keeps} validated optimization(s)",
                )
            )

    facts = [
        f"{len(entries)} adopted optimization entr"
        f"{'y' if len(entries) == 1 else 'ies'} recorded."
    ]
    validated = [entry for entry in entries if entry.get("validated") is True]
    facts.append(f"{len(validated)} entr{'y is' if len(validated) == 1 else 'ies are'} validated.")
    positive = [
        bucket
        for bucket in summary.values()
        if isinstance(bucket, dict) and bucket.get("total_gain_pct")
    ]
    if positive:
        facts.append(
            "Validated gain represented in canonical source summaries: "
            + fmt_pct(
                sum(float(bucket.get("total_gain_pct") or 0.0) for bucket in positive),
                plus=True,
            )
            + "."
        )

    return RenderedSection(
        section_id="optimizations",
        title="Adopted Optimizations",
        key_facts=facts,
        markdown_block=md_table(
            ["source", "validated_keeps", "total_gain_pct"],
            rows,
        ),
        decisions=decisions,
        warnings=[],
        skipped=not entries,
    )

