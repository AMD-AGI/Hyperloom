# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Adopted-optimization renderer, read off the canonical stack ledger.

Every figure here is read from ``outcome.validation``, which the stack timeline
event recorded as each adoption was accepted. The section used to be projected
from three v4 entity streams and published eight guard counts to report where
the three disagreed; six of those counts have no referent once a fact is
recorded at the moment it becomes true, and the two that remain are about the
measurements rather than about the bookkeeping.
"""

from __future__ import annotations

from typing import Any

from ..base import Decision, RenderedSection, as_dict, fmt_pct, md_table, register_renderer, validation_of


@register_renderer("optimizations")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render adopted optimizations from the stack ledger.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered section. Skipped when the ledger holds no
            adoptions, which is a session that kept nothing rather than a
            session whose records are missing -- the two are told apart by
            ``attribution.available``.
    """
    validation = validation_of(breakdown)
    attribution = as_dict(validation.get("attribution"))

    if not attribution.get("available"):
        # Rendering nothing here is what let a session with no records pass for
        # a session with no optimizations.
        return RenderedSection(
            section_id="optimizations",
            title="Adopted Optimizations",
            key_facts=["No stack ledger was recorded, so no adoption can be reported."],
            markdown_block="",
            decisions=[],
            warnings=["This section is absent, not empty. Nothing here says the session optimized nothing."],
            skipped=False,
        )

    rows: list[list[Any]] = []
    decisions: list[Decision] = []
    for source, bucket in as_dict(attribution.get("by_source")).items():
        bucket = as_dict(bucket)
        keeps = int(bucket.get("keep_count") or 0)
        gain = bucket.get("total_gain_pct")
        unmeasured = int(bucket.get("unmeasured_keep_count") or 0)
        rows.append([source, keeps, gain, unmeasured or None])
        if keeps > 0:
            decisions.append(
                Decision(
                    kind="kept",
                    subject=f"optimizations:{source}",
                    metric_pct=float(gain or 0.0),
                    rationale=f"{keeps} adoption(s) in the ledger",
                )
            )

    adopted = int(validation.get("adoption_count") or 0)
    facts = [f"{adopted} adoption(s) recorded in the stack ledger."]
    attributed = validation.get("attributed_gain_pct")
    if isinstance(attributed, (int, float)):
        # Every contribution is measured against the session baseline, the one
        # denominator they share, so this sum is exact rather than indicative.
        facts.append(f"Their contributions sum to {fmt_pct(attributed, plus=True)} against the session baseline.")
    stack_len = validation.get("validated_at_stack_len")
    if stack_len is not None:
        facts.append(
            f"The whole stack was last measured at length {stack_len}: "
            f"{fmt_pct(validation.get('validated_total_gain_pct'), plus=True)}."
        )

    return RenderedSection(
        section_id="optimizations",
        title="Adopted Optimizations",
        key_facts=facts,
        markdown_block=md_table(
            ["source", "adoptions", "total_gain_pct", "unmeasured"],
            rows,
        ),
        decisions=decisions,
        # Findings the ledger's own figures support, computed where the ledger
        # is read rather than restated per renderer.
        warnings=[str(note) for note in validation.get("notes") or []],
        skipped=not adopted,
    )
