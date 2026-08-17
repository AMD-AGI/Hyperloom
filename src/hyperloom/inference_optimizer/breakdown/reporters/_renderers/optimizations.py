# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical optimization renderer."""

from __future__ import annotations

from typing import Any

from ..base import Decision, RenderedSection, fmt_pct, md_table, register_renderer


def _count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _reconciliation_notes(validation: dict[str, Any]) -> list[str]:
    """Say what the per-source totals do not add up to, and why.

    The table sums what each source claims. The session moved by a different
    amount, and a reader with only the table in front of them has no way to
    see the difference or where it went.

    Args:
        validation: The ``optimizations.validation`` block.

    Returns:
        Notes to attach to the section, empty when nothing was left out.
    """
    notes: list[str] = []
    reconciliation = validation.get("reconciliation_gap_pct")
    if isinstance(reconciliation, (int, float)) and abs(float(reconciliation)) > 0.01:
        notes.append(
            f"The run promoted {fmt_pct(validation.get('validated_total_gain_pct'), plus=True)} "
            "as validated, but the adopted steps add up to "
            f"{fmt_pct(validation.get('ledger_total_gain_pct'), plus=True)}. Steps are "
            "missing from the ledger, or their recorded throughputs disagree "
            "with the end-to-end measurement."
        )
    elif validation.get("method") == "ledger_sum":
        notes.append(
            "The session total is the sum of the steps below, not an "
            "independent measurement, so the two cannot be checked against "
            "each other."
        )
    unattributed = float(validation.get("unattributed_gain_pct") or 0.0)
    if abs(unattributed) > 0.01:
        validated = float(validation.get("validated_total_gain_pct") or 0.0)
        notes.append(
            f"The session moved {fmt_pct(validated, plus=True)} end to end, and "
            f"{fmt_pct(unattributed, plus=True)} of that belongs to no adopted "
            "step: the workload was already somewhere other than where the "
            "previous step left it. It is reported here rather than credited "
            "to whichever step ran next."
        )
    unmeasured = _count(validation.get("unmeasured_keep_count"))
    if unmeasured:
        notes.append(
            f"{unmeasured} adopted step(s) recorded neither a throughput nor a "
            "gain, so they contribute nothing to the totals above."
        )
    projected = _count(validation.get("projected_keep_count"))
    if projected:
        notes.append(
            f"{projected} adopted step(s) recorded no finishing throughput; "
            "their contribution was reconstructed from the executor's own "
            "percentage, so any drift across them is invisible."
        )
    unscored = _count(validation.get("unscored_keep_count"))
    if unscored:
        notes.append(
            f"{unscored} adopted step(s) were kept on the verdict alone, with "
            "no accuracy gate having ruled on them."
        )
    stale = _count(validation.get("stale_evidence_count"))
    if stale:
        notes.append(
            f"{stale} adoption(s) cite measurements that were later written "
            "over. The frozen numbers still stand; the trail back to them "
            "does not."
        )
    unclaimed = _count(validation.get("unclaimed_integration_count"))
    if unclaimed:
        # Placed after the unattributed note on purpose: this is the reason to
        # doubt that figure rather than another item beside it.
        notes.append(
            f"{unclaimed} change(s) are recorded as integrated with nothing "
            "crediting them. Whatever they earned is inside the unattributed "
            "figure above, so that figure overstates how much of the session "
            "genuinely belongs to no step."
        )
    return notes


@register_renderer("optimizations")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render adopted optimizations from the single canonical read model."""

    optimizations = breakdown.get("optimizations") or {}
    entries = [
        entry
        for entry in optimizations.get("entries") or []
        if isinstance(entry, dict)
    ]
    validation = optimizations.get("validation") or {}

    if optimizations.get("available") is False:
        # Rendering nothing here is what let a session with no records pass for
        # a session with no optimizations.
        reason = str(optimizations.get("unavailable_reason") or "no reason recorded")
        return RenderedSection(
            section_id="optimizations",
            title="Adopted Optimizations",
            key_facts=[f"No optimization read model was produced: {reason}."],
            markdown_block="",
            decisions=[],
            warnings=[
                "This section is absent, not empty. Nothing here says the "
                "session optimized nothing."
            ],
            skipped=False,
        )

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
        warnings=_reconciliation_notes(validation),
        skipped=not entries,
    )
