# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Attribution renderer — gain split across optimization sources."""

from __future__ import annotations

from typing import Any

from ..base import Decision, RenderedSection, as_dict, fmt_pct, md_table, register_renderer, validation_of

#: Below this the residue is float noise from re-serialized throughputs, well
#: under any measurement's own repeatability.
_NOISE_PP = 0.01


def _claimed_gains(validation: dict[str, Any], attribution: dict[str, Any]) -> list[list[Any]]:
    """Each source's ``[source, gain]``, then the residue between adopted steps when it is more than noise."""
    claimed = [
        [source, as_dict(bucket).get("total_gain_pct")]
        for source, bucket in as_dict(attribution.get("by_source")).items()
    ]
    unattributed = validation.get("unattributed_gain_pct")
    if isinstance(unattributed, (int, float)) and abs(float(unattributed)) > _NOISE_PP:
        claimed.append(["unattributed (between adopted steps)", unattributed])
    return claimed


def _share_rows(claimed: list[list[Any]], total: Any) -> list[list[Any]]:
    """Each claimed gain as ``[source, gain, share_pct]``, the share taken of ``total`` or else of the claims' sum."""
    share_total = total
    if not isinstance(share_total, (int, float)) or not share_total:
        share_total = sum(float(gain) for _source, gain in claimed if isinstance(gain, (int, float)))
    return [
        [
            source,
            gain,
            (float(gain) / float(share_total) * 100.0 if isinstance(gain, (int, float)) and share_total else None),
        ]
        for source, gain in claimed
    ]


def _contributes(gain: Any) -> bool:
    """Whether a source's gain is a measurable contribution."""
    return gain not in (None, 0, 0.0)


def _contribution_facts(rows: list[list[Any]]) -> list[str]:
    """One fact per contributing source, with its share when one could be taken."""
    facts: list[str] = []
    for src, pct, share in rows:
        if not _contributes(pct):
            continue
        facts.append(f"{src}: {fmt_pct(pct)} of total" + (f" ({fmt_pct(share)} share)" if share is not None else ""))
    return facts


def _kept_decisions(rows: list[list[Any]]) -> list[Decision]:
    """A kept decision per source that contributed positive validated gain."""
    # The unattributed row is a residue, not a contributor; crediting it as a decision would put "nobody" on the
    # leaderboard.
    return [
        Decision(
            kind="kept",
            subject=f"attribution:{src}",
            metric_pct=float(pct),
            rationale="contributed positive validated gain",
        )
        for src, pct, _share in rows
        if pct and pct > 0 and not src.startswith("unattributed")
    ]


@register_renderer("attribution")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the source-attribution section: gain split across sources.

    The shares are taken against what the session actually moved, so a source
    that claims half the gain reads as half. That denominator is larger than
    the sum of the claims whenever the workload moved between adopted steps,
    and the difference gets its own row: dropping it would leave shares that
    silently fail to reach 100% with nothing to say why.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered section, marked skipped when there is no
            per-source split to show.
    """
    validation = validation_of(breakdown)
    attribution = as_dict(validation.get("attribution"))
    total_v = validation.get("validated_total_gain_pct")
    rows = _share_rows(_claimed_gains(validation, attribution), total_v)

    facts: list[str] = []
    if total_v is not None:
        facts.append(f"Validated total gain attributed: {fmt_pct(total_v, plus=True)}.")
    has_any_split = any(_contributes(gain) for _src, gain, _share in rows)
    if not attribution.get("available"):
        facts.append("No stack ledger was recorded, so nothing can be attributed.")
    elif not has_any_split:
        facts.append(
            "No per-source split available — the ledger holds no adoption with "
            "a measurable contribution, either because the session kept nothing "
            "or because every adoption is missing one of its throughputs."
        )
    facts.extend(_contribution_facts(rows))
    facts.extend(f"Note: {note}" for note in validation.get("notes") or [])
    decisions = _kept_decisions(rows)
    return RenderedSection(
        section_id="attribution",
        title="Source Attribution",
        key_facts=facts,
        markdown_block=md_table(["source", "pct_of_total", "share_pct"], rows) if has_any_split else "",
        decisions=decisions,
        warnings=[],
        skipped=not has_any_split,
    )
