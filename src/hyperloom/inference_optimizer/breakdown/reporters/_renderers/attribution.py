# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Attribution renderer — gain split across optimization sources."""

from __future__ import annotations

from typing import Any

from ..base import Decision, RenderedSection, fmt_pct, md_table, register_renderer

#: Below this the residue is float noise from re-serialized throughputs, well
#: under any measurement's own repeatability.
_NOISE_PP = 0.01


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
    optimizations = breakdown.get("optimizations") or {}
    validation = optimizations.get("validation") or {}
    notes = validation.get("notes") or []
    # Render ``validation.method`` verbatim; never substitute a different label.
    method_raw = validation.get("method")
    method = str(method_raw) if isinstance(method_raw, str) else ""
    method_display = "unknown attribution method" if method in ("", "missing") else method

    summary = optimizations.get("summary_by_source") or {}
    claimed = [[source, bucket.get("total_gain_pct")] for source, bucket in summary.items() if isinstance(bucket, dict)]
    total_v = validation.get("validated_total_gain_pct")
    unattributed = validation.get("unattributed_gain_pct")
    if isinstance(unattributed, (int, float)) and abs(float(unattributed)) > _NOISE_PP:
        claimed.append(["unattributed (between adopted steps)", unattributed])

    share_total = total_v
    if not isinstance(share_total, (int, float)) or not share_total:
        share_total = sum(float(gain) for _source, gain in claimed if isinstance(gain, (int, float)))
    rows = [
        [
            source,
            gain,
            (float(gain) / float(share_total) * 100.0 if isinstance(gain, (int, float)) and share_total else None),
        ]
        for source, gain in claimed
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
        facts.append(f"{src}: {fmt_pct(pct)} of total" + (f" ({fmt_pct(share)} share)" if share is not None else ""))
    for n in notes:
        facts.append(f"Note: {n}")

    decisions: list[Decision] = []
    for src, pct, _share in rows:
        # The unattributed row is a residue, not a contributor; crediting it as
        # a decision would put "nobody" on the leaderboard.
        if pct and pct > 0 and not src.startswith("unattributed"):
            decisions.append(
                Decision(
                    kind="kept",
                    subject=f"attribution:{src}",
                    metric_pct=float(pct),
                    rationale="contributed positive validated gain",
                )
            )

    # Only emit the table when some source is non-zero.
    md = md_table(["source", "pct_of_total", "share_pct"], rows) if has_any_split else ""
    return RenderedSection(
        section_id="attribution",
        title="Source Attribution",
        key_facts=facts,
        markdown_block=md,
        decisions=decisions,
        warnings=[],
        skipped=not has_any_split,
    )
