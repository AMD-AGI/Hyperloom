# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parameter / backend explore-search renderer.

Read off the configuration arm's attempts on the ``framework_agent`` timeline
events, each recorded as its variant was measured. The flat ledger this
replaces published four counts, so a reader could see that eleven variants
were tried but not which, nor what any of them measured.
"""

from __future__ import annotations

from typing import Any

from ..base import RenderedSection, as_dict, fmt_pct, md_table, register_renderer
from ._framework import config_attempts, config_tally

#: How many measured variants the table lists before it stops. Ordered by gain,
#: so the cut drops the least interesting rows.
_MAX_ROWS = 15


@register_renderer("param_search")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the parameter / backend explore-search section.

    Summarizes what the configuration arm measured and lists the variants by
    the gain each one produced. Skipped when the arm measured nothing.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered parameter/backend search section.
    """
    tally = config_tally(breakdown)
    if not tally.attempted:
        return RenderedSection(
            section_id="param_search",
            title="Parameter / Backend Search",
            key_facts=[],
            markdown_block="",
            warnings=["The configuration arm measured no variant this session."],
            skipped=True,
        )

    facts = [f"explore: {tally.tested} variant(s) measured across {len(tally.rounds)} round(s), {tally.keeps} kept."]
    if tally.best_gain_pct is not None:
        facts.append(f"Best measured variant gain: {fmt_pct(tally.best_gain_pct, plus=True)}.")
    if tally.keep_unstable:
        facts.append(
            f"{tally.keep_unstable} variant(s) won their round and were withheld "
            "when the confirmation did not reproduce the win."
        )

    rows = []
    for attempt, _proposal in config_attempts(breakdown):
        measurement = as_dict(attempt.get("measurement"))
        rows.append(
            [
                str(attempt.get("variant_name") or attempt.get("fingerprint") or ""),
                str(attempt.get("round_id") or ""),
                str(attempt.get("outcome") or ""),
                measurement.get("gain_pct"),
                measurement.get("after_tput"),
            ]
        )
    rows.sort(key=lambda row: -(row[3] if isinstance(row[3], (int, float)) else float("-inf")))
    md_parts = ["**Explore Search:**"]
    if len(rows) > _MAX_ROWS:
        md_parts.append(f"_Showing the {_MAX_ROWS} best of {len(rows)} measured variants._")
    md_parts.append(md_table(["variant", "round", "outcome", "gain_pct", "throughput"], rows[:_MAX_ROWS]))

    return RenderedSection(
        section_id="param_search",
        title="Parameter / Backend Search",
        key_facts=facts,
        markdown_block="\n".join(md_parts).strip(),
        warnings=[],
        skipped=False,
    )
