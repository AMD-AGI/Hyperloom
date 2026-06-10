# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Critic robustness renderer — captures pass-rate of LLM critic /
self-consistency checks on candidate optimizations.
"""

from __future__ import annotations

from typing import Any

from ..base import RenderedSection, md_table, register_renderer

_MAX_ROWS = 20


@register_renderer("critic_robustness")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the critic-robustness section: critic/self-consistency checks.

    Normalizes both legacy (prompt-only string) and structured entry
    shapes, surfaces a pass/fail table when verdicts exist, and marks the
    section skipped (with a warning) when entries carry no actionable
    payload.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered critic-robustness section.
    """
    cr = breakdown.get("critic_robustness") or []
    if not cr:
        return RenderedSection(
            section_id="critic_robustness",
            title="Critic Robustness",
            key_facts=["No critic robustness samples recorded."],
            markdown_block="",
            warnings=[],
            skipped=True,
        )

    # Normalize both historical shapes (V2 list[dict], V1 list[str]) into dicts.
    cr_norm: list[dict[str, Any]] = []
    for c in cr:
        if isinstance(c, dict):
            cr_norm.append(c)
        else:
            cr_norm.append({"prompt": str(c)})

    total = len(cr_norm)
    populated = [c for c in cr_norm if any(
        c.get(k) not in (None, "", [], {}) for k in
        ("response", "decision", "score", "rationale", "pass_count", "fail_count")
    )]
    facts: list[str] = []
    facts.append(f"Recorded {total} critic robustness entries.")
    string_only = total > 0 and all(
        list(c.keys()) == ["prompt"] for c in cr_norm
    )
    if string_only:
        facts.append(
            "All entries are raw prompt strings — collector ran on a "
            "session that did not persist verdicts. Decision / pass-fail "
            "data is non-actionable for this session."
        )
        warnings = [
            "critic_robustness collector produced prompt-only entries "
            "(no decisions); section skipped from narrative."
        ]
        skipped = True
        md = ""
    elif not populated:
        facts.append(
            "All entries have empty payloads — collector ran but the underlying "
            "state arrays were never populated (older Hyperloom build)."
        )
        warnings = [
            "critic_robustness has entries but every payload is empty; the "
            "section is non-actionable for this session."
        ]
        skipped = True
        md = ""
    else:
        warnings = []
        skipped = False
        head = populated[:_MAX_ROWS]
        rows = []
        for c in head:
            rows.append([
                c.get("ts") or "",
                c.get("action") or "",
                c.get("decision") or "",
                c.get("pass_count"),
                c.get("fail_count"),
                (c.get("rationale") or c.get("response") or "")[:80],
            ])
        md = md_table(
            ["ts", "action", "decision", "pass", "fail", "rationale (truncated)"],
            rows,
        )
        if len(populated) > _MAX_ROWS:
            md = f"_Showing first {_MAX_ROWS} of {len(populated)} populated entries._\n\n" + md

    return RenderedSection(
        section_id="critic_robustness",
        title="Critic Robustness",
        key_facts=facts,
        markdown_block=md,
        warnings=warnings,
        skipped=skipped,
    )
