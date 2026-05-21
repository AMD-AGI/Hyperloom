"""Decision journal renderer (v1.1).

One markdown block per params/backends round: round-level promotion
verdict plus a variant table (gain, outcome, reject reason).
"""

from __future__ import annotations

from typing import Any

from ..base import (
    Decision,
    RenderedSection,
    fmt_pct,
    md_table,
    register_renderer,
)

_MAX_ROUNDS_STANDARD = 20


def _variant_rows(variants: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for v in variants:
        rows.append([
            v.get("name") or "",
            v.get("outcome") or "",
            fmt_pct(v.get("gain_pct_vs_base"), plus=True),
            v.get("output_throughput"),
            v.get("reject_reason") or "—",
            v.get("status") or "",
        ])
    return rows


@register_renderer("decision_journal")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    journal = breakdown.get("decision_journal") or []
    detail_level = str(breakdown.get("detail_level") or "standard")

    if not journal:
        return RenderedSection(
            section_id="decision_journal",
            title="Decision Journal",
            key_facts=["No params/backends rounds recorded this session."],
            markdown_block="",
            decisions=[Decision(
                kind="not_attempted",
                subject="decision_journal",
                rationale="empty decision_journal",
            )],
            warnings=[],
            skipped=True,
        )

    promoted = sum(
        1 for e in journal
        if (e.get("round_decision") or {}).get("outcome") == "promoted"
    )
    discarded = sum(
        1 for e in journal
        if (e.get("round_decision") or {}).get("outcome") == "discarded"
    )
    variant_total = sum(len(e.get("variants") or []) for e in journal)

    facts: list[str] = [
        f"{len(journal)} round(s): {promoted} promoted, {discarded} discarded, "
        f"{variant_total} variant row(s) exported.",
    ]
    if detail_level != "verbose" and variant_total:
        facts.append(
            "standard detail_level: variant list capped at promoted/rejected "
            "+ top 30 tested by |gain| (see session_breakdown.json)."
        )

    decisions: list[Decision] = []
    if promoted:
        decisions.append(Decision(
            kind="kept",
            subject="decision_journal:promoted",
            rationale=f"{promoted} round(s) promoted",
        ))
    if discarded and not promoted:
        decisions.append(Decision(
            kind="rejected",
            subject="decision_journal",
            rationale=f"{discarded} round(s) discarded without promotion",
        ))

    rounds = journal if detail_level == "verbose" else journal[-_MAX_ROUNDS_STANDARD:]
    parts: list[str] = []
    if len(rounds) < len(journal):
        parts.append(
            f"_Showing last {len(rounds)} of {len(journal)} rounds "
            f"(detail_level={detail_level})._"
        )
        parts.append("")

    headers = ["name", "outcome", "gain_vs_base", "tput", "reject_reason", "status"]
    for entry in rounds:
        phase = entry.get("phase") or "?"
        round_id = entry.get("round_id") or "—"
        rd = entry.get("round_decision") or {}
        parts.append(
            f"**{phase}** round `{round_id}` "
            f"(base={entry.get('baseline_ref_tput') or '—'} tok/s/gpu, "
            f"ts={entry.get('ts') or '—'})"
        )
        parts.append(md_table(
            ["field", "value"],
            [
                ["round outcome", rd.get("outcome") or "—"],
                ["best_variant", rd.get("best_variant_name") or "—"],
                ["gain_vs_cb", fmt_pct(rd.get("gain_vs_cb_pct"), plus=True)],
                ["promotion_rule", rd.get("promotion_rule") or "—"],
                ["rule_detail", (rd.get("promotion_rule_detail") or "—")[:200]],
                ["keep_threshold_pct", rd.get("keep_threshold_pct")],
                ["variants_tested", rd.get("variants_tested_count")],
                ["accuracy_gate", rd.get("accuracy_gate_passed")],
            ],
        ))
        variants = entry.get("variants") or []
        if variants:
            parts.append("")
            parts.append(md_table(headers, _variant_rows(variants)))
        parts.append("")

    return RenderedSection(
        section_id="decision_journal",
        title="Decision Journal",
        key_facts=facts,
        markdown_block="\n".join(parts).strip(),
        decisions=decisions,
        warnings=[],
        skipped=False,
    )
