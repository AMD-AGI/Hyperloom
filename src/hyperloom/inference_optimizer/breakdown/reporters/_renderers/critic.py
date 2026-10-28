# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Critic renderer — the review agent's own run, pass by pass.

The per-proposal verdicts are reported by the sections that own the proposals
they judge; this one reports the agent itself: how often it was asked, what it
was asked about, how its rulings fell each time, and what it left on disk.

A pass that only spoke is kept rather than filtered out. A session where the
critic was asked forty times and ruled on nothing reads very differently from
one where it was never asked, and dropping the silent passes would make those
two look the same.

Silently skipped when the critic never ran.
"""

from __future__ import annotations

from typing import Any

from hyperloom.common.coerce import dict_rows, to_mapping
from hyperloom.inference_optimizer.protocol.intent import DEFAULT_IDLE_BODY

from ..base import (
    RenderedSection,
    critic_iterations_of,
    md_kv_list,
    md_table,
    register_renderer,
)

#: How much of a pass's prose reaches the table, which has to stay readable
#: next to the columns beside it. The full text is in the recorded emit.
_SUMMARY_CELL = 120


def _iteration_rows(iterations: list[dict[str, Any]]) -> list[list[Any]]:
    """One table row per pass: when it ran, where, and how it ruled."""
    rows: list[list[Any]] = []
    for seq, it in enumerate(iterations, start=1):
        said = str(it.get("summary") or "")
        rows.append(
            [
                seq,
                it.get("phase") or "—",
                it.get("macro_cycle"),
                it.get("topic") or "—",
                it.get("verdict") or "—",
                "—" if said == DEFAULT_IDLE_BODY else said[:_SUMMARY_CELL] or "—",
            ]
        )
    return rows


def _review_rows(iterations: list[dict[str, Any]]) -> list[list[Any]]:
    """One row per framework ruling, tagged with the pass that handed it down.

    Both verdicts are carried: a reject the loop held to an advise-only rule
    still ran, and reporting either alone misreads what happened.
    """
    rows: list[list[Any]] = []
    for seq, it in enumerate(iterations, start=1):
        for review in dict_rows(it.get("framework_reviews")):
            verdict = str(review.get("verdict") or "")
            effective = str(review.get("effective_verdict") or "")
            rows.append(
                [
                    seq,
                    review.get("proposal_msg_id") or "—",
                    review.get("variant_name") or review.get("candidate_id") or "—",
                    review.get("arm") or "—",
                    verdict or "—",
                    effective if effective and effective != verdict else "",
                    review.get("confidence"),
                    review.get("failure_reason_code") or "",
                    str(review.get("reasoning") or "")[:_SUMMARY_CELL],
                ]
            )
    return rows


def _totals(iterations: list[dict[str, Any]]) -> dict[str, int]:
    """The session-wide ruling distribution, summed over every pass."""
    totals: dict[str, int] = {}
    for it in iterations:
        counts = to_mapping(it.get("verdict_counts"))
        for verdict, count in counts.items():
            try:
                totals[str(verdict)] = totals.get(str(verdict), 0) + int(count)
            except (TypeError, ValueError):
                continue
    return dict(sorted(totals.items()))


def _priors_facts(iterations: list[dict[str, Any]]) -> list[str]:
    """What the knowledge base contributed, when it was wired up at all."""
    configured = [it for it in iterations if to_mapping(it.get("kb_priors")).get("configured")]
    if not configured:
        return []
    referenced = sum(1 for it in configured if to_mapping(it.get("kb_priors")).get("referenced_in_verdict"))
    priors = 0
    for it in configured:
        try:
            priors += int(to_mapping(it.get("kb_priors")).get("prior_count") or 0)
        except (TypeError, ValueError):
            continue
    return [
        f"Knowledge base was configured for {len(configured)} of {len(iterations)} passes, "
        f"supplied {priors} prior(s), and was cited in {referenced} verdict(s)."
    ]


@register_renderer("critic")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the critic-agent section.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered critic section, or a skipped placeholder
            when the critic never ran.
    """
    iterations = critic_iterations_of(breakdown)
    if not iterations:
        return RenderedSection(section_id="critic", title="Critic", skipped=True)

    totals = _totals(iterations)
    ruling_passes = sum(1 for it in iterations if to_mapping(it.get("verdict_counts")))
    facts = [
        f"The critic ran {len(iterations)} pass(es); {ruling_passes} of them ruled on at least one proposal.",
    ]
    if totals:
        facts.append("Rulings across the session: " + ", ".join(f"{n} {v}" for v, n in totals.items()) + ".")
    else:
        facts.append("The critic was asked but never ruled on a proposal.")
    facts.extend(_priors_facts(iterations))

    passes = md_table(
        ["#", "phase", "macro_cycle", "topic", "rulings", "said"],
        _iteration_rows(iterations),
    )
    parts: list[str] = ["**Passes**\n\n" + passes, ""]

    reviews = md_table(
        ["#", "proposal", "candidate", "arm", "verdict", "effective", "conf", "failure_code", "reasoning"],
        _review_rows(iterations),
    )
    if reviews:
        parts.append("**Framework rulings**\n\n" + reviews)
        parts.append("")

    last = iterations[-1]
    artifacts = md_kv_list(
        [
            ("request", last.get("request_path")),
            ("judge_bundle", last.get("judge_bundle_path")),
            ("emit", last.get("emit_path")),
            ("review", last.get("review_path")),
        ]
    )
    if artifacts:
        parts.append(f"**Artifacts of the last pass** (pass {len(iterations)})\n\n" + artifacts)
        parts.append("")

    return RenderedSection(
        section_id="critic",
        title="Critic",
        key_facts=facts,
        markdown_block="\n".join(parts).strip(),
        skipped=False,
    )
