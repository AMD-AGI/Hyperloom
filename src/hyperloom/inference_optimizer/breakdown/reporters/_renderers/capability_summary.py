# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Capability summary renderer — status/attempts/keeps table + per-capability decisions.

Each row is built from the events the capability recorded as it ran: the two
kernel routes from the per-source tallies on the ``kernel`` events, the
configuration arm and the specialists from the ``framework_agent`` events. The
flat projection this replaces re-derived the same counts at export time by
pairing ``<action>_attempts`` rows against the optimization stack, which is
why it needed a fallback that credited a capability from the stack alone.

A capability the session never invoked still gets a row. That absence is the
section's most load-bearing output -- it is what tells a reader that an
untested area is untested rather than clean.
"""

from __future__ import annotations

from typing import Any

from ..base import Decision, RenderedSection, fmt_pct, md_table, register_renderer
from ._framework import config_tally, specialist_tally
from ._kernels import FORGE_SOURCES, GEAK_SOURCES, source_counters

_CAPABILITY_ORDER = ("explore", "geak", "specialist", "forge")


def _status(*, keeps: int, attempts: int) -> str:
    """The standing a capability ends the session with.

    Args:
        keeps (int): Adoptions the capability's work survived to.
        attempts (int): Candidates or variants it measured.

    Returns:
        str: ``kept`` when something it produced was adopted, ``tried`` when it
            measured something that was not, and ``not_attempted`` when it
            never ran. The three are distinguished because a capability that
            ran and lost is evidence, where one that never ran is a gap.
    """
    if keeps > 0:
        return "kept"
    return "tried" if attempts > 0 else "not_attempted"


def _kernel_row(breakdown: dict[str, Any], sources: tuple[str, ...]) -> dict[str, Any]:
    """One kernel route's row, summed over every visit.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.
        sources (tuple[str, ...]): The source kinds the route owns.

    Returns:
        dict[str, Any]: The row's counts and extras.
    """
    counters = source_counters(breakdown, sources)
    attempts = int(counters.get("attempted") or 0)
    keeps = int(counters.get("keeps") or 0)
    return {
        "status": _status(keeps=keeps, attempts=attempts),
        "attempts": attempts,
        "keeps": keeps,
        "extras": {
            # Kernel-lane outcomes that ``keeps`` deliberately excludes.
            # Without them a reader cannot tell a lane that failed from one
            # whose wins are still waiting on the integrate gate.
            "micro_only": int(counters.get("micro_only_keeps") or 0),
            "pending_review": int(counters.get("needs_review") or 0),
            "reverts": int(counters.get("reverts") or 0),
            "rejected": int(counters.get("rejected") or 0),
            "e2e_gain": counters.get("e2e_gain_pct"),
        },
    }


def capability_rows(breakdown: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build every capability's row.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        dict[str, dict[str, Any]]: The rows, keyed by capability name.
    """
    explore = config_tally(breakdown)
    specialist, specialist_rounds = specialist_tally(breakdown)
    return {
        "explore": {
            "status": _status(keeps=explore.keeps, attempts=explore.tested),
            "attempts": explore.tested,
            "keeps": explore.keeps,
            "extras": {
                "tested": explore.tested,
                "rounds": len(explore.rounds),
                "best_gain": explore.best_gain_pct,
                "keep_unstable": explore.keep_unstable,
            },
        },
        "specialist": {
            # A round that came back empty proposed nothing, so the rounds are
            # counted apart from the attempts they led to: without them a
            # specialist that ran and found nothing reads as one that never ran.
            "status": _status(keeps=specialist.keeps, attempts=specialist.tested or specialist_rounds),
            "attempts": specialist.tested,
            "keeps": specialist.keeps,
            "extras": {
                "rounds": specialist_rounds,
                "best_gain": specialist.best_gain_pct,
            },
        },
        "geak": _kernel_row(breakdown, GEAK_SOURCES),
        "forge": _kernel_row(breakdown, FORGE_SOURCES),
    }


def _extras_cell(extras: dict[str, Any]) -> str:
    """Render a row's extras as one compact cell.

    Args:
        extras (dict[str, Any]): The row's extra counts, where ``0`` and
            ``None`` both mean "nothing to report" and are dropped.

    Returns:
        str: The joined cell, empty when nothing is worth reporting.
    """
    parts: list[str] = []
    for name, value in extras.items():
        if not value:
            continue
        parts.append(f"{name}={fmt_pct(value) if name.endswith('gain') else value}")
    return " · ".join(parts)


@register_renderer("capability_summary")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the capability-summary section.

    Produces a status/attempts/keeps table for each capability in a
    stable order, one structured :class:`Decision` per non-``not_attempted``
    capability, and a one-line fact per row. Skipped when no capability
    recorded anything.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered capability-summary section.
    """
    built = capability_rows(breakdown)
    rows: list[list[Any]] = []
    facts: list[str] = []
    decisions: list[Decision] = []
    warnings: list[str] = []

    for name in _CAPABILITY_ORDER:
        row = built[name]
        status = str(row["status"])
        attempts = int(row["attempts"])
        keeps = int(row["keeps"])
        extras_str = _extras_cell(row["extras"])
        rows.append([name, status, attempts, keeps, extras_str])
        facts.append(
            f"`{name}` status={status}, attempts={attempts}, keeps={keeps}" + (f" ({extras_str})" if extras_str else "")
        )
        if status != "not_attempted":
            decisions.append(
                Decision(
                    kind=status,
                    subject=name,
                    metric_pct=None,
                    rationale=f"{keeps}/{attempts} attempts promoted"
                    if attempts
                    else "ran without measuring a candidate",
                )
            )

    if not any(row[2] or row[3] for row in rows):
        warnings.append(
            "No capability measured a candidate this session — neither kernel "
            "route produced one and the configuration arm ran no variant."
        )

    md = md_table(["capability", "status", "attempts", "keeps", "extra"], rows)
    return RenderedSection(
        section_id="capability_summary",
        title="Capability Summary",
        key_facts=facts,
        markdown_block=md,
        decisions=decisions,
        warnings=warnings,
        skipped=not rows,
    )
