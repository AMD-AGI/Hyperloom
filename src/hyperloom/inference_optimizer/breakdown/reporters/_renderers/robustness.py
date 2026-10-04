# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Robustness renderer — what the side-channel agent raised, turn by turn.

The agent watches the session from outside the optimization loop, so its turns
belong to no phase and no macro cycle and are reported on their own rather than
folded into the timeline.

The turn outcome is reported even when the agent raised nothing, because a turn
the agent could not complete and a turn with nothing to say are different
findings that a bare intent count would render identically.

Silently skipped when the agent never took a turn.
"""

from __future__ import annotations

from typing import Any

from ..base import (
    RenderedSection,
    as_dict,
    dict_rows,
    md_table,
    register_renderer,
    robustness_turns_of,
)

#: How much of an intent payload reaches the table before it crowds out the
#: columns that say which turn raised it.
_PAYLOAD_CELL = 100

#: Outcomes that mean the agent's turn did not produce a usable envelope.
_FAILED_OUTCOMES = ("invalid_envelope", "no_envelope")


def _intent_label(intent: dict[str, Any]) -> str:
    """A one-line reading of an intent: its type, and its topic when it has one."""
    kind = str(intent.get("type") or "") or "—"
    topic = str(intent.get("topic") or "")
    return f"{kind} ({topic})" if topic else kind


def _payload_text(intent: dict[str, Any]) -> str:
    """The human-facing part of an intent payload, if it carries one."""
    payload = as_dict(intent.get("payload"))
    for key in ("body_md", "summary", "detail", "reason"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value[:_PAYLOAD_CELL]
    return ""


def _turn_rows(turns: list[dict[str, Any]]) -> list[list[Any]]:
    """One row per turn, listing what it raised and anything that failed to parse."""
    rows: list[list[Any]] = []
    for turn in turns:
        intents = dict_rows(turn.get("intents"))
        warnings = [str(w) for w in (turn.get("parse_warnings") or []) if str(w)]
        rows.append(
            [
                turn.get("turn_idx"),
                turn.get("tick_index"),
                turn.get("outcome") or "—",
                ", ".join(_intent_label(i) for i in intents) or "—",
                next((_payload_text(i) for i in intents if _payload_text(i)), ""),
                "; ".join(warnings),
            ]
        )
    return rows


def _severity_counts(turns: list[dict[str, Any]]) -> dict[str, int]:
    """How many intents were raised at each severity across the session."""
    counts: dict[str, int] = {}
    for turn in turns:
        for intent in dict_rows(turn.get("intents")):
            severity = str(intent.get("severity") or "").strip().lower()
            if severity:
                counts[severity] = counts.get(severity, 0) + 1
    return dict(sorted(counts.items()))


@register_renderer("robustness")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the robustness-agent section.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered robustness section, or a skipped
            placeholder when the agent never took a turn.
    """
    turns = robustness_turns_of(breakdown)
    if not turns:
        return RenderedSection(section_id="robustness", title="Robustness", skipped=True)

    intent_total = sum(len(dict_rows(turn.get("intents"))) for turn in turns)
    failed = [t for t in turns if str(t.get("outcome") or "") in _FAILED_OUTCOMES]
    warned = [t for t in turns if t.get("parse_warnings")]

    facts = [f"The robustness agent took {len(turns)} turn(s) and raised {intent_total} intent(s)."]
    severities = _severity_counts(turns)
    if severities:
        facts.append("Intents by severity: " + ", ".join(f"{n} {s}" for s, n in severities.items()) + ".")
    if failed:
        facts.append(
            f"{len(failed)} turn(s) produced no usable envelope, so the agent was "
            "silent on those ticks for a reason of its own rather than for lack of anything to raise."
        )
    if warned:
        facts.append(f"{len(warned)} turn(s) carried parse warnings.")

    parts: list[str] = []
    table = md_table(
        ["turn", "tick", "outcome", "intents", "said", "parse_warnings"],
        _turn_rows(turns),
    )
    if table:
        parts.append(table)

    return RenderedSection(
        section_id="robustness",
        title="Robustness",
        key_facts=facts,
        markdown_block="\n".join(parts).strip(),
        skipped=False,
    )
