# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared rollup over the ``framework_agent`` timeline events.

The capability summary and the parameter-search section both need the same
account of what the configuration arm tried and what it kept, and the summary
additionally needs it split by which producer proposed the attempt. Building
it once here is what keeps the two sections from disagreeing.

An attempt names the proposal it came from and a proposal names its producer,
so attributing a KEEP to the specialist that suggested it is a join rather
than a guess. Attempts with no proposal behind them -- the seeded grid -- are
attributed to the arm itself.

The ``_`` prefix marks this as a helper: it registers no renderer.
"""

from __future__ import annotations

from typing import Any

from ..base import as_dict, dict_rows, events_of

__all__ = ["FrameworkTally", "config_attempts", "config_tally", "specialist_tally"]

#: The per-variant outcome the executor reports for a variant it promoted.
_KEEP = "KEEP"

#: Reachable only for a session recorded before the per-KEEP confirmation round
#: was removed: a variant that won its round and was withheld when the
#: confirmation did not reproduce it. Counted apart from a plain revert, which
#: is a variant that lost on measurement.
_KEEP_UNSTABLE = "KEEP_UNSTABLE"

#: The producer label a specialist-proposed attempt carries on its proposal.
_PRODUCER_SPECIALIST = "specialist"


class FrameworkTally:
    """What one slice of the configuration arm tried and what came of it."""

    def __init__(self) -> None:
        """Start an empty tally."""
        self.tested = 0
        self.keeps = 0
        self.keep_unstable = 0
        self.best_gain_pct: float | None = None
        self.rounds: set[str] = set()

    def add(self, attempt: dict[str, Any]) -> None:
        """Fold one attempt row into the tally.

        Args:
            attempt (dict[str, Any]): One ``attempts`` row off a
                ``framework_agent`` event.
        """
        self.tested += 1
        outcome = str(attempt.get("outcome") or "").upper()
        if outcome == _KEEP:
            self.keeps += 1
        elif outcome == _KEEP_UNSTABLE:
            self.keep_unstable += 1
        round_id = str(attempt.get("round_id") or "")
        if round_id:
            self.rounds.add(round_id)
        gain = as_dict(attempt.get("measurement")).get("gain_pct")
        if isinstance(gain, (int, float)) and (self.best_gain_pct is None or gain > self.best_gain_pct):
            self.best_gain_pct = float(gain)

    @property
    def attempted(self) -> bool:
        """bool: Whether this slice measured anything at all."""
        return self.tested > 0


def config_attempts(breakdown: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Every configuration-arm attempt, paired with the proposal behind it.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        list[tuple[dict[str, Any], dict[str, Any]]]: One pair per attempt; the
            proposal is ``{}`` for an attempt nothing proposed.
    """
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for event in events_of(breakdown, "framework_agent"):
        ext = as_dict(event.get("ext"))
        proposals = {
            str(row.get("proposal_id") or ""): row for row in dict_rows(ext.get("proposals")) if row.get("proposal_id")
        }
        for attempt in dict_rows(ext.get("attempts")):
            if str(attempt.get("arm") or "") != "config":
                continue
            pairs.append((attempt, as_dict(proposals.get(str(attempt.get("proposal_ref") or "")))))
    return pairs


def config_tally(breakdown: dict[str, Any]) -> FrameworkTally:
    """What the configuration arm tried across the whole session.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.
    """
    tally = FrameworkTally()
    for attempt, _proposal in config_attempts(breakdown):
        tally.add(attempt)
    return tally


def specialist_tally(breakdown: dict[str, Any]) -> tuple[FrameworkTally, int]:
    """What the specialists proposed, and what the arm made of it.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        tuple[FrameworkTally, int]: The tally over attempts a specialist
            proposed, and the number of specialist rounds that were dispatched.
            The two are separate counts because a round that came back empty
            produced no attempt, and reporting only the attempts would leave a
            specialist that ran and found nothing indistinguishable from one
            that never ran.
    """
    tally = FrameworkTally()
    for attempt, proposal in config_attempts(breakdown):
        if str(proposal.get("producer") or "") == _PRODUCER_SPECIALIST:
            tally.add(attempt)

    rounds = 0
    for event in events_of(breakdown, "phase"):
        for row in dict_rows(as_dict(as_dict(event.get("ext")).get("actions")).get("rows")):
            if str(row.get("action") or "") == "specialist":
                rounds += 1
    for event in events_of(breakdown, "framework_agent"):
        for row in dict_rows(as_dict(event.get("ext")).get("runs")):
            if str(row.get("role") or "") in {"discovery", "authoring"}:
                rounds += 1
    return tally, rounds
