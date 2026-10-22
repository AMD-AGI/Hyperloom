# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""When a campaign may stop before its budget: every known direction mined out, recent rounds flat, and one
audit for directions nobody listed has come back empty."""

from __future__ import annotations

from dataclasses import dataclass

from kernelforge.loop.direction_ledger import TERMINAL, DirectionStanding

MEASURED_DECISIONS = frozenset({"KEEP", "REVERT_PERF"})


@dataclass(frozen=True)
class EarlyStopPolicy:
    """The thresholds of the rule. ``enabled`` False keeps the loop's own behaviour: it never stops on a stall."""

    enabled: bool = False
    min_directions: int = 10
    window_rounds: int = 3
    min_gain: float = 0.02
    exhaust_after: int = 2


@dataclass(frozen=True)
class EarlyStopStanding:
    """Where the rule stands: each criterion, and what the loop should do about it."""

    directions: int
    terminal: int
    window_gain: float | None
    criteria_met: bool
    audit_due: bool
    stop: bool
    reasons: tuple[str, ...]


def window_gain(results: list[dict], gains: dict[int, float], rounds: int) -> float | None:
    """The best score's gain over the last ``rounds`` measured iterations, or None when fewer were measured."""
    measured = [event for event in results if event.get("decision") in MEASURED_DECISIONS]
    if len(measured) < rounds:
        return None
    product = 1.0
    for event in measured[-rounds:]:
        product *= 1.0 + gains.get(int(event.get("iter") or 0), 0.0)
    return product - 1.0


def early_stop_standing(
    policy: EarlyStopPolicy,
    standings: tuple[DirectionStanding, ...],
    *,
    gain: float | None,
    audits: list[dict],
    lane_queue_empty: bool,
    inbox_pending: bool,
) -> EarlyStopStanding:
    """Whether the campaign has explored what it knows and stopped improving, and whether an audit has agreed."""
    terminal = sum(standing.status in TERMINAL for standing in standings)
    unmet = []
    if len(standings) < policy.min_directions:
        unmet.append(f"{len(standings)} directions known, {policy.min_directions} required")
    if terminal < len(standings):
        unmet.append(f"{len(standings) - terminal} directions not yet exhausted")
    if gain is None:
        unmet.append(f"fewer than {policy.window_rounds} measured rounds")
    elif gain >= policy.min_gain:
        unmet.append(f"last {policy.window_rounds} rounds gained {gain:.2%}, at least {policy.min_gain:.2%}")
    if not lane_queue_empty:
        unmet.append("lane candidates still queued")
    if inbox_pending:
        unmet.append("operator messages not yet read")
    criteria_met = policy.enabled and not unmet
    # An audit counts only if it ran against the ledger as it stands now (``ledger_size`` is recorded after its
    # own additions). One that added directions re-arms the rule: once those are mined out too, audit again.
    current = [audit for audit in audits if int(audit.get("ledger_size", -1)) == len(standings)]
    last_added = bool(current and current[-1].get("added"))
    audit_due = criteria_met and (not current or last_added)
    stop = criteria_met and bool(current) and not last_added
    return EarlyStopStanding(len(standings), terminal, gain, criteria_met, audit_due, stop, tuple(unmet))
