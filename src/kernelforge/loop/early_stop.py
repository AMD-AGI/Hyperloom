# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""When a campaign may stop before its budget.

Time decides first: over the last ``window_hours`` of rounds that tested something, the best score gained less than
``min_gain``. Rounds that tested nothing -- a bench that failed, a provider error -- add no time to the window. Then
the directions: none still in play was expected, when it was listed, to gain ``priority_gain`` or more, and none
lacks that estimate. Then one audit, run against the ledger as it stands, finds no direction the ledger lacks.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from kernelforge.loop.direction_ledger import (
    DirectionLedger,
    DirectionStanding,
    attempted,
    by_priority,
    direction_standings,
    keep_gains,
)


@dataclass(frozen=True)
class EarlyStopPolicy:
    """The thresholds of the rule. ``enabled`` False keeps the loop's own behaviour: it never stops on a stall."""

    enabled: bool = False
    window_hours: float = 3.0
    min_gain: float = 0.02
    priority_gain: float = 0.03
    exhaust_after: int = 2


@dataclass(frozen=True)
class TimeWindow:
    """The latest rounds that tested something, reaching back over at least the window, and what they gained."""

    hours: float
    rounds: int
    gain: float


@dataclass(frozen=True)
class EarlyStopStanding:
    """Where the rule stands: each criterion, and what the loop should do about it."""

    standings: tuple[DirectionStanding, ...]
    window: TimeWindow | None
    flat: bool
    promising: tuple[str, ...]
    criteria_met: bool
    audit_due: bool
    stop: bool
    reasons: tuple[str, ...]

    @property
    def focus(self) -> tuple[str, ...]:
        """The directions every plan must pursue: the promising ones, once progress has gone flat."""
        return self.promising if self.flat else ()


def time_window(
    results: list[dict], round_costs: list[dict], gains: dict[int, float], hours: float
) -> TimeWindow | None:
    """The best score's gain over the latest ``hours`` of rounds that tested something; None until there were
    that many. A round's time is its ``round_cost``; the iterations it ran are those up to the next round's."""
    seconds = {int(cost["iter"]): float(cost.get("total_sec") or 0.0) for cost in round_costs}
    starts = sorted(seconds)
    rounds: dict[int, list[dict]] = {}
    for event in results:
        index = bisect_right(starts, int(event.get("iter") or 0)) - 1
        if index >= 0:
            rounds.setdefault(starts[index], []).append(event)
    covered, counted, product = 0.0, 0, 1.0
    for start in sorted(rounds, reverse=True):
        if not any(attempted(event) for event in rounds[start]):
            continue
        covered += seconds[start] / 3600.0
        counted += 1
        for event in rounds[start]:
            product *= 1.0 + gains.get(int(event.get("iter") or 0), 0.0)
        if covered >= hours:
            return TimeWindow(covered, counted, product - 1.0)
    return None


def early_stop_standing(
    policy: EarlyStopPolicy,
    ledger: DirectionLedger,
    events: list[dict],
    *,
    start_score: float | None,
    lane_queue_empty: bool,
    inbox_pending: bool,
) -> EarlyStopStanding:
    """Whether the campaign has stopped improving with nothing promising left, and whether an audit has agreed."""
    results = [event for event in events if event.get("type") == "iteration_result"]
    gains = keep_gains(results, start_score)
    standings = by_priority(
        direction_standings(ledger.directions(), results, exhaust_after=policy.exhaust_after, gains=gains)
    )
    round_costs = [event for event in events if event.get("type") == "round_cost"]
    window = time_window(results, round_costs, gains, policy.window_hours)
    flat = window is not None and window.gain < policy.min_gain
    promising = tuple(
        standing.direction.id
        for standing in standings
        if standing.in_play
        and (standing.direction.priority is None or standing.direction.priority >= policy.priority_gain)
    )
    unmet = []
    if window is None:
        unmet.append(f"fewer than {policy.window_hours:g} h of rounds that tested something")
    elif not flat:
        unmet.append(f"the last {window.hours:.1f} h gained {window.gain:.2%}, at least {policy.min_gain:.2%}")
    if promising:
        unmet.append(
            f"{len(promising)} directions still expected to gain {policy.priority_gain * 100:g}% or more, or with no estimate"
        )
    if not lane_queue_empty:
        unmet.append("lane candidates still queued")
    if inbox_pending:
        unmet.append("operator messages not yet read")
    criteria_met = policy.enabled and not unmet
    # An audit counts only if it ran against the ledger as it stands now (``ledger_size`` is recorded after its
    # own additions). One that added directions re-arms the rule: once those are below the bar or mined out,
    # audit again.
    audits = ledger.audits()
    current = [audit for audit in audits if int(audit.get("ledger_size", -1)) == len(standings)]
    last_added = bool(current and current[-1].get("added"))
    audit_due = criteria_met and (not current or last_added)
    stop = criteria_met and bool(current) and not last_added
    return EarlyStopStanding(standings, window, flat, promising, criteria_met, audit_due, stop, tuple(unmet))
