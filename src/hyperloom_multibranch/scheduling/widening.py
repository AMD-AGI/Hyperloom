# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Opening branches during EXPLORE: whether one opened now would still pay off, and what each is briefed to attack.

Whether a branch may be opened at all is the roster's answer (its free slots, a closed roster, a stopping session).
The signals here are what the judge weighs, never a cap the scheduler enforces: whether a branch opened now has the
time and the test capacity to show progress, as the least EXPLORE time a new branch needs to show a trend and how many
more tests the toolbox would run at once. A branch opened after the opening batch is briefed on a focus: the op the
judge names for it, or else one of the ops the session ranks for a new branch.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from hyperloom_multibranch.contracts.blackboard.records import OpShares
from hyperloom_multibranch.contracts.toolbox import MeasurementCapacity, QueueState, Toolbox
from hyperloom_multibranch.blackboard.rsi.frontier import FrontierPolicy
from hyperloom_multibranch.scheduling.branches import SPECIALIST_START_SECONDS, BranchFocus
from hyperloom_multibranch.scheduling.roster import Roster, RosterCaps

#: Sandbox cold starts a new branch waits through before its tests start paying off: its first test's, and one more.
MB_MIN_USEFUL_COLD_STARTS = 2


@dataclass(frozen=True)
class WideningSignals:
    """``cold_start_seconds`` is ``None`` before the toolbox's first sandbox cold start, and ``min_useful_seconds``
    then leaves the cold starts out, so it reads low. ``toolbox_headroom`` is negative while more tests wait than the
    cap in force leaves room for."""

    min_useful_seconds: float
    cold_start_seconds: float | None
    toolbox_headroom: int


def min_useful_seconds(frontier: FrontierPolicy, queue_state: QueueState) -> float:
    """The least EXPLORE time a branch opened now needs: the frontier's trend window, or its start-up and cold starts
    when they take longer."""

    cold_start = queue_state.cold_start_seconds
    start_up = SPECIALIST_START_SECONDS + (MB_MIN_USEFUL_COLD_STARTS * cold_start if cold_start is not None else 0.0)
    return max(frontier.window_seconds, start_up)


def toolbox_headroom(
    caps: RosterCaps, queue_state: QueueState, capacity: MeasurementCapacity, cards_per_test: int
) -> int:
    """Tests the toolbox would still start at once, net of what runs and waits: under the session's global cap, and
    under the tests the measurement cards allow at once, since a branch sandbox's lease holds cards no test runs on.
    A toolbox with no card pool of its own reads under the cap alone.
    """

    allowed = caps.max_parallel_tests
    if capacity.known:
        allowed = min(allowed, capacity.measurement_cards // cards_per_test)
    return allowed - queue_state.running - queue_state.queued


def refill_focus(
    asked: tuple[str, ...], count: int, shares: OpShares | None, ranked: Sequence[str]
) -> tuple[BranchFocus, ...]:
    """The focus of each of ``count`` branches opened after the opening batch, in order: what the judge ``asked`` for
    it, else one of the ``ranked`` ops; ``""`` when neither names one.

    The judge's text is kept as written, and its op is the op of ``shares``, the profile the ops come from, it names,
    so one op is one focus however the judge annotates it. The branches the judge left without a focus take the ranked
    ops in turn, wrapping when the batch outruns them, so a refill spreads over them instead of sending every branch it
    opens at the first.
    """

    focus, taken = [], 0
    for index in range(count):
        chosen = asked[index] if index < len(asked) else ""
        if not chosen and ranked:
            chosen, taken = ranked[taken % len(ranked)], taken + 1
        focus.append(BranchFocus(chosen, _profile_op(chosen, shares)))
    return tuple(focus)


def _profile_op(text: str, shares: OpShares | None) -> str:
    """The op of ``shares`` that ``text`` names first as a whole word, with or without its ``.kd`` suffix, the longer
    of two named at one place; ``text`` itself when it names none."""

    named = []
    for op in shares.shares if shares is not None else ():
        found = _named(text, op.removesuffix(".kd"))
        if found is not None:
            named.append((found.start(), -len(op), op))
    return min(named)[2] if named else text


def _named(text: str, op: str) -> re.Match[str] | None:
    return re.search(rf"(?<!\w){re.escape(op)}(?!\w)", text) if op else None


class Widening:
    """Reads the signals off the toolbox's queue, the roster's caps and the frontier policy the session scores with.

    ``frontier`` is the scoring's own policy, so the trend window here is the one the judge's expected gains rest on.
    """

    def __init__(self, toolbox: Toolbox, roster: Roster, frontier: FrontierPolicy, cards_per_test: int) -> None:
        self._toolbox = toolbox
        self._roster = roster
        self._frontier = frontier
        self._cards_per_test = cards_per_test

    async def signals(self) -> WideningSignals:
        state = await self._toolbox.queue_state()
        capacity = await self._toolbox.measurement_capacity()
        return WideningSignals(
            min_useful_seconds=min_useful_seconds(self._frontier, state),
            cold_start_seconds=state.cold_start_seconds,
            toolbox_headroom=toolbox_headroom(self._roster.caps, state, capacity, self._cards_per_test),
        )


__all__ = [
    "MB_MIN_USEFUL_COLD_STARTS",
    "Widening",
    "WideningSignals",
    "min_useful_seconds",
    "refill_focus",
    "toolbox_headroom",
]
