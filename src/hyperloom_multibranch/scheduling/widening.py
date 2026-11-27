# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Opening branches during EXPLORE: whether one opened now would still pay off, and what each is briefed to attack.

Whether a branch may be opened at all is the roster's answer (its free slots, a closed roster, a stopping session).
The signals here are what the judge weighs, never a cap the scheduler enforces: whether a branch opened now has the
time to show progress and how long its tests would wait, as the least EXPLORE time a new branch needs to show a trend,
how many more tests the toolbox would start at once and how many tests wait for a card. A branch opened after the
opening batch is briefed on a focus: the op the judge names for it, or else one of the ops the session ranks for a new
branch.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from hyperloom_multibranch.contracts.blackboard.records import OpShares
from hyperloom_multibranch.contracts.toolbox import QueueState, Toolbox
from hyperloom_multibranch.blackboard.rsi.frontier import FrontierPolicy
from hyperloom_multibranch.scheduling.branches import SPECIALIST_START_SECONDS, BranchFocus

#: Sandbox cold starts a new branch waits through before its tests start paying off: its first test's, and one more.
MB_MIN_USEFUL_COLD_STARTS = 2


@dataclass(frozen=True)
class WideningSignals:
    """``cold_start_seconds`` is ``None`` before the toolbox's first sandbox cold start, and ``min_useful_seconds``
    then leaves the cold starts out, so it reads low. ``idle_test_slots`` is how many more tests the toolbox would
    start at once and ``queued_tests`` how many wait for a card: both are above 0 while the tests that wait belong to
    branches at their K. ``running_by_branch`` and ``queued_by_branch`` count each branch's own, a branch with none
    left out."""

    min_useful_seconds: float
    cold_start_seconds: float | None
    idle_test_slots: int
    queued_tests: int
    running_by_branch: Mapping[str, int] = field(default_factory=dict)
    queued_by_branch: Mapping[str, int] = field(default_factory=dict)


def min_useful_seconds(frontier: FrontierPolicy, queue_state: QueueState) -> float:
    """The least EXPLORE time a branch opened now needs: the frontier's trend window, or its start-up and cold starts
    when they take longer."""

    cold_start = queue_state.cold_start_seconds
    start_up = SPECIALIST_START_SECONDS + (MB_MIN_USEFUL_COLD_STARTS * cold_start if cold_start is not None else 0.0)
    return max(frontier.window_seconds, start_up)


def idle_test_slots(queue_state: QueueState) -> int:
    """Tests the toolbox would still start at once beside those running, under the limit its queue starts tests
    under; 0 while a lease holds cards the running tests were started under."""

    return max(queue_state.running_limit - queue_state.running, 0)


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
    """Reads the signals off the toolbox's queue and the frontier policy the session scores with.

    ``frontier`` is the scoring's own policy, so the trend window here is the one the judge's expected gains rest on.
    """

    def __init__(self, toolbox: Toolbox, frontier: FrontierPolicy) -> None:
        self._toolbox = toolbox
        self._frontier = frontier

    async def signals(self) -> WideningSignals:
        state = await self._toolbox.queue_state()
        return WideningSignals(
            min_useful_seconds=min_useful_seconds(self._frontier, state),
            cold_start_seconds=state.cold_start_seconds,
            idle_test_slots=idle_test_slots(state),
            queued_tests=state.queued,
            running_by_branch=state.running_by_branch,
            queued_by_branch=state.queued_by_branch,
        )


__all__ = [
    "MB_MIN_USEFUL_COLD_STARTS",
    "Widening",
    "WideningSignals",
    "idle_test_slots",
    "min_useful_seconds",
    "refill_focus",
]
