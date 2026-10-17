# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a branch opened now would still pay off: signals the judge weighs, never a cap the scheduler enforces.

Whether a branch may be opened at all is the roster's answer (its free slots, a closed roster, a stopping session).
These signals say whether one opened now has the time and the test capacity to show progress: the least EXPLORE time
a new branch needs to show a trend, and how many more tests the toolbox would run at once.
"""

from __future__ import annotations

from dataclasses import dataclass

from hyperloom_multibranch.contracts.toolbox import QueueState, Toolbox
from hyperloom_multibranch.notebook.rsi.frontier import FrontierPolicy
from hyperloom_multibranch.scheduling.branches import SPECIALIST_START_SECONDS
from hyperloom_multibranch.scheduling.roster import Roster, RosterCaps

#: Sandbox cold starts a new branch waits through before its tests start paying off: its first test's, and one more.
MB_MIN_USEFUL_COLD_STARTS = 2


@dataclass(frozen=True)
class WideningSignals:
    """``cold_start_seconds`` is ``None`` before the toolbox's first sandbox cold start, and ``min_useful_seconds``
    then leaves the cold starts out, so it reads low. ``toolbox_headroom`` is negative while more tests wait than the
    global cap leaves room for."""

    min_useful_seconds: float
    cold_start_seconds: float | None
    toolbox_headroom: int


def min_useful_seconds(frontier: FrontierPolicy, queue_state: QueueState) -> float:
    """The least EXPLORE time a branch opened now needs: the frontier's trend window, or its start-up and cold starts
    when they take longer."""

    cold_start = queue_state.cold_start_seconds
    start_up = SPECIALIST_START_SECONDS + (MB_MIN_USEFUL_COLD_STARTS * cold_start if cold_start is not None else 0.0)
    return max(frontier.window_seconds, start_up)


def toolbox_headroom(caps: RosterCaps, queue_state: QueueState) -> int:
    """Tests the toolbox would still start at once under the session's global cap, net of what runs and waits."""

    return caps.max_parallel_tests - queue_state.running - queue_state.queued


class Widening:
    """Reads the signals off the toolbox's queue, the roster's caps and the frontier policy the session scores with.

    ``frontier`` is the scoring's own policy, so the trend window here is the one the judge's expected gains rest on.
    """

    def __init__(self, toolbox: Toolbox, roster: Roster, frontier: FrontierPolicy) -> None:
        self._toolbox = toolbox
        self._roster = roster
        self._frontier = frontier

    async def signals(self) -> WideningSignals:
        state = await self._toolbox.queue_state()
        return WideningSignals(
            min_useful_seconds=min_useful_seconds(self._frontier, state),
            cold_start_seconds=state.cold_start_seconds,
            toolbox_headroom=toolbox_headroom(self._roster.caps, state),
        )


__all__ = [
    "MB_MIN_USEFUL_COLD_STARTS",
    "Widening",
    "WideningSignals",
    "min_useful_seconds",
    "toolbox_headroom",
]
