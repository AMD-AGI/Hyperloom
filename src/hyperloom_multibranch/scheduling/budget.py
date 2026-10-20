# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's wall-clock budget, split the way the single-lineage flow splits its own.

The baseline-prelude child gets all the remaining time; whatever it leaves, R, is shared
out by the main flow's phase shares: FINAL keeps R*5/92, but never less than the longest
root test of the baseline (one test of this workload), CLOSE keeps R*2/92 and EXPLORE
runs until both reserves are all that is left. The deadline plus the CLOSE grace is the
hard stop. The constants are re-declared because their owners are modules this package
must not import; each names its source. A GPU-hour budget keeps FINAL's share the same way:
EXPLORE ends once the tests have used the rest of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hyperloom_multibranch.scheduling.clock import Clock

# inference_optimizer/cli/parser.py DEFAULT_MAX_HOURS
MB_DEFAULT_MAX_HOURS = 2.0
# orchestrator/phases/machine_state.py DEFAULT_LONGRUN_MAX_MINUTES (14 days); a budget of 0 means unlimited
MB_UNLIMITED_HOURS = 14 * 24
# orchestrator/phases/machine_state.py DEFAULT_PHASE_BUDGET_PCT: FRAMEWORK_AGENT 38 + KERNEL_AGENT 47, SWEEP, CLOSE
MB_EXPLORE_SHARE = 85
MB_FINAL_SHARE = 5
MB_CLOSE_SHARE = 2
# orchestrator/state/shared_state.py effective_closing_grace_sec: min(120, total * 0.02)
MB_CLOSE_GRACE_CAP_SECONDS = 120.0
MB_CLOSE_GRACE_FRACTION = 0.02
# The main flow truncates --max-hours to int(h * 60) minutes, and 0 minutes would mean unlimited.
MB_MIN_PRELUDE_HOURS = 1 / 60

MB_SECONDS_PER_HOUR = 3600.0
_SHARE_TOTAL = MB_EXPLORE_SHARE + MB_FINAL_SHARE + MB_CLOSE_SHARE


def budget_seconds(max_hours: float) -> float:
    """``--max-hours`` in seconds; 0 means unlimited, capped like the main flow."""
    return (max_hours or MB_UNLIMITED_HOURS) * MB_SECONDS_PER_HOUR


def explore_gpu_seconds(max_gpu_hours: float) -> float:
    """The GPU seconds the session's tests may use before EXPLORE ends: ``--max-gpu-hours`` less FINAL's share,
    ``math.inf`` when it is 0. FINAL is not cut when it uses more than its share."""

    if not max_gpu_hours:
        return math.inf
    return max_gpu_hours * MB_SECONDS_PER_HOUR * MB_EXPLORE_SHARE / (MB_EXPLORE_SHARE + MB_FINAL_SHARE)


class BudgetExhausted(RuntimeError):
    """Too little wall-clock time is left to start the phase that asked."""


@dataclass(frozen=True)
class PhaseDeadlines:
    """Absolute clock times at which EXPLORE and FINAL end; CLOSE ends at the budget's deadline."""

    explore_end: float
    final_end: float


@dataclass(frozen=True)
class SessionBudget:
    """A wall-clock budget of ``total_seconds`` that started at ``started_at`` (clock seconds)."""

    started_at: float
    total_seconds: float

    @classmethod
    def from_hours(cls, max_hours: float, started_at: float) -> SessionBudget:
        """Build from ``--max-hours``, which the command line checked."""
        return cls(started_at=started_at, total_seconds=budget_seconds(max_hours))

    @property
    def deadline(self) -> float:
        return self.started_at + self.total_seconds

    @property
    def close_grace_seconds(self) -> float:
        return min(MB_CLOSE_GRACE_CAP_SECONDS, self.total_seconds * MB_CLOSE_GRACE_FRACTION)

    @property
    def hard_stop_at(self) -> float:
        """The deadline plus the CLOSE grace: nothing of the session may run past it."""
        return self.deadline + self.close_grace_seconds

    def remaining(self, now: float) -> float:
        return max(0.0, self.deadline - now)

    def prelude_max_hours(self, now: float) -> float:
        """The child's ``--max-hours``: all the remaining time."""
        hours = self.remaining(now) / MB_SECONDS_PER_HOUR
        if hours < MB_MIN_PRELUDE_HOURS:
            raise BudgetExhausted(f"{hours * 60:.2f} min left, the baseline prelude needs at least one minute")
        return hours

    def phases_after_prelude(self, now: float, final_floor_seconds: float) -> PhaseDeadlines:
        """Split what is left at ``now``: FINAL keeps its share or ``final_floor_seconds``, whichever is longer, and
        EXPLORE ends that much before FINAL does, at ``now`` at the earliest."""

        left = self.remaining(now)
        close_reserve = left * MB_CLOSE_SHARE / _SHARE_TOTAL
        final_reserve = min(max(left * MB_FINAL_SHARE / _SHARE_TOTAL, final_floor_seconds), left - close_reserve)
        return PhaseDeadlines(
            explore_end=self.deadline - final_reserve - close_reserve,
            final_end=self.deadline - close_reserve,
        )


class HardStop:
    """When nothing of the session may run any longer: the budget's hard stop until :meth:`bring_forward` moves it.

    A stop signal or an unhandled error brings it forward to the CLOSE grace from that moment, once, so the unwinding
    of the phases, a cancelled PRELUDE's child and CLOSE all share that one grace.
    """

    def __init__(self, budget: SessionBudget, clock: Clock) -> None:
        self._grace = budget.close_grace_seconds
        self._clock = clock
        self._at = budget.hard_stop_at
        self._brought_forward = False

    @property
    def at(self) -> float:
        return self._at

    def bring_forward(self) -> float:
        """Move the hard stop to the CLOSE grace from now, unless it is sooner or was moved already; return it."""

        if not self._brought_forward:
            self._at = min(self._at, self._clock.now() + self._grace)
            self._brought_forward = True
        return self._at


__all__ = [
    "MB_CLOSE_GRACE_CAP_SECONDS",
    "MB_CLOSE_GRACE_FRACTION",
    "MB_CLOSE_SHARE",
    "MB_DEFAULT_MAX_HOURS",
    "MB_EXPLORE_SHARE",
    "MB_FINAL_SHARE",
    "MB_MIN_PRELUDE_HOURS",
    "MB_UNLIMITED_HOURS",
    "BudgetExhausted",
    "HardStop",
    "PhaseDeadlines",
    "SessionBudget",
    "budget_seconds",
    "explore_gpu_seconds",
]
