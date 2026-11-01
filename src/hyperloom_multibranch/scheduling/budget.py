# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's wall-clock budget: PRELUDE gets all of it, and EXPLORE ends as late as still lets FINAL start.

The baseline-prelude child gets all the remaining time. EXPLORE then runs until one longest root test of the baseline
(one test of this workload) and one CLOSE grace are left: the grace for EXPLORE to wind down so that FINAL starts
before the deadline, the test for FINAL's validation, whose tests run concurrently. FINAL waits for the validation tests
it started, past the deadline and the hard stop if need be, so its reserve is the time it is expected to take, not a
bound on it. CLOSE keeps nothing before the deadline: the deadline plus the CLOSE grace is the hard stop, and a FINAL
that ends later leaves CLOSE one CLOSE grace from its end. The constants are re-declared because their owners are
modules this package must not import; each names its source. A GPU-hour budget keeps FINAL the main flow's share:
EXPLORE ends once the tests' GPU time plus the card time any lease holds has used the rest of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hyperloom_multibranch.scheduling.clock import Clock

# inference_optimizer/cli/parser.py DEFAULT_MAX_HOURS
MB_DEFAULT_MAX_HOURS = 2.0
# orchestrator/phases/machine_state.py DEFAULT_LONGRUN_MAX_MINUTES (14 days); a budget of 0 means unlimited
MB_UNLIMITED_HOURS = 14 * 24
# orchestrator/phases/machine_state.py DEFAULT_PHASE_BUDGET_PCT: FRAMEWORK_AGENT 38 + KERNEL_AGENT 47, SWEEP
MB_EXPLORE_SHARE = 85
MB_FINAL_SHARE = 5
# orchestrator/state/shared_state.py effective_closing_grace_sec: min(120, total * 0.02)
MB_CLOSE_GRACE_CAP_SECONDS = 120.0
MB_CLOSE_GRACE_FRACTION = 0.02
# The main flow truncates --max-hours to int(h * 60) minutes, and 0 minutes would mean unlimited.
MB_MIN_PRELUDE_HOURS = 1 / 60

MB_SECONDS_PER_HOUR = 3600.0


def budget_seconds(max_hours: float) -> float:
    """``--max-hours`` in seconds; 0 means unlimited, capped like the main flow."""
    return (max_hours or MB_UNLIMITED_HOURS) * MB_SECONDS_PER_HOUR


def explore_gpu_seconds(max_gpu_hours: float) -> float:
    """The GPU seconds the session may use before EXPLORE ends, tests' GPU time plus card time any lease holds:
    ``--max-gpu-hours`` less FINAL's share, ``math.inf`` when it is 0. FINAL is not cut when it uses more than its
    share."""

    if not max_gpu_hours:
        return math.inf
    return max_gpu_hours * MB_SECONDS_PER_HOUR * MB_EXPLORE_SHARE / (MB_EXPLORE_SHARE + MB_FINAL_SHARE)


class BudgetExhausted(RuntimeError):
    """Too little wall-clock time is left to start the phase that asked."""


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
        """The deadline plus the CLOSE grace: nothing of the session but FINAL may run past it."""
        return self.deadline + self.close_grace_seconds

    def remaining(self, now: float) -> float:
        return max(0.0, self.deadline - now)

    def prelude_max_hours(self, now: float) -> float:
        """The child's ``--max-hours``: all the remaining time."""
        hours = self.remaining(now) / MB_SECONDS_PER_HOUR
        if hours < MB_MIN_PRELUDE_HOURS:
            raise BudgetExhausted(f"{hours * 60:.2f} min left, the baseline prelude needs at least one minute")
        return hours

    def explore_end(self, now: float, final_test_seconds: float) -> float:
        """When EXPLORE ends: ``final_test_seconds`` and one CLOSE grace before the deadline, but no sooner than
        ``now``."""

        reserve = final_test_seconds + self.close_grace_seconds
        return self.deadline - min(reserve, self.remaining(now))


class HardStop:
    """When nothing of the session may run any longer: the budget's hard stop until :meth:`bring_forward` or
    :meth:`follow_final` moves it.

    A stop signal or an unhandled error brings it forward to the CLOSE grace from that moment, once, so the unwinding
    of the phases, a cancelled PRELUDE's child and CLOSE all share that one grace. FINAL does not stop at it: once
    FINAL ends, the hard stop is no sooner than one CLOSE grace from then.
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

    def follow_final(self) -> float:
        """Move the hard stop to the CLOSE grace from now, as FINAL ends, unless it is later or was brought forward;
        return it."""

        if not self._brought_forward:
            self._at = max(self._at, self._clock.now() + self._grace)
        return self._at

    def extend(self, seconds: float) -> float:
        """Move the hard stop ``seconds`` later, as the deadline moves past a person's review, unless it was brought
        forward; return it."""

        if not self._brought_forward:
            self._at += max(seconds, 0.0)
        return self._at


__all__ = [
    "MB_CLOSE_GRACE_CAP_SECONDS",
    "MB_CLOSE_GRACE_FRACTION",
    "MB_DEFAULT_MAX_HOURS",
    "MB_EXPLORE_SHARE",
    "MB_FINAL_SHARE",
    "MB_MIN_PRELUDE_HOURS",
    "MB_UNLIMITED_HOURS",
    "BudgetExhausted",
    "HardStop",
    "SessionBudget",
    "budget_seconds",
    "explore_gpu_seconds",
]
