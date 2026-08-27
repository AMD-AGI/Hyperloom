# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Self-supervision trigger for the forge-loop (AVO-style stall detection).

Decides WHEN to consult the supervisor using a cheap, purely FACTUAL signal: the
search has produced no new best for N consecutive iterations. It deliberately
does NOT judge WHY the search stalled — whether the implementer is circling one axis,
repeating a variant of a failed idea, or genuinely dead-ended is a strong-
semantic judgment left to the LLM supervisor, which reads the full trajectory
(plans + diffs + lessons) and decides persist-vs-pivot. See
:func:`kernelforge.orchestrator.supervisor.make_supervisor_fn` (a heterogeneous
model — e.g. codex/GPT — for diversity vs the Claude implementer).

This module is PURE state logic: no LLM, no I/O (easy to unit-test).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SupervisionMonitor:
    """Tracks the stall streak and decides WHEN to call the supervisor.

    One instance per run. ``record`` is called after every iteration; the loop
    consults ``should_intervene`` before each iteration and calls
    ``mark_intervened`` when it runs the supervisor. The trigger is purely the
    no-improvement streak (a budget signal); the semantic "is it circling /
    dead-ended" judgment is made by the LLM supervisor, not here.
    """

    supervise_after: int = 3  # consecutive no-improvement iters that trigger
    cooldown: int = 3  # min iterations between interventions

    no_improve_streak: int = 0
    intervention_count: int = 0
    last_intervention_iter: int = -10_000
    last_attempt_iter: int = -10_000

    def record(self, *, kept: bool) -> None:
        """Update the stall streak after an iteration completes."""
        self.no_improve_streak = 0 if kept else self.no_improve_streak + 1

    def should_intervene(self, iteration: int) -> tuple[bool, str]:
        """Whether to consult the supervisor now, plus a factual reason.

        Triggers only on a no-improvement stall. The reason is deliberately
        factual (a budget signal); the supervisor makes the semantic call about
        why the search stalled and whether to persist or pivot.
        """
        if iteration - self.last_attempt_iter < self.cooldown:
            return False, ""
        if self.no_improve_streak >= self.supervise_after:
            return True, (f"no new best for {self.no_improve_streak} consecutive iterations")
        return False, ""

    def mark_attempted(self, iteration: int) -> None:
        """Anchor cooldown when the loop actually calls the Supervisor."""
        self.last_attempt_iter = iteration

    def mark_intervened(self, iteration: int) -> None:
        """Record that an intervention just happened (resets the streak so the
        new directions get a fair chance before the next trigger)."""
        self.intervention_count += 1
        self.last_intervention_iter = iteration
        self.last_attempt_iter = iteration
        self.no_improve_streak = 0
