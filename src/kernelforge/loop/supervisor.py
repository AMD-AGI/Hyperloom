# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Self-supervision trigger for the forge-loop (AVO-style stall detection)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SupervisionMonitor:
    """Tracks the stall streak and decides WHEN to call the supervisor."""

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
        """Whether to consult the supervisor now, plus a factual reason."""
        if iteration - self.last_attempt_iter < self.cooldown:
            return False, ""
        if self.no_improve_streak >= self.supervise_after:
            return True, (f"no new best for {self.no_improve_streak} consecutive iterations")
        return False, ""

    def mark_attempted(self, iteration: int) -> None:
        """Anchor cooldown when the loop actually calls the Supervisor."""
        self.last_attempt_iter = iteration

    def mark_intervened(self, iteration: int) -> None:
        """Record that an intervention just happened (resets the streak so the"""
        self.intervention_count += 1
        self.last_intervention_iter = iteration
        self.last_attempt_iter = iteration
        self.no_improve_streak = 0
