# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a session runs, what it calls out to, and how it ended."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient
from hyperloom_multibranch.contracts.blackboard.interfaces import UsageSink
from hyperloom_multibranch.contracts.llm import Completion
from hyperloom_multibranch.rulegate.critic import FixReviewer
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.close import CloseReport
from hyperloom_multibranch.scheduling.final import FinalChoice
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.stops import SessionStopReason

#: Stop reasons a session ends with when it ran its course, a budget spent as set included; any other one, or an
#: error, is a failed session.
COMPLETED_REASONS = frozenset(
    {
        SessionStopReason.TIME_EXHAUSTED,
        SessionStopReason.GPU_HOURS_EXHAUSTED,
        SessionStopReason.LLM_BUDGET_EXHAUSTED,
        SessionStopReason.JUDGE_STOPPED,
        SessionStopReason.FINISHED_AFTER_RESTART,
    }
)


@dataclass(frozen=True)
class SessionOutcome:
    """How the session ended; ``close`` is ``None`` only while CLOSE has not run.

    The session completed (exit code 0) only when it ran its course without an error, FINAL chose a commit, and the
    stop left no process and no test behind; or when, asked to stop after its map, it kept one without an error and the
    stop left nothing behind; or when a person rejected its agent map at its review and the stop left nothing behind.
    """

    stop_reason: SessionStopReason
    final: FinalChoice | None
    close: CloseReport | None
    error: str

    @property
    def completed(self) -> bool:
        clean = not self.error and self.close is not None and self.close.stop.clean
        if self.stop_reason in (SessionStopReason.MAPPED, SessionStopReason.REVIEW_REJECTED):
            return clean
        chose = self.final is not None and self.final.chosen is not None
        return self.stop_reason in COMPLETED_REASONS and chose and clean

    @property
    def exit_code(self) -> int:
        return 0 if self.completed else 1


@dataclass(frozen=True)
class SessionHost:
    """Where and when the session runs: its directory, the registry of every process it starts itself (production
    passes a registry on ``layout.pids``) and its clock."""

    layout: SessionLayout
    processes: ProcessRegistry
    clock: Clock


@dataclass(frozen=True)
class SessionServices:
    """What the session calls out to besides its peers: the completions the judge and the patch resolver ask, the
    Recipe KB PRELUDE reads the warm-start leads from and CLOSE writes the final recipe back to (``None`` when none is
    configured), what builds the critic the rule gate asks whether a proven correctness fix matters under the
    workload, on the scheduler's blackboard handle its spend is recorded through, and the completion CLOSE asks to change
    the scheduling playbook and to curate the RSI lesson pool, with ``author_model`` the model it runs on;
    ``resolver_model`` is the model the patch resolver runs on, which every patch it writes names."""

    judge_completion: Completion
    resolver_completion: Completion
    recipe_kb: RemoteRecipeClient | None
    critic: Callable[[UsageSink], FixReviewer]
    author_completion: Completion
    author_model: str
    resolver_model: str


__all__ = ["COMPLETED_REASONS", "SessionHost", "SessionOutcome", "SessionServices"]
