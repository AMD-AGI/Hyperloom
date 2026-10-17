# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""EXPLORE: open the opening batch, then let the judge steer the branches until the EXPLORE deadline or a soft stop.

The loop wakes every :data:`EXPLORE_TICK_SECONDS`: it stops the repair branches that used up their time, starts the
forks and repairs the fork desk admitted, consults the MB2 placeholders, and every judge interval asks the judge,
applies what it decided and rebuilds the dashboard. The judge reads the frontier-scored progress and how similar
archived sessions went; a judge call that has not answered by the EXPLORE deadline counts as failed, and a branch that
has not opened by then is withdrawn, so neither eats into FINAL. The integrator merges in a background task of its own, in rounds as wide as the integration branch's K. When EXPLORE ends
that task is cancelled and the integrator abandons its round, so the integration head the FINAL candidates are read
from no longer moves and no merge test competes with FINAL's measurements.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import BranchProgress
from hyperloom_multibranch.notebook.rsi.frontier import FrontierScoring
from hyperloom_multibranch.notebook.rsi.session import judge_history
from hyperloom_multibranch.scheduling.branches import BranchControl
from hyperloom_multibranch.scheduling.context import SchedulerContext
from hyperloom_multibranch.scheduling.dashboard import SessionDashboard
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind
from hyperloom_multibranch.scheduling.forks import ForkDesk
from hyperloom_multibranch.scheduling.integrator import Integrator
from hyperloom_multibranch.scheduling.judge import Judge, JudgeDecision, JudgeView
from hyperloom_multibranch.scheduling.mb_fakes import mb_fake_branches_to_add, mb_fake_similar_branches
from hyperloom_multibranch.scheduling.roster import BranchEntry, BranchOrigin, BranchStop, StopCause
from hyperloom_multibranch.scheduling.settings import PRIMARY_METRIC
from hyperloom_multibranch.scheduling.stops import SessionStopReason
from hyperloom_multibranch.scheduling.widening import Widening

logger = logging.getLogger(__name__)

#: How often the loop starts admitted forks and repairs and checks the deadline; the judge runs on its own, longer interval.
EXPLORE_TICK_SECONDS = 5.0
SIMILAR_BRANCH_REASON = "duplicates another branch"
OPEN_CUT_REASON = "branch slots taken"


@dataclass(frozen=True)
class ExploreWindow:
    """The commit branches start from, the roofline ceiling, the clock times EXPLORE and the budget end at, and the
    session's whole budget."""

    root_commit: str
    roofline_ceiling: Mapping[str, float]
    explore_end: float
    session_deadline: float
    budget_seconds: float


@dataclass(frozen=True)
class Steering:
    """How the judge steers EXPLORE: the judge, how often it is asked, the frontier-scored progress it reads, against
    the archived sessions that progress holds, the dashboard rebuilt once the opening batch is open and after every
    later judge call, and the signals of whether a branch opened now would still pay off."""

    judge: Judge
    interval_seconds: float
    progress: FrontierScoring
    dashboard: SessionDashboard
    widening: Widening


def integration_row(progress: tuple[BranchProgress, ...]) -> BranchProgress | None:
    return next((row for row in progress if row.branch == INTEGRATION_BRANCH), None)


class Explorer:
    """Runs EXPLORE once; the scheduler's notebook handle gives the snapshot the judge's digest of the archived
    sessions compares with them."""

    def __init__(
        self,
        context: SchedulerContext,
        branches: BranchControl,
        desk: ForkDesk,
        integrator: Integrator,
        steering: Steering,
    ) -> None:
        self._roster = context.roster
        self._branches = branches
        self._desk = desk
        self._judge = steering.judge
        self._integrator = integrator
        self._progress = steering.progress
        self._notebook = context.notebook
        self._dashboard = steering.dashboard
        self._decisions = context.decisions
        self._interval = steering.interval_seconds
        self._clock = context.clock
        self._widening = steering.widening

    async def run(self, window: ExploreWindow) -> SessionStopReason:
        """Explore until the deadline or the judge's soft stop; the integrator is stopped before this returns.

        Nothing is opened when the opening judge call asks for the soft stop or answers only once EXPLORE is over.
        """

        opening = await self._ask(window, opening=True)
        if opening is None:
            await self._open_batch(1, 1, window.root_commit, BranchOrigin.FIRST_BATCH, window.explore_end)
        elif await self._apply(opening, window.root_commit, BranchOrigin.FIRST_BATCH, window.explore_end):
            return SessionStopReason.GLOBAL_CONVERGED
        if self._clock.now() >= window.explore_end:
            return SessionStopReason.TIME_EXHAUSTED
        await self._dashboard.refresh()
        merging = asyncio.ensure_future(self._merge_forever())
        try:
            return await self._steer(window, merging)
        finally:
            merging.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await merging
            await self._integrator.abandon()

    async def _steer(self, window: ExploreWindow, merging: asyncio.Future[None]) -> SessionStopReason:
        next_judge = self._clock.now() + self._interval
        while (now := self._clock.now()) < window.explore_end:
            if merging.done():
                merging.result()
            await self._branches.stop_overrun_repairs()
            await self._start_admitted(window.explore_end)
            await self._consult_placeholders(window)
            if now >= next_judge:
                next_judge = now + self._interval
                decision = await self._ask(window, opening=False)
                if decision is not None and await self._apply(
                    decision, window.root_commit, BranchOrigin.REFILL, window.explore_end
                ):
                    return SessionStopReason.GLOBAL_CONVERGED
                await self._dashboard.refresh()
            await self._clock.sleep(min(EXPLORE_TICK_SECONDS, window.explore_end - now))
        return SessionStopReason.TIME_EXHAUSTED

    async def _merge_forever(self) -> None:
        while True:
            width = self._roster.entry(INTEGRATION_BRANCH).k
            if not await self._integrator.integrate_next(width):
                await self._clock.sleep(EXPLORE_TICK_SECONDS)

    async def _start_admitted(self, deadline: float) -> None:
        for admitted in self._desk.waiting():
            if await self._branches.start_admitted(admitted, deadline):
                self._desk.release(admitted.branch)

    async def _consult_placeholders(self, window: ExploreWindow) -> None:
        for branch in mb_fake_similar_branches(self._roster):
            await self._stop(branch, SIMILAR_BRANCH_REASON)
        extra = min(mb_fake_branches_to_add(self._roster), self._roster.free_slots())
        if extra:
            await self._branches.open_branches(extra, window.root_commit, 1, BranchOrigin.REFILL, window.explore_end)

    async def _ask(self, window: ExploreWindow, *, opening: bool) -> JudgeDecision | None:
        now = self._clock.now()
        progress = await self._progress.progress()
        integration = integration_row(progress)
        view = JudgeView(
            now=now,
            explore_remaining_seconds=max(window.explore_end - now, 0.0),
            session_remaining_seconds=max(window.session_deadline - now, 0.0),
            caps=self._roster.caps,
            roster=self._roster.entries(),
            progress=progress,
            integration_value=dict(integration.head_value) if integration is not None else {},
            roofline_ceiling=window.roofline_ceiling,
            metric=PRIMARY_METRIC,
            opening=opening,
            free_slots=self._roster.free_slots(),
            history=judge_history(
                await self._notebook.snapshot(),
                self._progress.history,
                PRIMARY_METRIC,
                budget_seconds=window.budget_seconds,
            ),
            widening=await self._widening.signals(),
        )
        try:
            decision = await self._clock.wait_for(self._judge.decide(view), window.explore_end - now)
        except asyncio.TimeoutError:
            self._decisions.record(Decision(DecisionKind.JUDGE_FAILED, "no answer by the EXPLORE deadline"))
            return None
        if decision is None:
            self._decisions.record(Decision(DecisionKind.JUDGE_FAILED, "kept the status quo"))
        return decision

    async def _apply(self, decision: JudgeDecision, start_commit: str, origin: BranchOrigin, deadline: float) -> bool:
        """Apply the stop and K orders, then open the branches the judge asked for, of ``origin`` from
        ``start_commit``; True, and nothing opened, when the judge asked to stop the session."""

        self._decisions.record(
            Decision(
                DecisionKind.JUDGE,
                decision.session_stop or "applied",
                detail={
                    "open": decision.open_count,
                    "open_k": decision.open_k,
                    "k": {order.branch: order.k for order in decision.k},
                    "stop": {order.branch: order.reason for order in decision.stop},
                },
            )
        )
        for order in decision.stop:
            await self._stop(order.branch, order.reason)
        if decision.k:
            await self._branches.apply_k({order.branch: order.k for order in decision.k})
        if decision.session_stop:
            return True
        await self._open_batch(decision.open_count, decision.open_k, start_commit, origin, deadline)
        return False

    async def _open_batch(
        self, count: int, k: int, start_commit: str, origin: BranchOrigin, deadline: float
    ) -> tuple[BranchEntry, ...]:
        """Open up to ``count`` branches of ``origin`` at K ``k`` from ``start_commit``, each by ``deadline``; none
        once the deadline has passed. The only place a batch is cut to the free slots: a fork or repair admitted while
        the batch starts can take the last ones, and the branches that found no slot are recorded as ``open_cut``."""

        if self._clock.now() >= deadline:
            return ()
        opened = await self._branches.open_branches(count, start_commit, k, origin, deadline)
        if len(opened) < count:
            detail = {"requested": count, "opened": len(opened), "open_cut": count - len(opened)}
            self._decisions.record(Decision(DecisionKind.OPEN, OPEN_CUT_REASON, detail=detail))
        return opened

    async def _stop(self, branch: str, reason: str) -> None:
        """A judge stop; a step that fails is logged and exploring goes on, as CLOSE kills what the stop left."""

        stop = BranchStop(StopCause.JUDGE, reason, self._clock.now())
        for failure in await self._branches.stop_branch(branch, stop):
            logger.error("the judge's stop of %s did not finish cleanly", branch, exc_info=failure)


__all__ = ["EXPLORE_TICK_SECONDS", "OPEN_CUT_REASON", "ExploreWindow", "Explorer", "Steering", "integration_row"]
