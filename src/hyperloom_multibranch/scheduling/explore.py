# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""EXPLORE: open the opening batch, then let the judge steer the branches until the EXPLORE deadline or a soft stop.

The branches the user wrote a direction for, then those of the agents the user named, open first, from the start
commit the window names -- the root, or the Recipe KB champion PRELUDE replayed onto it -- and so does the opening
batch; the opening judge call sizes that batch into the slots those branches leave. Gains are still measured from the
root, which the window names apart.

The loop wakes every :data:`EXPLORE_TICK_SECONDS`: it stops the repair branches that used up their time and the branches
whose specialist gave up, once no candidate of theirs waits for its confirm, moves the forks and repairs queued on the
fork desk onto the free slots, has the rule gate land each admitted fork and starts the admitted forks and repairs, and
every judge interval asks the judge, applies what it decided and rebuilds the dashboard. A slot the judge's stops free
goes to the queued requests first and only then to the branches the judge opens. The judge is asked at once, not at its
next interval, when the scheduler itself frees a slot: a repair branch stopped at its time limit, a branch whose
specialist gave up, a fork the gate did not land, or an admitted fork or repair that failed to start; and when an event
changes what the branches are worth: a merge that moved the integration head by a large gain, the bottleneck of the
profile taken of the integration head after such a merge moving, or a quarantined measurement node. A branch the
judge opens after the opening batch starts from the integration head, briefed on a focus. The judge reads the
frontier-scored progress, how similar archived sessions went, which running branches made much the same change, the
events since its last call, how far each branch lies behind the integration head, whether a profile of the integration
head runs, the GPU hours and the LLM dollars the specialists and the scheduler itself spent so far, and what is known of
each running branch's specialist (given up, stopped because its judging failed, its heartbeat lost
again and again), and decides what to stop; a judge call that has not answered by the EXPLORE deadline counts as failed,
and a branch that has not opened by then is withdrawn, so neither eats into FINAL. The integrator merges in a background
task of its own, in rounds as wide as the integration branch's K. When EXPLORE ends that task is cancelled and the
integrator abandons its round, so the integration head the FINAL candidates are read from no longer moves and no merge
test competes with FINAL's measurements; a profile of the integration head still running is cancelled before that,
taking its test back. Under ``--max-gpu-hours`` EXPLORE also ends, at a tick, once the session's tests have used the GPU
hours FINAL's share leaves it, and under ``--max-llm-usd`` once the LLM spend recorded by the specialists, the judge
and the patch resolver reaches it: FINAL runs no specialist and asks no judge, so it gets no share of that cap. EXPLORE
ends at a tick as well once the toolbox has fewer usable cards than one test needs, and the session then skips FINAL.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from hyperloom_multibranch.contracts.lifecycle import SpecialistLifecycle, SpecialistStatus
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import BranchProgress, priced_usd
from hyperloom_multibranch.notebook.rsi.calibration import PolicyChoice
from hyperloom_multibranch.notebook.rsi.frontier import FrontierScoring
from hyperloom_multibranch.notebook.rsi.session import judge_history
from hyperloom_multibranch.scheduling.budget import MB_SECONDS_PER_HOUR
from hyperloom_multibranch.scheduling.branches import SPECIALIST_START_SECONDS, BatchStart, BranchControl
from hyperloom_multibranch.scheduling.context import SchedulerContext
from hyperloom_multibranch.scheduling.dashboard import SessionDashboard
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind
from hyperloom_multibranch.scheduling.events import SessionEvents, behind_integration_pct
from hyperloom_multibranch.scheduling.integrator import Integrator
from hyperloom_multibranch.scheduling.judge import (
    HoldReason,
    Judge,
    JudgeDecision,
    JudgeView,
    held_branches,
    hold_decision,
)
from hyperloom_multibranch.scheduling.roster import (
    BranchEntry,
    BranchKind,
    BranchOrigin,
    BranchStop,
    StopCause,
    specialist_open,
)
from hyperloom_multibranch.scheduling.settle import failure_text, settled
from hyperloom_multibranch.scheduling.similarity import BranchSimilarity
from hyperloom_multibranch.scheduling.stops import SessionStopReason
from hyperloom_multibranch.scheduling.widening import Widening, refill_focus

logger = logging.getLogger(__name__)

#: How often the loop starts admitted forks and repairs and checks the deadline; the judge runs on its own, longer interval.
EXPLORE_TICK_SECONDS = 5.0
OPEN_CUT_REASON = "branch slots taken"
FORK_NOT_LANDED = "the rule gate did not land the fork"


def gave_up_reason(status: SpecialistStatus) -> str:
    """Why a branch whose specialist gave up is stopped, from the counts the runtime keeps: the crashes it gave up
    after and how many of them lost their heartbeat, or no crash at all, which leaves the sessions that found nothing
    to test as the only way it gives up.

    The counts are reported rather than the limit they crossed, which is the runtime's own to name.
    """

    if not status.crashes:
        return "its specialist gave up: its sessions in a row found nothing to test"
    lost = f", {status.lost} of them with a lost heartbeat" if status.lost else ""
    return f"its specialist gave up after {status.crashes} crashes{lost}"


@dataclass(frozen=True)
class ExploreWindow:
    """The commit every gain is measured against, the commit the first branches start from, the roofline ceiling, the
    clock times EXPLORE and the budget end at, the session's whole budget, the GPU seconds the session's tests may use
    before EXPLORE ends and the dollars the session's recorded LLM calls may cost (``math.inf`` for no limit, for
    either), the upper bound of the GPU seconds the PRELUDE child used, which no limit counts, and the directions the
    user wrote, one user branch each.

    ``root_commit`` is the session's measurement anchor and nothing moves it. ``start_commit`` is where EXPLORE opens:
    the root, or the Recipe KB champion PRELUDE replayed onto it.
    """

    root_commit: str
    start_commit: str
    roofline_ceiling: Mapping[str, float]
    explore_end: float
    session_deadline: float
    budget_seconds: float
    explore_gpu_seconds: float
    max_llm_usd: float
    prelude_gpu_seconds_upper_bound: float
    user_directions: tuple[str, ...]
    agents: tuple[str, ...]


@dataclass(frozen=True)
class Steering:
    """How the judge steers EXPLORE: the judge, how often it is asked, the frontier-scored progress it reads, against
    the archived sessions that progress holds, the dashboard rebuilt once the opening batch is open and after every
    later judge call, the signals of whether a branch opened now would still pay off, which running branches made
    much the same change, the events that have the judge asked at once, the specialist lifecycle, which reports
    each running branch's specialist, and the frontier policy ``progress`` scores with and why it was chosen."""

    judge: Judge
    interval_seconds: float
    progress: FrontierScoring
    dashboard: SessionDashboard
    widening: Widening
    similarity: BranchSimilarity
    events: SessionEvents
    specialists: SpecialistLifecycle
    frontier_policy: PolicyChoice


def spent_gpu_seconds(progress: Iterable[BranchProgress]) -> float:
    """The GPU seconds the session's tests used: every test of every branch that has a result, the integration
    branch's baseline on the root included; a test withdrawn before its result counts none."""

    return sum(row.gpu_seconds for row in progress)


def integration_row(progress: tuple[BranchProgress, ...]) -> BranchProgress | None:
    return next((row for row in progress if row.branch == INTEGRATION_BRANCH), None)


class Explorer:
    """Runs EXPLORE once; the scheduler's notebook handle gives the snapshot the judge's digest of the archived
    sessions compares with them, and its rule gate lands each fork the fork desk admits."""

    def __init__(
        self,
        context: SchedulerContext,
        branches: BranchControl,
        integrator: Integrator,
        steering: Steering,
    ) -> None:
        self._roster = context.roster
        self._branches = branches
        self._desk = context.desk
        self._reserved = context.reserved
        self._judge = steering.judge
        self._integrator = integrator
        self._progress = steering.progress
        self._metric = steering.progress.policy.metric
        self._notebook = context.notebook
        self._dashboard = steering.dashboard
        self._decisions = context.decisions
        self._interval = steering.interval_seconds
        self._clock = context.clock
        self._widening = steering.widening
        self._similarity = steering.similarity
        self._events = steering.events
        self._specialists = steering.specialists
        self._frontier_policy = steering.frontier_policy

    async def run(self, window: ExploreWindow) -> SessionStopReason:
        """Explore until the deadline or the judge's soft stop; the integrator is stopped before this returns.

        The user branches open before the opening judge call; the opening batch opens nothing more when that call
        asks for the soft stop or answers only once EXPLORE is over, and opens one branch when it fails and no
        specialist branch is open.

        A start commit PRELUDE moved off the root is profiled as the head a merge moved is, so that the refills read
        the op shares of the commit they open on rather than the root's, which PRELUDE analysed from another recipe.
        """

        if window.start_commit != window.root_commit:
            self._events.start_profiling(window.start_commit, window.root_commit)
        await self._open_named(window)
        opening = await self._ask(window, opening=True)
        first_batch = BatchStart(BranchOrigin.FIRST_BATCH, window.start_commit, window.explore_end)
        if opening is None:
            await self._open_batch(0 if specialist_open(self._roster.entries()) else 1, 1, first_batch)
        elif await self._apply(opening, first_batch):
            return SessionStopReason.JUDGE_STOPPED
        if self._clock.now() >= window.explore_end:
            return SessionStopReason.TIME_EXHAUSTED
        await self._dashboard.refresh()
        merging = asyncio.ensure_future(self._merge_forever())
        try:
            return await self._steer(window, merging)
        finally:
            merging.cancel()
            try:
                with contextlib.suppress(asyncio.CancelledError):
                    await merging
                await self._events.stop_profiling()
            finally:
                await self._integrator.abandon()

    async def _open_named(self, window: ExploreWindow) -> None:
        if self._clock.now() < window.explore_end:
            await self._branches.open_user_branches(window.user_directions, window.start_commit, window.explore_end)
            await self._branches.open_agent_branches(window.agents, window.start_commit, window.explore_end)

    async def _steer(self, window: ExploreWindow, merging: asyncio.Future[None]) -> SessionStopReason:
        next_judge = self._clock.now() + self._interval
        while (now := self._clock.now()) < window.explore_end:
            if merging.done():
                merging.result()
            if (spent := await self._budget_spent(window)) is not None:
                return spent
            if await self._events.record_card_shortage():
                return SessionStopReason.NO_MEASUREMENT_CARDS
            overrun = await self._branches.stop_overrun_repairs()
            given_up = await self._stop_given_up()
            self._desk.fill_free_slots()
            freed = await self._start_admitted(window.explore_end) or overrun or given_up
            if await self._events.detect() or freed:
                next_judge = now
            if now >= next_judge:
                next_judge = now + self._interval
                decision = await self._ask(window, opening=False)
                if decision is not None and await self._apply(decision, await self._refill(window)):
                    return SessionStopReason.JUDGE_STOPPED
                await self._dashboard.refresh()
            await self._clock.sleep(min(EXPLORE_TICK_SECONDS, window.explore_end - now))
        return SessionStopReason.TIME_EXHAUSTED

    async def _budget_spent(self, window: ExploreWindow) -> SessionStopReason | None:
        """The budget the session used up, if any: the GPU seconds EXPLORE has, then the session's LLM dollars.
        Reads the notebook's own progress: a frontier-scored read appends a round to the frontier log."""

        gpu_limited = window.explore_gpu_seconds != math.inf
        if gpu_limited and spent_gpu_seconds(await self._notebook.progress()) >= window.explore_gpu_seconds:
            return SessionStopReason.GPU_HOURS_EXHAUSTED
        if window.max_llm_usd != math.inf and (await self._notebook.snapshot()).llm_usd >= window.max_llm_usd:
            return SessionStopReason.LLM_BUDGET_EXHAUSTED
        return None

    async def _merge_forever(self) -> None:
        while True:
            width = self._roster.entry(INTEGRATION_BRANCH).k
            reports = await self._integrator.integrate_next(width, EXPLORE_TICK_SECONDS)
            self._events.note_merges(reports)
            if not reports:
                await self._clock.sleep(EXPLORE_TICK_SECONDS)

    async def _start_admitted(self, deadline: float) -> bool:
        """Land the admitted forks and start the forks and repairs the fork desk admitted; true when a fork did not land
        or a branch failed to start, freeing its slot."""

        freed = False
        for admitted in self._desk.waiting():
            self._desk.release(admitted.branch)
            failure = await self._land(admitted.branch, deadline)
            if failure:
                await self._branches.void_admitted(admitted.branch, failure)
            else:
                await self._branches.start_admitted(admitted, deadline)
            stop = self._roster.entry(admitted.branch).stop
            freed = freed or (stop is not None and stop.cause is StopCause.START_FAILED)
        return freed

    async def _land(self, branch: str, deadline: float) -> str:
        """Have the rule gate land ``branch`` when it is a live fork; why it did not, or ``""``."""

        entry = self._roster.entry(branch)
        if entry.origin is not BranchOrigin.FORK or entry.stop is not None:
            return ""
        seconds = min(SPECIALIST_START_SECONDS, deadline - self._clock.now())
        landed = await settled(self._clock.wait_for(self._reserved.land_split(branch), seconds))
        if isinstance(landed, BaseException):
            logger.error("the fork %s was not landed", branch, exc_info=landed)
            return failure_text(landed)
        return "" if landed else FORK_NOT_LANDED

    async def _refill(self, window: ExploreWindow) -> BatchStart:
        """Where the branches a steering judge call opens start: the integration head, which no judge stop moves."""

        return BatchStart(BranchOrigin.REFILL, await self._branches.integration_head(), window.explore_end)

    async def _ask(self, window: ExploreWindow, *, opening: bool) -> JudgeDecision | None:
        now = self._clock.now()
        progress = await self._progress.progress()
        integration = integration_row(progress)
        integration_value = dict(integration.head_value) if integration is not None else {}
        root = await self._notebook.recorded_value(window.root_commit)
        snapshot = await self._notebook.snapshot()
        specialists = await self._specialist_states()
        view = JudgeView(
            now=now,
            explore_remaining_seconds=max(window.explore_end - now, 0.0),
            session_remaining_seconds=max(window.session_deadline - now, 0.0),
            caps=self._roster.caps,
            roster=self._roster.entries(),
            progress=progress,
            integration_value=integration_value,
            roofline_ceiling=window.roofline_ceiling,
            metric=self._metric,
            opening=opening,
            free_slots=self._roster.free_slots(),
            pending=self._desk.pending(),
            history=judge_history(
                snapshot,
                self._progress.history,
                self._metric,
                budget_seconds=window.budget_seconds,
            ),
            widening=await self._widening.signals(),
            similar_to=await self._similarity.similar_to(),
            events=self._events.hand_over(),
            behind_integration_pct=behind_integration_pct(progress, integration_value, root.values, self._metric),
            profile_running=self._events.profiling,
            gpu_hours_used=spent_gpu_seconds(progress) / MB_SECONDS_PER_HOUR,
            explore_gpu_hours=_hours_or_none(window.explore_gpu_seconds),
            prelude_gpu_hours_upper_bound=window.prelude_gpu_seconds_upper_bound / MB_SECONDS_PER_HOUR,
            specialist_llm_usd=sum(row.llm_usd for row in progress),
            max_llm_usd=None if window.max_llm_usd == math.inf else window.max_llm_usd,
            scheduler_llm_usd=priced_usd(record.usage for record in snapshot.llm_usage if not record.usage.branch),
            specialists=specialists,
            held=await self._held(specialists),
            frontier_policy=self._frontier_policy,
            frontier_scorer=self._progress.guard.label,
        )
        try:
            decision = await self._clock.wait_for(self._judge.decide(view), window.explore_end - now)
        except asyncio.TimeoutError:
            self._decisions.record(Decision(DecisionKind.JUDGE_FAILED, "no answer by the EXPLORE deadline"))
            return None
        if decision is None:
            self._decisions.record(Decision(DecisionKind.JUDGE_FAILED, "kept the status quo"))
        return decision

    async def _specialist_states(self) -> dict[str, SpecialistStatus]:
        """What the specialist lifecycle reports of the specialist of every live specialist branch; a branch it does
        not run is left out."""

        states: dict[str, SpecialistStatus] = {}
        for entry in self._roster.entries():
            if entry.kind is BranchKind.SPECIALIST and entry.stop is None:
                state = await self._specialists.status(entry.branch)
                if state is not None:
                    states[entry.branch] = state
        return states

    async def _held(self, specialists: Mapping[str, SpecialistStatus]) -> dict[str, HoldReason]:
        snapshot = await self._notebook.snapshot()
        return held_branches(
            self._roster.entries(), snapshot.progress, snapshot.records, specialists, self._clock.now()
        )

    async def _stop_given_up(self) -> bool:
        """Stop every live branch whose specialist gave up and has no candidate waiting for its confirm: nothing of
        it is left to measure but that confirm, which the rule gate judges without the specialist. True when one
        stopped."""

        states = await self._specialist_states()
        given_up = [(branch, state) for branch, state in states.items() if state.gave_up]
        if not given_up:
            return False
        held = await self._held(states)
        stopped = False
        for branch, state in given_up:
            if held.get(branch) is not HoldReason.PENDING_CONFIRM:
                await self._stop(branch, gave_up_reason(state), StopCause.GAVE_UP)
                stopped = True
        return stopped

    async def _apply(self, decision: JudgeDecision, start: BatchStart) -> bool:
        """Apply the stop and K orders, move the queued forks and repairs onto the slots the stops freed, then open the
        branches the judge asked for as ``start`` says into the slots left; True, and nothing moved or opened, when the
        judge asked to stop the session. The held branches are found again here, not taken from the judge's view: a
        candidate that started waiting for its confirm while the judge answered holds its branch as well."""

        held = await self._held(await self._specialist_states())
        decision = hold_decision(decision, held, self._roster.entries())
        self._decisions.record(
            Decision(
                DecisionKind.JUDGE,
                decision.session_stop or "applied",
                detail={
                    "open": decision.open_count,
                    "open_k": decision.open_k,
                    "k": {order.branch: order.k for order in decision.k},
                    "stop": {order.branch: order.reason for order in decision.stop},
                    "held": {order.branch: order.reason.value for order in decision.held},
                },
            )
        )
        for order in decision.stop:
            await self._stop(order.branch, order.reason)
        if decision.k:
            await self._branches.apply_k({order.branch: order.k for order in decision.k})
        if decision.session_stop:
            return True
        self._desk.fill_free_slots()
        focus = await self._focus(decision, start) if start.origin is BranchOrigin.REFILL else ()
        await self._open_batch(decision.open_count, decision.open_k, start, focus)
        return False

    async def _focus(self, decision: JudgeDecision, start: BatchStart) -> tuple[str, ...]:
        shares = await self._notebook.op_shares(start.start_commit)
        return refill_focus(decision.focus, decision.open_count, shares)

    async def _open_batch(
        self, count: int, k: int, start: BatchStart, focus: tuple[str, ...] = ()
    ) -> tuple[BranchEntry, ...]:
        """Open up to ``count`` branches at K ``k`` as ``start`` says, briefed on ``focus``; none once its deadline has
        passed. The only place a batch is cut to the free slots: a fork or repair admitted while the batch starts can
        take the last ones, and the branches that found no slot are recorded as ``open_cut``."""

        if self._clock.now() >= start.deadline:
            return ()
        opened = await self._branches.open_branches(count, k, start, focus)
        if len(opened) < count:
            detail = {"requested": count, "opened": len(opened), "open_cut": count - len(opened)}
            self._decisions.record(Decision(DecisionKind.OPEN, OPEN_CUT_REASON, detail=detail))
        return opened

    async def _stop(self, branch: str, reason: str, cause: StopCause = StopCause.JUDGE) -> None:
        """A stop the judge ordered or the scheduler made; a step that fails is logged and exploring goes on, as CLOSE
        kills what the stop left."""

        stop = BranchStop(cause, reason, self._clock.now())
        for failure in await self._branches.stop_branch(branch, stop):
            logger.error("the %s stop of %s did not finish cleanly", cause.value, branch, exc_info=failure)


def _hours_or_none(seconds: float) -> float | None:
    return None if seconds == math.inf else seconds / MB_SECONDS_PER_HOUR


__all__ = [
    "EXPLORE_TICK_SECONDS",
    "FORK_NOT_LANDED",
    "OPEN_CUT_REASON",
    "ExploreWindow",
    "Explorer",
    "Steering",
    "gave_up_reason",
    "integration_row",
    "spent_gpu_seconds",
]
