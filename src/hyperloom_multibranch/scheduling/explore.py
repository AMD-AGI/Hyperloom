# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""EXPLORE: open the opening batch, then let the judge steer the branches until the EXPLORE deadline or a soft stop.

The branches the user wrote a direction for, then those of the agents the user named, open first, from the start
commit the window names -- the root, or the Recipe KB champion PRELUDE replayed onto it -- and so does the opening
batch; the opening judge call sizes that batch into the slots those branches leave. Gains are still measured from the
root, which the window names apart.

The loop wakes every :data:`EXPLORE_TICK_SECONDS`: it hands the toolbox the quota again, which takes a stopped branch
out once the tests it was left to finish have ended, stops the repair branches that used up their time and the branches
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
events since its last call, how far each branch lies behind the integration head, which branches lead it while their
commits fail to merge, whether a profile of the integration head runs, the GPU hours (tests' GPU time plus card
time any lease holds) and the LLM dollars the specialists and the scheduler itself spent so far, and what is known
of each running branch's specialist (given up, stopped because its judging failed, its heartbeat lost again and
again), which explore tests each branch has queued and which native builds its tests wait for, and decides
what to stop; a judge call that has not answered by the EXPLORE deadline counts as failed,
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
from dataclasses import dataclass, replace
from pathlib import Path

from hyperloom_multibranch.contracts.lifecycle import SpecialistLifecycle, SpecialistStatus
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, ConflictingWrite
from hyperloom_multibranch.contracts.blackboard.records import BranchProgress, BlackboardSnapshot, priced_usd
from hyperloom_multibranch.contracts.toolbox import Toolbox
from hyperloom_multibranch.blackboard.rsi.frontier import FrontierScoring, probation_event
from hyperloom_multibranch.blackboard.rsi.playbook import PlaybookPlay
from hyperloom_multibranch.blackboard.rsi.session import judge_history, judge_open_areas
from hyperloom_multibranch.blackboard.rsi.stranded import stranded_leads
from hyperloom_multibranch.scheduling.budget import MB_SECONDS_PER_HOUR
from hyperloom_multibranch.scheduling.branches import SPECIALIST_START_SECONDS, BatchStart, BranchControl
from hyperloom_multibranch.scheduling.build_waits import waiting_builds
from hyperloom_multibranch.scheduling.context import SchedulerContext
from hyperloom_multibranch.scheduling.dashboard import SessionDashboard
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind
from hyperloom_multibranch.scheduling.events import SessionEvents, behind_integration_pct
from hyperloom_multibranch.scheduling.integrator import Integrator
from hyperloom_multibranch.scheduling.judge import (
    ENVELOPE_HOLDS,
    GRACE_KEY,
    HoldReason,
    HoldRules,
    Judge,
    JudgeDecision,
    JudgeView,
    envelope_holds,
    held_branches,
    hold_decision,
    queued_intents,
    session_grace,
)
from hyperloom_multibranch.scheduling.manifest import read_manifest
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
from hyperloom_multibranch.scheduling.widening import Widening

logger = logging.getLogger(__name__)

#: How often the loop starts admitted forks and repairs and checks the deadline; the judge runs on its own, longer interval.
EXPLORE_TICK_SECONDS = 5.0
OPEN_CUT_REASON = "branch slots taken"
FORK_NOT_LANDED = "the rule gate did not land the fork"
PLAYBOOK_FALLBACK_REASON = "the challenger playbook broke the safety envelope; the champion plays from now on"
PLAYBOOK_OPEN_CUT_REASON = "the playbook opens no more branches now"


def gave_up_reason(status: SpecialistStatus) -> str:
    """Why a branch whose specialist gave up is stopped, from what the runtime keeps: why it could not be started, why
    its agent's run ended of itself, the crashes it gave up after and how many of them lost their heartbeat, or no
    crash at all, which leaves the sessions that found nothing to test as the only way it gives up.

    The counts are reported rather than the limit they crossed, which is the runtime's own to name.
    """

    if status.unready:
        return f"its specialist could not be started: {status.unready}"
    if status.finished:
        return status.finished
    if not status.crashes:
        return "its specialist gave up: its sessions in a row found nothing to test"
    lost = f", {status.lost} of them with a lost heartbeat" if status.lost else ""
    return f"its specialist gave up after {status.crashes} crashes{lost}"


@dataclass(frozen=True)
class ExploreWindow:
    """The commit every gain is measured against, the commit the first branches start from, the roofline ceiling, the
    clock times EXPLORE and the budget end at, the session's whole budget, the GPU seconds the session's tests and
    leases may use before EXPLORE ends and the dollars the session's recorded LLM calls may cost (``math.inf`` for no
    limit, for either), the upper bound of the GPU seconds the PRELUDE child used outside a lease, which no limit
    counts (0 where it ran on leased cards, whose card time the integration branch's progress holds), the
    directions the user wrote, one user branch each, the agents the user named, one agent branch each, and in the same
    order each agent's share of the session's directions (:func:`~.agent_plans.agent_plans`; ``""`` for none).

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
    agent_plans: tuple[str, ...] = ()


@dataclass(frozen=True)
class Steering:
    """How the judge steers EXPLORE: the judge, how often it is asked, the frontier-scored progress it reads, against
    the archived sessions that progress holds, the dashboard rebuilt once the opening batch is open and after every
    later judge call, the signals of whether a branch opened now would still pay off, which running branches made
    much the same change, the events that have the judge asked at once, the specialist lifecycle, which reports
    each running branch's specialist, the toolbox the native builds each branch's tests wait for are read from, and
    the scheduling playbook the session plays, ``None`` for a session that plays none, with the session's manifest a
    fallback from it is recorded in."""

    judge: Judge
    interval_seconds: float
    progress: FrontierScoring
    dashboard: SessionDashboard
    widening: Widening
    similarity: BranchSimilarity
    events: SessionEvents
    specialists: SpecialistLifecycle
    toolbox: Toolbox
    play: PlaybookPlay | None
    manifest: Path


def spent_gpu_seconds(progress: Iterable[BranchProgress]) -> float:
    """The GPU seconds the session used: its tests' GPU time plus the card time any lease holds outside them, read
    from the same progress. Every test of every branch that has a result counts, the integration branch's baseline on
    the root included, and a test withdrawn before its result counts none; an open lease counts up to the reading."""

    return sum(row.gpu_seconds + row.held_card_seconds for row in progress)


def integration_row(progress: tuple[BranchProgress, ...]) -> BranchProgress | None:
    return next((row for row in progress if row.branch == INTEGRATION_BRANCH), None)


class Explorer:
    """Runs EXPLORE once; the scheduler's blackboard handle gives the snapshot the judge's digest of the archived
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
        self._blackboard = context.blackboard
        self._dashboard = steering.dashboard
        self._decisions = context.decisions
        self._interval = steering.interval_seconds
        self._clock = context.clock
        self._widening = steering.widening
        self._similarity = steering.similarity
        self._events = steering.events
        self._specialists = steering.specialists
        self._toolbox = steering.toolbox
        self._play = steering.play
        self._manifest = steering.manifest
        self._scored: tuple[BranchProgress, ...] = ()
        self._too_late = False

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
            await self._branches.open_agent_branches(
                window.agents, window.start_commit, window.explore_end, window.agent_plans
            )

    async def _steer(self, window: ExploreWindow, merging: asyncio.Future[None]) -> SessionStopReason:
        next_judge = self._clock.now() + self._interval
        while (now := self._clock.now()) < window.explore_end:
            if merging.done():
                merging.result()
            if (spent := await self._budget_spent(window)) is not None:
                return spent
            if await self._events.record_card_shortage():
                return SessionStopReason.NO_MEASUREMENT_CARDS
            await self._branches.push_quota()
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
        """The budget the session used up, if any: the GPU seconds EXPLORE has, tests' and leases' alike, then the
        session's LLM dollars.
        Reads the blackboard's own progress: a frontier-scored read appends a round to the frontier log."""

        gpu_limited = window.explore_gpu_seconds != math.inf
        if gpu_limited and spent_gpu_seconds(await self._blackboard.progress()) >= window.explore_gpu_seconds:
            return SessionStopReason.GPU_HOURS_EXHAUSTED
        if window.max_llm_usd != math.inf and (await self._blackboard.snapshot()).llm_usd >= window.max_llm_usd:
            return SessionStopReason.LLM_BUDGET_EXHAUSTED
        return None

    async def _merge_forever(self) -> None:
        while True:
            width = self._roster.entry(INTEGRATION_BRANCH).k
            leads = stranded_leads(await self._blackboard.snapshot(), self._metric)
            reports = await self._integrator.integrate_next(width, EXPLORE_TICK_SECONDS, leads)
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
        progress = self._scored = await self._progress.progress()
        integration = integration_row(progress)
        integration_value = dict(integration.head_value) if integration is not None else {}
        root = await self._blackboard.recorded_value(window.root_commit)
        snapshot = await self._blackboard.snapshot()
        specialists = await self._specialist_states()
        await self._record_probations(snapshot, progress)
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
            history=judge_history(snapshot, self._progress.history, self._metric, budget_seconds=window.budget_seconds),
            widening=await self._widening.signals(),
            similar_to=await self._similarity.similar_to(snapshot.lineage),
            events=self._events.hand_over(),
            behind_integration_pct=behind_integration_pct(progress, integration_value, root.values, self._metric),
            stranded_leads=stranded_leads(snapshot, self._metric),
            profile_running=self._events.profiling,
            gpu_hours_used=spent_gpu_seconds(progress) / MB_SECONDS_PER_HOUR,
            explore_gpu_hours=_hours_or_none(window.explore_gpu_seconds),
            prelude_gpu_hours_upper_bound=window.prelude_gpu_seconds_upper_bound / MB_SECONDS_PER_HOUR,
            specialist_llm_usd=sum(row.llm_usd for row in progress),
            max_llm_usd=None if window.max_llm_usd == math.inf else window.max_llm_usd,
            scheduler_llm_usd=priced_usd(record.usage for record in snapshot.llm_usage if not record.usage.branch),
            specialists=specialists,
            held=await self._held(specialists, progress),
            queued=queued_intents(snapshot.records, snapshot.intents),
            building=await waiting_builds(self._toolbox, snapshot.records),
        )
        view = self._with_playbook(view, snapshot)
        try:
            decision = await self._clock.wait_for(self._judge.decide(view), window.explore_end - now)
        except asyncio.TimeoutError:
            self._decisions.record(Decision(DecisionKind.JUDGE_FAILED, "no answer by the EXPLORE deadline"))
            return None
        if decision is None:
            self._decisions.record(Decision(DecisionKind.JUDGE_FAILED, "kept the status quo"))
        return decision

    def _with_playbook(self, view: JudgeView, snapshot: BlackboardSnapshot) -> JudgeView:
        """``view`` with the playbook the session plays and where a new branch could open under it in what is left of
        EXPLORE, kept for the round's opens; as it is for a session that plays none."""

        if self._play is None:
            return view
        left, playbook = view.explore_remaining_seconds, self._play.playbook
        areas = judge_open_areas(snapshot, self._progress.history, self._metric, seconds_left=left, playbook=playbook)
        self._too_late = areas.too_late
        return replace(view, open_areas=areas.render(), playbook=self._play.render())

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

    async def _held(
        self, specialists: Mapping[str, SpecialistStatus], scored: Iterable[BranchProgress]
    ) -> dict[str, HoldReason]:
        """The held branches (:func:`held_branches`), under the grace of the playbook the session plays
        (:func:`session_grace`), with those its safety envelope keeps, read off the frontier ``scored`` rows of the
        round, where a hold of their own does not come first. A session that plays no playbook has the grace it always
        had, and no envelope, as its rows carry no stall evidence."""

        snapshot = await self._blackboard.snapshot()
        rules = HoldRules(self._clock.now(), session_grace(self._play))
        held = held_branches(self._roster.entries(), snapshot.progress, snapshot.records, specialists, rules)
        return {**envelope_holds(scored, specialists), **held}

    async def _record_probations(self, snapshot: BlackboardSnapshot, scored: Iterable[BranchProgress]) -> None:
        """Record each branch the round's stall evidence holds on probation under the stop rule the scoring read, the
        blackboard keeping one a stall; nothing in a session that plays no playbook, whose rows carry no stall
        evidence."""

        if self._progress.stop_rule is None:
            return
        rule = self._progress.stop_rule()
        for row in scored:
            event = probation_event(snapshot, row, rule)
            if event is not None:
                with contextlib.suppress(ConflictingWrite):
                    await self._blackboard.record_lineage(event)

    async def _stop_given_up(self) -> bool:
        """Stop every live branch whose specialist gave up and has no candidate waiting for its confirm: nothing of
        it is left to measure but that confirm, which the rule gate judges without the specialist. True when one
        stopped."""

        states = await self._specialist_states()
        given_up = [(branch, state) for branch, state in states.items() if state.gave_up]
        if not given_up:
            return False
        held = await self._held(states, self._scored)
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

        held = await self._held(await self._specialist_states(), self._scored)
        decision = hold_decision(decision, held, self._roster.entries())
        breaches = [
            f"stop of {item.branch} refused for {item.reason.value}"
            for item in decision.held
            if item.reason in ENVELOPE_HOLDS
        ]
        if self._play is not None and self._play.fall_back(breaches):
            await self._fell_back(self._play)
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
        await self._open_batch(self._playbook_room(decision.open_count, start), decision.open_k, start, focus)
        return False

    def _playbook_room(self, count: int, start: BatchStart) -> int:
        """How many of the ``count`` branches the judge asks for the playbook the session plays lets open, a cut
        recorded: no more specialist branches running at once than its ``max_open``, and no refill once the round's
        open areas found too little time left for a new branch's first commit. The forks and repairs the rule gate asks
        for are not the judge's opens."""

        if self._play is None or count <= 0:
            return count
        cap = self._play.playbook.numbers.max_open
        room = count
        if cap is not None:
            entries = self._roster.entries()
            running = sum(1 for entry in entries if entry.kind is BranchKind.SPECIALIST and entry.stop is None)
            room = min(room, max(0, cap - running))
        if start.origin is BranchOrigin.REFILL and self._too_late:
            room = 0
        if room < count:
            detail = {"requested": count, "opened": room, "max_open": cap, "too_late": self._too_late}
            self._decisions.record(Decision(DecisionKind.OPEN, PLAYBOOK_OPEN_CUT_REASON, detail=detail))
        return room

    async def _fell_back(self, play: PlaybookPlay) -> None:
        """Record that the challenger playbook broke the safety envelope and the champion plays from now on: in the
        decisions, and with the champion's grace in a manifest fingerprint, which CLOSE reads the session's trial
        from, and in the session's manifest."""

        detail = {"breach": play.breach, "playing": play.playbook.label}
        self._decisions.record(Decision(DecisionKind.EVENT, PLAYBOOK_FALLBACK_REASON, detail=detail))
        played = {**play.versions(), GRACE_KEY: str(session_grace(play))}
        fingerprints = (await self._blackboard.snapshot()).fingerprints
        if fingerprints:
            latest = fingerprints[-1]
            versions = {**latest.policy_versions, **played}
            await self._blackboard.write_manifest_fingerprint(replace(latest, policy_versions=versions))
        await asyncio.to_thread(_record_versions, self._manifest, played)

    async def _focus(self, decision: JudgeDecision, start: BatchStart) -> tuple[str, ...]:
        return await self._events.focus_on(start.start_commit, decision.focus, decision.open_count)

    async def _open_batch(
        self, count: int, k: int, start: BatchStart, focus: tuple[str, ...] = ()
    ) -> tuple[BranchEntry, ...]:
        """Open up to ``count`` branches at K ``k`` as ``start`` says, briefed on ``focus``; none once its deadline has
        passed. The only place a batch is cut to the free slots: a fork or repair admitted while the batch starts can
        take the last ones, and the branches that found no slot are recorded as ``open_cut``. A start commit a large
        merge moved the head to is profiled once branches opened on it, unless its merge had it profiled."""

        if self._clock.now() >= start.deadline:
            return ()
        opened = await self._branches.open_branches(count, k, start, focus)
        if opened:
            self._events.branches_opened_on(start.start_commit)
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


def _record_versions(path: Path, versions: Mapping[str, str]) -> None:
    manifest = read_manifest(path)
    replace(manifest, policy_versions={**manifest.policy_versions, **versions}).write(path)


__all__ = [
    "EXPLORE_TICK_SECONDS",
    "FORK_NOT_LANDED",
    "OPEN_CUT_REASON",
    "PLAYBOOK_FALLBACK_REASON",
    "PLAYBOOK_OPEN_CUT_REASON",
    "ExploreWindow",
    "Explorer",
    "Steering",
    "gave_up_reason",
    "integration_row",
    "spent_gpu_seconds",
]
