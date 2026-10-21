# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Opening, re-sizing and stopping branches: the roster, the repository, the specialists, the toolbox quota and the
notebook in step.

Opening a branch records its ``OPEN`` lineage, cuts it in the experiment repository at its start commit, starts its
specialist there and gives it its K; the session's briefs keep the focus a branch the scheduler opens in a batch was
opened with. A user branch opens the same way at the root commit, on the direction the user wrote for it, which its
started decision records. A repair the fork desk admitted is opened the same way at the commit its blocked
candidate was measured against, however far the branch that asked for it has committed past that commit meanwhile:
the repair lands on its own branch, and the integrator carries a commit's own change onto the integration head. A
fork is cut by the rule gate at that same commit, which lands the fix on it first and records its
``SPLIT`` lineage, so it gets no ``OPEN``: it is opened at its head once the gate has landed it, and a fork the gate
does not land is stopped as not started. A repair's specialist starts on the request it was admitted for,
and a repair branch is stopped :data:`REPAIR_BRANCH_SECONDS` after its entry into the roster. Stopping a branch
withdraws everything of it: its specialist, its queued and running tests and tool jobs, the processes the scheduler
registered for it, its claims, and the forks and repairs of it that are queued, or admitted but not started yet, as
the fork desk admits nothing for a stopped branch. Every fork and repair voided that way, stopped before it started,
or whose start failed is handed to the rule gate, which records its candidate as only recorded unless it already
landed the fork; the notebook
gets the ``STOP`` lineage of a branch that started, the integration branch included, which only the end of the session
stops. A specialist counts as possibly running from the moment its start is asked for until the runtime confirms its
stop; only a start the runtime itself failed leaves nothing to stop. A call to a specialist gets
:data:`SPECIALIST_CALL_SECONDS`; a stop or a new K that overruns them, or fails, is recorded as a failed step, the rest
of the stop still runs, and the session stop asks once more for every stop that was not confirmed. The commits a
specialist already made stay in the notebook, so the integrator may still merge them. The roster entry is stopped
first, so an admitted branch whose parent is being stopped is never started. The toolbox gets a branch's K
before its specialist starts, so the first tests it submits already queue by it. A peer failure on one branch never
ends the session: a branch that does not open, its specialist's start included, within
:data:`SPECIALIST_START_SECONDS` or by the deadline its caller gives, whichever comes first, is withdrawn like a
stopped one.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hyperloom_multibranch.contracts.fork import RepairRequest
from hyperloom_multibranch.contracts.lifecycle import BranchTask, SpecialistLifecycle
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import LineageKind
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.scheduling.context import SchedulerContext
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind
from hyperloom_multibranch.scheduling.forks import DECISION_KINDS, AdmittedBranch
from hyperloom_multibranch.scheduling.settle import failure_text, settled
from hyperloom_multibranch.scheduling.roster import (
    BranchEntry,
    BranchOrigin,
    BranchStatus,
    BranchStop,
    StopCause,
)

if TYPE_CHECKING:
    from hyperloom_multibranch.notebook.rsi.briefing import SessionBriefs
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox

logger = logging.getLogger(__name__)

#: Clock seconds a specialist gets to stop or to take a new K before the call counts as failed.
SPECIALIST_CALL_SECONDS = 60.0
#: Clock seconds a branch gets to open, its specialist's start included, before the branch counts as not started.
SPECIALIST_START_SECONDS = 300.0
#: The K the lifecycle contract gives a specialist that has just started.
STARTED_K = 1
#: Clock seconds a repair branch may hold its slot, from its entry into the roster, before it is stopped: two hours,
#: the session owner's decision (board #54); a repair that waited in the fork desk's queue counts from leaving it.
REPAIR_BRANCH_SECONDS = 2 * 3600.0
REPAIR_TIME_LIMIT_REASON = f"a repair branch runs at most {REPAIR_BRANCH_SECONDS / 3600:g} hours"


@dataclass(frozen=True)
class BatchStart:
    """Where a batch of specialist branches opens: their origin, the commit they start from, and the clock time each
    must have started by."""

    origin: BranchOrigin
    start_commit: str
    deadline: float


@dataclass(frozen=True)
class Launch:
    """What a branch's specialist starts on: the repair the branch was opened for, the focus the session's briefs keep
    for it unless ``None``, the direction the user wrote for a user branch, and the agent the user named for an agent's
    branch."""

    repair: RepairRequest | None = None
    focus: str | None = None
    direction: str = ""
    agent: str = ""


class BranchControl:
    """Every change to a branch goes through here; ``briefs`` keep the focus of each branch it opens in a batch."""

    def __init__(
        self,
        context: SchedulerContext,
        lifecycle: SpecialistLifecycle,
        toolbox: RealToolbox,
        repo: ExperimentRepo,
        briefs: SessionBriefs,
    ) -> None:
        self._roster = context.roster
        self._notebook = context.notebook
        self._lifecycle = lifecycle
        self._toolbox = toolbox
        self._repo = repo
        self._briefs = briefs
        self._processes = context.processes
        self._decisions = context.decisions
        self._clock = context.clock
        self._desk = context.desk
        self._reserved = context.reserved
        self._specialists: set[str] = set()

    async def integration_head(self) -> str:
        return await self._repo.head(INTEGRATION_BRANCH)

    async def open_branches(
        self, count: int, k: int, start: BatchStart, focus: tuple[str, ...] = ()
    ) -> tuple[BranchEntry, ...]:
        """Open up to ``count`` specialist branches at K ``k`` as ``start`` says, record ``OPEN``, start them; the
        branches opened before the roster first refuses one. The n-th branch is briefed on ``focus[n]``, on none past
        its end or where it is ``""``.

        Forks and repairs are admitted while a branch starts, so the roster is asked again before each branch, right
        before its branch id is issued: a refused branch takes no id."""

        opened = []
        for index in range(count):
            entry = self._issue(start, k)
            if entry is None:
                break
            aim = focus[index] if index < len(focus) else ""
            opened.append(await self._open(entry, start.start_commit, start.deadline, Launch(focus=aim)))
        return tuple(opened)

    async def open_user_branches(
        self, directions: tuple[str, ...], root_commit: str, deadline: float
    ) -> tuple[BranchEntry, ...]:
        """Open one :attr:`BranchOrigin.USER` branch at ``root_commit`` for each of the user's ``directions`` as
        :meth:`open_branches` opens a batch, unfocused at K :data:`STARTED_K`; the roster entries issued before the
        roster first refuses one, a branch whose specialist did not start among them, already stopped."""

        launches = tuple(Launch(focus="", direction=direction) for direction in directions)
        return await self._open_at_root(BranchOrigin.USER, launches, root_commit, deadline)

    async def open_agent_branches(
        self, agents: tuple[str, ...], root_commit: str, deadline: float
    ) -> tuple[BranchEntry, ...]:
        """Open one :attr:`BranchOrigin.AGENT` branch at ``root_commit`` for each agent source the user named, as
        :meth:`open_user_branches` opens the user's directions; no judge changes its K of :data:`STARTED_K`."""

        launches = tuple(Launch(focus="", agent=agent) for agent in agents)
        return await self._open_at_root(BranchOrigin.AGENT, launches, root_commit, deadline)

    async def _open_at_root(
        self, origin: BranchOrigin, launches: tuple[Launch, ...], root_commit: str, deadline: float
    ) -> tuple[BranchEntry, ...]:
        start = BatchStart(origin, root_commit, deadline)
        opened = []
        for launch in launches:
            entry = self._issue(start, STARTED_K)
            if entry is None:
                break
            opened.append(await self._open(entry, root_commit, deadline, launch))
        return tuple(opened)

    def _issue(self, start: BatchStart, k: int) -> BranchEntry | None:
        """A new branch on the roster as ``start`` says, or ``None`` when the roster refuses one; a refused branch
        takes no id."""

        if self._roster.open_refusal():
            return None
        branch = self._roster.next_branch_id()
        return self._roster.open(branch, origin=start.origin, start_commit=start.start_commit, k=k)

    async def start_admitted(self, admitted: AdmittedBranch, deadline: float) -> None:
        """Open a repair, or a fork the rule gate has landed, that the fork desk admitted, and start its specialist by
        ``deadline`` (clock seconds); a branch stopped meanwhile stays stopped."""

        entry = self._roster.entry(admitted.branch)
        if entry.stop is not None:
            return
        start_commit = await settled(self._admitted_start(entry, admitted.parent_commit))
        if isinstance(start_commit, BaseException):
            await self._fail_start(entry, failure_text(start_commit))
            return
        started = self._roster.started(entry.branch, start_commit)
        await self._open(started, start_commit, deadline, Launch(repair=admitted.repair))

    async def void_admitted(self, branch: str, reason: str) -> None:
        """Void a fork the rule gate did not land, for ``reason``: it is stopped as not started, freeing its slot, and
        handed back to the gate like every admitted branch that stops."""

        await self._fail_start(self._roster.entry(branch), reason)

    async def _admitted_start(self, entry: BranchEntry, parent_commit: str) -> str:
        """The commit an admitted branch starts at: a repair's parent commit, a landed fork's head."""

        return parent_commit if entry.origin.cut_by_scheduler else await self._repo.head(entry.branch)

    async def _open(self, entry: BranchEntry, start_commit: str, deadline: float, launch: Launch) -> BranchEntry:
        """Record ``OPEN`` and cut a branch the scheduler cuts, give the toolbox its K and start its specialist on
        ``launch``; a branch any of that fails for, or that is not done within :data:`SPECIALIST_START_SECONDS` or by
        ``deadline``, is stopped with the failure as the reason. The started decision records the launch's focus, when
        it has one, and a user branch's direction."""

        seconds = min(SPECIALIST_START_SECONDS, deadline - self._clock.now())
        started = await settled(self._clock.wait_for(self._launch(entry, start_commit, launch), seconds))
        if isinstance(started, BaseException):
            logger.error("the specialist of %s did not start", entry.branch, exc_info=started)
            await self._fail_start(entry, failure_text(started))
            return entry
        detail: dict[str, object] = {"start_commit": start_commit, "k": entry.k, "from_branch": entry.from_branch}
        if launch.focus is not None:
            detail["focus"] = launch.focus
        if launch.direction:
            detail["direction"] = launch.direction
        kind = DECISION_KINDS.get(entry.origin, DecisionKind.OPEN)
        self._decisions.record(Decision(kind, f"{entry.origin.value} branch started", entry.branch, detail))
        return entry

    async def _fail_start(self, entry: BranchEntry, reason: str) -> None:
        stop = BranchStop(StopCause.START_FAILED, reason, self._clock.now())
        for failure in await self.stop_branch(entry.branch, stop):
            logger.error("%s did not stop cleanly after its failed start", entry.branch, exc_info=failure)

    async def apply_k(self, orders: Mapping[str, int]) -> None:
        """Give each live branch its new K, clamped by the roster; stopped branches are left alone.

        A specialist that does not take its new K keeps running; its ``SET_K`` decision names the failure.
        """

        for branch, k in orders.items():
            if self._roster.entry(branch).stop is not None:
                continue
            entry = self._roster.set_k(branch, k)
            failure = await self._send_k(entry) if branch in self._specialists else ""
            detail = {"asked": k, "failure": failure}
            self._decisions.record(Decision(DecisionKind.SET_K, f"K {entry.k}", branch, detail))
        await self.push_quota()

    async def stop_overrun_repairs(self) -> tuple[str, ...]:
        """Stop every repair branch that entered the roster :data:`REPAIR_BRANCH_SECONDS` ago or earlier and return
        them; a step that fails is logged, as CLOSE kills what the stop left."""

        now = self._clock.now()
        stopped = []
        for entry in self._roster.entries():
            if entry.origin is not BranchOrigin.REPAIR or entry.stop is not None:
                continue
            if now - entry.opened_at < REPAIR_BRANCH_SECONDS:
                continue
            stop = BranchStop(StopCause.REPAIR_TIME_LIMIT, REPAIR_TIME_LIMIT_REASON, now)
            for failure in await self.stop_branch(entry.branch, stop):
                logger.error("the time-limit stop of %s did not finish cleanly", entry.branch, exc_info=failure)
            stopped.append(entry.branch)
        return tuple(stopped)

    async def stop_branch(self, branch: str, stop: BranchStop) -> tuple[BaseException, ...]:
        """Withdraw everything of ``branch`` and return the steps that failed; stopping it again does nothing.

        Every step runs even when an earlier one fails, so the tests, processes and claims of the branch are always
        withdrawn and, once the branch is in the lineage, its ``STOP`` recorded; the ``STOP`` decision names each
        failure, and the caller decides what one means to it.
        """

        entry = self._roster.entry(branch)
        if entry.stop is not None:
            return ()
        self._roster.stop(branch, stop)
        forks = await self._stop_admitted_forks(branch, stop)
        orphaned = f"parent {branch} stopped: {stop.reason}"
        steps = await self._settle_voided(self._desk.void_from(branch, orphaned), orphaned)
        if entry.origin.reserved_by_gate and (
            entry.status is BranchStatus.ADMITTED or stop.cause is StopCause.START_FAILED
        ):
            steps += await self._settle_voided((branch,), stop.reason)
        steps.append(await settled(self.push_quota()))
        if branch in self._specialists:
            steps.append(await self._stop_specialist(branch))
        withdrawn = await settled(self._toolbox.cancel_branch(branch, stop.summary))
        steps.append(await settled(self._processes.kill_branch(branch)))
        released = await settled(self._notebook.release_branch(branch))
        if entry.start_commit:
            event = LineageEvent(LineageKind.STOP, branch, reason=stop.summary)
            steps.append(await settled(self._notebook.record_lineage(event)))
        raised = [outcome for outcome in (*steps, withdrawn, released) if isinstance(outcome, BaseException)]
        detail = {
            "withdrawn": 0 if isinstance(withdrawn, BaseException) else withdrawn,
            "released_claims": 0 if isinstance(released, BaseException) else len(released),
            "failures": [failure_text(error) for error in raised],
        }
        self._decisions.record(Decision(DecisionKind.STOP, stop.summary, branch, detail))
        return (*raised, *forks)

    async def void_queued(self, reason: str) -> list[BaseException]:
        """Void every fork and repair still queued, as the session stops for ``reason``, and have the gate record
        each; the records that failed."""

        settled_voids = await self._settle_voided(self._desk.void_all(reason), reason)
        return [outcome for outcome in settled_voids if isinstance(outcome, BaseException)]

    async def _settle_voided(self, branches: tuple[str, ...], reason: str) -> list[None | BaseException]:
        return [await settled(self._reserved.settle_voided(branch, reason)) for branch in branches]

    async def stop_unconfirmed_specialists(self) -> tuple[str, ...]:
        """Ask again, concurrently, for every specialist stop the runtime has not confirmed; returns the branches whose
        specialist still did not confirm it."""

        await asyncio.gather(*map(self._stop_specialist, sorted(self._specialists)))
        return tuple(sorted(self._specialists))

    async def _stop_specialist(self, branch: str) -> None | BaseException:
        stopped = await settled(self._clock.wait_for(self._lifecycle.stop(branch), SPECIALIST_CALL_SECONDS))
        if not isinstance(stopped, BaseException):
            self._specialists.discard(branch)
        return stopped

    async def _stop_admitted_forks(self, parent: str, stop: BranchStop) -> list[BaseException]:
        reason = f"parent {parent} stopped: {stop.reason}"
        failures: list[BaseException] = []
        for entry in self._roster.entries():
            if entry.from_branch == parent and entry.status is BranchStatus.ADMITTED:
                failures.extend(await self.stop_branch(entry.branch, BranchStop(stop.cause, reason, stop.at)))
        return failures

    async def push_quota(self) -> None:
        await self._toolbox.set_policy(self._roster.quota_policy())

    async def _launch(self, entry: BranchEntry, start_commit: str, launch: Launch) -> None:
        """Start the specialist unbriefed, as the runtime briefs each launch itself; a ``launch.focus`` is the one the
        session's briefs keep for the branch."""

        if launch.focus is not None:
            self._briefs.for_branch(entry.branch, launch.focus)
        if entry.origin.cut_by_scheduler:
            reason = f"{entry.origin.value} branch" + (f" of {entry.from_branch}" if entry.from_branch else "")
            await self._notebook.record_lineage(
                LineageEvent(LineageKind.OPEN, entry.branch, from_commit=start_commit, reason=reason)
            )
            await self._repo.create_branch(entry.branch, start_commit)
        await self.push_quota()
        self._specialists.add(entry.branch)
        task = BranchTask(repair=launch.repair, direction=launch.direction, agent=launch.agent)
        started = await settled(self._lifecycle.start(entry.branch, start_commit, task))
        if isinstance(started, BaseException):
            self._specialists.discard(entry.branch)
            raise started
        if entry.k != STARTED_K:
            await self._lifecycle.set_k(entry.branch, entry.k)

    async def _send_k(self, entry: BranchEntry) -> str:
        """Hand the specialist its K; the failure's text, or ``""``."""

        sent = await settled(
            self._clock.wait_for(self._lifecycle.set_k(entry.branch, entry.k), SPECIALIST_CALL_SECONDS)
        )
        if not isinstance(sent, BaseException):
            return ""
        logger.error("the specialist of %s did not take K %d", entry.branch, entry.k, exc_info=sent)
        return failure_text(sent)


__all__ = [
    "REPAIR_BRANCH_SECONDS",
    "REPAIR_TIME_LIMIT_REASON",
    "SPECIALIST_CALL_SECONDS",
    "SPECIALIST_START_SECONDS",
    "BatchStart",
    "BranchControl",
]
