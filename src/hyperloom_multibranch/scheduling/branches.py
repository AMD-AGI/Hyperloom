# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Opening, re-sizing and stopping branches: the roster, the repository, the specialists, the toolbox quota and the
notebook in step.

Opening a branch records its ``OPEN`` lineage, cuts it in the experiment repository at its start commit, starts its
specialist there and gives it its K. A repair the fork desk admitted is opened the same way at the commit its blocked
candidate was measured against; a fork is cut by the rule gate, which lands the fix on it first, so it is opened at its
head once that head has moved past the commit it was cut at, and stopped as not started when that has not happened
within :data:`SPECIALIST_START_SECONDS` of its admission. Stopping a branch withdraws everything of it: its specialist, its
queued and running tests and tool jobs, the processes the scheduler registered for it, its claims, and the forks and
repairs of it that are admitted but not started yet, as the fork desk admits nothing for a stopped branch; the notebook
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
from typing import TYPE_CHECKING

from hyperloom_multibranch.contracts.lifecycle import SpecialistLifecycle
from hyperloom_multibranch.contracts.notebook import Notebook
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import LineageKind
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.forks import DECISION_KINDS, AdmittedBranch
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.settle import failure_text, settled
from hyperloom_multibranch.scheduling.roster import (
    BranchEntry,
    BranchOrigin,
    BranchStatus,
    BranchStop,
    Roster,
    StopCause,
)

if TYPE_CHECKING:
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox

logger = logging.getLogger(__name__)

#: Clock seconds a specialist gets to stop or to take a new K before the call counts as failed.
SPECIALIST_CALL_SECONDS = 60.0
#: Clock seconds a branch gets to open, its specialist's start included, before the branch counts as not started.
SPECIALIST_START_SECONDS = 300.0
#: The K the lifecycle contract gives a specialist that has just started.
STARTED_K = 1


class BranchControl:
    """Every change to a branch goes through here; ``notebook`` is the scheduler's handle."""

    def __init__(
        self,
        roster: Roster,
        notebook: Notebook,
        lifecycle: SpecialistLifecycle,
        toolbox: RealToolbox,
        repo: ExperimentRepo,
        processes: ProcessRegistry,
        decisions: DecisionLog,
        clock: Clock,
    ) -> None:
        self._roster = roster
        self._notebook = notebook
        self._lifecycle = lifecycle
        self._toolbox = toolbox
        self._repo = repo
        self._processes = processes
        self._decisions = decisions
        self._clock = clock
        self._specialists: set[str] = set()

    async def open_branches(
        self, count: int, start_commit: str, k: int, origin: BranchOrigin, deadline: float
    ) -> tuple[BranchEntry, ...]:
        """Open ``count`` specialist branches of ``origin`` from ``start_commit``, record ``OPEN``, start them, each
        by ``deadline`` (clock seconds)."""

        opened = []
        for _ in range(count):
            entry = self._roster.open(self._roster.next_branch_id(), origin=origin, start_commit=start_commit, k=k)
            opened.append(await self._open(entry, start_commit, deadline))
        return tuple(opened)

    async def start_admitted(self, admitted: AdmittedBranch, deadline: float) -> bool:
        """Open a fork or repair the fork desk admitted and start its specialist by ``deadline`` (clock seconds).

        False while a fork waits for its fix; true once the branch started, failed to, or was stopped meanwhile.
        """

        entry = self._roster.entry(admitted.branch)
        if entry.stop is not None:
            return True
        start_commit = await settled(self._admitted_start(entry, admitted.parent_commit))
        if isinstance(start_commit, BaseException):
            await self._fail_start(entry, failure_text(start_commit))
        elif start_commit:
            await self._open(self._roster.started(entry.branch, start_commit), start_commit, deadline)
        elif self._clock.now() - entry.opened_at < SPECIALIST_START_SECONDS:
            return False
        else:
            await self._fail_start(entry, f"no fix landed on the fork past {admitted.parent_commit}")
        return True

    async def _admitted_start(self, entry: BranchEntry, parent_commit: str) -> str:
        """The commit an admitted branch starts at; ``""`` while the rule gate has not landed a fork's fix."""

        if entry.origin.cut_by_scheduler:
            return parent_commit
        try:
            head = await self._repo.head(entry.branch)
        except KeyError:
            return ""
        return "" if head == parent_commit else head

    async def _open(self, entry: BranchEntry, start_commit: str, deadline: float) -> BranchEntry:
        """Record ``OPEN``, give the toolbox the branch's K, cut the branch and start its specialist; a branch any of
        that fails for, or that is not done within :data:`SPECIALIST_START_SECONDS` or by ``deadline``, is stopped
        with the failure as the reason."""

        seconds = min(SPECIALIST_START_SECONDS, deadline - self._clock.now())
        started = await settled(self._clock.wait_for(self._launch(entry, start_commit), seconds))
        if isinstance(started, BaseException):
            logger.error("the specialist of %s did not start", entry.branch, exc_info=started)
            await self._fail_start(entry, failure_text(started))
            return entry
        detail = {"start_commit": start_commit, "k": entry.k, "from_branch": entry.from_branch}
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
        steps = [await settled(self.push_quota())]
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

    async def _launch(self, entry: BranchEntry, start_commit: str) -> None:
        reason = f"{entry.origin.value} branch" + (f" of {entry.from_branch}" if entry.from_branch else "")
        await self._notebook.record_lineage(
            LineageEvent(LineageKind.OPEN, entry.branch, from_commit=start_commit, reason=reason)
        )
        await self.push_quota()
        if entry.origin.cut_by_scheduler:
            await self._repo.create_branch(entry.branch, start_commit)
        self._specialists.add(entry.branch)
        started = await settled(self._lifecycle.start(entry.branch, start_commit))
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


__all__ = ["SPECIALIST_CALL_SECONDS", "SPECIALIST_START_SECONDS", "BranchControl"]
