# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Opening, re-sizing and stopping branches: the roster, the repository, the specialists, the toolbox quota and the
notebook in step.

Opening a branch records its ``OPEN`` lineage, cuts it in the experiment repository at its start commit and starts its
specialist there with its K; a fork or repair the fork desk admitted is opened the same way once it is drained, its
specialist briefed with the change it builds on. Stopping a branch withdraws everything of it: its specialist unless it
never started, its queued and running tests and tool jobs, the processes the scheduler registered for it, its claims,
and the forks and repairs of it that are admitted but not started yet, as the fork desk admits nothing for a stopped
branch; the notebook gets the ``STOP`` lineage of a branch it has the ``OPEN`` of. The commits it already made stay in
the notebook, so the integrator may still merge them. The scheduler records no ``STOP`` of the integration branch,
which only the end of the session stops. The roster entry is stopped first, so an admitted branch drained while its
parent is being stopped is never started. The toolbox gets a branch's K before its specialist starts, so the first
tests it submits already queue by it. A peer failure on one branch never ends the session: a branch whose specialist
does not start is withdrawn like a stopped one, and a failed step of a stop or of a new K is recorded in its decision.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from hyperloom_multibranch.contracts.lifecycle import SpecialistLaunch, SpecialistLifecycle
from hyperloom_multibranch.contracts.notebook import Notebook
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import LineageKind
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.forks import DECISION_KINDS, AdmittedBranch
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.settle import settled
from hyperloom_multibranch.scheduling.roster import (
    BranchEntry,
    BranchKind,
    BranchOrigin,
    BranchStatus,
    BranchStop,
    Roster,
    StopCause,
)

if TYPE_CHECKING:
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox

logger = logging.getLogger(__name__)


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

    async def open_branches(
        self, count: int, start_commit: str, k: int, origin: BranchOrigin
    ) -> tuple[BranchEntry, ...]:
        """Open ``count`` specialist branches of ``origin`` from ``start_commit``, record ``OPEN``, start them."""

        opened = []
        for _ in range(count):
            entry = self._roster.open(self._roster.next_branch_id(), origin=origin, start_commit=start_commit, k=k)
            opened.append(await self._open(entry, start_commit, None))
        return tuple(opened)

    async def start_admitted(self, admitted: AdmittedBranch) -> BranchEntry | None:
        """Open a fork or repair the fork desk admitted and start its specialist; ``None`` when it was stopped
        meanwhile."""

        if self._roster.entry(admitted.branch).stop is not None:
            return None
        entry = self._roster.started(admitted.branch, admitted.start_commit)
        return await self._open(entry, admitted.start_commit, admitted.brief)

    async def _open(self, entry: BranchEntry, start_commit: str, brief: str | None) -> BranchEntry:
        reason = f"{entry.origin.value} branch" + (f" of {entry.from_branch}" if entry.from_branch else "")
        await self._notebook.record_lineage(
            LineageEvent(LineageKind.OPEN, entry.branch, from_commit=start_commit, reason=reason)
        )
        await self.push_quota()
        await self._start(entry, start_commit, brief)
        return entry

    async def apply_k(self, orders: Mapping[str, int]) -> None:
        """Give each live branch its new K, clamped by the roster; stopped branches are left alone.

        A specialist that does not take its new K keeps running; its ``SET_K`` decision names the failure.
        """

        for branch, k in orders.items():
            if self._roster.entry(branch).stop is not None:
                continue
            entry = self._roster.set_k(branch, k)
            failure = await self._send_k(entry) if entry.kind is BranchKind.SPECIALIST and entry.start_commit else ""
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
        if entry.kind is BranchKind.SPECIALIST and entry.start_commit and stop.cause is not StopCause.START_FAILED:
            steps.append(await settled(self._lifecycle.stop(branch)))
        withdrawn = await settled(self._toolbox.cancel_branch(branch, stop.summary))
        steps.append(await settled(self._processes.kill_branch(branch)))
        released = await settled(self._notebook.release_branch(branch))
        if entry.kind is BranchKind.SPECIALIST and entry.start_commit:
            event = LineageEvent(LineageKind.STOP, branch, reason=stop.summary)
            steps.append(await settled(self._notebook.record_lineage(event)))
        raised = [outcome for outcome in (*steps, withdrawn, released) if isinstance(outcome, BaseException)]
        detail = {
            "withdrawn": 0 if isinstance(withdrawn, BaseException) else withdrawn,
            "released_claims": 0 if isinstance(released, BaseException) else len(released),
            "failures": [_failure_text(error) for error in raised],
        }
        self._decisions.record(Decision(DecisionKind.STOP, stop.summary, branch, detail))
        return (*raised, *forks)

    async def _stop_admitted_forks(self, parent: str, stop: BranchStop) -> list[BaseException]:
        reason = f"parent {parent} stopped: {stop.reason}"
        failures: list[BaseException] = []
        for entry in self._roster.entries():
            if entry.from_branch == parent and entry.status is BranchStatus.ADMITTED:
                failures.extend(await self.stop_branch(entry.branch, BranchStop(stop.cause, reason, stop.at)))
        return failures

    async def push_quota(self) -> None:
        await self._toolbox.set_policy(self._roster.quota_policy())

    async def _start(self, entry: BranchEntry, start_commit: str, brief: str | None) -> None:
        """Cut the branch and start its specialist; one that does not start is stopped, with the failure as the
        reason."""

        started = await settled(self._launch(SpecialistLaunch(entry.branch, start_commit, entry.k, brief)))
        if isinstance(started, BaseException):
            logger.error("the specialist of %s did not start", entry.branch, exc_info=started)
            stop = BranchStop(StopCause.START_FAILED, _failure_text(started), self._clock.now())
            for failure in await self.stop_branch(entry.branch, stop):
                logger.error("%s did not stop cleanly after its failed start", entry.branch, exc_info=failure)
            return
        detail = {"start_commit": start_commit, "k": entry.k, "from_branch": entry.from_branch}
        kind = DECISION_KINDS.get(entry.origin, DecisionKind.OPEN)
        self._decisions.record(Decision(kind, f"{entry.origin.value} branch started", entry.branch, detail))

    async def _launch(self, launch: SpecialistLaunch) -> None:
        await self._repo.create_branch(launch.branch, launch.start_commit)
        await self._lifecycle.start(launch)

    async def _send_k(self, entry: BranchEntry) -> str:
        """Hand the specialist its K; the failure's text, or ``""``."""

        sent = await settled(self._lifecycle.set_k(entry.branch, entry.k))
        if not isinstance(sent, BaseException):
            return ""
        logger.error("the specialist of %s did not take K %d", entry.branch, entry.k, exc_info=sent)
        return _failure_text(sent)


def _failure_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


__all__ = ["BranchControl"]
