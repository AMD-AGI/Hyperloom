# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Opening, re-sizing and stopping branches: the roster, the specialists, the toolbox quota and the notebook in step.

Stopping a branch withdraws everything of it: its specialist, its queued and running tests, the processes the
scheduler registered for it, its claims, and the forks of it that are admitted but not started yet, as the fork desk
admits no fork of a stopped branch; the notebook gets the ``STOP`` lineage. The commits it already made stay
in the notebook, so the integrator may still merge them. The notebook records no ``STOP`` of the integration
branch, which only the end of the session stops. The roster entry is stopped first, so a fork that commits
its fix while the branch is being stopped is never started; when that fork's ``SPLIT`` comes after its ``STOP``, the
fork desk records the ``STOP`` again. A branch gets its quota before its specialist starts,
because the toolbox treats a branch without one as stopped. A peer failure on one branch never ends the session: a
specialist that does not start is stopped, and a failed step of a stop or of a new K is recorded in its decision.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from hyperloom_multibranch.contracts.notebook import Notebook
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import LineageKind
from hyperloom_multibranch.contracts.scheduling import SpecialistLifecycle, Toolbox
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
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

logger = logging.getLogger(__name__)


class BranchControl:
    """Every change to a branch goes through here; ``notebook`` is the scheduler's handle."""

    def __init__(
        self,
        roster: Roster,
        notebook: Notebook,
        lifecycle: SpecialistLifecycle,
        toolbox: Toolbox,
        processes: ProcessRegistry,
        decisions: DecisionLog,
        clock: Clock,
    ) -> None:
        self._roster = roster
        self._notebook = notebook
        self._lifecycle = lifecycle
        self._toolbox = toolbox
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
            await self._notebook.record_lineage(
                LineageEvent(LineageKind.OPEN, entry.branch, from_commit=start_commit, reason=f"{origin.value} branch")
            )
            await self.push_quota()
            await self._start(entry, start_commit, DecisionKind.OPEN)
            opened.append(entry)
        return tuple(opened)

    async def start_fork(self, branch: str, fix_commit: str) -> BranchEntry | None:
        """Start the specialist of an admitted fork at its fix commit; ``None`` when the fork was stopped meanwhile.

        No ``OPEN`` is recorded: the rule gate's ``SPLIT`` opened the branch.
        """

        if self._roster.entry(branch).stop is not None:
            return None
        entry = self._roster.started(branch, fix_commit)
        await self.push_quota()
        await self._start(entry, fix_commit, DecisionKind.FORK)
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
        withdrawn and, once the branch is in the lineage, its ``STOP`` recorded (an admitted fork's ``STOP`` waits for
        its ``SPLIT``, see :class:`ForkDesk`); the ``STOP`` decision names each failure, and the caller decides what
        one means to it.
        """

        entry = self._roster.entry(branch)
        if entry.stop is not None:
            return ()
        self._roster.stop(branch, stop)
        forks = await self._stop_admitted_forks(branch, stop)
        steps = [await settled(self.push_quota())]
        if entry.kind is BranchKind.SPECIALIST and entry.start_commit:
            steps.append(await settled(self._lifecycle.stop(branch)))
        withdrawn = await settled(self._toolbox.withdraw(branch))
        steps.append(await settled(self._processes.kill_branch(branch)))
        released = await settled(self._notebook.release_branch(branch))
        if entry.kind is BranchKind.SPECIALIST and entry.start_commit:
            event = LineageEvent(LineageKind.STOP, branch, reason=stop.summary)
            steps.append(await settled(self._notebook.record_lineage(event)))
        raised = [outcome for outcome in (*steps, withdrawn, released) if isinstance(outcome, BaseException)]
        detail = {
            "withdrawn": [] if isinstance(withdrawn, BaseException) else list(withdrawn),
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
        await self._toolbox.apply_quota(self._roster.quota_policy())

    async def _start(self, entry: BranchEntry, start_commit: str, kind: DecisionKind) -> None:
        """Start the branch's specialist; one that does not start is stopped, with the failure as the reason."""

        started = await settled(self._lifecycle.start(entry.branch, start_commit))
        if isinstance(started, BaseException):
            logger.error("the specialist of %s did not start", entry.branch, exc_info=started)
            stop = BranchStop(StopCause.START_FAILED, _failure_text(started), self._clock.now())
            for failure in await self.stop_branch(entry.branch, stop):
                logger.error("%s did not stop cleanly after its failed start", entry.branch, exc_info=failure)
            return
        failure = await self._send_k(entry)
        detail = {"start_commit": start_commit, "k": entry.k, "from_branch": entry.from_branch, "failure": failure}
        self._decisions.record(Decision(kind, f"{entry.origin.value} branch started", entry.branch, detail))

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
