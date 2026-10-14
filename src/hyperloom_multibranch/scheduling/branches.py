# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Opening, re-sizing and stopping branches: the roster, the specialists, the toolbox quota and the notebook in step.

Stopping a branch withdraws everything of it: its specialist, its queued and running tests, the processes the
scheduler registered for it, and its claims; the notebook gets the ``STOP`` lineage. The commits it already made stay
in the notebook, so the integrator may still merge them. The notebook records no ``STOP`` of the integration
branch, which only the end of the session stops. The roster entry is stopped first, so a fork that commits
its fix while the branch is being stopped is never started. A branch gets its quota before its specialist starts,
because the toolbox treats a branch without one as stopped.
"""

from __future__ import annotations

from collections.abc import Mapping

from hyperloom_multibranch.contracts.notebook import Notebook
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import LineageKind
from hyperloom_multibranch.contracts.scheduling import SpecialistLifecycle, Toolbox
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.settle import settled
from hyperloom_multibranch.scheduling.roster import (
    BranchEntry,
    BranchKind,
    BranchOrigin,
    BranchStop,
    Roster,
)


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
    ) -> None:
        self._roster = roster
        self._notebook = notebook
        self._lifecycle = lifecycle
        self._toolbox = toolbox
        self._processes = processes
        self._decisions = decisions

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
        """Give each live branch its new K, clamped by the roster; stopped branches are left alone."""

        for branch, k in orders.items():
            if self._roster.entry(branch).stop is not None:
                continue
            entry = self._roster.set_k(branch, k)
            if entry.kind is BranchKind.SPECIALIST and entry.start_commit:
                await self._lifecycle.set_k(branch, entry.k)
            self._decisions.record(Decision(DecisionKind.SET_K, f"K {entry.k}", branch, {"asked": k}))
        await self.push_quota()

    async def stop_branch(self, branch: str, stop: BranchStop) -> None:
        """Withdraw everything of ``branch``; stopping it again does nothing.

        Every step runs even when an earlier one fails, so the tests, processes and claims of the branch are always
        withdrawn and the ``STOP`` recorded; the first failure is raised afterwards.
        """

        entry = self._roster.entry(branch)
        if entry.stop is not None:
            return
        self._roster.stop(branch, stop)
        failures = [await settled(self.push_quota())]
        if entry.kind is BranchKind.SPECIALIST and entry.start_commit:
            failures.append(await settled(self._lifecycle.stop(branch)))
        withdrawn = await settled(self._toolbox.withdraw(branch))
        failures.append(await settled(self._processes.kill_branch(branch)))
        released = await settled(self._notebook.release_branch(branch))
        reason = f"{stop.cause.value}: {stop.reason}"
        if entry.kind is BranchKind.SPECIALIST:
            failures.append(
                await settled(self._notebook.record_lineage(LineageEvent(LineageKind.STOP, branch, reason=reason)))
            )
        raised = [outcome for outcome in (*failures, withdrawn, released) if isinstance(outcome, BaseException)]
        detail = {
            "withdrawn": [] if isinstance(withdrawn, BaseException) else list(withdrawn),
            "released_claims": 0 if isinstance(released, BaseException) else len(released),
            "failures": [f"{type(error).__name__}: {error}" for error in raised],
        }
        self._decisions.record(Decision(DecisionKind.STOP, reason, branch, detail))
        if raised:
            raise raised[0]

    async def push_quota(self) -> None:
        await self._toolbox.apply_quota(self._roster.quota_policy())

    async def _start(self, entry: BranchEntry, start_commit: str, kind: DecisionKind) -> None:
        await self._lifecycle.start(entry.branch, start_commit)
        await self._lifecycle.set_k(entry.branch, entry.k)
        detail = {"start_commit": start_commit, "k": entry.k, "from_branch": entry.from_branch}
        self._decisions.record(Decision(kind, f"{entry.origin.value} branch started", entry.branch, detail))


__all__ = ["BranchControl"]
