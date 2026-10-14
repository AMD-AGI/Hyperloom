# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The fork admission desk the rule gate awaits.

A fork is admitted when the roster may open a branch; what a fork the roster cannot take gets is
:func:`mb_fake_refused_fork`. An accepted fork is put on the roster at once, admitted but not started, so it takes a slot of the branch cap while
the gate cuts the branch and commits the fix. The gate then reports the fix commit, and the lifecycle hands what
:meth:`ForkDesk.drain` returns to the branch control, which starts the fork's specialist there. The desk needs no
peer, so it is built before them and handed to the rule gate. A fork of a branch the roster does not hold, or of
one already stopped, is refused whatever the policy says: stopping a branch withdraws everything of it.

An admitted fork can be stopped before the gate records its ``SPLIT``; the notebook's lineage has no such branch yet
and drops that ``STOP``. The desk therefore records the ``STOP`` again once the ``SPLIT`` is in, so a stopped fork
never reads as running.
"""

from __future__ import annotations

from dataclasses import dataclass

from hyperloom_multibranch.contracts.notebook import Notebook
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import LineageKind
from hyperloom_multibranch.contracts.scheduling import AdmissionStatus, ForkAdmission, ForkRequest
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.mb_fakes import mb_fake_refused_fork
from hyperloom_multibranch.scheduling.roster import BranchOrigin, Roster


@dataclass(frozen=True)
class CommittedFork:
    """An admitted fork whose fix commit the rule gate has made."""

    branch: str
    fix_commit: str


def _parent_refusal(roster: Roster, from_branch: str) -> str:
    parent = next((entry for entry in roster.entries() if entry.branch == from_branch), None)
    if parent is None:
        return f"unknown parent branch {from_branch!r}"
    return "parent branch is stopped" if parent.stop is not None else ""


class ForkDesk:
    """Implements :class:`~hyperloom_multibranch.contracts.scheduling.ForkAdmissionService` on the roster.

    ``notebook`` is the scheduler's handle.
    """

    def __init__(self, roster: Roster, notebook: Notebook, decisions: DecisionLog) -> None:
        self._roster = roster
        self._notebook = notebook
        self._decisions = decisions
        self._committed: list[CommittedFork] = []

    async def admit(self, request: ForkRequest) -> ForkAdmission:
        parent_refusal = _parent_refusal(self._roster, request.from_branch)
        roster_refusal = self._roster.open_refusal()
        if parent_refusal:
            answer = ForkAdmission(AdmissionStatus.REJECTED, parent_refusal)
        elif roster_refusal:
            answer = mb_fake_refused_fork(roster_refusal)
        else:
            parent = self._roster.entry(request.from_branch)
            entry = self._roster.open(
                self._roster.next_branch_id(),
                origin=BranchOrigin.FORK,
                start_commit="",
                k=parent.k,
                from_branch=request.from_branch,
            )
            answer = ForkAdmission(AdmissionStatus.ACCEPTED, request.reason, entry.branch)
        self._decisions.record(
            Decision(
                DecisionKind.FORK,
                answer.reason,
                answer.branch,
                {
                    "status": answer.status.value,
                    "from_branch": request.from_branch,
                    "parent_commit": request.parent_commit,
                },
            )
        )
        return answer

    async def fork_committed(self, branch: str, fix_commit: str) -> None:
        """Queue the fork to start, or, when it was stopped while admitted, record its ``STOP`` after the ``SPLIT``."""

        stop = self._roster.entry(branch).stop
        if stop is None:
            self._committed.append(CommittedFork(branch, fix_commit))
            return
        await self._notebook.record_lineage(LineageEvent(LineageKind.STOP, branch, reason=stop.summary))

    def drain(self) -> tuple[CommittedFork, ...]:
        """The forks committed since the last call, oldest first; never waits."""

        drained, self._committed = tuple(self._committed), []
        return drained


__all__ = ["CommittedFork", "ForkDesk"]
