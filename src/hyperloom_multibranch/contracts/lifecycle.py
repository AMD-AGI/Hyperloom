# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How the scheduler starts, stops and resizes the long-running specialist on a branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hyperloom_multibranch.contracts.fork import RepairRequest


@dataclass(frozen=True)
class SpecialistStatus:
    """What the runtime knows of a branch's specialist.

    ``running`` says a session of it runs or will be restarted. ``crashes`` counts its sessions that crashed, the lost
    ones included, not those that ended finding nothing to test; ``lost`` counts those that lost their heartbeat.
    ``gave_up`` says the runtime stopped restarting it after repeated crashes, losses or sessions in a row that found
    nothing to test, ``judge_failed`` that it was stopped because judging the branch failed; the branch keeps its slot
    either way. ``unready`` says why its specialist could not be started (its session not built, or its preflight
    failed), empty when it was. ``finished`` says why a third-party agent's run ended of itself (its exit, such as its
    deadline), the other way such a specialist is not restarted; empty when it did not.
    """

    running: bool
    crashes: int
    lost: int
    gave_up: bool
    judge_failed: bool
    unready: str = ""
    finished: str = ""


@dataclass(frozen=True)
class BranchTask:
    """What a branch's specialist is started on; see :class:`SpecialistLifecycle`."""

    brief: str = ""
    repair: RepairRequest | None = None
    direction: str = ""
    agent: str = ""
    deadline: float = 0.0
    #: The file of an agent branch's share of the session's directions, ``""`` for none.
    plan: str = ""


class SpecialistLifecycle(Protocol):
    """Long-running specialists, one per branch; the runtime restarts a crashed one on the same branch by itself.

    A branch is started on its :class:`BranchTask`. Its ``brief`` is the startup brief the scheduler hands a new
    specialist (the warm-start leads for the first batch, the cross-session brief for a fill); the runtime puts it into
    the specialist's first prompt and every restart prompt. Its ``repair`` is the request an admitted repair branch was
    opened for, started at its intent's parent commit; every prompt of the branch's specialist shows it as the
    branch's task, whatever brief a launch is rebuilt with. Its ``direction`` is the text a user wrote for the branch;
    every launch and restart prompt shows it as the branch's task, the way ``repair`` is shown. Its ``agent`` names the
    third-party agent an autonomous branch runs, a git URL (``URL@ref`` for a ref) or a local path, and is empty for a
    specialist branch. Its ``deadline`` is the clock time (``time.time``) EXPLORE ends at, 0 when the branch has none:
    an agent that signals no keep has its workspace collected ahead of it. A started branch may
    have one test running until ``set_k`` says otherwise, and shrinking a branch to nothing is ``stop``, not
    ``set_k(branch, 0)``. Every process the runtime starts goes through the session's process registry, so the session
    stop kills it even when ``stop`` hangs.
    """

    async def start(self, branch: str, start_commit: str, task: BranchTask = BranchTask()) -> None: ...

    async def stop(self, branch: str) -> None:
        """Stop the branch's specialist; returns once no process of it is left."""

    async def set_k(self, branch: str, k: int) -> None:
        """How many tests the branch's specialist may have running at once."""

    async def status(self, branch: str) -> SpecialistStatus | None:
        """The state of the branch's specialist; ``None`` for a branch the runtime does not run (never started or
        stopped)."""


__all__ = ["BranchTask", "SpecialistLifecycle", "SpecialistStatus"]
