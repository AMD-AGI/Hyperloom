# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A mapping branch: the run that maps the directions a session's Arbor agents split, then hands them out as branches.

With two or more Arbor agents and no map given, or one under the session's review, EXPLORE opens one agent branch on
the first Arbor source whose task names the Arbor agents to map for (``BranchTask.map_for``) instead of opening them.
Its run is that Arbor's dossier pass (:mod:`dossier_pass`) on the branch's own lease and worktree, charged to the branch,
with the start and the end the scheduler gives (:class:`MapCalls`): ``start`` the pass's profile and deadline, ``None``
to run none; ``split`` what to do with the pass's outcome, which reviews the map when the session does, splits it and
asks the fork desk for one share branch per Arbor agent. The run then finishes, and the branch's slot goes to a share.
The other branches do not wait for any of this.

A run that crashes is restarted as any branch's and runs the pass again; a stop kills the pass and gives its cards
back.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from hyperloom_multibranch.specialist.dossier_pass import DossierOutcome, DossierStart
from hyperloom_multibranch.specialist.runtime import SessionFinished, SessionStart


@dataclass(frozen=True)
class MapCalls:
    """The scheduler's side of a mapping branch: ``start(source, start_commit, deadline)``, the dossier pass of the
    Arbor ``source`` to run by ``deadline``, ``None`` to run none; ``split(branch, start_commit, map_for, outcome)``,
    which hands the map out to the Arbor agents ``map_for`` (``outcome`` ``None`` when no pass ran) and says how."""

    start: Callable[[str, str, float], Awaitable[DossierStart | None]]
    split: Callable[[str, str, tuple[str, ...], DossierOutcome | None], Awaitable[str]]


class MapSession:
    """The run of the mapping branch ``start`` names, its pass run by ``run_pass``."""

    def __init__(
        self, start: SessionStart, calls: MapCalls, run_pass: Callable[[DossierStart], Awaitable[DossierOutcome]]
    ) -> None:
        self._start = start
        self._calls = calls
        self._run_pass = run_pass
        self._pass: asyncio.Future[DossierOutcome] | None = None

    async def preflight(self) -> None:
        return None

    async def run(self) -> None:
        start, task = self._start, self._start.task
        given = await self._calls.start(task.map_for[0], start.start_commit, task.deadline)
        outcome = None
        if given is not None:
            self._pass = asyncio.ensure_future(self._run_pass(given))
            outcome = await self._pass
        said = await self._calls.split(start.branch, start.start_commit, task.map_for, outcome)
        raise SessionFinished(f"the map is made and handed out: {said}")

    async def set_k(self, k: int) -> None:
        return None

    async def stop(self) -> None:
        if self._pass is not None and not self._pass.done():
            self._pass.cancel()
            await asyncio.gather(self._pass, return_exceptions=True)

    async def release(self) -> None:
        return None


__all__ = ["MapCalls", "MapSession"]
