# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The specialist lifecycle: on each branch, the specialist's session and the branch's judge.

A session that crashes is restarted on its branch, at most ``MAX_RESTARTS`` times within ``RESTART_WINDOW_S``; past
that the branch keeps its judge, which settles what was already measured, and the scheduler stops it as it stops any
branch that makes no progress. A judge that fails is a defect of this code, not of the specialist: it is not
restarted, and the branch's session is stopped so no candidate is measured that nothing would judge. Both leave a
notice in the branch log.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.toolbox import Toolbox
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.rules import Rule
from hyperloom_multibranch.specialist.judging import NOTICE, BranchJudge, JudgeNotebook

logger = logging.getLogger(__name__)

MAX_RESTARTS = 3
RESTART_WINDOW_S = 600.0


class Session(Protocol):
    """One run of a branch's specialist: ``run`` returns once its work is done and raises when it crashed."""

    async def run(self) -> None: ...

    async def set_k(self, k: int) -> None: ...

    async def stop(self) -> None:
        """Stop the run, more than once if need be; returns once none of its processes is left."""


#: A new run of the specialist on ``branch``, started at ``start_commit`` and allowed ``k`` tests running at once.
SessionFactory = Callable[[str, str, int], Session]


@dataclass
class _Branch:
    name: str
    start_commit: str
    notebook: JudgeNotebook
    k: int = 1
    session: Session | None = None
    tasks: list[asyncio.Task[None]] = field(default_factory=list)


class SpecialistRuntime:
    """The session's :class:`~hyperloom_multibranch.contracts.lifecycle.SpecialistLifecycle`."""

    def __init__(
        self,
        notebook_for: Callable[[str], JudgeNotebook],
        gate: RuleGate,
        toolbox: Toolbox,
        rule: Rule,
        sessions: SessionFactory,
        *,
        poll_seconds: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._notebook_for = notebook_for
        self._gate = gate
        self._toolbox = toolbox
        self._rule = rule
        self._sessions = sessions
        self._poll_seconds = poll_seconds
        self._clock = clock
        self._branches: dict[str, _Branch] = {}

    async def start(self, branch: str, start_commit: str) -> None:
        if branch in self._branches:
            raise ValueError(f"branch {branch!r} already has a specialist")
        state = _Branch(branch, start_commit, self._notebook_for(branch))
        self._branches[branch] = state
        judge = BranchJudge(branch, state.notebook, self._gate, self._toolbox, self._rule)
        state.tasks = [
            asyncio.create_task(self._supervise(state), name=f"specialist of {branch}"),
            asyncio.create_task(self._judge(state, judge), name=f"judge of {branch}"),
        ]
        for task in state.tasks:
            task.add_done_callback(_log_failure)

    async def set_k(self, branch: str, k: int) -> None:
        if k < 1:
            raise ValueError("stop the branch instead of giving it no room")
        state = self._branches[branch]
        state.k = k
        if state.session is not None:
            await state.session.set_k(k)

    async def stop(self, branch: str) -> None:
        state = self._branches.pop(branch)
        supervisor, judge = state.tasks
        supervisor.cancel()
        await asyncio.gather(supervisor, return_exceptions=True)
        if state.session is not None:
            await state.session.stop()
        judge.cancel()
        await asyncio.gather(judge, return_exceptions=True)

    async def _supervise(self, state: _Branch) -> None:
        crashes: deque[float] = deque()
        while True:
            state.session = self._sessions(state.name, state.start_commit, state.k)
            try:
                await state.session.run()
                return
            except Exception as error:
                logger.exception("the specialist of %s crashed", state.name)
                await state.session.stop()
                now = self._clock()
                crashes.append(now)
                while now - crashes[0] > RESTART_WINDOW_S:
                    crashes.popleft()
                if len(crashes) > MAX_RESTARTS:
                    state.session = None
                    window = f"{len(crashes)} times in {RESTART_WINDOW_S / 60:g} minutes"
                    await _notice(state, f"the specialist crashed {window} ({error!r}); it is not restarted")
                    return
                await _notice(state, f"the specialist crashed ({error!r}) and is restarted on this branch")

    async def _judge(self, state: _Branch, judge: BranchJudge) -> None:
        try:
            await judge.run(self._poll_seconds)
        except Exception as error:
            logger.exception("judging %s failed", state.name)
            supervisor = state.tasks[0]
            supervisor.cancel()
            await asyncio.gather(supervisor, return_exceptions=True)
            if state.session is not None:
                await state.session.stop()
            await _notice(state, f"judging this branch failed ({error!r}); its specialist is stopped")


async def _notice(state: _Branch, content: str) -> None:
    await state.notebook.append_branch_log(BranchLogEntry(state.name, BranchLogType.RESULT, f"{NOTICE} {content}"))


def _log_failure(task: asyncio.Task[None]) -> None:
    """What ends a branch's task unexpectedly, a notice that could not be written included, is never lost."""
    if not task.cancelled() and task.exception() is not None:
        logger.error("the %s ended", task.get_name(), exc_info=task.exception())


__all__ = ["MAX_RESTARTS", "RESTART_WINDOW_S", "Session", "SessionFactory", "SpecialistRuntime"]
