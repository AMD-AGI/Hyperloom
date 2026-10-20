# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The specialist lifecycle: on each branch, the specialist's session and the branch's judge.

A session that crashes is restarted on its branch, at most ``MAX_RESTARTS`` times within ``RESTART_WINDOW_S``, and a
lost one at most ``MAX_LOSS_RESTARTS`` times in a row with no session working in between: a loss takes the heartbeat
timeout, so losses alone never fill the window. Past either the branch keeps its judge, which settles what was already measured,
and the scheduler stops it as it stops any branch that makes no progress; ``status`` tells it. A judge that fails is a defect of this code, not of the specialist: it is not
restarted, and the branch's session is stopped so no candidate is measured that nothing would judge. Both leave a
notice in the branch log. While the specialist writes a change, the runtime renews the claims it holds, so a claim does
not lapse while a model thinks; once a test carries a claim, the toolbox renews it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.lifecycle import BranchTask, SpecialistStatus
from hyperloom_multibranch.contracts.notebook.interfaces import ClaimService, NotebookError
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus
from hyperloom_multibranch.specialist.judging import NOTICE, BranchJudge, JudgeNotebook, Judging

logger = logging.getLogger(__name__)

MAX_RESTARTS = 3
RESTART_WINDOW_S = 600.0
MAX_LOSS_RESTARTS = 2


class RuntimeNotebook(JudgeNotebook, ClaimService, Protocol):
    """A branch's runtime handle: what its judge reads and writes, and the claims its specialist holds."""


class SessionCrashed(RuntimeError):
    """A run of a branch's specialist ended the way no working session ends.

    ``lost`` says its session lost its heartbeat; ``worked`` says an earlier session of the same run ended well.
    """

    def __init__(self, reason: str, *, lost: bool = False, worked: bool = False) -> None:
        super().__init__(reason)
        self.lost, self.worked = lost, worked


class Session(Protocol):
    """One run of a branch's specialist: ``run`` returns once its work is done and raises when it crashed, a
    :class:`SessionCrashed` when it can say how."""

    async def run(self) -> None: ...

    async def set_k(self, k: int) -> None: ...

    async def stop(self) -> None:
        """Stop the run, more than once if need be; returns once none of its processes is left."""


@dataclass(frozen=True)
class SessionStart:
    """What a run of a branch's specialist starts from: the commit the branch was started at, how many tests it may
    have running at once, and the task the branch was started on."""

    branch: str
    start_commit: str
    k: int
    task: BranchTask = BranchTask()


SessionFactory = Callable[[SessionStart], Session]


@dataclass(frozen=True)
class RuntimePacing:
    """How often the runtime looks: a judge polls its inbox every ``poll_seconds``, a claim the specialist holds is
    renewed for ``claim_ttl_seconds`` a third of that apart, ``clock`` times the restart window and ``sleep`` waits
    out both intervals, on the same time base as ``clock``."""

    poll_seconds: float = 5.0
    claim_ttl_seconds: float = 600.0
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


@dataclass
class _Branch:
    name: str
    start_commit: str
    task: BranchTask
    notebook: RuntimeNotebook
    k: int = 1
    session: Session | None = None
    tasks: list[asyncio.Task[None]] = field(default_factory=list)
    crashes: int = 0
    lost: int = 0
    gave_up: bool = False
    judge_failed: bool = False


class SpecialistRuntime:
    """The session's :class:`~hyperloom_multibranch.contracts.lifecycle.SpecialistLifecycle`."""

    def __init__(
        self,
        notebook_for: Callable[[str], RuntimeNotebook],
        judging: Judging,
        sessions: SessionFactory,
        *,
        pacing: RuntimePacing = RuntimePacing(),
    ) -> None:
        self._notebook_for = notebook_for
        self._judging = judging
        self._sessions = sessions
        self._pacing = pacing
        self._branches: dict[str, _Branch] = {}

    async def start(self, branch: str, start_commit: str, task: BranchTask = BranchTask()) -> None:
        if branch in self._branches:
            raise ValueError(f"branch {branch!r} already has a specialist")
        state = _Branch(branch, start_commit, task, self._notebook_for(branch))
        self._branches[branch] = state
        judge = BranchJudge(branch, state.notebook, self._judging)
        state.tasks = [
            asyncio.create_task(self._supervise(state), name=f"specialist of {branch}"),
            asyncio.create_task(self._judge(state, judge), name=f"judge of {branch}"),
            asyncio.create_task(self._renew(state), name=f"claim renewal of {branch}"),
        ]
        for task in state.tasks:
            task.add_done_callback(_log_failure)

    async def set_k(self, branch: str, k: int) -> None:
        if k < 1:
            raise ValueError("stop the branch instead of giving it no room")
        state = self._branches[branch]
        if k == state.k:
            return
        state.k = k
        if state.session is not None:
            await state.session.set_k(k)
        await _notice(state, f"you may now have {k} tests running at once")

    async def status(self, branch: str) -> SpecialistStatus | None:
        state = self._branches.get(branch)
        if state is None:
            return None
        running = not state.tasks[0].done()
        return SpecialistStatus(running, state.crashes, state.lost, state.gave_up, state.judge_failed)

    async def stop(self, branch: str) -> None:
        state = self._branches.pop(branch)
        supervisor, *others = state.tasks
        supervisor.cancel()
        await asyncio.gather(supervisor, return_exceptions=True)
        if state.session is not None:
            await state.session.stop()
        for task in others:
            task.cancel()
        await asyncio.gather(*others, return_exceptions=True)

    async def _supervise(self, state: _Branch) -> None:
        crashes: deque[float] = deque()
        losses = 0
        while True:
            start = SessionStart(state.name, state.start_commit, state.k, state.task)
            state.session = self._sessions(start)
            try:
                await state.session.run()
                return
            except Exception as error:
                logger.exception("the specialist of %s crashed", state.name)
                await state.session.stop()
                lost = isinstance(error, SessionCrashed) and error.lost
                if isinstance(error, SessionCrashed) and error.worked:
                    losses = 0
                losses += lost
                state.crashes += 1
                state.lost += lost
                now = self._pacing.clock()
                crashes.append(now)
                while now - crashes[0] > RESTART_WINDOW_S:
                    crashes.popleft()
                if len(crashes) > MAX_RESTARTS:
                    why = f"crashed {len(crashes)} times in {RESTART_WINDOW_S / 60:g} minutes"
                elif losses > MAX_LOSS_RESTARTS:
                    why = f"lost {losses} sessions in a row, none working in between"
                else:
                    await _notice(state, f"the specialist crashed ({error!r}) and is restarted on this branch")
                    continue
                state.session, state.gave_up = None, True
                await _notice(state, f"the specialist {why} ({error!r}); it is not restarted")
                return

    async def _judge(self, state: _Branch, judge: BranchJudge) -> None:
        try:
            await judge.run(self._pacing.poll_seconds, self._pacing.sleep)
        except Exception as error:
            logger.exception("judging %s failed", state.name)
            supervisor, _, renewal = state.tasks
            supervisor.cancel()
            renewal.cancel()
            await asyncio.gather(supervisor, renewal, return_exceptions=True)
            if state.session is not None:
                await state.session.stop()
            state.judge_failed = True
            await _notice(state, f"judging this branch failed ({error!r}); its specialist is stopped")

    async def _renew(self, state: _Branch) -> None:
        ttl = self._pacing.claim_ttl_seconds
        while True:
            await self._pacing.sleep(ttl / 3)
            snapshot = await state.notebook.snapshot()
            for claim in snapshot.claims:
                if claim.holder != state.name or claim.status is not ClaimStatus.HELD or claim.request_id is not None:
                    continue
                try:
                    await state.notebook.renew(claim.key, ttl_seconds=ttl)
                except NotebookError:
                    logger.warning("the claim %s of %s ended before its renewal", claim.key, state.name, exc_info=True)


async def _notice(state: _Branch, content: str) -> None:
    await state.notebook.append_branch_log(BranchLogEntry(state.name, BranchLogType.RESULT, f"{NOTICE} {content}"))


def _log_failure(task: asyncio.Task[None]) -> None:
    """What ends a branch's task unexpectedly, a notice that could not be written included, is never lost."""
    if not task.cancelled() and task.exception() is not None:
        logger.error("the %s ended", task.get_name(), exc_info=task.exception())


__all__ = [
    "MAX_LOSS_RESTARTS",
    "MAX_RESTARTS",
    "RESTART_WINDOW_S",
    "RuntimePacing",
    "Session",
    "SessionCrashed",
    "SessionFactory",
    "SessionStart",
    "SpecialistRuntime",
]
