# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The specialist lifecycle: on each branch, the specialist's session and the branch's judge.

A branch started with no direction, no repair and no third-party agent takes one of the session's
:data:`~hyperloom_multibranch.specialist.directions.DIRECTIONS` that no branch has held yet, so the branches split the
work instead of all sweeping the same settings, and a stopped branch's direction is not handed out again while another
was never tried. Once each was held, it takes the one the fewest branches held over the session, then the one the
fewest running branches hold. A direction given to a branch counts when it is one of these. A direction whose
specialist gave up finding nothing to test is not dealt again on that start commit while another is left there, nor on
the head its last :mod:`~hyperloom_multibranch.specialist.exhausted` record names, and neither is one any branch's
record names done on the commit a branch starts at: a later head may open it again, the same one would not.

A branch's first session is made ready once (``preflight``); one that cannot be gives the specialist up with every
reason in a notice, and a restart does not repeat it. A session that crashes is restarted on its branch, at most
``MAX_RESTARTS`` times within ``RESTART_WINDOW_S``, and a lost one at most ``MAX_LOSS_RESTARTS`` times in a row with no
session working in between: a loss takes the heartbeat timeout, so losses alone never fill the window. A loss while
the session waited on its branch's long dev-sandbox process counts toward neither: a model waiting on it is silent,
not hung, and giving it up would end that process. Whether the
restart resumes the crashed conversation is the session's to decide; the notice only says which. A session that
ends idle, its model finding nothing to test, is no crash: the branch gets one more session, which goes on with the
idle conversation or starts fresh as the session decides, and ``MAX_IDLE_SESSIONS`` idle sessions in a row end it, a
session that submitted a test starting the count again. Then
the specialist gives up and the scheduler stops the branch, freeing its slot for a fresh branch with its own direction.
A session that finished, a third-party agent that ended its run and whose last keep settled, ends its specialist the
same way. Past any of these the branch keeps its judge, which settles what was already measured, and the scheduler stops
it once nothing of it waits for a confirm; ``status`` tells it. A judge that fails is a defect of this code, not of the
specialist: it is not restarted, and the branch's session is stopped so no candidate is measured that nothing would
judge. Both leave a notice in the branch log. A branch stopped with ``finish_tests`` keeps its judge, without its
specialist, until no test of the branch is left in flight, so a test that wins after the stop still commits and is
offered to the integrator; a plain stop ends that judge too. While the specialist writes a change, the runtime
renews the claims it holds, so a claim does not lapse while a model thinks; once a test carries a claim, the toolbox
renews it. When the scheduler stops a branch, its last session gives back what the branch's runs held across them.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.lifecycle import BranchTask, SpecialistStatus
from hyperloom_multibranch.contracts.blackboard.interfaces import ClaimService, BlackboardError
from hyperloom_multibranch.contracts.blackboard.vocab import ClaimStatus
from hyperloom_multibranch.specialist.directions import DIRECTIONS
from hyperloom_multibranch.specialist.exhausted import exhausted_records, latest_exhausted, session_direction
from hyperloom_multibranch.specialist.judging import NOTICE, BranchJudge, JudgeBlackboard, Judging

logger = logging.getLogger(__name__)

MAX_RESTARTS = 3
#: A stop that takes this long is logged step by step: half of what the scheduler waits for one before it counts it
#: as failed (``SPECIALIST_CALL_SECONDS``).
SLOW_STOP_SECONDS = 30.0
RESTART_WINDOW_S = 600.0
MAX_LOSS_RESTARTS = 2
MAX_IDLE_SESSIONS = 2


class RuntimeBlackboard(JudgeBlackboard, ClaimService, Protocol):
    """A branch's runtime handle: what its judge reads and writes, and the claims its specialist holds."""


class SessionCrashed(RuntimeError):
    """A run of a branch's specialist ended the way no working session ends.

    ``lost`` says its session lost its heartbeat; ``waiting`` says it was lost while its branch's long dev-sandbox
    process ran; ``worked`` says an earlier session of the same run ended well; ``resumable`` says the next run goes on
    with the crashed session's conversation.
    """

    def __init__(
        self,
        reason: str,
        *,
        lost: bool = False,
        waiting: bool = False,
        worked: bool = False,
        resumable: bool = False,
    ) -> None:
        super().__init__(reason)
        self.lost, self.waiting, self.worked, self.resumable = lost, waiting, worked, resumable


class SessionIdle(RuntimeError):
    """A run of a branch's specialist ended because its model kept finding nothing to test; ``tested`` says whether
    the run submitted a test before it did, ``resumable`` whether the next run goes on with its conversation."""

    def __init__(self, message: str, *, tested: bool = False, resumable: bool = False) -> None:
        super().__init__(message)
        self.tested, self.resumable = tested, resumable


class SessionFinished(RuntimeError):
    """A run of a branch's specialist that has no work left: a third-party agent that ended its run, once the last
    change it kept settled."""


class SessionRefused(SessionFinished):
    """A run of a third-party agent that refused the config it was given, as Arbor's exit 2 says: started again, it
    would refuse it again, so it ends its specialist as a finished run does, with why."""


class PreflightFailed(RuntimeError):
    """A branch's specialist could not be made ready to run; ``reasons`` says why, one line per attempt."""

    def __init__(self, reasons: Sequence[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = tuple(reasons)


class Session(Protocol):
    """One run of a branch's specialist: ``preflight`` makes it ready, once a branch, and raises
    :class:`PreflightFailed` when it cannot; ``run`` returns once its work is done, raises :class:`SessionIdle` when
    its model kept finding nothing to test, :class:`SessionFinished` when it has no work left, and raises when it
    crashed, a :class:`SessionCrashed` when it can say how."""

    async def preflight(self) -> None: ...

    async def run(self) -> None: ...

    async def set_k(self, k: int) -> None: ...

    async def stop(self) -> None:
        """Stop the run, more than once if need be; returns once none of its processes is left."""

    async def release(self) -> None:
        """The branch's runs are over: give back what they hold across runs, such as a sandbox's cards."""


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


class _StepTimes:
    """How long each step of a stop took, logged when the whole stop took :data:`SLOW_STOP_SECONDS` or more."""

    def __init__(self) -> None:
        self._seconds: dict[str, float] = {}

    async def __call__(self, step: str, awaitable: Awaitable[object]) -> None:
        started = time.monotonic()
        try:
            await awaitable
        finally:
            self._seconds[step] = time.monotonic() - started

    def report_if_slow(self, branch: str) -> None:
        total = sum(self._seconds.values())
        if total >= SLOW_STOP_SECONDS:
            steps = ", ".join(f"{step} {seconds:.1f} s" for step, seconds in self._seconds.items())
            logger.warning("the stop of %s took %.1f s: %s", branch, total, steps)


@dataclass
class _Branch:
    name: str
    start_commit: str
    task: BranchTask
    blackboard: RuntimeBlackboard
    k: int = 1
    session: Session | None = None
    stopping: asyncio.Task[None] | None = None
    last: Session | None = None
    tasks: list[asyncio.Task[None]] = field(default_factory=list)
    crashes: int = 0
    lost: int = 0
    gave_up: bool = False
    judge_failed: bool = False
    unready: str = ""
    finished: str = ""
    judge: BranchJudge | None = None


@dataclass
class _Restarts:
    """A branch's specialist crashes within the restart window, and its sessions lost since one last worked."""

    crashes: deque[float] = field(default_factory=deque)
    losses: int = 0

    def give_up(self, error: Exception, now: float) -> str:
        """Count the crash ``error`` at ``now``: why the specialist is not restarted, or "" when it is. A loss while
        waiting on a dev-sandbox process is not counted."""

        if isinstance(error, SessionCrashed) and error.worked:
            self.losses = 0
        if isinstance(error, SessionCrashed) and error.lost and error.waiting:
            return ""
        self.losses += isinstance(error, SessionCrashed) and error.lost
        self.crashes.append(now)
        while now - self.crashes[0] > RESTART_WINDOW_S:
            self.crashes.popleft()
        if len(self.crashes) > MAX_RESTARTS:
            return f"crashed {len(self.crashes)} times in {RESTART_WINDOW_S / 60:g} minutes"
        if self.losses > MAX_LOSS_RESTARTS:
            return f"lost {self.losses} sessions in a row, none working in between"
        return ""


class SpecialistRuntime:
    """The session's :class:`~hyperloom_multibranch.contracts.lifecycle.SpecialistLifecycle`."""

    def __init__(
        self,
        blackboard_for: Callable[[str], RuntimeBlackboard],
        judging: Judging,
        sessions: SessionFactory,
        *,
        pacing: RuntimePacing = RuntimePacing(),
    ) -> None:
        self._blackboard_for = blackboard_for
        self._judging = judging
        self._sessions = sessions
        self._pacing = pacing
        self._branches: dict[str, _Branch] = {}
        self._held: Counter[str] = Counter()
        self._closed: set[tuple[str, str]] = set()
        self._finishing: dict[str, asyncio.Task[None]] = {}

    async def start(self, branch: str, start_commit: str, task: BranchTask = BranchTask()) -> None:
        if branch in self._branches or branch in self._finishing:
            raise ValueError(f"branch {branch!r} already has a specialist")
        if not task.direction.strip() and task.repair is None and not task.agent:
            for state in self._branches.values():
                await self._learn_exhausted(state)
            task = replace(task, direction=self._least_held(start_commit))
        if task.direction in DIRECTIONS:
            self._held[task.direction] += 1
        state = _Branch(branch, start_commit, task, self._blackboard_for(branch))
        self._branches[branch] = state
        state.judge = BranchJudge(branch, state.blackboard, self._judging)
        state.tasks = [
            asyncio.create_task(self._supervise(state), name=f"specialist of {branch}"),
            asyncio.create_task(self._judge(state, state.judge), name=f"judge of {branch}"),
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
        return SpecialistStatus(
            running, state.crashes, state.lost, state.gave_up, state.judge_failed, state.unready, state.finished
        )

    async def stop(self, branch: str, *, finish_tests: bool = False) -> None:
        """Stop the branch's specialist, at most once whoever asks; a branch already stopped is left alone.

        With ``finish_tests`` the branch's judge goes on until no test of the branch is left in flight; without it the
        judge ends with the stop, and so does one an earlier stop left finishing. The first stop decides for the
        specialist. The stop runs in a task of its own that a caller who stops waiting does not cancel, as a cancelled
        stop would leave the branch's run and what it holds behind: the next call waits for that same stop, and one
        that failed is run again."""

        state = self._branches.get(branch)
        if state is not None:
            if state.stopping is None:
                state.stopping = asyncio.create_task(self._stopped(state, finish_tests), name=f"stop of {branch}")
            await asyncio.shield(state.stopping)
        if not finish_tests:
            finishing = self._finishing.pop(branch, None)
            if finishing is not None:
                finishing.cancel()
                await asyncio.gather(finishing, return_exceptions=True)

    async def _stopped(self, state: _Branch, finish_tests: bool) -> None:
        try:
            await self._stop_steps(state, finish_tests)
        except BaseException:
            state.stopping = None
            raise
        self._branches.pop(state.name, None)

    async def _stop_steps(self, state: _Branch, finish_tests: bool) -> None:
        timed = _StepTimes()
        supervisor, judge, renewal = state.tasks
        supervisor.cancel()
        await timed("supervisor", asyncio.gather(supervisor, return_exceptions=True))
        if state.session is not None:
            await timed("session stop", state.session.stop())
        ending = [renewal]
        if finish_tests and not judge.done() and state.judge is not None:
            state.judge.finish()
            self._finishing[state.name] = judge
            judge.add_done_callback(lambda _: self._finishing.pop(state.name, None))
        else:
            ending.append(judge)
        for task in ending:
            task.cancel()
        await timed("judge and claims", asyncio.gather(*ending, return_exceptions=True))
        if state.last is not None:
            await timed("release", state.last.release())
        await timed("learn", self._learn_exhausted(state))
        timed.report_if_slow(state.name)

    async def _supervise(self, state: _Branch) -> None:
        restarts = _Restarts()
        idle = 0
        ready = False
        while True:
            start = SessionStart(state.name, state.start_commit, state.k, state.task)
            state.session = await self._built(state, start)
            if state.session is None or not ready and not await self._preflight(state, state.session):
                return
            ready = True
            try:
                await state.session.run()
                return
            except SessionFinished as error:
                await state.session.stop()
                state.session, state.gave_up, state.finished = None, True, str(error)
                await _notice(state, f"{error}; it is not restarted")
                return
            except SessionIdle as error:
                await state.session.stop()
                logger.info("the specialist of %s found nothing to test: %s", state.name, error)
                idle = 0 if error.tested else idle + 1
                if idle < MAX_IDLE_SESSIONS:
                    await _notice(state, _idle_restarted(error))
                    continue
                state.session, state.gave_up = None, True
                record = await self._close_direction(state)
                await _notice(state, f"the specialist found nothing to test ({error}); it is not restarted{record}")
                return
            except Exception as error:
                logger.exception("the specialist of %s crashed", state.name)
                await state.session.stop()
                state.crashes += 1
                state.lost += isinstance(error, SessionCrashed) and error.lost
                why = restarts.give_up(error, self._pacing.clock())
                if not why:
                    await _notice(state, _restarted(error))
                    continue
                state.session, state.gave_up = None, True
                await _notice(state, f"the specialist {why} ({error!r}); it is not restarted")
                return

    async def _built(self, state: _Branch, start: SessionStart) -> Session | None:
        """The branch's next session; ``None``, its specialist given up with the reason, when it cannot be built."""

        try:
            state.last = self._sessions(start)
        except Exception as error:
            logger.exception("the session of %s cannot be built", state.name)
            state.gave_up, state.unready = True, f"its session cannot be built ({error})"
            await _notice(state, f"the specialist's session cannot be built ({error}); it is not started")
            return None
        return state.last

    async def _preflight(self, state: _Branch, session: Session) -> bool:
        """Make the branch's first session ready; give the specialist up, with every reason, when it cannot be: a
        failure the preflight names, or one it did not foresee, which leaves the branch as unready."""

        try:
            await session.preflight()
        except PreflightFailed as error:
            why = str(error)
            logger.warning("the specialist of %s could not be made ready: %s", state.name, error)
        except Exception as error:
            why = repr(error)
            logger.exception("the preflight of %s failed", state.name)
        else:
            return True
        await session.stop()
        state.session, state.gave_up, state.unready = None, True, why
        await _notice(state, f"the specialist's preflight failed ({why}); it is not started")
        return False

    async def _close_direction(self, state: _Branch) -> str:
        """Deal the given-up branch's direction to no new branch on its start commit or on the head its specialist's
        last exhausted record names; that record as the give-up notice quotes it, ``""`` without one."""
        self._closed.add((state.start_commit, state.task.direction))
        exhausted = latest_exhausted(await state.blackboard.branch_log(state.name))
        if exhausted is None:
            return ""
        self._closed.add((exhausted.head, state.task.direction))
        return f"; its last exhausted record: {exhausted.summary()}"

    async def _learn_exhausted(self, state: _Branch) -> None:
        """Close, on its head, each of the session's directions that ``state``'s specialist recorded exhausted."""
        for exhausted in exhausted_records(await state.blackboard.branch_log(state.name)):
            direction = session_direction(exhausted.direction)
            if direction is None:
                logger.warning(
                    "an exhausted record of %s names none of the session's directions: %r",
                    state.name,
                    exhausted.direction,
                )
                continue
            self._closed.add((exhausted.head, direction))

    def _least_held(self, start_commit: str) -> str:
        running = [state.task.direction for state in self._branches.values()]
        open_here = [d for d in DIRECTIONS if (start_commit, d) not in self._closed] or list(DIRECTIONS)
        return min(open_here, key=lambda d: (self._held[d], running.count(d)))

    async def _judge(self, state: _Branch, judge: BranchJudge) -> None:
        try:
            await judge.run(self._pacing.poll_seconds, self._pacing.sleep)
        except Exception as error:
            logger.exception("judging %s failed", state.name)
            state.judge_failed = True
            if state.stopping is not None:
                await _notice(state, f"judging this stopped branch's last tests failed ({error!r})")
                return
            supervisor, _, renewal = state.tasks
            supervisor.cancel()
            renewal.cancel()
            await asyncio.gather(supervisor, renewal, return_exceptions=True)
            if state.session is not None:
                await state.session.stop()
            await _notice(state, f"judging this branch failed ({error!r}); its specialist is stopped")

    async def _renew(self, state: _Branch) -> None:
        ttl = self._pacing.claim_ttl_seconds
        while True:
            await self._pacing.sleep(ttl / 3)
            snapshot = await state.blackboard.snapshot()
            for claim in snapshot.claims:
                if claim.holder != state.name or claim.status is not ClaimStatus.HELD or claim.request_id is not None:
                    continue
                try:
                    await state.blackboard.renew(claim.key, ttl_seconds=ttl)
                except BlackboardError:
                    logger.warning("the claim %s of %s ended before its renewal", claim.key, state.name, exc_info=True)


async def _notice(state: _Branch, content: str) -> None:
    await state.blackboard.append_branch_log(BranchLogEntry(state.name, BranchLogType.RESULT, f"{NOTICE} {content}"))


#: How the notices of a crashed and of an idle session's restart open.
_CRASHED, _IDLE = "the specialist crashed (", "the specialist found nothing to test ("


def restart_notice(content: str) -> bool:
    """Whether the branch-log ``content`` is the runtime's notice that a session ended and the next one starts: the
    next session is told why in its own first message, so the notice brings it nothing new."""
    return content.startswith((f"{NOTICE} {_CRASHED}", f"{NOTICE} {_IDLE}"))


def _restarted(error: Exception) -> str:
    resumed = isinstance(error, SessionCrashed) and error.resumable
    how = "resuming its conversation" if resumed else "with a fresh session"
    return f"{_CRASHED}{error!r}) and is restarted on this branch, {how}"


def _idle_restarted(error: SessionIdle) -> str:
    how = "resumes its conversation" if error.resumable else "starts a fresh session"
    return f"{_IDLE}{error}); the branch {how}"


def _log_failure(task: asyncio.Task[None]) -> None:
    """What ends a branch's task unexpectedly, a notice that could not be written included, is never lost."""
    if not task.cancelled() and task.exception() is not None:
        logger.error("the %s ended", task.get_name(), exc_info=task.exception())


__all__ = [
    "MAX_IDLE_SESSIONS",
    "MAX_LOSS_RESTARTS",
    "MAX_RESTARTS",
    "RESTART_WINDOW_S",
    "RuntimePacing",
    "Session",
    "SessionCrashed",
    "PreflightFailed",
    "SessionFinished",
    "SessionIdle",
    "SessionFactory",
    "SessionRefused",
    "SessionStart",
    "SpecialistRuntime",
    "restart_notice",
]
