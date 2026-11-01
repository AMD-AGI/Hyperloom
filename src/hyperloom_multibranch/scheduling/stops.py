# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Why a session stops, the operator's stop signals, and the two session-wide stops.

Ending exploration stops every specialist branch, withdraws the tests the branches stopped earlier were left to finish,
and keeps the integration branch for FINAL; a branch stop or a withdrawal that fails there is logged, not raised.
Stopping everything stops every branch including integration, which withdraws every test, then releases what the
session holds: it stops the MCP service the specialists reach their tools on, closes the toolbox, which releases the
sandboxes it holds, and kills every process the scheduler started, then reports what is still alive. The toolbox close
gets :data:`TOOLBOX_CLOSE_SECONDS`, or half the time left before the hard stop when that is less, the branch stops
the rest, and the release runs whether or not they finished. Both close the roster first, so no fork is admitted and
no branch opened afterwards, and void the forks and repairs still queued for a slot.
The main flow's signal drain lives in its loop package, which this package does not import.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

from hyperloom_multibranch.contracts.toolbox import QueueState
from hyperloom_multibranch.scheduling.branches import BranchControl
from hyperloom_multibranch.scheduling.context import SchedulerContext
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind
from hyperloom_multibranch.scheduling.processes import ProcessRecord
from hyperloom_multibranch.scheduling.roster import BranchKind, BranchStop, StopCause
from hyperloom_multibranch.scheduling.settle import failure_text, settled

if TYPE_CHECKING:
    from hyperloom.hosted.mcp_server import McpServer
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox

logger = logging.getLogger(__name__)

STOP_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
STOP_OVERRAN = "the stop overran the hard stop"
TOOLBOX_CLOSE_OVERRAN = "the toolbox did not close in time"
#: Clock seconds the toolbox gets to close; its close cancels its own work and releases its backend and sandboxes, so
#: it is quick even while a specialist hangs.
TOOLBOX_CLOSE_SECONDS = 10.0


class SessionStopReason(str, Enum):
    TIME_EXHAUSTED = "time_exhausted"
    #: Tests' GPU time plus card time any lease holds used the GPU hours ``--max-gpu-hours`` leaves EXPLORE; FINAL
    #: still runs.
    GPU_HOURS_EXHAUSTED = "gpu_hours_exhausted"
    #: The specialists' recorded LLM spend reached ``--max-llm-usd``; FINAL still runs.
    LLM_BUDGET_EXHAUSTED = "llm_budget_exhausted"
    #: The judge answered ``stop_session``: nothing measures convergence, so this is its judgement and not a
    #: measurement. FINAL still runs, and the session counts as one that ran its course.
    JUDGE_STOPPED = "judge_stopped"
    SIGNAL = "signal"
    PRELUDE_FAILED = "prelude_baseline_failed"
    #: The toolbox has fewer usable cards than one test needs; FINAL is skipped and the session failed.
    NO_MEASUREMENT_CARDS = "no_measurement_cards"
    #: The run asked to stop once its Arbor agents' directions were mapped (``--stop-after-map``): PRELUDE and the map
    #: ran, EXPLORE and FINAL did not, and the map and dossier were left for a person to review.
    MAPPED = "mapped"
    #: The session raised an error it does not handle; CLOSE still runs before it propagates.
    ABORTED = "aborted"
    #: The session was killed with the machine it ran on, so it never closed and wrote no report; only the emergency
    #: ``final-only`` recovery reports it, and finishing a dead session does not make it one that ran its course.
    KILLED = "killed"


class StopSignals:
    """SIGINT, SIGTERM and SIGHUP routed to the event loop; the first one received is the one reported.

    Children lead sessions of their own, so a terminal's Ctrl-C reaches only the scheduler, which stops them itself;
    a second signal tells it to cut the orderly stop short, and it is remembered whenever it arrives.
    """

    def __init__(self) -> None:
        self._first: asyncio.Future[int] | None = None
        self._second: asyncio.Future[int] | None = None

    def install(self) -> None:
        """Register the handlers on the running loop."""

        loop = asyncio.get_running_loop()
        self._first = loop.create_future()
        self._second = loop.create_future()
        for sig in STOP_SIGNALS:
            loop.add_signal_handler(sig, self._on_signal, sig)

    def uninstall(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in STOP_SIGNALS:
            loop.remove_signal_handler(sig)

    @property
    def received(self) -> int | None:
        return self._first.result() if self._first is not None and self._first.done() else None

    async def wait(self) -> int:
        """Return the first signal received, waiting for one if none has come yet; needs :meth:`install` first."""

        return await asyncio.shield(self._installed(self._first))

    async def wait_second(self) -> int:
        """Return the second signal received, waiting for it if it has not come yet."""

        return await asyncio.shield(self._installed(self._second))

    def _on_signal(self, sig: int) -> None:
        pending = next((future for future in (self._first, self._second) if future and not future.done()), None)
        if pending is not None:
            pending.set_result(sig)

    @staticmethod
    def _installed(future: asyncio.Future[int] | None) -> asyncio.Future[int]:
        if future is None:
            raise RuntimeError("StopSignals waited on before install()")
        return future


@dataclass(frozen=True)
class StopReport:
    """What outlived a session stop: registered processes still running and tests the toolbox still holds.

    ``queue`` is what the toolbox held once every branch was stopped, ``None`` when it did not report it in time;
    ``failures`` are the branch stops that failed, branch stops that overran their time, specialists that never
    confirmed their stop, and a toolbox close that raised or overran its own.
    """

    leftovers: tuple[ProcessRecord, ...]
    queue: QueueState | None
    failures: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        idle = self.queue is not None and self.queue.queued == 0 and self.queue.running == 0
        return not self.leftovers and not self.failures and idle


class SessionStopper:
    def __init__(
        self, context: SchedulerContext, branches: BranchControl, toolbox: RealToolbox, mcp: McpServer
    ) -> None:
        self._roster = context.roster
        self._branches = branches
        self._toolbox = toolbox
        self._mcp = mcp
        self._processes = context.processes
        self._decisions = context.decisions
        self._clock = context.clock

    async def end_exploration(self, reason: SessionStopReason) -> None:
        """Close the roster, void the queued forks and repairs, stop every specialist branch and withdraw the tests the
        branches stopped earlier were left to finish; the integration branch keeps its quota for FINAL.

        A branch stop that fails is logged and does not hold FINAL up: the branch's own ``STOP`` decision names the
        failure, and CLOSE kills whatever it left behind and reports what outlived that.
        """

        self._roster.close()
        for failure in await self._branches.void_queued(reason.value):
            logger.error("a queued fork or repair was not voided cleanly", exc_info=failure)
        for failure in await self._stop_open(reason, frozenset({BranchKind.SPECIALIST})):
            logger.error("a specialist branch did not stop cleanly at the end of EXPLORE", exc_info=failure)
        withdrawn = await settled(self._branches.withdraw_finishing_tests(reason.value))
        if isinstance(withdrawn, BaseException):
            logger.error("the tests the stopped branches were left to finish were not withdrawn", exc_info=withdrawn)

    async def stop_everything(self, reason: SessionStopReason, hard_stop_at: float) -> StopReport:
        """Close the roster, stop every branch, read the queue, :meth:`release` what the session holds, and report.

        Branches stop concurrently, each withdrawing its tests, and a failed branch stop is reported, not raised, so
        every other branch is still stopped. That gets the time before ``hard_stop_at`` (clock seconds) the toolbox
        close does not; the release runs afterwards whatever became of it.
        """

        self._roster.close()
        left = max(hard_stop_at - self._clock.now(), 0.0)
        closing = min(TOOLBOX_CLOSE_SECONDS, left / 2)
        try:
            halted = await self._clock.wait_for(self._halt(reason), left - closing)
        except asyncio.TimeoutError:
            halted = StopReport((), None, (STOP_OVERRAN,))
        released = await self.release(closing)
        report = replace(halted, leftovers=await self._processes.leftovers(), failures=(*halted.failures, *released))
        self._decisions.record(
            Decision(
                DecisionKind.SESSION_STOP,
                reason.value,
                detail={
                    "leftover_pids": [record.pid for record in report.leftovers],
                    "failures": list(report.failures),
                    "clean": report.clean,
                },
            )
        )
        return report

    async def release(self, closing: float) -> tuple[str, ...]:
        """Stop the MCP service, close the toolbox, giving it ``closing`` clock seconds, then kill every registered
        process; returns the close's failure, if it had one.

        A CLOSE cut short releases through here too, so the sandboxes and the processes go whatever else did not
        finish.
        """

        await asyncio.to_thread(self._mcp.stop)
        closed = await settled(self._clock.wait_for(self._toolbox.close(), closing))
        await self._processes.kill_all()
        if isinstance(closed, asyncio.TimeoutError):
            logger.error(TOOLBOX_CLOSE_OVERRAN)
            return (TOOLBOX_CLOSE_OVERRAN,)
        if isinstance(closed, BaseException):
            logger.error("the toolbox did not close", exc_info=closed)
            return (failure_text(closed),)
        return ()

    async def _halt(self, reason: SessionStopReason) -> StopReport:
        """What the branch stops leave; the queue is read before the toolbox closes and forgets it.

        Every test is withdrawn before any specialist is asked to stop: a specialist that hangs in its stop would
        otherwise hold its branch's withdrawal past the hard stop. Every specialist whose stop the runtime has not
        confirmed, at the end of EXPLORE or here, is asked once more, and one that still does not confirm it is a
        failure of the session stop.
        """

        failures = await self._branches.void_queued(reason.value)
        withdrawn = await settled(self._toolbox.withdraw_all(reason.value))
        if isinstance(withdrawn, BaseException):
            failures.append(withdrawn)
        failures += await self._stop_open(reason, frozenset(BranchKind))
        unconfirmed = await self._branches.stop_unconfirmed_specialists()
        status = await settled(self._toolbox.queue_state())
        if isinstance(status, BaseException):
            logger.error("the toolbox did not report its queue after the stop", exc_info=status)
        for failure in failures:
            logger.error("a step of the session stop failed", exc_info=failure)
        queue = None if isinstance(status, BaseException) else status
        unstopped = tuple(f"the specialist of {branch} did not confirm its stop" for branch in unconfirmed)
        return StopReport((), queue, (*map(failure_text, failures), *unstopped))

    async def _stop_open(self, reason: SessionStopReason, kinds: frozenset[BranchKind]) -> list[BaseException]:
        stop = BranchStop(StopCause.SESSION_END, reason.value, self._clock.now())
        open_branches = [entry.branch for entry in self._roster.entries() if entry.stop is None and entry.kind in kinds]
        outcomes = await asyncio.gather(*(self._branches.stop_branch(branch, stop) for branch in open_branches))
        return [failure for failures in outcomes for failure in failures]


__all__ = [
    "STOP_OVERRAN",
    "STOP_SIGNALS",
    "TOOLBOX_CLOSE_OVERRAN",
    "TOOLBOX_CLOSE_SECONDS",
    "SessionStopReason",
    "SessionStopper",
    "StopReport",
    "StopSignals",
]
