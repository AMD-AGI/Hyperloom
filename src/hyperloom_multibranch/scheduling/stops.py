# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Why a session stops, the operator's stop signals, and the two session-wide stops.

Ending exploration stops every specialist branch and keeps the integration branch for FINAL; a branch stop that fails
there is logged, not raised. Stopping everything stops every branch including integration, which withdraws every
test, closes the toolbox, which releases the sandboxes it holds, and kills every process the scheduler started, then
reports what is still alive; it is bounded by the hard stop, and the kill runs whether or not the rest finished. Both
close the roster first, so no fork is admitted and no branch opened afterwards.
The main flow's signal drain lives in its loop package, which this package does not import.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass, replace
from enum import Enum

from hyperloom_multibranch.contracts.toolbox import QueueState
from hyperloom_multibranch.scheduling.branches import BranchControl
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.peers import SessionToolbox
from hyperloom_multibranch.scheduling.processes import ProcessRecord, ProcessRegistry
from hyperloom_multibranch.scheduling.roster import BranchKind, BranchStop, Roster, StopCause
from hyperloom_multibranch.scheduling.settle import settled

logger = logging.getLogger(__name__)

STOP_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
STOP_OVERRAN = "the stop overran the hard stop"


class SessionStopReason(str, Enum):
    TIME_EXHAUSTED = "time_exhausted"
    GLOBAL_CONVERGED = "global_converged"
    SIGNAL = "signal"
    PRELUDE_FAILED = "prelude_baseline_failed"
    #: The session raised an error it does not handle; CLOSE still runs before it propagates.
    ABORTED = "aborted"


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

    ``queue`` is what the toolbox held once every branch was stopped, ``None`` when it did not report it before the
    hard stop; ``failures`` are the branch stops and the toolbox close that raised and a stop that overran the hard
    stop.
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
        self,
        roster: Roster,
        branches: BranchControl,
        toolbox: SessionToolbox,
        processes: ProcessRegistry,
        decisions: DecisionLog,
        clock: Clock,
    ) -> None:
        self._roster = roster
        self._branches = branches
        self._toolbox = toolbox
        self._processes = processes
        self._decisions = decisions
        self._clock = clock

    async def end_exploration(self, reason: SessionStopReason) -> None:
        """Close the roster and stop every specialist branch; the integration branch keeps its quota for FINAL.

        A branch stop that fails is logged and does not hold FINAL up: the branch's own ``STOP`` decision names the
        failure, and CLOSE kills whatever it left behind and reports what outlived that.
        """

        self._roster.close()
        for failure in await self._stop_open(reason, frozenset({BranchKind.SPECIALIST})):
            logger.error("a specialist branch did not stop cleanly at the end of EXPLORE", exc_info=failure)

    async def stop_everything(self, reason: SessionStopReason, hard_stop_at: float) -> StopReport:
        """Close the roster, stop every branch, read the queue, close the toolbox, kill every registered process, and
        report.

        Branches stop concurrently, each withdrawing its tests, and a failed branch stop is reported, not raised, so
        every other branch is still stopped and the toolbox still closed, which releases the sandboxes it holds. All of
        that gets until ``hard_stop_at`` (clock seconds); every registered process is killed afterwards whatever became
        of it.
        """

        self._roster.close()
        try:
            halted = await self._clock.wait_for(self._halt(reason), hard_stop_at - self._clock.now())
        except asyncio.TimeoutError:
            halted = StopReport((), None, (STOP_OVERRAN,))
        finally:
            await self._processes.kill_all()
        report = replace(halted, leftovers=await self._processes.leftovers())
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

    async def _halt(self, reason: SessionStopReason) -> StopReport:
        """What the stop leaves before the kill; the queue is read before the toolbox closes and forgets it."""

        failures = await self._stop_open(reason, frozenset(BranchKind))
        status = await settled(self._toolbox.queue_state())
        closed = await settled(self._toolbox.close())
        if isinstance(status, BaseException):
            logger.error("the toolbox did not report its queue after the stop", exc_info=status)
        if isinstance(closed, BaseException):
            failures.append(closed)
        for failure in failures:
            logger.error("a step of the session stop failed", exc_info=failure)
        queue = None if isinstance(status, BaseException) else status
        return StopReport((), queue, tuple(f"{type(failure).__name__}: {failure}" for failure in failures))

    async def _stop_open(self, reason: SessionStopReason, kinds: frozenset[BranchKind]) -> list[BaseException]:
        stop = BranchStop(StopCause.SESSION_END, reason.value, self._clock.now())
        open_branches = [entry.branch for entry in self._roster.entries() if entry.stop is None and entry.kind in kinds]
        outcomes = await asyncio.gather(*(self._branches.stop_branch(branch, stop) for branch in open_branches))
        return [failure for failures in outcomes for failure in failures]


__all__ = ["STOP_OVERRAN", "STOP_SIGNALS", "SessionStopReason", "SessionStopper", "StopReport", "StopSignals"]
