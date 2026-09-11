# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Watch the coordinator from outside it, and act when it stops being one."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from hyperloom.common.proctree import running
from hyperloom.inference_optimizer.session.lock import read_owner
from hyperloom.orchestrator.bringup.reap import ReapBackend, select_reaper
from hyperloom.orchestrator.supervisor import store

log = logging.getLogger(__name__)

#: The coordinator is alive and its tick is advancing.
ALIVE = "alive"

#: The coordinator's process is alive but its tick has stopped advancing.
WEDGED = "wedged"

#: The coordinator's process is gone.
DEAD = "dead"

#: Nothing about the coordinator could be established. Never escalated on.
UNKNOWN = "unknown"

#: How long a tick may go without advancing before it counts as wedged. Well
#: above a legitimately slow tick -- each role turn is capped at
#: :data:`~..loop.coordinator.REACTOR_TURN_BUDGET_SEC` and long actions run as
#: dispatched tasks the tick does not wait on -- and well inside the default
#: session, which a window it cannot fit in would make unreachable.
DEFAULT_TICK_STALL_SEC: float = 3600.0

#: How long the coordinator is given to act on the stop it was asked for before
#: it is treated as unable to. Its teardown cancels in-flight actions and waits
#: for the threads running them to unwind.
DEFAULT_STOP_GRACE_SEC: float = 300.0

#: How often the watcher looks.
DEFAULT_POLL_SEC: float = 30.0

#: Maximum resumable watchdog restarts before the session becomes terminal.
DEFAULT_MAX_RESTARTS: int = 3

#: The stop reason recorded for a coordinator that stopped running its loop.
WEDGED_STOP_REASON = "supervisor_tick_stalled"

#: Internal handoff telling the launcher that the session remains resumable.
SUPERVISOR_RESTART_REASON = "supervisor_restart_requested"

#: The stop reason recorded in the terminal artifact for a dead coordinator.
DIED_STOP_REASON = "supervisor_coordinator_died"

__all__ = [
    "ALIVE",
    "DEAD",
    "DEFAULT_MAX_RESTARTS",
    "DEFAULT_POLL_SEC",
    "DEFAULT_TICK_STALL_SEC",
    "DIED_STOP_REASON",
    "SUPERVISOR_RESTART_REASON",
    "UNKNOWN",
    "WEDGED",
    "WEDGED_STOP_REASON",
    "Observation",
    "Supervisor",
]


@dataclass(frozen=True)
class Observation:
    """One reading of the coordinator.

    Attributes:
        verdict: :data:`ALIVE`, :data:`WEDGED`, :data:`DEAD` or :data:`UNKNOWN`.
        pid: The owner pid the lock names, ``0`` when there is none.
        detail: Why the verdict is what it is.
        tick: The last tick the coordinator stamped.
        tick_age_sec: How long ago it stamped it; ``-1`` when never.
    """

    verdict: str
    pid: int = 0
    detail: str = ""
    tick: int = 0
    tick_age_sec: float = -1.0


@dataclass
class SupervisorReport:
    """What a supervisor run did, for the log and for tests.

    Attributes:
        asked: Reasons the coordinator was asked to stop.
        terminal_path: Where the terminal artifact was written, if it was.
        refusals: Actions declined, and why.
    """

    asked: list[str] = field(default_factory=list)
    terminal_path: str = ""
    refusals: list[str] = field(default_factory=list)


class Supervisor:
    """The out-of-band watcher.

    Attributes:
        session_dir (Path): The session being watched.
        tick_stall_sec (float): How long a tick may go without advancing.
        stop_grace_sec (float): How long a stop that was asked for is given.
        poll_sec (float): How often it looks.
    """

    def __init__(
        self,
        session_dir: Path | str,
        *,
        tick_stall_sec: float = DEFAULT_TICK_STALL_SEC,
        stop_grace_sec: float = DEFAULT_STOP_GRACE_SEC,
        poll_sec: float = DEFAULT_POLL_SEC,
        max_restarts: int = DEFAULT_MAX_RESTARTS,
        reaper: ReapBackend | None = None,
        now: Callable[[], float] | None = None,
    ):
        """Initialise the supervisor.

        Args:
            session_dir: The session root directory.
            tick_stall_sec: How long a tick may go without advancing.
            stop_grace_sec: How long the coordinator is given to act on a stop.
            poll_sec: Seconds between readings.
            max_restarts: Resumable restart attempts allowed for this session.
            reaper: The reap unit; defaults to whatever this host selects.
            now: Wall-clock source, for tests.
        """
        self.session_dir = Path(session_dir)
        self.tick_stall_sec = max(0.0, tick_stall_sec)
        self.stop_grace_sec = max(0.0, stop_grace_sec)
        self.poll_sec = max(0.1, poll_sec)
        self.max_restarts = max(0, max_restarts)
        self._reaper: ReapBackend = reaper if reaper is not None else select_reaper()
        self._now: Callable[[], float] = now if now is not None else time.time
        self._report = SupervisorReport()
        self._ask_attempted = False
        self._asked_unix = 0.0
        self._end_attempted = False
        self._restart_count = self._load_restart_count()

    def _load_restart_count(self) -> int:
        """Load the durable restart count, failing closed on corrupt status."""
        try:
            status = store.read_status(self.session_dir)
            return max(0, int((status or {}).get("restart_count") or 0))
        except (OSError, TypeError, ValueError):
            log.error("SUPERVISOR: restart count is unreadable; refusing another resumable restart")
            return self.max_restarts

    @property
    def unit(self) -> ReapBackend:
        """ReapBackend: The reap unit this supervisor would use."""
        return self._reaper

    @property
    def report(self) -> SupervisorReport:
        """SupervisorReport: What this supervisor has done so far."""
        return self._report

    def observe(self) -> Observation:
        """Take one reading of the coordinator.

        Returns:
            Observation: What the lock and the tick stamp say together. The lock
            alone names a process that may be stuck; the stamp alone may belong
            to a coordinator that has since exited.
        """
        owner = read_owner(self.session_dir)
        if owner is None:
            return Observation(UNKNOWN, detail="no optimizer lock owner recorded")
        pid = int(owner["pid"])
        host = str(owner["hostname"])
        if host != socket.gethostname():
            # A pid is only meaningful on the host that issued it.
            return Observation(UNKNOWN, pid=pid, detail=f"owner recorded on another host ({host})")
        stamp = store.read_tick(self.session_dir)
        if stamp is not None and stamp.pid != pid:
            # The stamp file outlives the leg that wrote it, so a resume finds
            # the previous owner's stamp already older than any stall window:
            # aging a stamp this owner did not write would end the session it
            # just restarted. Both files are written by the coordinator itself,
            # so a pid mismatch means this owner has not reached its first tick.
            stamp = None
        age = -1.0 if stamp is None else max(0.0, self._now() - stamp.stamped_unix)
        tick = 0 if stamp is None else stamp.tick
        if not running(pid):
            return Observation(DEAD, pid=pid, detail="owner pid is gone", tick=tick, tick_age_sec=age)
        if stamp is None:
            # A coordinator that has never stamped is starting up, not wedged.
            return Observation(ALIVE, pid=pid, detail="no tick stamp yet", tick=tick, tick_age_sec=age)
        if age > self.tick_stall_sec:
            return Observation(
                WEDGED,
                pid=pid,
                detail=f"tick {tick} last advanced {age:.0f}s ago",
                tick=tick,
                tick_age_sec=age,
            )
        return Observation(ALIVE, pid=pid, detail=f"tick {tick}", tick=tick, tick_age_sec=age)

    async def act(self, observation: Observation) -> bool:
        """Respond to one reading.

        Args:
            observation: What was just observed.

        Returns:
            bool: True when the session is over and the watch should stop.
        """
        over = False
        if observation.verdict == WEDGED:
            over = await self._escalate_wedged(observation)
        elif observation.verdict == DEAD:
            if self._asked_unix and self._restart_count <= self.max_restarts:
                over = True
            else:
                reason = WEDGED_STOP_REASON if self._asked_unix else DIED_STOP_REASON
                over = await self._end(observation, reason)
        self._write_status(observation)
        return over

    async def _escalate_wedged(self, observation: Observation) -> bool:
        """Ask a live-but-stuck coordinator to stop, and end it if it will not.

        Args:
            observation: The wedged reading being escalated.

        Returns:
            bool: True once the coordinator has been ended and recorded.
        """
        if not self._ask_attempted:
            self._ask_attempted = True
            self._ask_to_stop(observation)
            return False
        if self._asked_unix == 0.0:
            # The ask never went out, so nothing has been given a grace to run
            # in and there is no stop to escalate past.
            return False
        if self._end_attempted or self._now() - self._asked_unix < self.stop_grace_sec:
            return False
        # Attempted once: a coordinator that survives the attempt is left
        # running and reported as such, not re-signalled every poll.
        self._end_attempted = True
        return await self._end(observation, WEDGED_STOP_REASON)

    def _ask_to_stop(self, observation: Observation) -> None:
        """Send the coordinator the stop signal its drain thread is waiting on."""
        reason = f"{WEDGED_STOP_REASON}: {observation.detail}"
        next_restart_count = self._restart_count + 1
        stop_signal = signal.SIGHUP if next_restart_count <= self.max_restarts else signal.SIGTERM
        try:
            os.kill(observation.pid, stop_signal)
        except ProcessLookupError:
            # It exited between the reading and the signal; the next reading
            # sees a dead coordinator and ends the session on that.
            return
        except PermissionError as exc:
            self._report.refusals.append(f"cannot signal coordinator {observation.pid}: {exc}")
            log.error("SUPERVISOR: refusing to escalate -- pid %d is not ours to signal", observation.pid)
            return
        self._asked_unix = self._now()
        self._restart_count = next_restart_count
        self._report.asked.append(reason)
        log.error("SUPERVISOR: asked coordinator %d to stop (%s)", observation.pid, reason)

    async def _end(self, observation: Observation, stop_reason: str) -> bool:
        """Reap what the coordinator left and write the terminal artifact.

        Args:
            observation: The reading that ended the session.
            stop_reason: The terminal reason to record.

        Returns:
            bool: True when the terminal artifact was written, which it is only
            once nothing of the coordinator is still running.
        """
        if running(observation.pid):
            return False
        self._write_terminal(observation, stop_reason)
        return True

    def _write_terminal(self, observation: Observation, stop_reason: str) -> None:
        """Write the terminal artifact the coordinator never got to write.

        Args:
            observation: The reading that ended the session.
            stop_reason: The terminal reason to record.

        Raises:
            OSError: If the artifact cannot be written.
        """
        from hyperloom.inference_optimizer.breakdown import (
            FINAL_PRODUCER_SUPERVISOR,
            write_minimal_final_json,
        )

        path = write_minimal_final_json(
            self.session_dir,
            producer=FINAL_PRODUCER_SUPERVISOR,
            extra={
                "stop_reason": stop_reason,
                "supervisor": {
                    "detail": observation.detail,
                    "coordinator_pid": observation.pid,
                    "last_tick": observation.tick,
                    "tick_age_sec": observation.tick_age_sec,
                    "restart_count": self._restart_count,
                    "refused": list(self._report.refusals),
                },
            },
        )
        self._report.terminal_path = str(path)
        log.error("SUPERVISOR: coordinator %d is gone; wrote %s", observation.pid, path)

    def _write_status(self, observation: Observation) -> None:
        """Rewrite the status snapshot after a reading."""
        store.write_status(
            self.session_dir,
            {
                "verdict": observation.verdict,
                "detail": observation.detail,
                "coordinator_pid": observation.pid,
                "last_tick": observation.tick,
                "tick_age_sec": observation.tick_age_sec,
                "observed_unix": self._now(),
                "tick_stall_sec": self.tick_stall_sec,
                "restart_count": self._restart_count,
                "max_restarts": self.max_restarts,
                "stop_asked": list(self._report.asked),
                "refused": list(self._report.refusals),
                "terminal_path": self._report.terminal_path,
            },
        )

    async def run(self, *, max_polls: int = 0) -> SupervisorReport:
        """Poll until the coordinator is gone, or until ``max_polls`` readings.

        Args:
            max_polls: Stop after this many readings; ``0`` means run until the
                coordinator dies.

        Returns:
            SupervisorReport: What the run did.
        """
        polls = 0
        while True:
            done = await self.act(self.observe())
            polls += 1
            if done or (max_polls and polls >= max_polls):
                return self._report
            await asyncio.sleep(self.poll_sec)
