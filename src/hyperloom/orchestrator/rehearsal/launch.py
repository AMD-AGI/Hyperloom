# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A launch backend that plays a written-down round instead of booting a server.

A round consumes the server log at the path the launch was handed, the
artifacts left in its slot, the returncode, and the budget the attempt spent.
The scenario states all four and this produces them on disk.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    stamp_server_ready,
)
from hyperloom.orchestrator.rehearsal.clock import VirtualClock
from hyperloom.orchestrator.rehearsal.scenario import (
    DIED_SILENTLY,
    HANG,
    READY,
    LaunchAttempt,
    LaunchScenario,
    ScenarioError,
)

__all__ = ["RecordedLaunch", "ScenarioExhausted", "ScriptedLaunchBackend"]

#: Returncode a stage failure reports when the scenario does not name one.
_FAILED_RETURNCODE = 1

#: How many liveness callbacks an attempt makes, spread across its duration.
_OUTPUT_TICKS = 4


class ScenarioExhausted(ScenarioError):
    """The round asked for more launches than the scenario described."""


@dataclass(frozen=True)
class RecordedLaunch:
    """One launch as it was requested, kept for assertions.

    Attributes:
        attempt: The scenario attempt that answered it.
        cmd: The command the caller passed.
        cwd: The working directory the caller passed, if any.
        server_log_path: Where the caller expected the server's log, if named.
        session_deadline_sec: The session budget instant in force, if any.
        timeout: The hard timeout in force, if any.
        started_at: Clock elapsed when the launch began.
        finished_at: Clock elapsed when it returned.
        returncode: What the caller was told; ``-1`` for a reaped hang.
    """

    attempt: LaunchAttempt
    cmd: tuple[str, ...]
    cwd: str | None
    server_log_path: str | None
    session_deadline_sec: float | None
    timeout: float | None
    started_at: float
    finished_at: float
    returncode: int


@dataclass
class ScriptedLaunchBackend:
    """Serves one scenario attempt per launch, in order.

    Attributes:
        scenario: The round to play.
        clock: The clock attempts are charged to.
        calls: Every launch served, in order.
    """

    scenario: LaunchScenario
    clock: VirtualClock = field(default_factory=VirtualClock)
    calls: list[RecordedLaunch] = field(default_factory=list)

    @property
    def served(self) -> int:
        """int: How many launches have been answered."""
        return len(self.calls)

    @property
    def exhausted(self) -> bool:
        """bool: Whether every attempt in the scenario has been played."""
        return self.served >= len(self.scenario.attempts)

    def run(
        self,
        cmd: Sequence[str],
        *,
        cwd: str | None = None,
        server_log_path: str | None = None,
        timeout: int | float | None = None,
        session_deadline_sec: float | None = None,
        on_output: Callable[[], None] | None = None,
        **_ignored: Any,
    ) -> subprocess.CompletedProcess:
        """Play the next attempt and report what it did.

        Args:
            cmd: The command the caller would have run.
            cwd: Working directory; the round slot when no server log is named.
            server_log_path: Where the attempt writes the server's own log. Its
                parent is the round slot the artifacts are materialised into.
            timeout: Hard timeout; a hanging attempt is reaped on it.
            session_deadline_sec: Absolute instant the session budget expires,
                read off ``time.monotonic()``.
            on_output: Liveness callback, driven across the attempt's duration.
            **_ignored: The rest of the launch surface, which a scripted attempt
                has no process to apply.

        Returns:
            subprocess.CompletedProcess: The attempt's outcome.

        Raises:
            ScenarioExhausted: When the scenario has no attempt left.
            ScenarioError: When a session deadline is supplied while this
                backend's clock is not the one ``time.monotonic`` reads, or
                when an attempt has artifacts and names no directory to put
                them in.
            subprocess.TimeoutExpired: When a hanging attempt hits ``timeout``
                with budget to spare.
        """
        budget_left = self._budget_left(session_deadline_sec)
        attempt = self._next_attempt()
        started = self.clock.elapsed
        if server_log_path:
            slot: Path | None = Path(server_log_path).parent
        else:
            slot = Path(cwd) if cwd is not None else None
        self._materialise(attempt, slot=slot, server_log_path=server_log_path)

        reap_at = float(timeout) if attempt.outcome == HANG and timeout is not None else None
        spend = reap_at if reap_at is not None else attempt.duration_sec

        # Production tests the session budget ahead of the hard-timeout gate, so
        # an attempt that outlives the budget stops for the budget whatever else
        # it was about to do -- including a hang the timeout would have reaped.
        if budget_left is not None and budget_left <= spend:
            self._spend(budget_left, on_output)
            returncode = SESSION_TIME_EXHAUSTED_RETURNCODE
        elif reap_at is not None:
            self._spend(reap_at, on_output)
            self.clock.mark(f"launch:{_label(attempt)}:reaped")
            self.calls.append(
                self._record(attempt, cmd, cwd, server_log_path, session_deadline_sec, timeout, started, -1)
            )
            raise subprocess.TimeoutExpired(cmd=list(cmd), timeout=reap_at)
        else:
            self._spend(spend, on_output)
            returncode = _FAILED_RETURNCODE if attempt.outcome == HANG else self._returncode(attempt)

        if attempt.outcome == READY and server_log_path and returncode == 0:
            stamp_server_ready(server_log_path, attempt.duration_sec)

        self.clock.mark(f"launch:{_label(attempt)}:rc={returncode}")
        self.calls.append(
            self._record(attempt, cmd, cwd, server_log_path, session_deadline_sec, timeout, started, returncode)
        )
        return subprocess.CompletedProcess(
            args=list(cmd),
            returncode=returncode,
            stdout=attempt.wrapper_stdout,
            stderr=attempt.rendered_stderr(),
        )

    def _budget_left(self, session_deadline_sec: float | None) -> float | None:
        """Return the seconds of session budget left, or ``None`` with no deadline.

        Raises:
            ScenarioError: When this backend's clock is not the one
                ``time.monotonic`` reads, so the deadline and the attempt
                durations sit on different timelines.
        """
        if session_deadline_sec is None:
            return None
        if not self.clock.installed:
            raise ScenarioError(
                "session_deadline_sec is an instant on time.monotonic()'s timeline; "
                "play this round inside installed_clock(clock) so the deadline and "
                "the attempt durations are read off the same clock"
            )
        return float(session_deadline_sec) - self.clock.monotonic()

    def _next_attempt(self) -> LaunchAttempt:
        """Return the next unplayed attempt, or raise :class:`ScenarioExhausted`."""
        if self.exhausted:
            raise ScenarioExhausted(
                f"scenario {self.scenario.name or '<unnamed>'} described "
                f"{len(self.scenario.attempts)} attempts; launch "
                f"{self.served + 1} was requested"
            )
        return self.scenario.attempts[self.served]

    def _spend(self, seconds: float, on_output: Callable[[], None] | None) -> None:
        """Charge ``seconds`` to the clock, reporting liveness as it goes."""
        step = max(0.0, float(seconds)) / _OUTPUT_TICKS
        for _ in range(_OUTPUT_TICKS):
            self.clock.advance(step)
            if on_output is not None:
                on_output()

    @staticmethod
    def _returncode(attempt: LaunchAttempt) -> int:
        """Return the attempt's declared returncode, or the one it implies."""
        if attempt.returncode is not None:
            return int(attempt.returncode)
        return 0 if attempt.outcome == READY else _FAILED_RETURNCODE

    def _materialise(self, attempt: LaunchAttempt, *, slot: Path | None, server_log_path: str | None) -> None:
        """Write the attempt's server log and artifacts into the round slot.

        Raises:
            ScenarioError: When the attempt has artifacts and there is no slot
                to write them into.
        """
        log_text = attempt.rendered_log()
        if server_log_path and (log_text or attempt.outcome != DIED_SILENTLY):
            target = Path(server_log_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(log_text, encoding="utf-8")
        if not attempt.artifacts:
            return
        if slot is None:
            raise ScenarioError(
                f"attempt {_label(attempt)!r} declares artifacts, "
                "but the launch named neither a server log nor a working directory"
            )
        for relative, content in attempt.artifacts.items():
            path = slot / str(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, (Mapping, list)):
                path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            else:
                path.write_text(str(content), encoding="utf-8")

    def _record(
        self,
        attempt: LaunchAttempt,
        cmd: Sequence[str],
        cwd: str | None,
        server_log_path: str | None,
        session_deadline_sec: float | None,
        timeout: int | float | None,
        started: float,
        returncode: int,
    ) -> RecordedLaunch:
        """Build the :class:`RecordedLaunch` kept for this launch."""
        return RecordedLaunch(
            attempt=attempt,
            cmd=tuple(str(c) for c in cmd),
            cwd=cwd,
            server_log_path=server_log_path,
            session_deadline_sec=session_deadline_sec,
            timeout=None if timeout is None else float(timeout),
            started_at=started,
            finished_at=self.clock.elapsed,
            returncode=returncode,
        )


def _label(attempt: LaunchAttempt) -> str:
    """Return the attempt's name, falling back to its outcome when unnamed."""
    return attempt.name or attempt.outcome
