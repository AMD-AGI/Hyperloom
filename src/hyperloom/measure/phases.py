# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where the time of a test goes: the phases a runner and the hosted backend stamp, and how they add up.

A test is about ten minutes of wall time, and the measured bench is a small part of it. The runner stamps what it does
(waiting for its sheet, checking it, rebuilding the environment, booting the server, each round, signing the result), the
hosted backend stamps what it does around the runner (starting the sandbox, landing the result, closing the sandbox),
and the executor reads the whole as ``TestResult.phases`` with the seconds nothing covers kept apart.

Two rules keep the seconds honest. A phase ends where the next begins, so the phases of one clock tile its span and a
mark that cannot be placed makes a phase coarser, never loses time. And clocks are never subtracted across machines: the
runner states its phases as lengths on its own clock, the host places them on its own by their order, and what is left of
the host's wait is a phase of its own (``runner_handoff``).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import NamedTuple

from hyperloom.contracts.measurement import PhaseTiming

__all__ = [
    "ACCURACY_EVAL",
    "ACQUIRED",
    "BOOTSTRAP",
    "BOOT_COMPILE",
    "BOOT_GRAPHS",
    "BOOT_OTHER",
    "BOOT_WEIGHTS",
    "CLIENT_START",
    "ENV_REBUILD",
    "EVAL_DEPS",
    "JIT_CACHE",
    "KERNEL_TABLES",
    "LANDED",
    "LONG_CONTEXT",
    "MEASURED",
    "OTHER",
    "PHASE_ORDER",
    "PREFLIGHT",
    "REPRO_CASE",
    "RESULT_LANDING",
    "RESULT_READ",
    "RESULT_SIGNING",
    "ROUND",
    "ROUND_COLLECT",
    "ROUND_SETUP",
    "RUNNER_DONE",
    "RUNNER_HANDOFF",
    "RUNNER_START",
    "RUNNER_WAIT",
    "SANDBOX_RELEASE",
    "SANDBOX_START",
    "SERVER_BOOT",
    "SHEET_WAIT",
    "SHORT_PROMPTS",
    "TEARDOWN",
    "WARMUP",
    "Accounted",
    "HostTimeline",
    "PhaseRecorder",
    "account",
    "manifest_document",
]

# -- the vocabulary ------------------------------------------------------------------------------------------------
# The host's own phases.
SANDBOX_START = "sandbox_start"
RUNNER_WAIT = "runner_wait"
RUNNER_HANDOFF = "runner_handoff"
RESULT_LANDING = "result_landing"
SANDBOX_RELEASE = "sandbox_release"
RESULT_READ = "result_read"
# The runner's phases, before the server boots, and after the last round.
BOOTSTRAP = "bootstrap"
RUNNER_START = "runner_start"
SHEET_WAIT = "sheet_wait"
PREFLIGHT = "preflight"
ENV_REBUILD = "env_rebuild"
KERNEL_TABLES = "kernel_tables"
JIT_CACHE = "jit_cache"
#: Installing the accuracy evaluator before the server boots (``measure.runner.eval_deps``).
EVAL_DEPS = "eval_deps"
RESULT_SIGNING = "result_signing"
# One round.
ROUND_SETUP = "round_setup"
SERVER_BOOT = "server_boot"
BOOT_OTHER = "server_boot.other"
BOOT_WEIGHTS = "server_boot.weights_load"
BOOT_COMPILE = "server_boot.compile"
BOOT_GRAPHS = "server_boot.graph_capture"
CLIENT_START = "client_start"
WARMUP = "warmup"
ACCURACY_EVAL = "accuracy_eval"
MEASURED = "measured"
ROUND_COLLECT = "round_collect"
TEARDOWN = "teardown"
REPRO_CASE = "repro_case"
#: The short-prompt consistency check's questions to a server it has just booted (the boots are ``server_boot`` as always).
SHORT_PROMPTS = "short_prompts"
#: The long-context check's questions to a server it has just booted (the boot is ``server_boot`` as always).
LONG_CONTEXT = "long_context"
#: A round whose marks cannot say where its parts began, and time between stamped phases that nothing stamps.
ROUND = "round"
OTHER = "other"

#: The phases in the order a test goes through them; a reader that lists them sorts by it.
PHASE_ORDER = (
    SANDBOX_START,
    BOOTSTRAP,
    RUNNER_START,
    SHEET_WAIT,
    PREFLIGHT,
    ENV_REBUILD,
    EVAL_DEPS,
    JIT_CACHE,
    KERNEL_TABLES,
    ROUND_SETUP,
    SERVER_BOOT,
    CLIENT_START,
    WARMUP,
    ACCURACY_EVAL,
    MEASURED,
    REPRO_CASE,
    SHORT_PROMPTS,
    LONG_CONTEXT,
    ROUND_COLLECT,
    TEARDOWN,
    ROUND,
    OTHER,
    RESULT_SIGNING,
    RUNNER_WAIT,
    RUNNER_HANDOFF,
    RESULT_LANDING,
    SANDBOX_RELEASE,
    RESULT_READ,
)

#: The instants the hosted backend marks on a :class:`HostTimeline`.
ACQUIRED = "acquired"
RUNNER_DONE = "runner_done"
LANDED = "landed"


class PhaseRecorder:
    """The phases of one clock, in order: :meth:`begin` ends the open phase where the next one starts.

    A phase that took no time is not recorded. A mark that lies before the phase it closes (a clock stepped back, a file
    time ahead of its neighbours) is moved up to it, so no phase has a negative length and the others keep their time.
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._done: list[PhaseTiming] = []
        self._open: tuple[str, float, int | None] | None = None

    def begin(self, name: str, *, at: float | None = None, point: int | None = None) -> None:
        """End the open phase and open ``name`` at ``at`` (now by default)."""
        now = self._clock() if at is None else at
        if self._open is not None:
            now = max(now, self._open[1])
        self._close(now)
        self._open = (name, now, point)

    def finish(self, at: float | None = None) -> tuple[PhaseTiming, ...]:
        """End the open phase and give every phase recorded."""
        self._close(self._clock() if at is None else at)
        self._open = None
        return self.phases

    @property
    def phases(self) -> tuple[PhaseTiming, ...]:
        """The phases that have ended; the open one is not among them."""
        return tuple(self._done)

    def _close(self, now: float) -> None:
        if self._open is None:
            return
        name, started, point = self._open
        seconds = max(0.0, now - started)
        if seconds > 0.0:
            self._done.append(PhaseTiming(name, started, seconds, point=point))


def manifest_document(phases: Sequence[PhaseTiming]) -> list[dict[str, object]]:
    """The phases as the signed manifest states them: lengths on the runner's own clock, no card time."""
    return [{"name": p.name, "started_unix": p.started_at, "seconds": p.seconds, "point": p.point} for p in phases]


# -- the host's side -------------------------------------------------------------------------------------------------


@dataclass
class HostTimeline:
    """What the hosted backend saw of one attempt on its own clock, and the phases that make of it.

    ``mark`` records the instants the backend passes: the sandbox acquired, the runner's result seen, the result landed.
    :meth:`compose` puts the runner's phases between the host's own.
    """

    clock: Callable[[], float] = time.time
    began: float = field(init=False)
    marks: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.began = self.clock()

    def mark(self, name: str) -> None:
        self.marks[name] = self.clock()

    def compose(self, runner: Sequence[PhaseTiming]) -> tuple[PhaseTiming, ...]:
        """The attempt's phases: the sandbox starting, the runner's (or one wait where it stated none), what is left of the
        host's wait, the result landing and the sandbox closing.

        The runner's phases are placed after the sandbox start by their order and lengths, back to back as the runner
        stamped them; their own clock's times are not used, so what a runner signs there cannot move them. The wait the host saw for the runner, less what the runner accounts for, is ``runner_handoff``: the
        sandbox's start command before the runner's first stamp, and the host polling for a result already written.
        """
        ended = self.clock()
        acquired = self.marks.get(ACQUIRED)
        if acquired is None:
            return (PhaseTiming(SANDBOX_START, self.began, max(0.0, ended - self.began)),)
        seen = self.marks.get(RUNNER_DONE)
        waited_until = ended if seen is None else seen
        phases = [PhaseTiming(SANDBOX_START, self.began, max(0.0, acquired - self.began))]
        phases.extend(self._runner_phases(runner, acquired, waited_until))
        if seen is not None:
            landed = self.marks.get(LANDED, ended)
            phases.append(PhaseTiming(RESULT_LANDING, seen, max(0.0, landed - seen)))
            phases.append(PhaseTiming(SANDBOX_RELEASE, landed, max(0.0, ended - landed)))
        return tuple(p for p in phases if p.seconds > 0.0)

    @staticmethod
    def _runner_phases(runner: Sequence[PhaseTiming], acquired: float, until: float) -> list[PhaseTiming]:
        waited = max(0.0, until - acquired)
        if not runner:
            return [PhaseTiming(RUNNER_WAIT, acquired, waited)]
        at, placed = acquired, []
        for phase in runner:
            placed.append(replace(phase, started_at=at))
            at += phase.seconds
        handoff = max(0.0, waited - sum(phase.seconds for phase in runner))
        return placed + [PhaseTiming(RUNNER_HANDOFF, at, handoff)]


class Accounted(NamedTuple):
    """A test's phases with the card time each held, and the seconds of its wall time they do not cover."""

    phases: tuple[PhaseTiming, ...]
    unaccounted_seconds: float


def account(phases: Sequence[PhaseTiming], wall: float, gpus: int) -> Accounted:
    """``phases`` with the card time each held, and the part of ``wall`` they leave uncovered.

    The cards a test holds are held by every phase, as ``TestResult.gpu_seconds`` counts them over the whole wall time.
    The uncovered part is negative when the phases total more than the wall time.
    """
    held = tuple(replace(phase, gpu_seconds=phase.seconds * gpus) for phase in phases)
    return Accounted(held, wall - sum(phase.seconds for phase in phases))
