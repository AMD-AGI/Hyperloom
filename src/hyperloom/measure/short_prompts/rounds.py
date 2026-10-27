# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The passes of the check: start a fresh server process, ask it every prompt, stop it, once per planned process.

How a server is started and stopped is the backend's (``PassHost``): the local backend boots the point's own launch the way
a ``repro`` test does. This module owns the order, the time budget and what is recorded. A process that fails to start, or
fails to answer a prompt, is recorded as failed with its reason and the next one is tried; the check stops early, and says
why, when the phase cap leaves no room for another process. A failed or skipped process is never a clean one.

The check runs after the test's own rounds, so the test's server has booted and its ready stamp below the test's output
directory says how long that took. The cap is sized from it: ``processes * (boot + answers) * (1 + CAP_MARGIN_FRACTION) +
CAP_MARGIN_SEC``, ``answers`` being the longest a process of this check took to answer the prompts (nothing before the
first has), and ``boot`` the longest boot measured (the test's, then each process's). It is sized again after every
process. The plan's ``phase_cap_sec`` is the cap only while no boot is measured. A model whose server boots slowly would
fit one or two processes in a fixed cap, and on a fast one a hung process would hold the cards long after the others.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hyperloom.measure.boot_watch import boot_seconds
from hyperloom.measure.short_prompts.client import CompletionError, discover_model, request_timeout, run_prompts
from hyperloom.measure.short_prompts.outputs import FAILED, OK, ProcessEntry, completions_to_outputs, write_outputs
from hyperloom.measure.short_prompts.plan import ShortPromptPlan
from hyperloom.measure.short_prompts.prompts import Prompt

__all__ = ["CAP_MARGIN_FRACTION", "CAP_MARGIN_SEC", "PassFailed", "PassHost", "Serving", "model_of", "run_passes"]

_PHASE_CAP_STOP = "the phase cap leaves no room for another process"
_RUN_STOPPED = "the run was stopped"
#: The margin a cap sized from measured times leaves over them: a share of them, and a fixed time on top.
CAP_MARGIN_FRACTION = 0.25
CAP_MARGIN_SEC = 120.0

_LOG = logging.getLogger(__name__)


class PassFailed(Exception):
    """A server process that could not be started; the message is the reason recorded for it."""


@dataclass(frozen=True)
class Serving:
    """A running server: where it listens, and the model name to fall back on when it does not list its own."""

    base_url: str
    model: str = ""


class PassHost(Protocol):
    """Starts and stops the server processes of a check."""

    async def start(self, index: int, deadline: float) -> Serving:
        """Start process ``index``, to be ready before ``deadline`` (a ``time.monotonic()`` instant).

        Raises:
            PassFailed: The process did not come up.
        """
        ...

    async def stop(self) -> None:
        """Stop the process ``start`` started; called after every start, also one that raised."""
        ...

    def should_stop(self) -> bool:
        """Whether the run is being stopped, so no further process is to be started."""
        ...


@dataclass(frozen=True)
class _Budget:
    """The check's time: when it must end on the clock ``run_passes`` was given."""

    deadline: float
    clock: Callable[[], float]

    def monotonic_deadline(self) -> float:
        """The same instant on ``time.monotonic()``, which the requests are timed by."""
        return time.monotonic() + (self.deadline - self.clock())


@dataclass
class _Measured:
    """The longest server boot and prompt-answering time measured so far, which the check's cap is sized from."""

    boot: float | None
    answers: float = 0.0

    def saw(self, entry: ProcessEntry) -> None:
        """A process of the check that answered every prompt measured a boot and an answering time."""
        if entry.status != OK:
            return
        boot = _positive(entry.boot_seconds)
        if boot is not None:
            self.boot = max(self.boot or 0.0, boot)
        self.answers = max(self.answers, _positive(entry.query_seconds) or 0.0)

    def cap(self, plan: ShortPromptPlan) -> float:
        """The check's cap: sized from the measured times once a boot is measured, the plan's until then."""
        if self.boot is None:
            return plan.phase_cap_sec
        return plan.processes * (self.boot + self.answers) * (1 + CAP_MARGIN_FRACTION) + CAP_MARGIN_SEC


def _positive(seconds: float | None) -> float | None:
    return seconds if seconds is not None and math.isfinite(seconds) and seconds > 0 else None


def model_of(serving: Serving, deadline: float) -> str:
    try:
        return discover_model(serving.base_url, request_timeout(deadline))
    except CompletionError:
        if serving.model:
            return serving.model
        raise


async def _answer(plan: ShortPromptPlan, prompts: Sequence[Prompt], serving: Serving, budget: _Budget) -> dict:
    deadline = budget.monotonic_deadline()
    model = await asyncio.to_thread(model_of, serving, deadline)
    return await asyncio.to_thread(
        run_prompts, serving.base_url, model, prompts, max_new_tokens=plan.max_new_tokens, deadline=deadline
    )


async def _one_process(
    plan: ShortPromptPlan, prompts: Sequence[Prompt], host: PassHost, index: int, budget: _Budget
) -> ProcessEntry:
    clock = budget.clock
    started = clock()
    booted = started
    try:
        serving = await host.start(index, budget.deadline)
        booted = clock()
        answers = await _answer(plan, prompts, serving, budget)
    except (PassFailed, CompletionError) as exc:
        return ProcessEntry(index, FAILED, str(exc), booted - started, clock() - booted, {})
    except Exception as exc:  # noqa: BLE001 - one process's unforeseen end is that process's failure, not the test's
        reason = f"{type(exc).__name__} while running the process"
        return ProcessEntry(index, FAILED, reason, booted - started, clock() - booted, {})
    finally:
        await host.stop()
    return ProcessEntry(index, OK, "", booted - started, clock() - booted, completions_to_outputs(answers))


async def run_passes(
    plan: ShortPromptPlan,
    prompts: Sequence[Prompt],
    host: PassHost,
    output_dir: Path,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[ProcessEntry, ...]:
    """Run the planned processes one after another, write ``outputs.json`` below ``output_dir`` and return the entries.

    ``output_dir`` is the test's own output directory, whose ready stamps say how long the test's server took to boot.
    ``clock`` is the ``time.monotonic()`` that the deadline handed to the host is on.
    """
    started = clock()
    measured = _Measured(_positive(boot_seconds(Path(output_dir))))
    budget = _Budget(started + measured.cap(plan), clock)
    _LOG.info(
        "short-prompt check capped at %.0f s (the plan's cap is %.0f s)", budget.deadline - started, plan.phase_cap_sec
    )
    infos = [prompt.info() for prompt in prompts]
    entries: list[ProcessEntry] = []
    longest = 0.0
    stopped = ""
    for index in range(plan.processes):
        left = budget.deadline - clock()
        if host.should_stop():
            stopped = _RUN_STOPPED
        elif left <= 0 or (entries and left < longest):
            stopped = _PHASE_CAP_STOP
        if stopped:
            break
        began = clock()
        entries.append(await _one_process(plan, prompts, host, index, budget))
        longest = max(longest, clock() - began)
        measured.saw(entries[-1])
        budget = _Budget(started + measured.cap(plan), clock)
        write_outputs(output_dir, plan, infos, "", entries)
    write_outputs(output_dir, plan, infos, stopped, entries)
    return tuple(entries)
