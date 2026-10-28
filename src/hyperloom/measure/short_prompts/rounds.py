# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The check's one process: the test's own server, asked after its measured rounds and before its teardown.

The server is the one the test itself brought up (:func:`answer_held`); the check boots none, so it costs the seconds the
prompts take. How the server is reached is the backend's (``PassHost``). This module owns the time cap and what is
recorded: a server that cannot be reached, or fails to answer a prompt, is recorded as failed with its reason, and a failed
process is never a clean one. :func:`write_answered` leaves the answers in ``outputs.json``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hyperloom.measure.short_prompts.client import CompletionError, discover_model, request_timeout, run_prompts
from hyperloom.measure.short_prompts.outputs import FAILED, OK, ProcessEntry, completions_to_outputs, write_outputs
from hyperloom.measure.short_prompts.plan import ShortPromptPlan
from hyperloom.measure.short_prompts.prompts import Prompt

__all__ = [
    "Answered",
    "PassFailed",
    "PassHost",
    "Serving",
    "answer_held",
    "model_of",
    "write_answered",
]


class PassFailed(Exception):
    """A server that could not be reached; the message is the reason recorded for it."""


@dataclass(frozen=True)
class Serving:
    """A running server: where it listens, and the model name to fall back on when it does not list its own."""

    base_url: str
    model: str = ""


class PassHost(Protocol):
    """Gives a check the test's own server; it starts no server of its own."""

    async def start(self, index: int, deadline: float) -> Serving:
        """The server of process ``index`` (the test's own: 0), to answer before ``deadline`` (a ``time.monotonic()``
        instant).

        Raises:
            PassFailed: There is no server to ask.
        """
        ...

    async def stop(self) -> None:
        """Release what ``start`` gave; called after every start, also one that raised."""
        ...

    def should_stop(self) -> bool:
        """Whether the run is being stopped, so the check is to end."""
        ...


@dataclass(frozen=True)
class _Budget:
    """The check's time: when it must end on the clock :func:`answer_held` was given."""

    deadline: float
    clock: Callable[[], float]

    def monotonic_deadline(self) -> float:
        """The same instant on ``time.monotonic()``, which the requests are timed by."""
        return time.monotonic() + (self.deadline - self.clock())


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


@dataclass(frozen=True)
class Answered:
    """Process 0 of a check: the plan, the prompts it was asked, and its entry (the test's own server's, or a failed one
    that says why that server could not answer)."""

    plan: ShortPromptPlan
    prompts: Sequence[Prompt]
    entry: ProcessEntry


async def answer_held(
    plan: ShortPromptPlan, prompts: Sequence[Prompt], host: PassHost, *, clock: Callable[[], float] = time.monotonic
) -> Answered:
    """Process 0: ``host`` is the test's own server, still up after its measured rounds; it answers every prompt within the
    plan's ``phase_cap_sec``. A host that cannot give the server makes the process failed with its reason."""
    entry = await _one_process(plan, prompts, host, 0, _Budget(clock() + plan.phase_cap_sec, clock))
    return Answered(plan, prompts, entry)


def write_answered(answered: Answered, output_dir: Path) -> ProcessEntry:
    """Write ``outputs.json`` below ``output_dir`` (the test's own output directory) for ``answered``, the test's own
    server's answers (:func:`answer_held`), and return its entry."""
    write_outputs(output_dir, answered.plan, [prompt.info() for prompt in answered.prompts], [answered.entry])
    return answered.entry
