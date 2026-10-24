# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pass of the check: one server of the test's own launch answers every prompt, and what it answered is recorded.

How a server is started and stopped is the backend's (``short_prompts.rounds.PassHost``: the local backend boots the point's
own launch as a boot-only round). This module owns the order, the time budget and what is recorded. A launch that serves
less context than the prompts need is refused before a server starts; a server that does not come up, a request that fails
and a phase cap that leaves prompts unsent each end the check ``incomplete`` with the reason, never as a pass.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

from hyperloom.contracts.long_context import PROMPT_ERROR, PROMPT_NOT_RUN, PROMPT_OK
from hyperloom.measure.long_context.client import complete_text
from hyperloom.measure.long_context.outputs import OutputsFile, PromptEntry, RunFacts, read_outputs, write_outputs
from hyperloom.measure.long_context.plan import LongContextPlan
from hyperloom.measure.long_context.prompts import Prompt
from hyperloom.measure.short_prompts.client import CompletionError, request_timeout
from hyperloom.measure.short_prompts.rounds import PassFailed, PassHost, Serving, model_of

__all__ = ["run_long_context"]

_PHASE_CAP = "the phase cap ended the check before every prompt was sent"
_REQUEST_CAP = "the phase cap ended this prompt's request"


def _unanswered(prompt: Prompt, status: str, reason: str, seconds: float = 0.0) -> PromptEntry:
    return PromptEntry(prompt.id, prompt.task, prompt.references, prompt.tokens, status, reason, seconds=seconds)


def _ask(plan: LongContextPlan, serving: Serving, model: str, deadline: float, prompt: Prompt) -> PromptEntry:
    """One prompt's entry: answered, failed, or not sent because the deadline had passed."""
    began = time.monotonic()
    if began >= deadline:
        return _unanswered(prompt, PROMPT_NOT_RUN, _PHASE_CAP)
    try:
        answer = complete_text(
            serving.base_url,
            model,
            prompt.text,
            max_new_tokens=plan.max_new_tokens,
            timeout_sec=request_timeout(deadline),
        )
    except CompletionError as exc:
        reason = _REQUEST_CAP if time.monotonic() >= deadline else str(exc)
        return _unanswered(prompt, PROMPT_ERROR, reason, time.monotonic() - began)
    return PromptEntry(
        prompt.id, prompt.task, prompt.references, prompt.tokens, PROMPT_OK, "", answer.text, answer.finish_reason,
        answer.prompt_tokens, time.monotonic() - began,
    )  # fmt: skip


def _ask_all(plan: LongContextPlan, prompts: Sequence[Prompt], serving: Serving, deadline: float) -> list[PromptEntry]:
    """Every prompt's entry, ``plan.concurrency`` requests at a time, in the order of ``prompts``."""
    model = model_of(serving, deadline)
    with ThreadPoolExecutor(max_workers=plan.concurrency) as pool:
        return list(pool.map(lambda prompt: _ask(plan, serving, model, deadline, prompt), prompts))


def _too_short(plan: LongContextPlan, served: int | None) -> str:
    if served is None or served >= plan.needed_context:
        return ""
    return (
        f"served context too short: the launch serves {served} tokens, the check needs {plan.needed_context} "
        f"({plan.length} for the prompt and {plan.needed_context - plan.length} for the answer)"
    )


class _Answered(NamedTuple):
    """What a server gave the check: the entries, why it ended early (or ``""``) and the seconds the server took to come up."""

    entries: list[PromptEntry]
    stopped: str
    boot_seconds: float


async def _answered(plan: LongContextPlan, prompts: Sequence[Prompt], host: PassHost, deadline: float) -> _Answered:
    """Start the server, ask it every prompt before ``deadline`` (a ``time.monotonic()`` instant) and stop it."""
    started = time.monotonic()
    try:
        serving = await host.start(0, deadline)
        booted = time.monotonic() - started
        entries = await asyncio.to_thread(_ask_all, plan, prompts, serving, deadline)
    except (PassFailed, CompletionError) as exc:
        return _Answered([], str(exc), time.monotonic() - started)
    except Exception as exc:  # noqa: BLE001 - an unforeseen end is the check's incomplete, not the test's failure
        return _Answered([], f"{type(exc).__name__} while running the check", time.monotonic() - started)
    finally:
        await host.stop()
    unsent = any(entry.status == PROMPT_NOT_RUN for entry in entries)
    return _Answered(entries, _PHASE_CAP if unsent else "", booted)


async def run_long_context(
    plan: LongContextPlan, prompts: Sequence[Prompt], host: PassHost, output_dir: Path, *, served_context: int | None
) -> OutputsFile:
    """Start one server, ask it every prompt, write ``outputs.json`` below ``output_dir`` and return what was written.

    ``served_context`` is the context the point's launch serves, when it states one: a launch that serves less than
    ``plan.needed_context`` starts no server.
    """
    started = time.monotonic()
    stopped = _too_short(plan, served_context)
    if not stopped and host.should_stop():
        stopped = "the run was stopped"
    answered = _Answered([], stopped, 0.0)
    if not stopped:
        answered = await _answered(plan, prompts, host, started + plan.phase_cap_sec)
    entries = answered.entries or [_unanswered(prompt, PROMPT_NOT_RUN, answered.stopped) for prompt in prompts]
    spent = max(0.0, time.monotonic() - started - answered.boot_seconds)
    facts = RunFacts(served_context, answered.stopped, answered.boot_seconds, spent)
    write_outputs(output_dir, plan, entries, facts)
    return read_outputs(output_dir)
