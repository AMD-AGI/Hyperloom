# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The accuracy checks a test runs on its own server, after its measured rounds and before the server's teardown.

Both checks judge the recipe's answers, not one server process, so the server the test measured on is as good as a fresh
one and costs no boot (InferenceX already evaluates on the measured server: ``local._eval_then_measure``). The local backend
holds the server up past its last measured round (Magpie would stop it at the end of that round) and runs
:func:`run_held_checks` before the teardown: the short-prompt check (``short_prompts.rounds.answer_held``) and the
long-context check. Neither boots a server of its own.

Before the checks the server's prefix cache, and with it HiCache's host tier, is flushed where the framework exposes a
flush (sglang's ``/flush_cache``). A server that exposes none, or refuses the flush, keeps its cache: the checks' prompts
are then disjoint from what the benchmark sent, being the staged synthetic sets, which share at most a chat template's
first tokens with a trace. Which of the two held is a note on the test.

A test whose server is not held up (a framework the lifecycle cannot hold, a fallback that restarted the server per level)
runs the checks on no server: :class:`NoServer` makes each one record the reason, never a pass and never a second boot.
"""

from __future__ import annotations

import asyncio
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from hyperloom.measure.eval_context import eval_context, served_context
from hyperloom.measure.long_context import staging as long_context_staging
from hyperloom.measure.long_context.rounds import run_long_context
from hyperloom.measure.phases import LONG_CONTEXT, OTHER, SHORT_PROMPTS, PhaseRecorder
from hyperloom.measure.short_prompts.outputs import FAILED, ProcessEntry, clean_reason
from hyperloom.measure.long_context.plan import LongContextPlan
from hyperloom.measure.short_prompts.plan import ShortPromptPlan
from hyperloom.measure.short_prompts.rounds import Answered, PassFailed, PassHost, Serving, answer_held
from hyperloom.measure.short_prompts.staging import load_prompt_set

__all__ = [
    "FLUSH_PATHS",
    "HeldAnswers",
    "HeldChecks",
    "HeldServer",
    "NoServer",
    "flush_prefix_cache",
    "long_context_unheld",
    "not_held",
    "run_held_checks",
    "served_by",
]

#: The endpoint that drops a framework's prefix cache, by the framework's name; a framework not named exposes none.
FLUSH_PATHS = {"sglang": "/flush_cache"}
FLUSHED = "prefix cache flushed before the accuracy checks"
_NOT_FLUSHED = "prefix cache not flushed before the accuracy checks ({why}); their prompts are disjoint from the trace"
_FLUSH_TIMEOUT_SEC = 60.0
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def flush_prefix_cache(base_url: str, framework: str) -> str:
    """Ask the server at ``base_url`` to drop its prefix cache; the note that says whether it did, and why not."""
    path = FLUSH_PATHS.get(framework)
    if path is None:
        return _NOT_FLUSHED.format(why=f"{framework or 'the framework'} exposes no flush")
    request = urllib.request.Request(base_url + path, data=b"", method="POST")
    try:
        with _OPENER.open(request, timeout=_FLUSH_TIMEOUT_SEC) as reply:
            reply.read()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return _NOT_FLUSHED.format(why=clean_reason(f"{path} answered {type(exc).__name__}: {exc}"))
    return FLUSHED


@dataclass(frozen=True)
class HeldServer:
    """The test's own server, still up: where it listens, the framework serving it, and whether the run is being stopped.
    A check's host (``short_prompts.rounds.PassHost``) whose start starts nothing and whose stop stops nothing."""

    serving: Serving
    framework: str
    stopping: Callable[[], bool]

    async def start(self, index: int, deadline: float) -> Serving:
        return self.serving

    async def stop(self) -> None:
        return None

    def should_stop(self) -> bool:
        return self.stopping()


@dataclass(frozen=True)
class NoServer:
    """The host of a check whose test held no server up: every start fails with ``reason``."""

    reason: str

    async def start(self, index: int, deadline: float) -> Serving:
        raise PassFailed(self.reason)

    async def stop(self) -> None:
        return None

    def should_stop(self) -> bool:
        return False


def served_by(envs: dict[str, Any], carried: str | None) -> int | None:
    """The tokens of context a launch serves, from its point's ``envs`` and the server arguments ``carried`` there (as
    ``points.launch_carriers`` gives them): its context flag, else its ``MAX_MODEL_LEN`` (as ``eval_context`` reads them);
    ``None`` when it states neither, which leaves the server's own default, and the requests then find out."""
    flag = served_context(carried or "")
    return flag.tokens if flag is not None else eval_context(envs)


@dataclass(frozen=True)
class HeldChecks:
    """The checks a test plans (``None`` for one it does not), the test's output directory, where their outputs go, and
    the context its launch serves (``None`` when it states none)."""

    short_prompts: ShortPromptPlan | None
    long_context: LongContextPlan | None
    output_dir: Path
    served_context: int | None


def not_held(checks: HeldChecks, reason: str) -> Answered | None:
    """The short-prompt check's process 0 of a test whose server was not held up for it; ``None`` when it plans none."""
    plan = checks.short_prompts
    if plan is None:
        return None
    return Answered(plan, load_prompt_set(plan, os.environ), ProcessEntry(0, FAILED, reason, 0.0, 0.0, {}))


@dataclass(frozen=True)
class HeldAnswers:
    """What the checks on the held server left besides their files: the short-prompt check's process 0 (``None`` when the
    test plans no short-prompt check) and the flush's note."""

    short: Answered | None
    flush: str


async def run_held_checks(checks: HeldChecks, server: HeldServer, phases: PhaseRecorder) -> HeldAnswers:
    """Flush the server's prefix cache, then ask it the ``checks``. The long-context check writes its outputs below the
    test's output directory here; the short-prompt check's answers are returned, to be written after the rounds.
    The phases are the checks'."""
    phases.begin(SHORT_PROMPTS if checks.short_prompts is not None else LONG_CONTEXT)
    flush = await asyncio.to_thread(flush_prefix_cache, server.serving.base_url, server.framework)
    short = None
    if checks.short_prompts is not None:
        short = await answer_held(checks.short_prompts, load_prompt_set(checks.short_prompts, os.environ), server)
    if checks.long_context is not None:
        if short is not None:
            phases.begin(LONG_CONTEXT)
        await _long_context(checks.long_context, checks, server)
    phases.begin(OTHER)
    return HeldAnswers(short, flush)


async def long_context_unheld(checks: HeldChecks, reason: str) -> None:
    """The long-context check of a test whose server was not held up for it: its outputs say ``reason``, no server starts."""
    if checks.long_context is not None:
        await _long_context(checks.long_context, replace(checks, served_context=None), NoServer(reason))


async def _long_context(plan: LongContextPlan, checks: HeldChecks, host: PassHost) -> None:
    prompts = long_context_staging.load_prompt_set(plan, os.environ)
    await run_long_context(plan, prompts, host, checks.output_dir, served_context=checks.served_context)
