# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The toolbox's side of the long-context check: when a test runs it, and what is recorded.

*When (phase 1).* A test runs the check when the toolbox has a staged prompt set (:class:`LongContextSetup`) and the request
sets ``long_context_check``. Which changes of a recipe make the toolbox ask for it itself, and that a final candidate always
does, come with the rule pack's lever list; a caller can add the check, never remove one the rule demands. A toolbox given no
setup runs no check and refuses nothing.

*Switched on.* A builder gives the toolbox the setup: from its options, or from the host environment's
``HYPERLOOM_LONG_CONTEXT`` (the key of the staged entry; :func:`environment_long_context`) when it names none. A builder whose
runners read this host's runner cache reads what it is given there first (:func:`staged_long_context`); a set that is asked
for and not staged is refused, never run without.

*What is recorded.* :func:`long_context_of` reads the landed ``outputs.json``, grades every answer by substring match and
states scores and outcomes (``LongContextResult``); it never says pass or fail. A prompt that errored or never ran counts as
not passed and makes the check ``incomplete``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from hyperloom.contracts.long_context import (
    LONG_CONTEXT_COMPLETE,
    LONG_CONTEXT_INCOMPLETE,
    PROMPT_OK,
    LongContextPrompt,
    LongContextResult,
    LongContextTask,
)
from hyperloom.contracts.measurement import TestKind, TestRequest
from hyperloom.measure.long_context.grade import passes
from hyperloom.measure.long_context.outputs import OutputsError, OutputsFile, PromptEntry, clean_reason, read_outputs
from hyperloom.measure.long_context.plan import LONG_CONTEXT_DIR, OUTPUTS_FILE, LongContextPlan
from hyperloom.measure.long_context.prompts import PromptSetError
from hyperloom.measure.long_context.resolve import StagedSet, staged_set

__all__ = [
    "LONG_CONTEXT_ENV",
    "LongContextNotStaged",
    "LongContextSetup",
    "decide",
    "environment_long_context",
    "first_difference",
    "long_context_of",
    "rederive",
    "staged_long_context",
]

#: The host environment variable that names the staged prompt set (its key) when a builder is given no setup.
LONG_CONTEXT_ENV = "HYPERLOOM_LONG_CONTEXT"

_CHECKED_KINDS = (TestKind.TEST, TestKind.SWEEP)
_FORCED = "request long_context_check"
_MAX_FAILED_SHOWN = 3
_NOT_IN_BUNDLE = frozenset({"trigger"})
_REBUILT = "rederived"


@dataclass(frozen=True)
class LongContextSetup:
    """The staged prompt set a deployment's checks use (``measure.long_context.build`` makes it): its key, the SHA-256 of its
    prompts file and the token length of its prompts. A toolbox that is given none does not run the check.

    Raises:
        ValueError: A value that no sheet could carry is refused here, not at the first test that would have run the check.
    """

    prompt_set: str
    prompts_sha256: str
    length: int

    def __post_init__(self) -> None:
        LongContextPlan(self.prompt_set, self.prompts_sha256, self.length)


class LongContextNotStaged(ValueError):
    """The check was asked for with a prompt set that is not staged where the runners read it; the message says why."""


def _setup_of(asked: str, read: Callable[[], StagedSet]) -> LongContextSetup:
    try:
        staged = read()
    except PromptSetError as exc:
        raise LongContextNotStaged(
            f"the long-context check is asked for with {asked}, which is not staged in the runner cache: {exc}"
        ) from exc
    return LongContextSetup(staged.prompt_set, staged.prompts_sha256, staged.length)


def environment_long_context(environ: Mapping[str, str]) -> LongContextSetup | None:
    """The setup ``HYPERLOOM_LONG_CONTEXT`` names in ``environ``, or ``None`` when it names none (a blank value names none).

    Raises:
        LongContextNotStaged: A key is named and its entry is not staged there, is unfinished or is not what its manifest
            recorded.
    """
    key = environ.get(LONG_CONTEXT_ENV, "").strip()
    return _setup_of(f"{LONG_CONTEXT_ENV}={key}", lambda: staged_set(key, environ)) if key else None


def staged_long_context(named: LongContextSetup | None, environ: Mapping[str, str]) -> LongContextSetup | None:
    """The setup a toolbox whose runners read the runner cache ``environ`` names runs the check with, or ``None`` for none.

    ``named`` is the setup the builder was given, read in that cache as a runner reads it; without one, the environment's.

    Raises:
        LongContextNotStaged: The set is asked for and is not staged there, is unfinished or is not the one named.
    """
    if named is None:
        return environment_long_context(environ)
    found = _setup_of(f"prompt set {named.prompt_set}", lambda: staged_set(named.prompt_set, environ))
    if found != named:
        raise LongContextNotStaged(
            f"the long-context set {named.prompt_set} is staged with other contents or another length than the setup names"
        )
    return found


def decide(request: TestRequest, setup: LongContextSetup | None) -> LongContextPlan | None:
    """The check ``request`` runs, or ``None``: the toolbox has a staged set and the caller forces the check."""
    if setup is None or request.kind not in _CHECKED_KINDS or not request.forces_long_context_check:
        return None
    return LongContextPlan(setup.prompt_set, setup.prompts_sha256, setup.length, trigger=(_FORCED,))


# --- what is recorded ----------------------------------------------------------------------------------------------


def _unread(plan: LongContextPlan, reason: str) -> LongContextResult:
    """A check whose outputs could not be read: no prompt, no score, and the reason."""
    return LongContextResult(
        status=LONG_CONTEXT_INCOMPLETE,
        reason=clean_reason(reason),
        prompt_set_sha256=plan.prompts_sha256,
        length=plan.length,
        max_new_tokens=plan.max_new_tokens,
        concurrency=plan.concurrency,
        served_context=None,
        n=0,
        passed=0,
        score=None,
        tasks=(),
        prompts=(),
        seconds=0.0,
        prefill_tokens=0,
        trigger=plan.trigger,
    )


def _prompt(entry: PromptEntry) -> LongContextPrompt:
    ok = entry.status == PROMPT_OK
    reached = (entry.prompt_tokens if entry.prompt_tokens is not None else entry.tokens) if ok else 0
    passed = ok and passes(entry.text, entry.references)
    return LongContextPrompt(entry.id, entry.task, entry.status, passed, reached, entry.seconds)


def _tasks(prompts: tuple[LongContextPrompt, ...]) -> tuple[LongContextTask, ...]:
    counts: dict[str, list[int]] = {}
    for prompt in prompts:
        total = counts.setdefault(prompt.task, [0, 0])
        total[0] += 1
        total[1] += prompt.passed
    return tuple(LongContextTask(task, n, passed, passed / n) for task, (n, passed) in counts.items())


def _why_incomplete(found: OutputsFile) -> str:
    unanswered = [entry for entry in found.prompts if entry.status != PROMPT_OK]
    if not unanswered:
        return ""
    if found.stopped:
        return found.stopped
    shown = "; ".join(f"{e.id}: {e.reason or e.status}" for e in unanswered[:_MAX_FAILED_SHOWN])
    return clean_reason(f"{len(unanswered)} of {len(found.prompts)} prompts were not answered ({shown})")


def _result(plan: LongContextPlan, found: OutputsFile) -> LongContextResult:
    prompts = tuple(map(_prompt, found.prompts))
    reason = _why_incomplete(found)
    passed = sum(prompt.passed for prompt in prompts)
    return LongContextResult(
        status=LONG_CONTEXT_INCOMPLETE if reason else LONG_CONTEXT_COMPLETE,
        reason=reason,
        prompt_set_sha256=found.prompts_sha256,
        length=found.length,
        max_new_tokens=found.max_new_tokens,
        concurrency=found.concurrency,
        served_context=found.served_context,
        n=len(prompts),
        passed=passed,
        score=passed / len(prompts),
        tasks=_tasks(prompts),
        prompts=prompts,
        seconds=found.query_seconds,
        prefill_tokens=sum(prompt.prompt_tokens for prompt in prompts),
        trigger=plan.trigger,
        outputs=f"{LONG_CONTEXT_DIR}/{OUTPUTS_FILE}",
    )


def long_context_of(plan: LongContextPlan, directory: Path) -> LongContextResult:
    """The check of a test whose output landed in ``directory``, as the result records it.

    Whatever the bundle lacks makes the check ``incomplete`` with the reason, never a pass: no outputs file, outputs of another
    prompt set, a prompt that errored or was never sent, a launch that serves too little context.
    """
    try:
        found = read_outputs(directory)
    except OutputsError as exc:
        return _unread(plan, str(exc))
    if (found.prompt_set, found.prompts_sha256, found.length) != (plan.prompt_set, plan.prompts_sha256, plan.length):
        return _unread(plan, "the bundle's outputs are for another prompt set than the one the test asked for")
    return _result(plan, found)


def rederive(recorded: LongContextResult, directory: Path) -> LongContextResult:
    """The check ``recorded`` claims, derived again from the outputs file of the verified bundle at ``directory`` alone."""
    plan = LongContextPlan(
        _REBUILT,
        recorded.prompt_set_sha256,
        recorded.length,
        max_new_tokens=recorded.max_new_tokens,
        concurrency=recorded.concurrency,
        trigger=recorded.trigger,
    )
    try:
        found = read_outputs(directory)
    except OutputsError as exc:
        return _unread(plan, str(exc))
    return _result(plan, found)


def first_difference(recorded: LongContextResult, derived: LongContextResult) -> str | None:
    """The first field of ``recorded`` that ``derived`` (from the verified bundle) does not support, or ``None``."""
    for field in fields(LongContextResult):
        if field.name not in _NOT_IN_BUNDLE and getattr(recorded, field.name) != getattr(derived, field.name):
            return (
                f"long_context.{field.name}: recorded {getattr(recorded, field.name)!r}, "
                f"the bundle's outputs give {getattr(derived, field.name)!r}"
            )
    return None
