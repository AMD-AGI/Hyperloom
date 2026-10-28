# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The toolbox's side of the long-context check: when a test runs it, and what is recorded.

*When.* The same lever list as the short-prompt check, with no second list (#142 c3016): a test runs the check when the
change from its parent recipe to its candidate touches what the rule pack's ``short_prompt_touch`` matches
(``short_prompts.lever_lines``), when its purpose is one of the pack's ``always_purposes`` (a final candidate, and a
baseline, whose checks are the root's reading every candidate is judged after; no test is added for it), or when the request
sets ``long_context_check``. A caller can add the check, never remove one the rule demands. A toolbox given no setup runs no
check and refuses nothing. The check runs on the test's own server after its measured rounds, at the context the canonical
launch serves (the launch is not widened for it).

*Switched on.* The check is on by default. A builder gives the toolbox the setup: from its options, or from the host
environment's ``HYPERLOOM_LONG_CONTEXT`` (the key of the staged entry; ``off`` turns the check off) when it names none, and
without either the set ``measure.long_context.build`` stages for the session model by default (its tokenizer, its own
context limit and the default seed; :func:`default_long_context`), staged at the build when the builder's runners read this
host's cache. A launch that serves less context than that set needs makes the check ``incomplete`` with the reason; a set
built with ``--served-context`` is named by its key. A set that is asked for, or found and not readable, is refused, never
run without.

*What is recorded.* :func:`long_context_of` reads the landed ``outputs.json``, grades every answer by substring match and
states scores and outcomes (``LongContextResult``); it never says pass or fail. A prompt that errored or never ran counts as
not passed and makes the check ``incomplete``, which never blocks. The toolbox adds the root's reading (``root``) when the
test ends; ``hyperloom.contracts.long_context.refuses`` is the bar the rule gate applies to them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path

from hyperloom.contracts.long_context import (
    LONG_CONTEXT_COMPLETE,
    LONG_CONTEXT_INCOMPLETE,
    PROMPT_OK,
    LongContextPrompt,
    LongContextResult,
    LongContextTask,
    root_reading,
)
from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest, TestResult
from hyperloom.contracts.recipe import Recipe
from hyperloom.inference_optimizer.model_config_utils import load_model_max_position_embeddings
from hyperloom.measure.long_context.build import DEFAULT_SEED, build_entry, set_length
from hyperloom.measure.long_context.grade import passes
from hyperloom.measure.long_context.outputs import OutputsError, OutputsFile, PromptEntry, clean_reason, read_outputs
from hyperloom.measure.long_context.plan import LONG_CONTEXT_DIR, OUTPUTS_FILE, LongContextPlan
from hyperloom.measure.long_context.prompts import GENERATOR, PromptSetError
from hyperloom.measure.long_context.resolve import StagedSet, staged_set
from hyperloom.measure.long_context.staging import KEY_PREFIX
from hyperloom.measure.short_prompts.build import tokenizer_hashes
from hyperloom.measure.short_prompts.staging import EVALSETS_DIR, cache_root, entry_key
from hyperloom_multibranch.toolbox.submission_log import LogEntry
from hyperloom_multibranch.toolbox.short_prompts import (
    MODEL_PATH_ENV,
    PatchFiles,
    bounded_reasons,
    lever_lines,
    purpose_reasons,
)

__all__ = [
    "LONG_CONTEXT_ENV",
    "LONG_CONTEXT_OFF",
    "LongContextNotStaged",
    "LongContextSetup",
    "decide",
    "default_long_context",
    "environment_long_context",
    "first_difference",
    "long_context_of",
    "rederive",
    "staged_long_context",
    "with_root_reading",
]

#: The host environment variable that names the staged prompt set (its key) when a builder is given no setup.
LONG_CONTEXT_ENV = "HYPERLOOM_LONG_CONTEXT"
#: The value of ``HYPERLOOM_LONG_CONTEXT`` that turns the check off where it would be on by default.
LONG_CONTEXT_OFF = "off"

_LOG = logging.getLogger(__name__)
_NOT_STAGED = "long-context check off: the model's prompt set %s is not staged in the runner cache: %s"

_CHECKED_KINDS = (TestKind.TEST, TestKind.SWEEP)
_FORCED = "request long_context_check"
_MAX_FAILED_SHOWN = 3
#: What a check records that its bundle cannot show: why it ran, and the root's reading, made from other tests' bundles.
_NOT_IN_BUNDLE = frozenset({"trigger", "root"})
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


def default_long_context(
    model_dir: Path | None, environ: Mapping[str, str], *, stage: bool = False
) -> LongContextSetup | None:
    """The set ``measure.long_context.build`` stages for the model in ``model_dir`` (else the directory ``MODEL_PATH`` in
    ``environ`` names) when it is given no length: the model's tokenizer, its own context limit and the default seed, read in
    the runner cache ``environ`` names. ``None``, and a log line that says why, when there is no model directory, its
    tokenizer or context limit cannot be read, or the set is not staged; with ``stage`` (a toolbox whose runners read the
    same cache) a set that is not staged is staged first, and the check is off with a warning when that fails.

    Raises:
        LongContextNotStaged: The set is there and cannot be read: a set that was found is not a check that is off.
    """
    directory = model_dir or (Path(environ[MODEL_PATH_ENV]) if environ.get(MODEL_PATH_ENV) else None)
    if directory is None:
        _LOG.info("long-context check off: no model directory to find its prompt set by")
        return None
    try:
        length = set_length(load_model_max_position_embeddings(str(directory)), None)
        inputs = {"seed": DEFAULT_SEED, "length": length, "tokenizer": tokenizer_hashes(directory)}
    except (PromptSetError, OSError) as exc:
        _LOG.warning("long-context check off: the model's tokenizer or context limit cannot be read: %s", exc)
        return None
    key = entry_key(inputs, generator=GENERATOR, prefix=KEY_PREFIX)
    if not (cache_root(environ) / EVALSETS_DIR / key).exists():
        if not stage:
            _LOG.warning(_NOT_STAGED, key, "the toolbox does not stage it")
            return None
        if not _staged_now(directory, environ, key, length):
            return None
    return _setup_of(f"the model's prompt set {key}", lambda: staged_set(key, environ))


def _staged_now(model_dir: Path, environ: Mapping[str, str], key: str, length: int) -> bool:
    """Stage the model's default prompt set (``key``, at ``length`` tokens) in the runner cache ``environ`` names; whether
    it was, the reason logged when it was not."""
    try:
        build_entry(model_dir, cache_root(environ), length=length)
    except Exception as exc:  # noqa: BLE001 - the tokenizer library raises anything; the check is off and says why
        _LOG.warning(_NOT_STAGED, key, f"staging it failed: {type(exc).__name__}: {exc}")
        return False
    _LOG.info("long-context check: staged the model's prompt set %s in the runner cache", key)
    return True


def environment_long_context(
    environ: Mapping[str, str], model_dir: Path | None = None, *, stage: bool = False
) -> LongContextSetup | None:
    """The setup ``HYPERLOOM_LONG_CONTEXT`` names in ``environ``: none for ``off``, the model's own set
    (:func:`default_long_context`) when it names none (a blank value names none).

    Raises:
        LongContextNotStaged: A key is named and its entry is not staged there, is unfinished or is not what its manifest
            recorded; or the model's own set is there and cannot be read.
    """
    key = environ.get(LONG_CONTEXT_ENV, "").strip()
    if key == LONG_CONTEXT_OFF:
        return None
    if not key:
        return default_long_context(model_dir, environ, stage=stage)
    return _setup_of(f"{LONG_CONTEXT_ENV}={key}", lambda: staged_set(key, environ))


def staged_long_context(
    named: LongContextSetup | None, environ: Mapping[str, str], model_dir: Path | None = None, *, stage: bool = False
) -> LongContextSetup | None:
    """The setup a toolbox whose runners read the runner cache ``environ`` names runs the check with, or ``None`` for none.

    ``named`` is the setup the builder was given, read in that cache as a runner reads it; without one, the environment's
    (:func:`environment_long_context`, with the session's ``model_dir``).

    Raises:
        LongContextNotStaged: The set is asked for and is not staged there, is unfinished or is not the one named.
    """
    if named is None:
        return environment_long_context(environ, model_dir, stage=stage)
    found = _setup_of(f"prompt set {named.prompt_set}", lambda: staged_set(named.prompt_set, environ))
    if found != named:
        raise LongContextNotStaged(
            f"the long-context set {named.prompt_set} is staged with other contents or another length than the setup names"
        )
    return found


def decide(
    request: TestRequest, parent: Recipe, candidate: Recipe, setup: LongContextSetup | None, files: PatchFiles
) -> LongContextPlan | None:
    """The check ``request`` runs, or ``None``: the toolbox has a staged set and the request's purpose is one the pack
    always checks, the caller forces the check, or the change from ``parent`` to ``candidate`` touches the pack's lever
    list (``files`` reads the files a patch changes): the short-prompt check's triggers, read from the same entries."""
    if setup is None or request.kind not in _CHECKED_KINDS:
        return None
    reasons = purpose_reasons(request)
    if request.forces_long_context_check:
        reasons.append(_FORCED)
    reasons += lever_lines(parent, candidate, files)
    if not reasons:
        return None
    return LongContextPlan(setup.prompt_set, setup.prompts_sha256, setup.length, trigger=bounded_reasons(reasons))


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


def _root_checks(entries: Iterable[LogEntry]) -> list[tuple[str, LongContextResult]]:
    """The long-context checks of the root's tests (the BASELINE tests) among ``entries``, in the order they were
    accepted, by request id."""
    checks = []
    for entry in entries:
        if entry.aborted or entry.outcome is None or entry.request.get("purpose") != Purpose.BASELINE.value:
            continue
        result = entry.outcome.get("result")
        document = result.get("long_context") if isinstance(result, Mapping) else None
        if document is not None:
            checks.append((entry.request_id, LongContextResult.from_document(document)))
    return checks


def with_root_reading(request: TestRequest, result: TestResult, entries: Iterable[LogEntry]) -> TestResult:
    """``result`` with the root's reading of the long-context check (``LongContextResult.root``), pooled from the root's
    tests the submit log ``entries`` hold as ended; as it is for a test that ran no check, for a root test itself, and
    while the root has no complete check (an AgentX session, whose root is first tested at FINAL, has none)."""
    check = result.long_context
    if check is None or request.purpose is Purpose.BASELINE:
        return result
    root = root_reading(_root_checks(entries))
    return result if root is None else replace(result, long_context=replace(check, root=root))
