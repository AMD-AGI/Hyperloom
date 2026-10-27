# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The toolbox's side of the short-prompt consistency check: when a test runs it, what it is compared with, what is recorded.

*When.* Acceptance renders the change from the parent recipe to the candidate as lines (``touch_lines``) and the rule pack's
``short_prompt_touch`` pattern says which of them touch attention, the KV cache or prefill. A patch is rendered with the
files it changes, read from the patch itself in the blob store (its path in the recipe is only its hash), and a patch whose
files cannot be read counts as touching. A test also runs the check when
its purpose is one of the pack's ``always_purposes`` (a baseline, whose inconsistency is the floor the rule gate holds every
candidate to, and a final candidate) or when the request sets ``consistency_check`` (scheduling does on a reverse merge). A caller can add the
check, never remove one the rule demands. A toolbox given no :class:`ShortPromptSetup` runs no check and refuses nothing.

*Switched on.* The check is on by default: a test that runs it answers the prompts on its own server after its rounds, in
seconds; only a merge test or a final candidate whose recipe's pool is short boots fresh processes. A builder gives the toolbox the setup: from its options, or from the host environment's
``HYPERLOOM_SHORT_PROMPTS`` (the key of the staged entry; ``off`` turns the check off) when it names none, and without
either the set ``measure.short_prompts.build`` stages for the session model's tokenizer with its default seed, when it is
staged (:func:`default_short_prompts`; the set depends on the tokenizer, so a host that cannot find it runs no check and
logs why). A builder whose runners read this host's runner cache reads what it is given there first
(:func:`staged_short_prompts`); a set that is asked for, or found and not readable, is refused, never run without.

*What against.* The outputs of server processes of one recipe are compared with each other only: the test's own, and
every process the toolbox pooled for the same recipe (``short_prompt_pool``). Every test that runs the check adds its own
server (process 0); a merge test (``integration``) or a final candidate whose recipe has fewer pooled processes than the
pack's ``processes`` also boots fresh ones up to that count (:func:`with_extra_processes`), and an explore is judged over
the processes it has. A comparison with the parent's outputs fires on every change that moves greedy tokens early (fp8
KV, quantized GEMMs), and a defect that is wrong in every process is the accuracy evaluation's to see.

*What is recorded.* :func:`consistency_of` reads the landed ``outputs.json`` and states counts and positions
(``ShortPromptConsistency``); it never says pass or fail, and an ``incomplete`` check holds nothing back.
"""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from hyperloom.common.outside_facts import SHORT_PROMPT_CHECK, SHORT_PROMPT_TOUCH
from hyperloom.contracts.consistency import (
    SHORT_PROMPTS_COMPLETE,
    SHORT_PROMPTS_INCOMPLETE,
    OutputCounts,
    OutputDifference,
    ShortPromptConsistency,
)
from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import PatchRef, Recipe, diff
from hyperloom.measure.compile import BlobStore, CompileError, patch_targets
from hyperloom.measure.short_prompts.compare import Comparison, Difference, Side, compare_outputs
from hyperloom.measure.short_prompts.outputs import (
    FAILED,
    OutputsError,
    OutputsFile,
    parse_outputs,
    read_outputs_document,
)
from hyperloom.measure.short_prompts.plan import (
    OUTPUTS_FILE,
    SHORT_PROMPTS_DIR,
    ShortPromptPlan,
    miss_probability,
)
from hyperloom.measure.short_prompts.build import DEFAULT_SEED, build_entry, tokenizer_hashes
from hyperloom.measure.short_prompts.prompts import PromptSetError
from hyperloom.measure.short_prompts.resolve import staged_entry, verified_entry
from hyperloom.measure.short_prompts.staging import EVALSETS_DIR, StagedEntry, cache_root, entry_key

__all__ = [
    "MODEL_PATH_ENV",
    "PatchFiles",
    "PatchFilesUnreadable",
    "SHORT_PROMPTS_ENV",
    "SHORT_PROMPTS_OFF",
    "ShortPromptRules",
    "ShortPromptSetup",
    "ShortPromptsNotStaged",
    "consistency_of",
    "decide",
    "output_counts",
    "blob_patch_files",
    "default_short_prompts",
    "environment_short_prompts",
    "no_patch_files",
    "rederive",
    "short_prompt_rules",
    "staged_short_prompts",
    "touch_lines",
    "with_extra_processes",
]

#: The host environment variable that names the staged prompt set (its key) when a builder is given no setup.
SHORT_PROMPTS_ENV = "HYPERLOOM_SHORT_PROMPTS"
#: The value of ``HYPERLOOM_SHORT_PROMPTS`` that turns the check off where it would be on by default.
SHORT_PROMPTS_OFF = "off"
#: The host environment variable a session exports the directory its model is served from in.
MODEL_PATH_ENV = "MODEL_PATH"

_LOG = logging.getLogger(__name__)
_NOT_STAGED = "short-prompt consistency check off: the model's prompt set %s is not staged in the runner cache: %s"

_UNSET = "(unset)"
_MAX_LINE_CHARS = 200
_MAX_REASONS = 32
_CHECKED_KINDS = (TestKind.TEST, TestKind.SWEEP)
_UTF8 = "utf-8"
_MAX_FAILED_SHOWN = 3
#: The tests whose recipe the gate reads over the pack's full count of processes: what reaches integration, and what
#: ships. A defect in one launch in three escapes two processes about 44% of the time and six about 9%.
_FULL_COUNT_PURPOSES = (Purpose.INTEGRATION, Purpose.FINAL)


@dataclass(frozen=True)
class ShortPromptSetup:
    """The staged prompt set a deployment's checks use (``measure.short_prompts.build`` makes it): its key and the SHA-256
    of its prompts file. A toolbox that is given none does not run the check.

    Raises:
        ValueError: The key is not a plain name or the hash is not a SHA-256; a setup that no sheet could carry is refused
            here, not at the first test that would have run the check.
    """

    prompt_set: str
    prompts_sha256: str

    def __post_init__(self) -> None:
        ShortPromptPlan(self.prompt_set, self.prompts_sha256)


class ShortPromptsNotStaged(ValueError):
    """The check was asked for with a prompt set that is not staged where the runners read it; the message says why."""


def _setup_of(asked: str, read: Callable[[], StagedEntry]) -> ShortPromptSetup:
    """The setup of the entry ``read`` finds; a refusal says what was asked for and why it cannot be used."""
    try:
        entry = read()
    except PromptSetError as exc:
        raise ShortPromptsNotStaged(
            f"the short-prompt consistency check is asked for with {asked}, which is not staged in the runner cache: {exc}"
        ) from exc
    return ShortPromptSetup(entry.key, entry.prompts_sha256)


def default_short_prompts(
    model_dir: Path | None, environ: Mapping[str, str], *, stage: bool = False
) -> ShortPromptSetup | None:
    """The set ``measure.short_prompts.build`` stages for the tokenizer in ``model_dir`` (else the directory
    ``MODEL_PATH`` in ``environ`` names) with its default seed, read in the runner cache ``environ`` names; ``None``, and
    a log line that says why, when there is no model directory (information: a toolbox of no model has nothing to find), it
    holds no tokenizer or that set is not staged (a warning: the model's own check is off).

    With ``stage`` (a toolbox whose runners read the same cache) a set that is not staged is staged first, on the CPU: the
    model's tokenizer and the seeded generator; when that fails the check is off and the warning says why.

    Raises:
        ShortPromptsNotStaged: The set is there and unfinished, or is not what its manifest recorded: a set that was found
            and cannot be read is not a check that is off.
    """
    directory = model_dir or (Path(environ[MODEL_PATH_ENV]) if environ.get(MODEL_PATH_ENV) else None)
    if directory is None:
        _LOG.info("short-prompt consistency check off: no model directory to find its prompt set by")
        return None
    try:
        key = entry_key({"seed": DEFAULT_SEED, "tokenizer": tokenizer_hashes(directory)})
    except (PromptSetError, OSError) as exc:
        _LOG.warning("short-prompt consistency check off: the model's tokenizer cannot be read: %s", exc)
        return None
    if not (cache_root(environ) / EVALSETS_DIR / key).exists():
        if not stage:
            _LOG.warning(_NOT_STAGED, key, "the toolbox does not stage it")
            return None
        if not _staged_now(directory, environ, key):
            return None
    return _setup_of(f"the model's prompt set {key}", lambda: staged_entry(key, environ))


def _staged_now(model_dir: Path, environ: Mapping[str, str], key: str) -> bool:
    """Stage the model's default prompt set (``key``) in the runner cache ``environ`` names; whether it was, the reason
    logged when it was not."""
    try:
        build_entry(model_dir, cache_root(environ))
    except Exception as exc:  # noqa: BLE001 - the tokenizer library raises anything; the check is off and says why
        _LOG.warning(_NOT_STAGED, key, f"staging it failed: {type(exc).__name__}: {exc}")
        return False
    _LOG.info("short-prompt consistency check: staged the model's prompt set %s in the runner cache", key)
    return True


def environment_short_prompts(
    environ: Mapping[str, str], model_dir: Path | None = None, *, stage: bool = False
) -> ShortPromptSetup | None:
    """The setup ``HYPERLOOM_SHORT_PROMPTS`` names in ``environ``: none for ``off``, the model's own set
    (:func:`default_short_prompts`) when it names none (a blank value names none).

    The variable is the key of an entry in the runner cache ``environ`` names; the entry's own manifest says its hash, and
    the entry is read as a runner reads it before it is accepted.

    Raises:
        ShortPromptsNotStaged: A key is named and its entry is not staged there, is unfinished or is not what its manifest
            recorded; or the model's own set is there and cannot be read.
    """
    key = environ.get(SHORT_PROMPTS_ENV, "").strip()
    if key == SHORT_PROMPTS_OFF:
        return None
    if not key:
        return default_short_prompts(model_dir, environ, stage=stage)
    return _setup_of(f"{SHORT_PROMPTS_ENV}={key}", lambda: staged_entry(key, environ))


def staged_short_prompts(
    named: ShortPromptSetup | None, environ: Mapping[str, str], model_dir: Path | None = None, *, stage: bool = False
) -> ShortPromptSetup | None:
    """The setup a toolbox whose runners read the runner cache ``environ`` names runs the check with, or ``None`` for none.

    ``named`` is the setup the builder was given, read in that cache as a runner reads it before it is accepted; without
    one, the environment's (:func:`environment_short_prompts`, with the session's ``model_dir``). Asking for the check
    never ends as a toolbox that quietly runs none.

    Raises:
        ShortPromptsNotStaged: The set is asked for and is not staged there, is unfinished or is not the one named.
    """
    if named is None:
        return environment_short_prompts(environ, model_dir, stage=stage)
    return _setup_of(
        f"prompt set {named.prompt_set}", lambda: verified_entry(named.prompt_set, named.prompts_sha256, environ)
    )


@dataclass(frozen=True)
class ShortPromptRules:
    """The rule pack's facts about the check, read as the toolbox uses them."""

    touch: re.Pattern[str]
    processes: int
    max_new_tokens: int
    phase_cap_sec: float
    always_purposes: frozenset[str]


@functools.cache
def short_prompt_rules() -> ShortPromptRules:
    """The check's rules, from the pack this process reads; a value that is not a number or a word is a refused pack."""
    values = SHORT_PROMPT_CHECK
    try:
        return ShortPromptRules(
            touch=re.compile(SHORT_PROMPT_TOUCH),
            processes=int(values["processes"]),
            max_new_tokens=int(values["max_new_tokens"]),
            phase_cap_sec=float(values["phase_cap_sec"]),
            always_purposes=frozenset(values["always_purposes"].split()),
        )
    except ValueError as exc:
        raise ValueError(f"the rule pack's short_prompt_check holds a value that is not a number: {exc}") from exc


# --- when ----------------------------------------------------------------------------------------------------------


def _shown(value: str | None) -> str:
    return _UNSET if value is None else value


class PatchFilesUnreadable(Exception):
    """The files a patch changes cannot be read; the message says why."""


#: The files a patch changes, repository-relative, read from the patch itself.
PatchFiles = Callable[[PatchRef], Sequence[str]]


def blob_patch_files(blobs: BlobStore) -> PatchFiles:
    """The files each patch changes, read from its blob in ``blobs`` (the paths of its ``diff --git`` headers)."""

    def files(patch: PatchRef) -> Sequence[str]:
        try:
            return patch_targets(blobs.get(patch.sha256).decode(_UTF8), "", patch.path)
        except (CompileError, OSError, ValueError) as exc:
            raise PatchFilesUnreadable(f"{type(exc).__name__}: {exc}") from exc

    return files


def no_patch_files(patch: PatchRef) -> Sequence[str]:
    """For an acceptance with no blob store: no patch's files can be read, so every patch counts as touching."""
    raise PatchFilesUnreadable("this toolbox has no blob store to read it from")


def _patch_lines(recipe: Recipe, hashes: frozenset[str], files: PatchFiles) -> list[str]:
    """``patch <component> <path>`` for each patch among ``hashes``, then ``patch-file <component> <file>`` for each file
    it changes, or ``unexpressed patch <path> ...`` when those cannot be read (a change that cannot be read touches)."""
    lines = []
    for component in recipe.components:
        for patch in component.patches:
            if patch.sha256 not in hashes:
                continue
            lines.append(f"patch {component.component} {patch.path}")
            try:
                lines += [f"patch-file {component.component} {path}" for path in files(patch)]
            except PatchFilesUnreadable as exc:
                lines.append(f"unexpressed patch {patch.path}: its files cannot be read ({exc})")
    return lines


def _kernel_lines(recipe: Recipe, hashes: frozenset[str]) -> list[str]:
    return [f"kernel {kernel.kind.value} {kernel.target}" for kernel in recipe.kernels if kernel.sha256 in hashes]


def touch_lines(parent: Recipe, candidate: Recipe, files: PatchFiles) -> list[str]:
    """The change from ``parent`` to ``candidate`` as the lines the rule pack's pattern reads; ``files`` reads the files a
    patch changes.

    A change the toolbox cannot read counts as touching, so it is rendered as a line the pattern matches: an entry the
    recipe names as unexpressed, or a patch whose files cannot be read.
    """
    change = diff(parent, candidate)
    lines = [f"arg {c.name} {_shown(c.after)}".rstrip() for c in change.args]
    lines += [f"env {c.name}={_shown(c.after)}" for c in change.envs]
    lines += [f"ref {r.component} {_shown(r.before)}->{_shown(r.after)}" for r in change.refs]
    lines += _patch_lines(candidate, change.patches_only_in_second, files)
    lines += _patch_lines(parent, change.patches_only_in_first, files)
    lines += _kernel_lines(candidate, change.kernels_only_in_second) + _kernel_lines(
        parent, change.kernels_only_in_first
    )
    if change.base_image_changed:
        lines.append("image changed")
    lines += [
        f"unexpressed {entry}" for entry in sorted(change.unexpressed_only_in_first | change.unexpressed_only_in_second)
    ]
    return lines


def _bounded(reasons: list[str]) -> tuple[str, ...]:
    shown = [reason[:_MAX_LINE_CHARS] for reason in reasons[: _MAX_REASONS - 1]]
    if len(reasons) >= _MAX_REASONS:
        shown.append(f"... and {len(reasons) - _MAX_REASONS + 1} more lines")
    else:
        shown = [reason[:_MAX_LINE_CHARS] for reason in reasons]
    return tuple(shown)


def decide(
    request: TestRequest,
    parent: Recipe,
    candidate: Recipe,
    setup: ShortPromptSetup | None,
    files: PatchFiles,
) -> ShortPromptPlan | None:
    """The check ``request`` runs, or ``None``: its purpose is one the pack always checks, the caller forces it, or the
    change from ``parent`` to ``candidate`` touches what the pack says affects attention, the KV cache or prefill
    (``files`` reads the files a patch changes). The plan asks the test's own server only; :func:`with_extra_processes`
    adds the fresh ones a short pool needs."""
    if setup is None or request.kind not in _CHECKED_KINDS:
        return None
    rules = short_prompt_rules()
    reasons = []
    if request.purpose.value in rules.always_purposes:
        reasons.append(f"purpose {request.purpose.value}")
    if request.forces_consistency_check:
        reasons.append("request consistency_check")
    reasons += [line for line in touch_lines(parent, candidate, files) if rules.touch.search(line)]
    if not reasons:
        return None
    return ShortPromptPlan(
        prompt_set=setup.prompt_set,
        prompts_sha256=setup.prompts_sha256,
        processes=1,
        max_new_tokens=rules.max_new_tokens,
        phase_cap_sec=rules.phase_cap_sec,
        trigger=_bounded(reasons),
    )


def with_extra_processes(plan: ShortPromptPlan, request: TestRequest, pooled: int) -> ShortPromptPlan:
    """``plan`` with the fresh processes ``request`` boots after its own server's answers: for a merge test or a final
    candidate, as many as the recipe's pool (``pooled`` processes so far) lacks of the pack's ``processes``; for any other
    test, none."""
    if request.purpose not in _FULL_COUNT_PURPOSES:
        return plan
    return replace(plan, processes=max(1, short_prompt_rules().processes - pooled))


# --- what is recorded ----------------------------------------------------------------------------------------------


def _difference(found: Difference | None) -> OutputDifference | None:
    return None if found is None else OutputDifference(**vars(found))


def output_counts(side: Side | None) -> OutputCounts | None:
    """A comparison's counts as the result records them; ``None`` for none."""
    return None if side is None else OutputCounts(side.mismatches, _difference(side.first))


def _why_incomplete(plan: ShortPromptPlan, found: OutputsFile, completed: int) -> str:
    if completed == plan.processes:
        return ""
    failed = [f"process {e.index}: {e.reason}" for e in found.processes if e.status == FAILED]
    parts = ([found.stopped] if found.stopped else []) + failed[:_MAX_FAILED_SHOWN]
    return "; ".join(parts) or f"{completed} of {plan.processes} processes answered every prompt"


def _incomplete(plan: ShortPromptPlan, reason: str) -> ShortPromptConsistency:
    return ShortPromptConsistency(
        status=SHORT_PROMPTS_INCOMPLETE,
        reason=reason,
        prompt_set_sha256=plan.prompts_sha256,
        prompts=0,
        max_new_tokens=plan.max_new_tokens,
        trigger=plan.trigger,
        processes_planned=plan.processes,
        processes=(),
        miss_probability_one_third=miss_probability(0),
        compared_by="",
        self_consistency=None,
    )


def consistency_of(plan: ShortPromptPlan, directory: Path) -> ShortPromptConsistency:
    """The check of a test whose output landed in ``directory``, as the result records it.

    Whatever the bundle lacks makes the check ``incomplete`` with the reason, never a clean one: no outputs file, outputs of
    another prompt set, a process that did not answer.
    """
    try:
        found = parse_outputs(read_outputs_document(directory))
    except OutputsError as exc:
        return _incomplete(plan, str(exc))
    if (found.prompt_set, found.prompts_sha256) != (plan.prompt_set, plan.prompts_sha256):
        return _incomplete(plan, "the bundle's outputs are for another prompt set than the one the test asked for")
    if not found.completed:
        return _incomplete(plan, _why_incomplete(plan, found, 0))
    return _consistency(plan, found, compare_outputs(found.completed, found.prompts))


def rederive(recorded: ShortPromptConsistency, directory: Path) -> ShortPromptConsistency:
    """The check ``recorded`` claims, derived again from the outputs file of the verified bundle at ``directory`` alone."""
    plan = ShortPromptPlan(
        "rederived",
        recorded.prompt_set_sha256,
        processes=recorded.processes_planned,
        max_new_tokens=recorded.max_new_tokens,
        trigger=recorded.trigger,
    )
    try:
        found = parse_outputs(read_outputs_document(directory))
    except OutputsError as exc:
        return _incomplete(plan, str(exc))
    plan = replace(plan, prompt_set=found.prompt_set, prompts_sha256=found.prompts_sha256)
    if not found.completed:
        return _incomplete(plan, _why_incomplete(plan, found, 0))
    return _consistency(plan, found, compare_outputs(found.completed, found.prompts))


def _consistency(plan: ShortPromptPlan, found: OutputsFile, comparison: Comparison) -> ShortPromptConsistency:
    completed = found.completed
    reason = _why_incomplete(plan, found, len(completed))
    return ShortPromptConsistency(
        status=SHORT_PROMPTS_INCOMPLETE if reason else SHORT_PROMPTS_COMPLETE,
        reason=reason,
        prompt_set_sha256=plan.prompts_sha256,
        prompts=len(found.prompts),
        max_new_tokens=found.max_new_tokens,
        trigger=plan.trigger,
        processes_planned=plan.processes,
        processes=tuple(sorted(completed)),
        miss_probability_one_third=miss_probability(len(completed)),
        compared_by=comparison.compared_by,
        self_consistency=output_counts(comparison.self_consistency),
        outputs=f"{SHORT_PROMPTS_DIR}/{OUTPUTS_FILE}",
    )
