# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The toolbox's side of the short-prompt consistency check: when a test runs it, what it is compared with, what is recorded.

*When.* Acceptance renders the change from the parent recipe to the candidate as lines (``touch_lines``) and the rule pack's
``short_prompt_touch`` pattern says which of them touch attention, the KV cache or prefill; a test also runs the check when
its purpose is one of the pack's ``always_purposes`` (a baseline, which gives its descendants their reference outputs, and a
final candidate) or when the request sets ``consistency_check`` (scheduling does on a reverse merge). A caller can add the
check, never remove one the rule demands. A toolbox given no :class:`ShortPromptSetup` runs no check and refuses nothing.

*Switched on.* The check is on by default: the triggers above bound its cost (six fresh processes, about 9 to 11
minutes, on a test that runs it). A builder gives the toolbox the setup: from its options, or from the host environment's
``HYPERLOOM_SHORT_PROMPTS`` (the key of the staged entry; ``off`` turns the check off) when it names none, and without
either the set ``measure.short_prompts.build`` stages for the session model's tokenizer with its default seed, when it is
staged (:func:`default_short_prompts`; the set depends on the tokenizer, so a host that cannot find it runs no check and
logs why). A builder whose runners read this host's runner cache reads what it is given there first
(:func:`staged_short_prompts`); a set that is asked for, or found and not readable, is refused, never run without.

*What against.* The outputs of the test's own fresh server processes are compared with each other and with the parent's
outputs from the parent's own run, which :class:`ReferenceStore` keeps by recipe digest. A parent with no stored outputs is
a reference that is ``absent`` and counts nothing, never a clean one.

*What is recorded.* :func:`consistency_of` reads the landed ``outputs.json`` and states counts and positions
(``ShortPromptConsistency``); it never says pass or fail.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from hyperloom.common.outside_facts import SHORT_PROMPT_CHECK, SHORT_PROMPT_TOUCH
from hyperloom.contracts.consistency import (
    REFERENCE_ABSENT,
    REFERENCE_COMPARED,
    SHORT_PROMPTS_COMPLETE,
    SHORT_PROMPTS_INCOMPLETE,
    OutputCounts,
    OutputDifference,
    ShortPromptConsistency,
    ShortPromptReference,
)
from hyperloom.contracts.measurement import TestKind, TestRequest
from hyperloom.contracts.recipe import Recipe, diff
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
from hyperloom.measure.short_prompts.build import DEFAULT_SEED, tokenizer_hashes
from hyperloom.measure.short_prompts.prompts import PromptSetError
from hyperloom.measure.short_prompts.resolve import staged_entry, verified_entry
from hyperloom.measure.short_prompts.staging import EVALSETS_DIR, StagedEntry, cache_root, entry_key

__all__ = [
    "MODEL_PATH_ENV",
    "SHORT_PROMPTS_ENV",
    "SHORT_PROMPTS_OFF",
    "ReferenceAbsent",
    "ReferenceStore",
    "ShortPromptRules",
    "ShortPromptSetup",
    "ShortPromptsNotStaged",
    "consistency_of",
    "decide",
    "default_short_prompts",
    "environment_short_prompts",
    "record_reference",
    "rederive",
    "short_prompt_rules",
    "staged_short_prompts",
    "touch_lines",
]

#: The host environment variable that names the staged prompt set (its key) when a builder is given no setup.
SHORT_PROMPTS_ENV = "HYPERLOOM_SHORT_PROMPTS"
#: The value of ``HYPERLOOM_SHORT_PROMPTS`` that turns the check off where it would be on by default.
SHORT_PROMPTS_OFF = "off"
#: The host environment variable a session exports the directory its model is served from in.
MODEL_PATH_ENV = "MODEL_PATH"

_LOG = logging.getLogger(__name__)

_UNSET = "(unset)"
_MAX_LINE_CHARS = 200
_MAX_REASONS = 32
_CHECKED_KINDS = (TestKind.TEST, TestKind.SWEEP)
_SCHEMA = 1
_UTF8 = "utf-8"
_MAX_FAILED_SHOWN = 3
_SHA_DIGITS = 16


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


def default_short_prompts(model_dir: Path | None, environ: Mapping[str, str]) -> ShortPromptSetup | None:
    """The set ``measure.short_prompts.build`` stages for the tokenizer in ``model_dir`` (else the directory
    ``MODEL_PATH`` in ``environ`` names) with its default seed, read in the runner cache ``environ`` names; ``None``, and
    a warning that says why, when there is no model directory, it holds no tokenizer or that set is not staged.

    Raises:
        ShortPromptsNotStaged: The set is there and unfinished, or is not what its manifest recorded: a set that was found
            and cannot be read is not a check that is off.
    """
    directory = model_dir or (Path(environ[MODEL_PATH_ENV]) if environ.get(MODEL_PATH_ENV) else None)
    if directory is None:
        _LOG.warning("short-prompt consistency check off: no model directory to find its prompt set by")
        return None
    try:
        key = entry_key({"seed": DEFAULT_SEED, "tokenizer": tokenizer_hashes(directory)})
    except (PromptSetError, OSError) as exc:
        _LOG.warning("short-prompt consistency check off: the model's tokenizer cannot be read: %s", exc)
        return None
    if not (cache_root(environ) / EVALSETS_DIR / key).exists():
        _LOG.warning(
            "short-prompt consistency check off: the model's prompt set %s is not staged in the runner cache", key
        )
        return None
    return _setup_of(f"the model's prompt set {key}", lambda: staged_entry(key, environ))


def environment_short_prompts(environ: Mapping[str, str], model_dir: Path | None = None) -> ShortPromptSetup | None:
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
        return default_short_prompts(model_dir, environ)
    return _setup_of(f"{SHORT_PROMPTS_ENV}={key}", lambda: staged_entry(key, environ))


def staged_short_prompts(
    named: ShortPromptSetup | None, environ: Mapping[str, str], model_dir: Path | None = None
) -> ShortPromptSetup | None:
    """The setup a toolbox whose runners read the runner cache ``environ`` names runs the check with, or ``None`` for none.

    ``named`` is the setup the builder was given, read in that cache as a runner reads it before it is accepted; without
    one, the environment's (:func:`environment_short_prompts`, with the session's ``model_dir``). Asking for the check
    never ends as a toolbox that quietly runs none.

    Raises:
        ShortPromptsNotStaged: The set is asked for and is not staged there, is unfinished or is not the one named.
    """
    if named is None:
        return environment_short_prompts(environ, model_dir)
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


def _patch_lines(recipe: Recipe, hashes: frozenset[str]) -> list[str]:
    return [
        f"patch {component.component} {patch.path}"
        for component in recipe.components
        for patch in component.patches
        if patch.sha256 in hashes
    ]


def _kernel_lines(recipe: Recipe, hashes: frozenset[str]) -> list[str]:
    return [f"kernel {kernel.kind.value} {kernel.target}" for kernel in recipe.kernels if kernel.sha256 in hashes]


def touch_lines(parent: Recipe, candidate: Recipe) -> list[str]:
    """The change from ``parent`` to ``candidate`` as the lines the rule pack's pattern reads.

    A change the toolbox cannot read counts as touching, so it is rendered as a line the pattern matches: an entry the
    recipe names as unexpressed.
    """
    change = diff(parent, candidate)
    lines = [f"arg {c.name} {_shown(c.after)}".rstrip() for c in change.args]
    lines += [f"env {c.name}={_shown(c.after)}" for c in change.envs]
    lines += [f"ref {r.component} {_shown(r.before)}->{_shown(r.after)}" for r in change.refs]
    lines += _patch_lines(candidate, change.patches_only_in_second) + _patch_lines(parent, change.patches_only_in_first)
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
    request: TestRequest, parent: Recipe, candidate: Recipe, setup: ShortPromptSetup | None
) -> ShortPromptPlan | None:
    """The check ``request`` runs, or ``None``: its purpose is one the pack always checks, the caller forces it, or the
    change from ``parent`` to ``candidate`` touches what the pack says affects attention, the KV cache or prefill."""
    if setup is None or request.kind not in _CHECKED_KINDS:
        return None
    rules = short_prompt_rules()
    reasons = []
    if request.purpose.value in rules.always_purposes:
        reasons.append(f"purpose {request.purpose.value}")
    if request.forces_consistency_check:
        reasons.append("request consistency_check")
    reasons += [line for line in touch_lines(parent, candidate) if rules.touch.search(line)]
    if not reasons:
        return None
    return ShortPromptPlan(
        prompt_set=setup.prompt_set,
        prompts_sha256=setup.prompts_sha256,
        processes=rules.processes,
        max_new_tokens=rules.max_new_tokens,
        phase_cap_sec=rules.phase_cap_sec,
        trigger=_bounded(reasons),
        recipe=candidate.digest(),
        parent_recipe=parent.digest(),
    )


# --- what against --------------------------------------------------------------------------------------------------


class ReferenceAbsent(Exception):
    """The parent has no usable outputs; the message is the reason the result records."""


@dataclass(frozen=True)
class StoredReference:
    request_id: str
    outputs: OutputsFile


class ReferenceStore:
    """The outputs of finished checks, by the digest of the recipe they were measured on and the prompt set.

    A test's outputs are stored under its own recipe's digest, so a test whose parent is that recipe finds them. The first
    complete check of a recipe is the one kept: a reference does not change under the tests that were compared with it.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)

    def _path(self, recipe: str, prompts_sha256: str) -> Path:
        return self._directory / f"{recipe.removeprefix('sha256:')}-{prompts_sha256[:_SHA_DIGITS]}.json"

    def get(self, recipe: str, prompts_sha256: str) -> StoredReference:
        """The outputs stored for ``recipe``.

        Raises:
            ReferenceAbsent: Nothing is stored, or what is stored cannot be read.
        """
        path = self._path(recipe, prompts_sha256)
        try:
            document = json.loads(path.read_text(encoding=_UTF8))
            if document["schema"] != _SCHEMA or document["recipe"] != recipe:
                raise ValueError("another recipe")
            return StoredReference(str(document["request_id"]), parse_outputs(document["outputs"]))
        except FileNotFoundError:
            raise ReferenceAbsent("the parent recipe has no recorded outputs for this prompt set") from None
        except (OSError, ValueError, KeyError, TypeError, OutputsError):
            raise ReferenceAbsent("the parent recipe's recorded outputs cannot be read") from None

    def put(self, recipe: str, prompts_sha256: str, request_id: str, outputs: Mapping[str, Any]) -> bool:
        """Store ``outputs`` (the document of a complete check) unless the recipe already has some; whether it did."""
        path = self._path(recipe, prompts_sha256)
        self._directory.mkdir(parents=True, exist_ok=True)
        document = {"schema": _SCHEMA, "recipe": recipe, "request_id": request_id, "outputs": outputs}
        fd, scratch = tempfile.mkstemp(dir=self._directory, prefix=".ref-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding=_UTF8) as handle:
                json.dump(document, handle, sort_keys=True)
            os.link(scratch, path)
        except FileExistsError:
            return False
        finally:
            Path(scratch).unlink(missing_ok=True)
        return True


# --- what is recorded ----------------------------------------------------------------------------------------------


def _difference(found: Difference | None) -> OutputDifference | None:
    return None if found is None else OutputDifference(**vars(found))


def _counts(side: Side | None) -> OutputCounts | None:
    if side is None:
        return None
    return OutputCounts(side.mismatches, _difference(side.first), side.unstable)


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
        reference=ShortPromptReference(REFERENCE_ABSENT, "no process answered every prompt, so nothing was compared"),
    )


def _reference_outputs(plan: ShortPromptPlan, store: ReferenceStore | None) -> tuple[StoredReference | None, str]:
    if store is None or not plan.parent_recipe:
        return None, "no reference store, or no parent recipe, is known to this toolbox"
    try:
        return store.get(plan.parent_recipe, plan.prompts_sha256), ""
    except ReferenceAbsent as exc:
        return None, str(exc)


def consistency_of(plan: ShortPromptPlan, directory: Path, store: ReferenceStore | None) -> ShortPromptConsistency:
    """The check of a test whose output landed in ``directory``, as the result records it.

    Whatever the bundle lacks makes the check ``incomplete`` with the reason, never a clean one: no outputs file, outputs of
    another prompt set, a process that did not answer, a parent with no outputs to compare with.
    """
    try:
        found = parse_outputs(read_outputs_document(directory))
    except OutputsError as exc:
        return _incomplete(plan, str(exc))
    if (found.prompt_set, found.prompts_sha256) != (plan.prompt_set, plan.prompts_sha256):
        return _incomplete(plan, "the bundle's outputs are for another prompt set than the one the test asked for")
    completed = found.completed
    if not completed:
        return _incomplete(plan, _why_incomplete(plan, found, 0))
    stored, absent = _reference_outputs(plan, store)
    comparison = compare_outputs(completed, None if stored is None else stored.outputs.completed, found.prompts)
    return _consistency(plan, found, comparison, stored, absent)


def rederive(recorded: ShortPromptConsistency, directory: Path) -> ShortPromptConsistency:
    """The check ``recorded`` claims, derived again from the outputs file of the verified bundle at ``directory`` alone: all
    of it except the reference, which is the parent's outputs as they were stored then and not part of this bundle."""
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
    comparison = compare_outputs(found.completed, None, found.prompts) if found.completed else None
    if comparison is None:
        return _incomplete(plan, _why_incomplete(plan, found, 0))
    return _consistency(plan, found, comparison, None, "not part of the bundle")


def record_reference(
    plan: ShortPromptPlan, check: ShortPromptConsistency, directory: Path, store: ReferenceStore, request_id: str
) -> bool:
    """Keep the outputs of a complete check as the reference for tests whose parent is the recipe it measured; whether it
    kept them. An incomplete check, or a test that knows no recipe digest, leaves nothing."""
    if check.status != SHORT_PROMPTS_COMPLETE or not plan.recipe:
        return False
    try:
        return store.put(plan.recipe, plan.prompts_sha256, request_id, read_outputs_document(directory))
    except (OutputsError, OSError):
        return False


def _consistency(
    plan: ShortPromptPlan, found: OutputsFile, comparison: Comparison, stored: StoredReference | None, absent: str
) -> ShortPromptConsistency:
    completed = found.completed
    reason = _why_incomplete(plan, found, len(completed))
    reference_counts = _counts(comparison.reference)
    if stored is not None and reference_counts is None:
        absent = "the parent's recorded outputs hold no process that answered every prompt"
    reference = (
        ShortPromptReference(REFERENCE_COMPARED, request_id=stored.request_id, counts=reference_counts)
        if stored is not None and reference_counts is not None
        else ShortPromptReference(REFERENCE_ABSENT, reason=absent)
    )
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
        self_consistency=_counts(comparison.self_consistency),
        reference=reference,
        outputs=f"{SHORT_PROMPTS_DIR}/{OUTPUTS_FILE}",
    )
