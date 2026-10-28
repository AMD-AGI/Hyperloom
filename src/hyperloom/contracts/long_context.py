# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The long-context check as a test result carries it.

A frozen set of long synthetic prompts (needle retrieval, multi-key retrieval, variable tracking, common-word extraction) is
answered greedily by the candidate's own server and graded by substring match. The types here state the scores and what
each prompt did, never a verdict: what refuses a candidate is the gate's rule, not this record. A prompt that errored or was
never run counts as not passed and makes the check ``incomplete``, which is never a pass and never blocks.

The root's reading (:class:`LongContextRoot`, made by :func:`root_reading`) pools the complete checks of the session root's
tests and states its score with a Wilson 95% interval; a candidate's result carries the reading the toolbox had when the
candidate ended. :func:`refuses` is the bar the board decided (#142 c3219, bar (b)) for the gate to apply: a drop from the
lower end of the root's interval of more than the check's own sampling noise, ``NOISE_SHARE`` of the items (3 of 64). Like the rest of the measurement
contract they are frozen dataclasses that validate on construction and round-trip through ``to_document()`` and
``from_document()``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, fields

from hyperloom.contracts._validate import (
    MeasurementError,
    _bool,
    _int,
    _items,
    _mapping,
    _number,
    _optional_int,
    _relative_dir,
    _sha256,
    _strings,
    _text,
)

__all__ = [
    "CI_Z",
    "LONG_CONTEXT_COMPLETE",
    "LONG_CONTEXT_INCOMPLETE",
    "LONG_CONTEXT_STATUSES",
    "PROMPT_ERROR",
    "PROMPT_NOT_RUN",
    "PROMPT_OK",
    "NOISE_SHARE",
    "PROMPT_STATUSES",
    "LongContextPrompt",
    "LongContextResult",
    "LongContextRoot",
    "LongContextTask",
    "refuses",
    "root_reading",
    "wilson_interval",
]

#: How far a long-context check got: every prompt was answered, or not (and then the result says why).
LONG_CONTEXT_COMPLETE = "complete"
LONG_CONTEXT_INCOMPLETE = "incomplete"
LONG_CONTEXT_STATUSES = (LONG_CONTEXT_COMPLETE, LONG_CONTEXT_INCOMPLETE)
#: What became of one prompt: answered, its request failed, or it was never sent.
PROMPT_OK = "ok"
PROMPT_ERROR = "error"
PROMPT_NOT_RUN = "not_run"
PROMPT_STATUSES = (PROMPT_OK, PROMPT_ERROR, PROMPT_NOT_RUN)
_EPSILON = 1e-9
_TESTS = "tests"


@dataclass(frozen=True)
class LongContextPrompt:
    """One prompt: its task, what became of it, whether its answer held every reference, the tokens the server read for it
    and the seconds its request took."""

    id: str
    task: str
    status: str
    passed: bool
    prompt_tokens: int
    seconds: float

    def __post_init__(self) -> None:
        _text(self.id, "a prompt's id")
        _text(self.task, "a prompt's task")
        if self.status not in PROMPT_STATUSES:
            raise MeasurementError(f"a prompt's status must be one of {PROMPT_STATUSES}, got {self.status!r}")
        if _bool(self.passed, "a prompt's passed") and self.status != PROMPT_OK:
            raise MeasurementError("a prompt that was not answered cannot have passed")
        _int(self.prompt_tokens, "a prompt's prompt_tokens", minimum=0)
        if _number(self.seconds, "a prompt's seconds") < 0:
            raise MeasurementError("a prompt's seconds cannot be negative")

    def to_document(self) -> dict[str, object]:
        return {
            "id": self.id,
            "task": self.task,
            "status": self.status,
            "passed": self.passed,
            "prompt_tokens": self.prompt_tokens,
            "seconds": self.seconds,
        }

    @classmethod
    def from_document(cls, document: object) -> LongContextPrompt:
        f = _mapping(document, "a long-context prompt")
        keys = ("id", "task", "status", "passed", "prompt_tokens", "seconds")
        return cls(**{key: f.get(key) for key in keys})  # type: ignore[arg-type]


@dataclass(frozen=True)
class LongContextTask:
    """One task: how many of its ``n`` prompts passed, and ``score`` (that share)."""

    task: str
    n: int
    passed: int
    score: float

    def __post_init__(self) -> None:
        _text(self.task, "a task's name")
        _int(self.n, "a task's n", minimum=1)
        if _int(self.passed, "a task's passed", minimum=0) > self.n:
            raise MeasurementError("a task cannot pass more prompts than it has")
        if abs(_number(self.score, "a task's score") - self.passed / self.n) > _EPSILON:
            raise MeasurementError("a task's score must be passed over n")

    def to_document(self) -> dict[str, object]:
        return {"task": self.task, "n": self.n, "passed": self.passed, "score": self.score}

    @classmethod
    def from_document(cls, document: object) -> LongContextTask:
        f = _mapping(document, "a long-context task")
        return cls(task=f.get("task"), n=f.get("n"), passed=f.get("passed"), score=f.get("score"))  # type: ignore[arg-type]


#: The z of the two-sided 95% interval on the root's score.
CI_Z = 1.959963984540054
#: The check's own sampling noise, as a share of its prompts: about 3 of 64 binary items.
NOISE_SHARE = 3 / 64


def wilson_interval(passed: int, n: int) -> tuple[float, float]:
    """The Wilson score interval at 95% (``CI_Z``) on ``passed`` of ``n`` binary items; ``n`` is at least 1."""
    share, spread = passed / n, CI_Z * CI_Z / n
    centre = (share + spread / 2) / (1 + spread)
    half = CI_Z * math.sqrt(share * (1 - share) / n + spread / (4 * n)) / (1 + spread)
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class LongContextRoot:
    """The session root's reading of the check, which a candidate's drop is judged after: the complete checks of the root's
    tests (``tests``, their request ids) pooled, ``passed`` of ``n`` prompts, ``score`` that share and ``ci_low`` and
    ``ci_high`` the Wilson 95% interval on it (:func:`wilson_interval`), on the prompt set ``prompt_set_sha256``."""

    prompt_set_sha256: str
    tests: tuple[str, ...]
    n: int
    passed: int
    score: float
    ci_low: float
    ci_high: float

    def __post_init__(self) -> None:
        _sha256(self.prompt_set_sha256, "the root's prompt_set_sha256")
        _strings(self.tests, "the root's tests")
        if not self.tests:
            raise MeasurementError("the root's reading names the tests it pools")
        _int(self.n, "the root's n", minimum=1)
        if _int(self.passed, "the root's passed", minimum=0) > self.n:
            raise MeasurementError("the root cannot pass more prompts than it has")
        if abs(_number(self.score, "the root's score") - self.passed / self.n) > _EPSILON:
            raise MeasurementError("the root's score must be passed over n")
        low, high = _number(self.ci_low, "the root's ci_low"), _number(self.ci_high, "the root's ci_high")
        if not 0 <= low <= self.score + _EPSILON or not self.score - _EPSILON <= high <= 1:
            raise MeasurementError("the root's interval must hold its score, within 0 and 1")

    def to_document(self) -> dict[str, object]:
        return {_TESTS: list(self.tests), **{key: getattr(self, key) for key in _root_scalars()}}

    @classmethod
    def from_document(cls, document: object) -> LongContextRoot:
        f = _mapping(document, "a long-context root reading")
        tests = tuple(_items(f.get(_TESTS), _TESTS))
        return cls(tests=tests, **{key: f.get(key) for key in _root_scalars()})  # type: ignore[arg-type]


def _root_scalars() -> tuple[str, ...]:
    """The fields of a root reading other than the tests it pools, as its document names them."""
    return tuple(field.name for field in fields(LongContextRoot) if field.name != _TESTS)


@dataclass(frozen=True)
class LongContextResult:
    """One long-context check.

    ``n`` is every prompt of the set and ``score`` the share that passed (so a prompt that errored counts against it);
    ``status`` is ``incomplete``, with ``reason``, unless every prompt was answered. A check whose outputs the toolbox could
    not read states no prompt: ``n`` is 0 and ``score`` is ``None``, never a score of zero. ``length`` is the prompts' token
    length, ``served_context`` the context the launch serves when it states one, ``seconds`` the time the questions took and
    ``prefill_tokens`` the prompt tokens of the answered prompts. ``outputs`` is where the raw answers are in the bundle.
    """

    status: str
    reason: str
    prompt_set_sha256: str
    length: int
    max_new_tokens: int
    concurrency: int
    served_context: int | None
    n: int
    passed: int
    score: float | None
    tasks: tuple[LongContextTask, ...]
    prompts: tuple[LongContextPrompt, ...]
    seconds: float
    prefill_tokens: int
    trigger: tuple[str, ...] = ()
    outputs: str = ""
    root: LongContextRoot | None = None

    def __post_init__(self) -> None:
        if self.root is not None and not isinstance(self.root, LongContextRoot):
            raise MeasurementError(f"root must be a LongContextRoot, got {self.root!r}")
        if self.status not in LONG_CONTEXT_STATUSES:
            raise MeasurementError(f"status must be one of {LONG_CONTEXT_STATUSES}, got {self.status!r}")
        if not isinstance(self.reason, str) or (self.status == LONG_CONTEXT_INCOMPLETE) != bool(self.reason):
            raise MeasurementError("a long-context check has a reason exactly when it is incomplete")
        _sha256(self.prompt_set_sha256, "prompt_set_sha256")
        for value, what in (
            (self.length, "length"),
            (self.max_new_tokens, "max_new_tokens"),
            (self.concurrency, "concurrency"),
        ):
            _int(value, what, minimum=1)
        _optional_int(self.served_context, "served_context")
        self._check_counts()
        _strings(self.trigger, "trigger")
        _relative_dir(self.outputs, "outputs")
        if _number(self.seconds, "seconds") < 0 or _int(self.prefill_tokens, "prefill_tokens", minimum=0) < 0:
            raise MeasurementError("seconds and prefill_tokens cannot be negative")

    def _check_counts(self) -> None:
        if not isinstance(self.tasks, tuple) or not all(isinstance(t, LongContextTask) for t in self.tasks):
            raise MeasurementError(f"tasks must be a tuple of LongContextTask, got {self.tasks!r}")
        if not isinstance(self.prompts, tuple) or not all(isinstance(p, LongContextPrompt) for p in self.prompts):
            raise MeasurementError(f"prompts must be a tuple of LongContextPrompt, got {self.prompts!r}")
        _int(self.n, "n", minimum=0)
        _int(self.passed, "passed", minimum=0)
        answered = bool(self.prompts) and all(prompt.status == PROMPT_OK for prompt in self.prompts)
        if (self.status == LONG_CONTEXT_COMPLETE) != answered:
            raise MeasurementError("a check is complete exactly when every prompt was answered")
        counts = (len(self.prompts), sum(p.passed for p in self.prompts))
        if (self.n, self.passed) != counts or (
            sum(t.n for t in self.tasks),
            sum(t.passed for t in self.tasks),
        ) != counts:
            raise MeasurementError("n and passed must be what the prompts and the tasks add up to")
        if (self.score is None) != (self.n == 0) or (
            self.score is not None and abs(_number(self.score, "score") - self.passed / self.n) > _EPSILON
        ):
            raise MeasurementError("score must be passed over n, and absent exactly when there is no prompt")

    def to_document(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "prompt_set_sha256": self.prompt_set_sha256,
            "length": self.length,
            "max_new_tokens": self.max_new_tokens,
            "concurrency": self.concurrency,
            "served_context": self.served_context,
            "n": self.n,
            "passed": self.passed,
            "score": self.score,
            "tasks": [task.to_document() for task in self.tasks],
            "prompts": [prompt.to_document() for prompt in self.prompts],
            "seconds": self.seconds,
            "prefill_tokens": self.prefill_tokens,
            "trigger": list(self.trigger),
            "outputs": self.outputs,
            **({} if self.root is None else {"root": self.root.to_document()}),
        }

    @classmethod
    def from_document(cls, document: object) -> LongContextResult:
        f = _mapping(document, "a long-context result")
        scalars = (
            "status reason prompt_set_sha256 length max_new_tokens concurrency served_context n passed score seconds "
            "prefill_tokens outputs"
        ).split()
        return cls(
            tasks=tuple(map(LongContextTask.from_document, _items(f.get("tasks"), "tasks"))),
            prompts=tuple(map(LongContextPrompt.from_document, _items(f.get("prompts"), "prompts"))),
            trigger=tuple(_items(f.get("trigger", ()), "trigger")),  # type: ignore[arg-type]
            root=None if f.get("root") is None else LongContextRoot.from_document(f.get("root")),
            **{key: f.get(key) for key in scalars},  # type: ignore[arg-type]
        )


def root_reading(checks: Iterable[tuple[str, LongContextResult]]) -> LongContextRoot | None:
    """The root's reading from its tests' checks, ``(request id, check)`` in the order they ended: the complete ones on the
    prompt set of the last of them, pooled; ``None`` when none is complete (an incomplete check is no reading)."""
    complete = [(test, check) for test, check in checks if check.status == LONG_CONTEXT_COMPLETE]
    if not complete:
        return None
    prompt_set = complete[-1][1].prompt_set_sha256
    pooled = [(test, check) for test, check in complete if check.prompt_set_sha256 == prompt_set]
    n, passed = sum(check.n for _, check in pooled), sum(check.passed for _, check in pooled)
    low, high = wilson_interval(passed, n)
    return LongContextRoot(prompt_set, tuple(test for test, _ in pooled), n, passed, passed / n, low, high)


def refuses(candidate: LongContextResult | None, root: LongContextRoot | None) -> bool:
    """Whether the candidate's check shows a drop beyond the check's own sampling noise (#142 c3219, bar (b)).

    The drop is taken after the CI on the root's score: from the lower end of the root's 95% interval, in the candidate's
    items, ``root.ci_low * n - passed``. It refuses when that drop exceeds ``NOISE_SHARE`` of the candidate's ``n`` (3
    items of 64); a drop of exactly that much does not. Nothing is refused without a complete check, without a root
    reading, or when the two were made on different prompt sets: an ``incomplete`` check is recorded and never blocks.
    """
    if candidate is None or root is None or candidate.status != LONG_CONTEXT_COMPLETE:
        return False
    if candidate.prompt_set_sha256 != root.prompt_set_sha256:
        return False
    return root.ci_low * candidate.n - candidate.passed > NOISE_SHARE * candidate.n + _EPSILON
