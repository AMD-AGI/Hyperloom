# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The long-context check as a test result carries it.

A frozen set of long synthetic prompts (needle retrieval, multi-key retrieval, variable tracking, common-word extraction) is
answered greedily by the candidate's own server and graded by substring match. The types here state the scores and what
each prompt did, never a verdict: what refuses a candidate is the gate's rule, not this record. A prompt that errored or was
never run counts as not passed and makes the check ``incomplete``, which is never a pass. Like the rest of the measurement
contract they are frozen dataclasses that validate on construction and round-trip through ``to_document()`` and
``from_document()``.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    "LONG_CONTEXT_COMPLETE",
    "LONG_CONTEXT_INCOMPLETE",
    "LONG_CONTEXT_STATUSES",
    "PROMPT_ERROR",
    "PROMPT_NOT_RUN",
    "PROMPT_OK",
    "PROMPT_STATUSES",
    "LongContextPrompt",
    "LongContextResult",
    "LongContextTask",
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

    def __post_init__(self) -> None:
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
            **{key: f.get(key) for key in scalars},  # type: ignore[arg-type]
        )
