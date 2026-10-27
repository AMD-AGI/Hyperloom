# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The short-prompt consistency check as a test result carries it.

A frozen set of short prompts is answered greedily by several fresh server processes of the candidate's launch; the check
compares the answers across those processes (self-consistency only: a comparison with the parent's answers fires on
every change that moves greedy tokens early, fp8 KV and quantized GEMMs among them, and a defect that is wrong in every
process is the accuracy evaluation's to see). The types here state counts and positions, never a verdict: what refuses a
candidate is the gate's rule, not this record. The gate holds a candidate's :attr:`ShortPromptConsistency.inconsistency`
to the one the session's baseline check measured. Like the rest of the
measurement contract they are frozen dataclasses that validate on construction and round-trip through ``to_document()``
and ``from_document()``.
"""

from __future__ import annotations

from dataclasses import dataclass

from hyperloom.contracts._validate import (
    MeasurementError,
    _int,
    _items,
    _mapping,
    _number,
    _relative_dir,
    _sha256,
    _strings,
    _text,
)

__all__ = [
    "COMPARED_BY",
    "SHORT_PROMPTS_COMPLETE",
    "SHORT_PROMPTS_INCOMPLETE",
    "SHORT_PROMPTS_STATUSES",
    "OutputCounts",
    "OutputDifference",
    "ShortPromptConsistency",
]

#: How far a short-prompt consistency check got: every planned server process answered every prompt, or not. An
#: incomplete check is recorded and is not a pass; it holds nothing back either.
SHORT_PROMPTS_COMPLETE = "complete"
SHORT_PROMPTS_INCOMPLETE = "incomplete"
SHORT_PROMPTS_STATUSES = (SHORT_PROMPTS_COMPLETE, SHORT_PROMPTS_INCOMPLETE)
#: What the outputs were compared by: the generated tokens, or the text when a server returned no tokens; ``""`` when no
#: comparison was made.
COMPARED_BY = ("", "tokens", "text")


def _ints(value: object, what: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise MeasurementError(f"{what} must be a tuple of integers, got {value!r}")
    return tuple(_int(item, what, minimum=0) for item in value)


@dataclass(frozen=True)
class OutputDifference:
    """Where one process's answer to a short prompt first departs from what it is compared with.

    ``position`` is the index of the first differing token (of the first differing character when the outputs were
    compared as texts); ``expected`` and ``actual`` are the starts of both texts. A corrupted output differs within its
    first tokens, a rounding drift late.
    """

    process: int
    prompt_id: str
    mode: str
    prompt_tokens: int
    position: int
    expected: str
    actual: str

    def __post_init__(self) -> None:
        for value, what in (
            (self.process, "process"),
            (self.prompt_tokens, "prompt_tokens"),
            (self.position, "position"),
        ):
            _int(value, f"a difference's {what}", minimum=0)
        _text(self.prompt_id, "a difference's prompt_id")
        _text(self.mode, "a difference's mode")
        if not isinstance(self.expected, str) or not isinstance(self.actual, str):
            raise MeasurementError("a difference's expected and actual must be strings")

    def to_document(self) -> dict[str, object]:
        return {
            "process": self.process,
            "prompt_id": self.prompt_id,
            "mode": self.mode,
            "prompt_tokens": self.prompt_tokens,
            "position": self.position,
            "expected": self.expected,
            "actual": self.actual,
        }

    @classmethod
    def from_document(cls, document: object) -> OutputDifference:
        f = _mapping(document, "an output difference")
        return cls(
            process=f.get("process"),
            prompt_id=f.get("prompt_id"),
            mode=f.get("mode"),
            prompt_tokens=f.get("prompt_tokens"),
            position=f.get("position"),
            expected=f.get("expected"),
            actual=f.get("actual"),
        )


def _optional_difference(value: object) -> OutputDifference | None:
    return None if value is None else OutputDifference.from_document(value)


@dataclass(frozen=True)
class OutputCounts:
    """The comparison of a short-prompt check's processes with each other: for each completed process (in the order of the
    check's ``processes``) how many prompts it answers differently from the consensus of the processes, and the first
    difference found."""

    mismatches: tuple[int, ...]
    first_difference: OutputDifference | None = None

    def __post_init__(self) -> None:
        _ints(self.mismatches, "mismatches")
        if self.first_difference is not None and not isinstance(self.first_difference, OutputDifference):
            raise MeasurementError(f"first_difference must be an OutputDifference, got {self.first_difference!r}")
        if (self.first_difference is not None) != any(self.mismatches):
            raise MeasurementError("a first difference is given exactly when some prompt differs")

    def to_document(self) -> dict[str, object]:
        return {
            "mismatches": list(self.mismatches),
            "first_difference": None if self.first_difference is None else self.first_difference.to_document(),
        }

    @classmethod
    def from_document(cls, document: object) -> OutputCounts:
        """Also reads a record written while the check compared with the parent: its ``unstable_prompts`` was always 0
        here and is left out."""
        f = _mapping(document, "output counts")
        return cls(
            mismatches=tuple(_items(f.get("mismatches"), "mismatches")),
            first_difference=_optional_difference(f.get("first_difference")),
        )


@dataclass(frozen=True)
class ShortPromptConsistency:
    """The short-prompt consistency check of one test: a frozen set of short prompts, answered greedily by fresh server
    processes of the candidate's launch, compared across those processes.

    It states counts and positions, never a verdict. ``status`` is ``complete`` when every one of the
    ``processes_planned`` processes answered every prompt and ``incomplete`` otherwise (``reason`` says why); ``processes``
    are the indices of those that did. ``miss_probability_one_third`` is the chance that that many launches all miss a
    defect that shows in one launch in three. ``self_consistency`` compares the candidate's processes with each other
    (``None`` below two processes). ``trigger`` says why the check ran, ``outputs`` where the raw outputs are in the
    test's output directory.
    """

    status: str
    reason: str
    prompt_set_sha256: str
    prompts: int
    max_new_tokens: int
    trigger: tuple[str, ...]
    processes_planned: int
    processes: tuple[int, ...]
    miss_probability_one_third: float
    compared_by: str
    self_consistency: OutputCounts | None
    outputs: str = ""

    def __post_init__(self) -> None:
        if self.status not in SHORT_PROMPTS_STATUSES:
            raise MeasurementError(f"a consistency check's status must be one of {list(SHORT_PROMPTS_STATUSES)}")
        if not isinstance(self.reason, str) or (self.status == SHORT_PROMPTS_INCOMPLETE and not self.reason):
            raise MeasurementError("an incomplete consistency check says why")
        _sha256(self.prompt_set_sha256, "the prompt set's sha256")
        _int(self.prompts, "prompts", minimum=0)
        _int(self.max_new_tokens, "max_new_tokens", minimum=1)
        _strings(self.trigger, "trigger")
        _int(self.processes_planned, "processes_planned", minimum=1)
        _ints(self.processes, "processes")
        if len(set(self.processes)) != len(self.processes) or len(self.processes) > self.processes_planned:
            raise MeasurementError("the completed processes are distinct and no more than planned")
        if (self.status == SHORT_PROMPTS_COMPLETE) != (len(self.processes) == self.processes_planned):
            raise MeasurementError("a check is complete exactly when every planned process completed")
        if not 0.0 <= _number(self.miss_probability_one_third, "miss_probability_one_third") <= 1.0:
            raise MeasurementError("miss_probability_one_third is a probability")
        if self.compared_by not in COMPARED_BY:
            raise MeasurementError(f"compared_by must be one of {list(COMPARED_BY)}, got {self.compared_by!r}")
        self._check_counts()
        _relative_dir(self.outputs, "outputs")

    def _check_counts(self) -> None:
        own = self.self_consistency
        if own is None:
            return
        if not isinstance(own, OutputCounts):
            raise MeasurementError(f"self_consistency must be OutputCounts, got {own!r}")
        if len(own.mismatches) != len(self.processes):
            raise MeasurementError("self-consistency counts every completed process")
        if len(self.processes) < 2:
            raise MeasurementError("self-consistency needs two processes")

    @property
    def inconsistency(self) -> int | None:
        """The most prompts one completed process answers differently from the consensus of the check's processes;
        ``None`` when fewer than two processes completed, so nothing was compared (never read as 0).

        The rule gate holds a candidate's to the one the session's baseline check measured (its floor). It is counted
        over the processes that completed, so an incomplete check has one too; its ``status`` says it is incomplete.
        """
        own = self.self_consistency
        return None if own is None else max(own.mismatches)

    def to_document(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "prompt_set_sha256": self.prompt_set_sha256,
            "prompts": self.prompts,
            "max_new_tokens": self.max_new_tokens,
            "trigger": list(self.trigger),
            "processes_planned": self.processes_planned,
            "processes": list(self.processes),
            "miss_probability_one_third": self.miss_probability_one_third,
            "compared_by": self.compared_by,
            "self_consistency": None if self.self_consistency is None else self.self_consistency.to_document(),
            "outputs": self.outputs,
        }

    @classmethod
    def from_document(cls, document: object) -> ShortPromptConsistency:
        """Also reads a record written while the check compared with the parent's outputs: its ``reference`` is left
        out."""
        f = _mapping(document, "a short-prompt consistency check")
        own = f.get("self_consistency")
        return cls(
            status=f.get("status"),
            reason=f.get("reason", ""),
            prompt_set_sha256=f.get("prompt_set_sha256"),
            prompts=f.get("prompts"),
            max_new_tokens=f.get("max_new_tokens"),
            trigger=tuple(_items(f.get("trigger", ()), "trigger")),
            processes_planned=f.get("processes_planned"),
            processes=tuple(_items(f.get("processes", ()), "processes")),
            miss_probability_one_third=f.get("miss_probability_one_third"),
            compared_by=f.get("compared_by", ""),
            self_consistency=None if own is None else OutputCounts.from_document(own),
            outputs=f.get("outputs", ""),
        )
