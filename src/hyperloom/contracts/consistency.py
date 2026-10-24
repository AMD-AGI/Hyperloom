# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The short-prompt consistency check as a test result carries it.

A frozen set of short prompts is answered greedily by several fresh server processes of the candidate's launch; the check
compares the answers across those processes and with the parent's own answers. The types here state counts and
positions, never a verdict: what refuses a candidate is the gate's rule, not this record. Like the rest of the
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
    "REFERENCE_ABSENT",
    "REFERENCE_COMPARED",
    "REFERENCE_STATUSES",
    "SHORT_PROMPTS_COMPLETE",
    "SHORT_PROMPTS_INCOMPLETE",
    "SHORT_PROMPTS_STATUSES",
    "OutputCounts",
    "OutputDifference",
    "ShortPromptConsistency",
    "ShortPromptReference",
]

#: How far a short-prompt consistency check got: every planned server process answered every prompt, or not. A reference is
#: the parent's own outputs: compared, or absent (never read as "no difference").
SHORT_PROMPTS_COMPLETE = "complete"
SHORT_PROMPTS_INCOMPLETE = "incomplete"
SHORT_PROMPTS_STATUSES = (SHORT_PROMPTS_COMPLETE, SHORT_PROMPTS_INCOMPLETE)
REFERENCE_COMPARED = "compared"
REFERENCE_ABSENT = "absent"
REFERENCE_STATUSES = (REFERENCE_COMPARED, REFERENCE_ABSENT)
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
    """One comparison of a short-prompt check: for each completed process (in the order of the check's ``processes``) how
    many prompts' outputs differ from what it is compared with, and the first difference found.

    ``unstable_prompts`` is only for a comparison with a reference: the prompts the reference's own processes disagree
    on, which it cannot say the answer to and which are left out of ``mismatches``.
    """

    mismatches: tuple[int, ...]
    first_difference: OutputDifference | None = None
    unstable_prompts: int = 0

    def __post_init__(self) -> None:
        _ints(self.mismatches, "mismatches")
        _int(self.unstable_prompts, "unstable_prompts", minimum=0)
        if self.first_difference is not None and not isinstance(self.first_difference, OutputDifference):
            raise MeasurementError(f"first_difference must be an OutputDifference, got {self.first_difference!r}")
        if (self.first_difference is not None) != any(self.mismatches):
            raise MeasurementError("a first difference is given exactly when some prompt differs")

    def to_document(self) -> dict[str, object]:
        return {
            "mismatches": list(self.mismatches),
            "first_difference": None if self.first_difference is None else self.first_difference.to_document(),
            "unstable_prompts": self.unstable_prompts,
        }

    @classmethod
    def from_document(cls, document: object) -> OutputCounts:
        f = _mapping(document, "output counts")
        return cls(
            mismatches=tuple(_items(f.get("mismatches"), "mismatches")),
            first_difference=_optional_difference(f.get("first_difference")),
            unstable_prompts=f.get("unstable_prompts", 0),
        )


@dataclass(frozen=True)
class ShortPromptReference:
    """What the parent's own outputs say: ``compared`` (``counts`` holds the comparison) or ``absent`` (``reason`` says
    why there is none; there is then no count, never a count of zero). ``request_id`` is the test the reference outputs
    come from."""

    status: str
    reason: str = ""
    request_id: str = ""
    counts: OutputCounts | None = None

    def __post_init__(self) -> None:
        if self.status not in REFERENCE_STATUSES:
            raise MeasurementError(
                f"a reference's status must be one of {list(REFERENCE_STATUSES)}, got {self.status!r}"
            )
        if not isinstance(self.reason, str) or not isinstance(self.request_id, str):
            raise MeasurementError("a reference's reason and request_id must be strings")
        if self.status == REFERENCE_COMPARED and not isinstance(self.counts, OutputCounts):
            raise MeasurementError("a compared reference has counts")
        if self.status == REFERENCE_ABSENT and (self.counts is not None or not self.reason):
            raise MeasurementError("an absent reference has no counts and says why")

    def to_document(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "request_id": self.request_id,
            "counts": None if self.counts is None else self.counts.to_document(),
        }

    @classmethod
    def from_document(cls, document: object) -> ShortPromptReference:
        f = _mapping(document, "a short-prompt reference")
        counts = f.get("counts")
        return cls(
            status=f.get("status"),
            reason=f.get("reason", ""),
            request_id=f.get("request_id", ""),
            counts=None if counts is None else OutputCounts.from_document(counts),
        )


@dataclass(frozen=True)
class ShortPromptConsistency:
    """The short-prompt consistency check of one test: a frozen set of short prompts, answered greedily by fresh server
    processes of the candidate's launch, compared across those processes and with the parent's outputs.

    It states counts and positions, never a verdict. ``status`` is ``complete`` when every one of the
    ``processes_planned`` processes answered every prompt and ``incomplete`` otherwise (``reason`` says why); ``processes``
    are the indices of those that did. ``miss_probability_one_third`` is the chance that that many launches all miss a
    defect that shows in one launch in three. ``self_consistency`` compares the candidate's processes with each other
    (``None`` below two processes), ``reference`` with the parent's. ``trigger`` says why the check ran, ``outputs``
    where the raw outputs are in the test's output directory.
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
    reference: ShortPromptReference
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
        if not isinstance(self.reference, ShortPromptReference):
            raise MeasurementError(f"reference must be a ShortPromptReference, got {self.reference!r}")
        counted = [self.self_consistency, self.reference.counts]
        if any(c is not None and not isinstance(c, OutputCounts) for c in counted):
            raise MeasurementError("a comparison is OutputCounts")
        if any(c is not None and len(c.mismatches) != len(self.processes) for c in counted):
            raise MeasurementError("each comparison counts every completed process")
        if self.self_consistency is not None and (len(self.processes) < 2 or self.self_consistency.unstable_prompts):
            raise MeasurementError("self-consistency needs two processes and has no unstable prompts")

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
            "reference": self.reference.to_document(),
            "outputs": self.outputs,
        }

    @classmethod
    def from_document(cls, document: object) -> ShortPromptConsistency:
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
            reference=ShortPromptReference.from_document(f.get("reference")),
            outputs=f.get("outputs", ""),
        )
