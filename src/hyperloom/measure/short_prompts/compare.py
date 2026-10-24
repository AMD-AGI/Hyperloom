# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compare the outputs of several server processes with each other and with a reference.

An output of a prompt is its generated tokens as the server printed them (its text when any output lacks tokens). For
each prompt the *consensus* of a set of processes is the output most of them gave (a tie goes to the lowest process
index). Two comparisons are made, and neither says pass or fail, only how many prompts differ and where:

* self-consistency: for each candidate process, the prompts whose output is not the consensus of the candidate's own
  processes. A server that gives different answers to the same greedy prompt in different launches has a defect whatever
  its parent does; the check needs at least two processes.
* against the reference (the parent's processes): for each candidate process, the prompts whose output is not the
  reference's consensus. A prompt the reference's own processes disagree on is *unstable* there: counted apart and left
  out, because the reference cannot say what the answer is.

Where two outputs differ, the position of the first difference (a token index, or a character index when the comparison is
by text) and the start of both texts say whether the output is corrupted from its first tokens or drifts late, as two
different kernels' rounding can make a greedy decode do.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


__all__ = [
    "BY_TEXT",
    "BY_TOKENS",
    "Comparison",
    "Difference",
    "Output",
    "Outputs",
    "PromptInfo",
    "Side",
    "compare_outputs",
]

BY_TOKENS = "tokens"
BY_TEXT = "text"
_EXCERPT_CHARS = 80
_Signature = tuple[str, ...] | str


@dataclass(frozen=True)
class PromptInfo:
    """What a comparison needs to know of a prompt: its id, its mode and its length in tokens."""

    id: str
    mode: str
    length: int


@dataclass(frozen=True)
class Output:
    """One prompt's output in one process."""

    text: str
    tokens: tuple[str, ...] | None = None
    finish_reason: str = ""


#: The outputs of one process, by prompt id.
Outputs = Mapping[str, Output]


@dataclass(frozen=True)
class Difference:
    """Where an output first departs from what it is compared with."""

    process: int
    prompt_id: str
    mode: str
    prompt_tokens: int
    position: int
    expected: str
    actual: str


@dataclass(frozen=True)
class Side:
    """One comparison: per process (in the order of its index) how many prompts differ, the first difference found, and
    for a reference the number of prompts it is unstable on."""

    mismatches: tuple[int, ...]
    first: Difference | None
    unstable: int = 0


@dataclass(frozen=True)
class Comparison:
    """``self_consistency`` is ``None`` below two candidate processes, ``reference`` when there is no usable reference."""

    compared_by: str
    self_consistency: Side | None
    reference: Side | None


def _compared_by(groups: Sequence[Mapping[int, Outputs]]) -> str:
    every = (out for group in groups for outputs in group.values() for out in outputs.values())
    return BY_TOKENS if all(out.tokens is not None for out in every) else BY_TEXT


def _signature(output: Output, by: str) -> _Signature:
    return output.tokens if by == BY_TOKENS and output.tokens is not None else output.text


def _consensus(signatures: Sequence[_Signature]) -> _Signature:
    """The signature most processes gave; the first of those with the highest count."""
    counts = Counter(signatures)
    best = max(counts.values())
    return next(sig for sig in signatures if counts[sig] == best)


def _position(first: _Signature, second: _Signature) -> int:
    for index, (a, b) in enumerate(zip(first, second)):
        if a != b:
            return index
    return min(len(first), len(second))


def _difference(process: int, prompt: PromptInfo, expected: Output, actual: Output, by: str) -> Difference:
    position = _position(_signature(expected, by), _signature(actual, by))
    return Difference(
        process,
        prompt.id,
        prompt.mode,
        prompt.length,
        position,
        expected.text[:_EXCERPT_CHARS],
        actual.text[:_EXCERPT_CHARS],
    )


def _side(
    candidate: Mapping[int, Outputs], reference: Mapping[int, Outputs], prompts: Sequence[PromptInfo], by: str
) -> Side:
    """The candidate processes against the consensus of ``reference`` (which may be the candidate itself)."""
    unstable = 0
    consensus: dict[str, Output] = {}
    for prompt in prompts:
        given = [outputs[prompt.id] for _, outputs in sorted(reference.items()) if prompt.id in outputs]
        if not given:
            continue
        sigs = [_signature(out, by) for out in given]
        if len(set(sigs)) > 1 and reference is not candidate:
            unstable += 1
            continue
        wanted = _consensus(sigs)
        consensus[prompt.id] = next(out for out, sig in zip(given, sigs) if sig == wanted)
    counts, first = [], None
    for index, outputs in sorted(candidate.items()):
        differing = [
            (prompt, consensus[prompt.id], outputs[prompt.id])
            for prompt in prompts
            if prompt.id in consensus and _signature(outputs[prompt.id], by) != _signature(consensus[prompt.id], by)
        ]
        counts.append(len(differing))
        if first is None and differing:
            first = _difference(index, *differing[0], by)
    return Side(tuple(counts), first, unstable)


def compare_outputs(
    candidate: Mapping[int, Outputs], reference: Mapping[int, Outputs] | None, prompts: Sequence[PromptInfo]
) -> Comparison:
    """The comparisons of the candidate's processes (by index) with each other and with ``reference``'s, if it has any.

    A candidate process must hold an output for every prompt in ``prompts``; the rounds only record such processes.
    """
    groups = [candidate] if not reference else [candidate, reference]
    by = _compared_by(groups)
    own = _side(candidate, candidate, prompts, by) if len(candidate) >= 2 else None
    against = _side(candidate, reference, prompts, by) if reference else None
    return Comparison(by, own, against)
