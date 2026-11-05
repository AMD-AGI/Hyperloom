# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The generators of the four tasks, RULER-style, built offline from a seed and a tokenizer.

* ``niah_single``: one needle (a magic number for a key) somewhere in a haystack of repeated filler sentences.
* ``niah_multikey``: four needles for four keys; the question names one, so the others are distractors.
* ``variable_tracking``: two chains of variable assignments (``VAR B = VAR A``) spread over the filler; the question asks
  for the variables that hold the value at the start of one chain.
* ``common_words``: a numbered list of words, five of which appear far more often than the rest; the question asks for them.

Each prompt is made as long as the tokenizer allows within ``length`` tokens (and within 1% of it), by searching for the
number of filler units; needle depths are spread evenly over the prompts of a task so every depth is sampled. All of the
randomness comes from the seed, so the same tokenizer, length and seed give the same prompts.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import NamedTuple

from hyperloom.measure.long_context.prompts import PER_TASK, TASKS, Prompt, PromptSetError

__all__ = ["build_prompts"]

Encode = Callable[[str], Sequence[int]]
_FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
_SYLLABLES = tuple(c + v for c in "bdfgklmnprstvz" for v in "aeiou")
_SLACK = 0.01
_FIT_ROUNDS = 8
_PROBE_UNITS = 16
_NEEDLE = "One of the special magic numbers for {key} is: {value}."
_NUMBER_DIGITS = 7
_CHAIN_HOPS = 4
#: The share of the way through a prompt its first chain statement is at most, and a chain's last one at most ``_CHAIN_END``.
_CHAIN_START_SHARE = 0.5
_CHAIN_END = 0.95
_POOL_MARGIN = 8
_COMMON_WORDS = 5
_COMMON_FREQ = 30
_UNCOMMON_FREQ = 3
_MULTIKEY_NEEDLES = 4


class _Made(NamedTuple):
    """One prompt made by a task: its text, the tokens the tokenizer counted for it and the references an answer holds."""

    text: str
    tokens: int
    references: list[str]


class _Chain(NamedTuple):
    """One chain of assignments: its statements with their depths, its variables and the value they come to."""

    statements: list[tuple[float, str]]
    names: list[str]
    value: str


_MAGIC_QUESTION = "What is the special magic number for {key} mentioned in the provided text?"
_MAGIC_ANSWER = "The special magic number for {key} mentioned in the provided text is"
_MAGIC_INTRO = "Some special magic numbers are hidden within the following text. Make sure to memorize them."


def _words(rng: random.Random, count: int, exclude: frozenset[str] = frozenset()) -> list[str]:
    """``count`` distinct pronounceable made-up words, none of them in ``exclude``."""
    syllables = max(3, math.ceil(math.log(_POOL_MARGIN * count + 1, len(_SYLLABLES))))
    found: dict[str, None] = {}
    while len(found) < count:
        word = "".join(rng.choice(_SYLLABLES) for _ in range(syllables))
        if word not in exclude:
            found[word] = None
    return list(found)


def _number(rng: random.Random) -> str:
    return str(rng.randrange(10 ** (_NUMBER_DIGITS - 1), 10**_NUMBER_DIGITS))


def _fit(encode: Encode, render: Callable[[int], str], length: int, what: str) -> tuple[str, int]:
    """The longest ``render(units)`` of at most ``length`` tokens, and no more than 1% under it; ``render`` grows with ``units``."""
    base = len(encode(render(_PROBE_UNITS)))
    per_unit = (len(encode(render(2 * _PROBE_UNITS))) - base) / _PROBE_UNITS
    if per_unit <= 0:
        raise PromptSetError(f"{what}: a longer text does not give more tokens")
    units = max(1, _PROBE_UNITS + math.floor((length - base) / per_unit))
    for _ in range(_FIT_ROUNDS):
        text = render(units)
        tokens = len(encode(text))
        if length * (1 - _SLACK) <= tokens <= length:
            return text, tokens
        units = max(1, units + math.floor((length - tokens) / per_unit))
    raise PromptSetError(f"{what}: no text of {length} tokens (within 1%) was found for this tokenizer")


def _haystack(units: int, needles: Sequence[tuple[float, str]]) -> str:
    """``units`` filler sentences with each ``(depth, text)`` of ``needles`` inserted at that share of the way through."""
    sentences = [_FILLER] * units
    for depth, text in sorted(needles, key=lambda needle: -needle[0]):
        sentences.insert(min(units, int(depth * units)), text)
    return " ".join(sentences)


def _depth(index: int) -> float:
    """The depth of prompt ``index``: evenly spread over the prompts of a task, never at an end."""
    return (index + 0.5) / PER_TASK


def _magic(rng: random.Random, encode: Encode, length: int, index: int, others: int) -> _Made:
    """A needle prompt: the asked-for needle at the prompt's own depth and ``others`` distractors at random depths."""
    keys = [f"{a}-{b}" for a, b in zip(_words(rng, others + 1), _words(rng, others + 1))]
    values = [_number(rng) for _ in keys]
    needles = [(_depth(index), _NEEDLE.format(key=keys[0], value=values[0]))]
    needles += [(rng.random(), _NEEDLE.format(key=k, value=v)) for k, v in zip(keys[1:], values[1:])]
    question = f"{_MAGIC_QUESTION.format(key=keys[0])}\nAnswer: {_MAGIC_ANSWER.format(key=keys[0])}"
    text, tokens = _fit(encode, lambda n: f"{_MAGIC_INTRO}\n{_haystack(n, needles)}\n{question}", length, "needle task")
    return _Made(text, tokens, [values[0]])


def _niah_single(rng: random.Random, encode: Encode, length: int, index: int) -> _Made:
    return _magic(rng, encode, length, index, 0)


def _niah_multikey(rng: random.Random, encode: Encode, length: int, index: int) -> _Made:
    return _magic(rng, encode, length, index, _MULTIKEY_NEEDLES - 1)


def _chain(rng: random.Random, start: float) -> _Chain:
    """The statements of one chain of assignments, spread from depth ``start`` on, its variables and its value."""
    names = _words(rng, _CHAIN_HOPS)
    value = _number(rng)
    spread = (_CHAIN_END - start) / _CHAIN_HOPS
    statements = [f"VAR {names[0].upper()} = {value}"] + [
        f"VAR {names[i].upper()} = VAR {names[i - 1].upper()}" for i in range(1, _CHAIN_HOPS)
    ]
    return _Chain([(start + i * spread, f"{s}.") for i, s in enumerate(statements)], [n.upper() for n in names], value)


def _variable_tracking(rng: random.Random, encode: Encode, length: int, index: int) -> _Made:
    target = _chain(rng, _depth(index) * _CHAIN_START_SHARE)
    distractor = _chain(rng, rng.random() * _CHAIN_START_SHARE)
    value = target.value
    intro = "Memorize and track the chains of variable assignment hidden in the following text."
    question = (
        f"Question: Find all variables that are assigned the value {value} in the text above.\n"
        f"Answer: According to the chains of variable assignments in the text above, the variables that are assigned "
        f"the value {value} are:"
    )
    needles = target.statements + distractor.statements
    text, tokens = _fit(encode, lambda n: f"{intro}\n{_haystack(n, needles)}\n{question}", length, "tracking task")
    return _Made(text, tokens, target.names)


def _common_words(rng: random.Random, encode: Encode, length: int, index: int) -> _Made:
    common = _words(rng, _COMMON_WORDS)
    rare = _words(rng, length // _UNCOMMON_FREQ + _COMMON_FREQ, exclude=frozenset(common))
    seed = rng.random()

    def render(rare_lines: int) -> str:
        pool = [w for w in common for _ in range(_COMMON_FREQ)]
        pool += [w for w in rare for _ in range(_UNCOMMON_FREQ)][:rare_lines]
        random.Random(seed).shuffle(pool)
        listing = "\n".join(f"{i}. {word}" for i, word in enumerate(pool, 1))
        return (
            "Below is a numbered list of words. In these words, some appear more often than others. "
            f"Memorize the ones that appear most often.\n{listing}\n"
            f"Question: What are the {_COMMON_WORDS} most common words in the above list?\n"
            f"Answer: The top {_COMMON_WORDS} words that appear most often in the list are:"
        )

    text, tokens = _fit(encode, render, length, "word-list task")
    return _Made(text, tokens, common)


_TASKS = {
    "niah_single": _niah_single,
    "niah_multikey": _niah_multikey,
    "variable_tracking": _variable_tracking,
    "common_words": _common_words,
}


def build_prompts(encode: Encode, *, length: int, seed: int) -> tuple[Prompt, ...]:
    """The ``PER_TASK`` prompts of each task for a tokenizer's ``encode`` (text to ids, no special tokens), each of at most
    ``length`` tokens, and a seed.

    Raises:
        PromptSetError: The tokenizer gives no text of that length within 1%, or the length is too short for a task.
    """
    prompts = []
    for task in TASKS:
        rng = random.Random(f"{seed}:{task}:{length}")
        for index in range(PER_TASK):
            made = _TASKS[task](rng, encode, length, index)
            prompts.append(Prompt(f"{task}-{index:02d}", task, made.text, tuple(made.references), made.tokens))
    return tuple(prompts)
