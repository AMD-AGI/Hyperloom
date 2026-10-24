# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The frozen set of short prompts, and the builder that makes it.

The set is 64 prompts of 1 to 64 tokens in two modes: ``sequential`` prompts are sent one at a time, ``batched`` prompts
all at once, so the server's scheduler forms prefill batches of several short sequences. Each prompt is the first
``length`` tokens of a passage of its own, tokenised once, and every prompt starts with a different token: the prefix
cache of a server then cannot serve one prompt from another, and each prompt really runs its short prefill. The
passages come from a fixed seeded generator over a fixed word list, so the same tokenizer and seed give the same file.

The prompts are token ids, sent as they are (no special tokens added): the file is what the server is asked, whatever its
tokenizer would have done to the text.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from hyperloom.measure.docs import DocumentError, canonical_json, parse_json_object
from hyperloom.measure.short_prompts.compare import PromptInfo

__all__ = [
    "BATCHED",
    "GENERATOR",
    "LENGTHS",
    "SEQUENTIAL",
    "Prompt",
    "PromptSetError",
    "build_prompts",
    "load_prompts",
    "prompts_bytes",
]

SEQUENTIAL = "sequential"
BATCHED = "batched"
#: The version of the generator: a change to the lengths, the word lists or the passage rule changes it.
GENERATOR = "hl-short-prompts/1"
#: The prompt lengths in tokens of each mode: single tokens (twice, with different tokens), every odd length up to 33 and
#: the lengths around the block sizes a prefill kernel tiles by.
LENGTHS = {
    SEQUENTIAL: tuple(
        map(int, "1 1 2 3 4 5 6 7 8 9 10 11 13 15 16 17 19 21 23 25 27 29 31 32 33 35 40 47 48 55 63 64".split())
    ),
    BATCHED: tuple(
        map(int, "1 2 3 5 7 9 11 12 14 17 18 20 22 24 26 28 30 31 33 34 36 37 39 41 43 45 49 51 57 59 61 64".split())
    ),
}
_MODE_PREFIX = {SEQUENTIAL: "s", BATCHED: "b"}
_MAX_LENGTH = 1024
_PASSAGE_WORDS = 120
_REFILLS = 3
_START_WORDS = tuple(
    """The When Water Every Bright Several People Under After Because Children Over Often Small Before Between Quiet Today
    Most Rivers Without Through Many Early Behind Green Farmers Some Winter Across Light During Nothing Silver Great Songs
    Inside Mountains Perhaps Stone Another Morning Slowly Books Along Heavy Toward Painters Whenever Gentle Within
    Travelers Open Beyond Fresh Machines Near Maps Against Kitchens Until Windows Beside Harbors Once Ancient Around
    Letters Clouds Lanterns Meadows Bridges Orchards Villages Engines Gardens Markets Towers""".split()
)
_WORDS = tuple(
    """river walked slowly through the old town while people talked about weather and market prices in a quiet voice that
    carried across square morning light fell on stone walls children played near fountain farmers brought bread apples
    cheese from valley beyond hills where snow melts early spring teacher wrote letters to her friends far away of with
    was were had never seen such large crowd before after long winter boats left harbor at dawn carrying salt wool timber
    engines hummed low tone painters mixed colors small rooms above bakery books lay open tables maps showed roads no
    longer used by travelers because new bridge opened last year some say it will rain tomorrow others think sky stay clear
    until evening lanterns were lit along banks songs drifted over water each step echoed empty halls everything seemed
    calm yet something felt different gardens grew wild behind houses nobody remembered who planted them""".split()
)


class PromptSetError(Exception):
    """A prompt set that cannot be built or read; the message says what is wrong."""


@dataclass(frozen=True)
class Prompt:
    """One prompt: its id, the mode it is sent in and its token ids."""

    id: str
    mode: str
    token_ids: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.token_ids)

    def info(self) -> PromptInfo:
        return PromptInfo(self.id, self.mode, self.length)

    def to_document(self) -> dict[str, object]:
        return {"id": self.id, "mode": self.mode, "token_ids": list(self.token_ids)}


def _passage(rng: random.Random, start: str) -> str:
    return " ".join([start, *(rng.choice(_WORDS) for _ in range(_PASSAGE_WORDS))])


def _prompt_ids(encode: Callable[[str], Sequence[int]], rng: random.Random, start: str, length: int) -> tuple[int, ...]:
    """The first ``length`` token ids of a passage that starts with ``start``, drawing more words if it is too short."""
    for _ in range(_REFILLS):
        ids = list(encode(_passage(rng, start)))
        if len(ids) >= length:
            return tuple(int(i) for i in ids[:length])
    raise PromptSetError(f"the tokenizer gives fewer than {length} tokens for a passage of {_PASSAGE_WORDS + 1} words")


def build_prompts(encode: Callable[[str], Sequence[int]], *, seed: int) -> tuple[Prompt, ...]:
    """The 64 prompts for a tokenizer's ``encode`` (text to ids, no special tokens) and a seed.

    A start word whose first token an earlier prompt already starts with is skipped, so no two prompts share a first
    token; running out of start words is an error.
    """
    rng = random.Random(seed)
    starts = iter(dict.fromkeys(_START_WORDS))
    used: set[int] = set()
    prompts = []
    for mode, lengths in LENGTHS.items():
        for index, length in enumerate(lengths):
            ids = _fresh_first_token(encode, rng, starts, length, used)
            used.add(ids[0])
            prompts.append(Prompt(f"{_MODE_PREFIX[mode]}{index:02d}", mode, ids))
    return tuple(prompts)


def _fresh_first_token(encode, rng: random.Random, starts, length: int, used: set[int]) -> tuple[int, ...]:
    for start in starts:
        ids = _prompt_ids(encode, rng, start, length)
        if ids[0] not in used:
            return ids
    raise PromptSetError("the tokenizer maps the start words to too few different first tokens")


def prompts_bytes(prompts: Sequence[Prompt]) -> bytes:
    """The file's content: one canonical JSON object per prompt, one per line."""
    return "".join(canonical_json(prompt.to_document()) + "\n" for prompt in prompts).encode("utf-8")


def _read_prompt(line: bytes, number: int) -> Prompt:
    try:
        document = parse_json_object(line, what=f"prompts line {number}")
    except DocumentError as exc:
        raise PromptSetError(str(exc)) from None
    ids = document.get("token_ids")
    if set(document) != {"id", "mode", "token_ids"} or not isinstance(document["id"], str) or not document["id"]:
        raise PromptSetError(f"prompts line {number}: expected the keys id, mode and token_ids")
    if document["mode"] not in LENGTHS:
        raise PromptSetError(f"prompts line {number}: mode {document['mode']!r} is not one of {sorted(LENGTHS)}")
    if not isinstance(ids, list) or not 1 <= len(ids) <= _MAX_LENGTH:
        raise PromptSetError(f"prompts line {number}: token_ids must be a list of 1 to {_MAX_LENGTH} integers")
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids):
        raise PromptSetError(f"prompts line {number}: token_ids must be non-negative integers")
    return Prompt(document["id"], document["mode"], tuple(ids))


def load_prompts(path: Path) -> tuple[Prompt, ...]:
    """The prompts of a staged file; a file that is not a prompt set, or repeats an id, is refused."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise PromptSetError(f"{path} cannot be read: {type(exc).__name__}") from None
    prompts = tuple(_read_prompt(line, number) for number, line in enumerate(raw.splitlines(), 1))
    ids = [prompt.id for prompt in prompts]
    if not prompts or len(set(ids)) != len(ids):
        raise PromptSetError(f"{path} holds no prompt, or an id twice")
    return prompts
