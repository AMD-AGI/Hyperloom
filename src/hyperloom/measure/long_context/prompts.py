# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One prompt of the frozen set, and the file that holds the set.

``data/prompts.jsonl`` of an entry has one canonical JSON object per line: ``{"id", "task", "input", "outputs", "tokens"}``.
``input`` is the whole prompt text (sent as it is, no chat template), ``outputs`` the references an answer must contain and
``tokens`` the length the builder's tokenizer measured for ``input``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hyperloom.measure.docs import DocumentError, canonical_json, parse_json_object
from hyperloom.measure.short_prompts.prompts import PromptSetError, load_lines

__all__ = ["GENERATOR", "PER_TASK", "TASKS", "Prompt", "PromptSetError", "load_prompts", "prompts_bytes"]

#: The version of the generators: a change to a task, the filler or the placement rule changes it.
GENERATOR = "hl-long-context/1"
TASKS = ("niah_single", "niah_multikey", "variable_tracking", "common_words")
PER_TASK = 16
_KEYS = {"id", "task", "input", "outputs", "tokens"}


@dataclass(frozen=True)
class Prompt:
    """One prompt: its id, its task, the text, the references an answer must contain and the text's length in tokens."""

    id: str
    task: str
    text: str
    references: tuple[str, ...]
    tokens: int

    def to_document(self) -> dict[str, object]:
        return {
            "id": self.id,
            "task": self.task,
            "input": self.text,
            "outputs": list(self.references),
            "tokens": self.tokens,
        }


def prompts_bytes(prompts: Sequence[Prompt]) -> bytes:
    """The file's content: one canonical JSON object per prompt, one per line."""
    return "".join(canonical_json(prompt.to_document()) + "\n" for prompt in prompts).encode("utf-8")


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _read_prompt(line: bytes, number: int) -> Prompt:
    where = f"prompts line {number}"
    try:
        document = parse_json_object(line, what=where)
    except DocumentError as exc:
        raise PromptSetError(str(exc)) from None
    refs = document.get("outputs")
    if set(document) != _KEYS or not all(isinstance(document[k], str) and document[k] for k in ("id", "input")):
        raise PromptSetError(f"{where}: expected the keys {sorted(_KEYS)}, with a non-empty id and input")
    if document["task"] not in TASKS:
        raise PromptSetError(f"{where}: task {document['task']!r} is not one of {list(TASKS)}")
    if not isinstance(refs, list) or not refs or not all(isinstance(r, str) and r for r in refs):
        raise PromptSetError(f"{where}: outputs must be a non-empty list of non-empty strings")
    if not _is_count(document["tokens"]):
        raise PromptSetError(f"{where}: tokens must be a positive integer")
    return Prompt(document["id"], document["task"], document["input"], tuple(refs), document["tokens"])


def load_prompts(path: Path) -> tuple[Prompt, ...]:
    """The prompts of a staged file; a file that is not a prompt set, or repeats an id, is refused."""
    return load_lines(path, _read_prompt)
