# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The offline entry of the long-context prompt set: the layout and checks of ``measure.short_prompts.staging``.

An entry is ``<cache>/evalsets/<key>/data/prompts.jsonl`` and a ``manifest`` written last, the same layout the short-prompt
set uses; the key hashes this generator and its inputs (seed, token length, tokenizer file hashes), so another length or
tokenizer is another entry. This module adds what is particular to the set: the generator, the key prefix and the length the
manifest records.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hyperloom.measure.long_context.plan import LongContextPlan
from hyperloom.measure.long_context.prompts import GENERATOR, Prompt, PromptSetError, parse_prompts, prompts_bytes
from hyperloom.measure.short_prompts.staging import (
    EVALSETS_DIR,
    MANIFEST_NAME,
    StagedEntry,
    cache_root,
    stage_file,
    verified_prompts,
)

__all__ = ["KEY_PREFIX", "load_prompt_set", "manifest_bound", "manifest_length", "stage_entry"]

KEY_PREFIX = "long-context-"
_UTF8 = "utf-8"


def stage_entry(root: Path, prompts: Sequence[Prompt], inputs: Mapping[str, Any]) -> StagedEntry:
    """Write the entry for ``prompts`` below ``root`` (``inputs`` holds the seed, the length and the tokenizer hashes)."""
    return stage_file(root, prompts_bytes(prompts), inputs, generator=GENERATOR, prefix=KEY_PREFIX)


def _manifest_inputs(entry: Path) -> Mapping[str, Any]:
    try:
        inputs = json.loads((entry / MANIFEST_NAME).read_text(encoding=_UTF8))["inputs"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PromptSetError(f"the entry {entry} records no inputs ({type(exc).__name__})") from exc
    if not isinstance(inputs, Mapping):
        raise PromptSetError(f"the entry {entry} records inputs that are not an object")
    return inputs


def _manifest_integer(entry: Path, name: str) -> int | None:
    """The integer ``name`` of the manifest's inputs, ``None`` when it records none."""
    value = _manifest_inputs(entry).get(name)
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise PromptSetError(f"the entry {entry} records a {name} that is not an integer")
    return value


def manifest_length(entry: Path) -> int:
    """The token length the manifest of ``entry`` records its prompts were built for.

    Raises:
        PromptSetError: The manifest cannot be read or records no length.
    """
    length = _manifest_integer(entry, "length")
    if length is None:
        raise PromptSetError(f"the entry {entry} records no length")
    return length


def manifest_bound(entry: Path) -> int | None:
    """The context bound the manifest of ``entry`` records its set was built under; ``None`` for a set that records none
    (one built before the bound).

    Raises:
        PromptSetError: The manifest cannot be read, or records a bound that is not an integer.
    """
    return _manifest_integer(entry, "bound")


def load_prompt_set(plan: LongContextPlan, environ: Mapping[str, str]) -> tuple[Prompt, ...]:
    """The prompts ``plan`` names, parsed from the very bytes whose hashes were checked.

    Raises:
        PromptSetError: The entry is missing, unfinished, or not the one the plan names; or a prompt is longer than the
            plan's length.
    """
    entry = cache_root(environ) / EVALSETS_DIR / plan.prompt_set
    prompts = parse_prompts(verified_prompts(entry, plan.prompts_sha256), str(entry))
    longest = max(prompt.tokens for prompt in prompts)
    if longest > plan.length:
        raise PromptSetError(
            f"the entry {entry} holds a prompt of {longest} tokens, the sheet's length is {plan.length}"
        )
    return prompts
