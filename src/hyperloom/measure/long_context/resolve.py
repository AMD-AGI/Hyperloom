# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Find a staged long-context set in the runner cache and show that a runner can read it: the read side of ``staging``.

A host that builds a toolbox names the set by its key. Before it agrees, it reads the entry the way a runner does: the
manifest, the prompts file's size and SHA-256, the prompts themselves, and the token length the entry was built for, so a
set that is missing, unfinished or not the named one is refused where it is asked for, not by a runner after a candidate's
rounds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from hyperloom.measure.docs import DocumentError
from hyperloom.measure.long_context.plan import MIN_LENGTH, LongContextPlan
from hyperloom.measure.long_context.prompts import PromptSetError
from hyperloom.measure.long_context.staging import load_prompt_set, manifest_length
from hyperloom.measure.short_prompts.staging import EVALSETS_DIR, cache_root, recorded_sha256

__all__ = ["StagedSet", "staged_set"]


@dataclass(frozen=True)
class StagedSet:
    """A set in the cache: its key, the SHA-256 of its prompts file and the token length its prompts were built for."""

    prompt_set: str
    prompts_sha256: str
    length: int


def staged_set(key: str, environ: Mapping[str, str]) -> StagedSet:
    """The set ``key`` in the runner cache ``environ`` names, with the hash and length its own manifest records, after the
    checks a runner makes.

    Raises:
        PromptSetError: ``key`` is not a plain name, or the entry is missing, unfinished, holds other prompts than its
            manifest recorded, or holds a prompt longer than its recorded length.
    """
    entry = cache_root(environ) / EVALSETS_DIR / key
    try:
        LongContextPlan(key, "0" * 64, MIN_LENGTH)  # the key as a name, before it is used as a path
        sha256, length = recorded_sha256(entry), manifest_length(entry)
        load_prompt_set(LongContextPlan(key, sha256, length), environ)
    except DocumentError as exc:
        raise PromptSetError(str(exc)) from exc
    return StagedSet(key, sha256, length)
