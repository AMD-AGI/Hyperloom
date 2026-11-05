# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Find a staged prompt set in the runner cache and show that a runner can read it: the read side of ``staging``.

A host that builds a toolbox says which prompt set the check uses by its key (and, from a sheet, by its hash). Before it
agrees, it reads the entry the way a runner does (``load_prompt_set``): the manifest, the file's size and SHA-256, and
the prompts themselves, so a set that is missing, unfinished or not the named one is refused where it is asked for,
not by a runner after a candidate's rounds.
"""

from __future__ import annotations

from collections.abc import Mapping

from hyperloom.measure.docs import DocumentError
from hyperloom.measure.short_prompts.plan import ShortPromptPlan
from hyperloom.measure.short_prompts.prompts import PromptSetError
from hyperloom.measure.short_prompts.staging import (
    EVALSETS_DIR,
    StagedEntry,
    cache_root,
    load_prompt_set,
    recorded_sha256,
)

__all__ = ["staged_entry", "verified_entry"]

#: A well-formed hash that names nothing: it lets a key be checked as a name before the entry's own hash is read.
_ANY_SHA256 = "0" * 64


def _plan(key: str, sha256: object) -> ShortPromptPlan:
    """The plan that names the entry, which is what refuses a key that is not a plain name or a value that is not a hash
    (``sha256`` is whatever a manifest recorded, so it may not be text)."""
    if not isinstance(sha256, str):
        raise PromptSetError(f"short prompts: prompts_sha256 is {type(sha256).__name__}, not a sha256")
    try:
        return ShortPromptPlan(key, sha256)
    except DocumentError as exc:
        raise PromptSetError(str(exc)) from exc


def verified_entry(key: str, sha256: str, environ: Mapping[str, str]) -> StagedEntry:
    """The entry ``key`` in the runner cache ``environ`` names, shown to hold the prompts with SHA-256 ``sha256``.

    Raises:
        PromptSetError: The entry is missing, unfinished, holds other prompts, or its files are not what its manifest
            recorded; or ``key`` or ``sha256`` is malformed.
    """
    plan = _plan(key, sha256)
    load_prompt_set(plan, environ)
    return StagedEntry(key, cache_root(environ) / EVALSETS_DIR / key, sha256)


def staged_entry(key: str, environ: Mapping[str, str]) -> StagedEntry:
    """The entry ``key`` in the runner cache ``environ`` names, with the hash its own manifest records, after the same checks
    as :func:`verified_entry`.

    Raises:
        PromptSetError: As :func:`verified_entry`, and no readable manifest.
    """
    _plan(key, _ANY_SHA256)
    return verified_entry(key, recorded_sha256(cache_root(environ) / EVALSETS_DIR / key), environ)
