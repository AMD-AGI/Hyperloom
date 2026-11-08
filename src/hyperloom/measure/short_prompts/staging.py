# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The offline entry of a prompt set, in the layout of the runner cache.

A measurement sandbox has no network and reads its eval data from the runner cache directory
(``hyperloom.measure.runner.cache_layout``), mounted read-only. An entry is::

    <cache>/evalsets/<key>/data/prompts.jsonl    the prompts, one JSON object per line
    <cache>/evalsets/<key>/manifest              JSON, written last: the generator, its inputs (seed, tokenizer file
                                                 hashes) and the prompts file's count, size and SHA-256

``<key>`` hashes the generator and its inputs, so a different tokenizer or seed is another entry. An entry that is partly
written has no manifest, and a rewrite that would differ from the entry in place is refused. A reader reads the prompts
file once, checks those bytes and parses the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom_common.shared_storage import make_shared_dirs
from hyperloom.measure.docs import canonical_json
from hyperloom.measure.short_prompts.plan import ShortPromptPlan
from hyperloom.measure.short_prompts.prompts import GENERATOR, Prompt, PromptSetError, parse_prompts, prompts_bytes

__all__ = [
    "CACHE_ENV",
    "DATA_DIR",
    "DEFAULT_CACHE_DIR",
    "EVALSETS_DIR",
    "KEY_PREFIX",
    "MANIFEST_NAME",
    "PROMPTS_FILE",
    "StagedEntry",
    "cache_root",
    "entry_key",
    "load_prompt_set",
    "recorded_sha256",
    "stage_entry",
    "stage_file",
    "verified_prompts",
]

#: The directory of the runner cache, as ``hyperloom.measure.runner.cache_layout`` names it. The two are siblings, so this
#: package keeps its own copy and a test pins it to that one.
CACHE_ENV = "HYPERLOOM_RUNNER_CACHE"
DEFAULT_CACHE_DIR = "/runner-cache"
EVALSETS_DIR = "evalsets"
MANIFEST_NAME = "manifest"
DATA_DIR = "data"
PROMPTS_FILE = "prompts.jsonl"
#: Where an entry's prompts are, the only file its manifest may name.
_PROMPTS_PATH = f"{DATA_DIR}/{PROMPTS_FILE}"
KEY_PREFIX = "short-prompts-"
_SCHEMA = 1
_KEY_DIGITS = 16
_ENTRY_DIR_MODE = 0o755
_ENTRY_FILE_MODE = 0o644
_UTF8 = "utf-8"


@dataclass(frozen=True)
class StagedEntry:
    """An entry in the cache: its key, its directory and the SHA-256 of its prompts file, which a sheet names."""

    key: str
    path: Path
    prompts_sha256: str


def cache_root(environ: Mapping[str, str]) -> Path:
    """The runner cache directory: the one ``HYPERLOOM_RUNNER_CACHE`` names, else where a sandbox mounts it."""
    return Path(environ.get(CACHE_ENV) or DEFAULT_CACHE_DIR)


def entry_key(inputs: Mapping[str, Any], *, generator: str = GENERATOR, prefix: str = KEY_PREFIX) -> str:
    """The key of the entry ``generator`` makes from ``inputs`` (the seed and the tokenizer's file hashes)."""
    digest = hashlib.sha256(canonical_json({"generator": generator, "inputs": inputs}).encode(_UTF8)).hexdigest()
    return f"{prefix}{digest[:_KEY_DIGITS]}"


def _manifest(inputs: Mapping[str, Any], count: int, raw: bytes, generator: str) -> bytes:
    document = {
        "schema": _SCHEMA,
        "generator": generator,
        "inputs": inputs,
        "prompts": {
            "file": _PROMPTS_PATH,
            "count": count,
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    }
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode(_UTF8)


def stage_entry(root: Path, prompts: Sequence[Prompt], inputs: Mapping[str, Any]) -> StagedEntry:
    """Write the entry for ``prompts`` below ``root`` (see :func:`stage_file`)."""
    return stage_file(root, prompts_bytes(prompts), inputs, generator=GENERATOR, prefix=KEY_PREFIX)


def stage_file(root: Path, raw: bytes, inputs: Mapping[str, Any], *, generator: str, prefix: str) -> StagedEntry:
    """Write the entry whose prompts file is ``raw`` (one prompt on each line) below ``root``; an entry already there that is the
    same one is left as it is.

    The data and then the manifest are written into a directory of their own, which becomes the entry in one rename, so
    a reader never sees half an entry. The entry is readable by every account whatever this process's umask (a runner
    sandbox reads it as an account of its own) and writable by none but its writer; ``evalsets`` is made group-shared
    (``2775``), so the next account of a run's group can stage another entry beside it.

    Raises:
        PromptSetError: An entry with this key exists and differs from what would be written.
    """
    manifest = _manifest(inputs, raw.count(b"\n"), raw, generator)
    key = entry_key(inputs, generator=generator, prefix=prefix)
    final = Path(root) / EVALSETS_DIR / key
    staged = StagedEntry(key, final, hashlib.sha256(raw).hexdigest())
    if final.exists():
        if (final / MANIFEST_NAME).read_bytes() != manifest or (final / DATA_DIR / PROMPTS_FILE).read_bytes() != raw:
            raise PromptSetError(f"the entry {final} exists and differs from the one that would be written")
        return staged
    incoming = make_shared_dirs(final.parent) / f".incoming-{uuid.uuid4().hex}"
    (incoming / DATA_DIR).mkdir(parents=True)
    try:
        _write_readable(incoming / DATA_DIR / PROMPTS_FILE, raw)
        _write_readable(incoming / MANIFEST_NAME, manifest)
        for directory in (incoming / DATA_DIR, incoming):
            os.chmod(directory, _ENTRY_DIR_MODE)
        try:
            os.rename(incoming, final)
        except OSError:
            if not final.exists():
                raise
    finally:
        shutil.rmtree(incoming, ignore_errors=True)
    return staged


def _write_readable(path: Path, data: bytes) -> None:
    """``data`` in a new file at ``path`` with mode ``644``, which the umask does not decide."""
    path.write_bytes(data)
    os.chmod(path, _ENTRY_FILE_MODE)


def recorded_sha256(entry: Path) -> str:
    """The SHA-256 of the prompts file the manifest of ``entry`` records.

    Raises:
        PromptSetError: The manifest cannot be read, or records no hash.
    """
    try:
        return json.loads((entry / MANIFEST_NAME).read_text(encoding=_UTF8))["prompts"]["sha256"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PromptSetError(
            f"the entry {entry} has no readable manifest ({type(exc).__name__}): not staged, or unfinished"
        ) from exc


def verified_prompts(entry: Path, wanted_sha256: str) -> bytes:
    """The bytes of the entry's ``data/prompts.jsonl``, read once and checked against its manifest and ``wanted_sha256``.

    The caller parses these bytes, so what it reads is what was hashed. The file is always ``data/prompts.jsonl``: a
    manifest that names another file is refused, never followed.

    Raises:
        PromptSetError: No manifest or prompts file, a manifest that names another file or other prompts than
            ``wanted_sha256``, or a prompts file that is not what the manifest recorded.
    """
    try:
        recorded = json.loads((entry / MANIFEST_NAME).read_text(encoding=_UTF8))["prompts"]
        named, size, sha256 = recorded["file"], recorded["size"], recorded["sha256"]
        raw = (entry / _PROMPTS_PATH).read_bytes()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PromptSetError(
            f"the entry {entry} has no readable manifest and prompts file ({type(exc).__name__}): not staged, or unfinished"
        ) from exc
    if sha256 != wanted_sha256:
        raise PromptSetError(f"the entry {entry} holds prompts {sha256}, the sheet names {wanted_sha256}")
    if named != _PROMPTS_PATH or len(raw) != size or hashlib.sha256(raw).hexdigest() != wanted_sha256:
        raise PromptSetError(f"the prompts file of {entry} is not what its manifest recorded")
    return raw


def load_prompt_set(plan: ShortPromptPlan, environ: Mapping[str, str]) -> tuple[Prompt, ...]:
    """The prompts ``plan`` names, parsed from the very bytes whose hashes were checked (:func:`verified_prompts`).

    Raises:
        PromptSetError: The entry is missing, unfinished, or not the one the plan names.
    """
    entry = cache_root(environ) / EVALSETS_DIR / plan.prompt_set
    return parse_prompts(verified_prompts(entry, plan.prompts_sha256), str(entry))
