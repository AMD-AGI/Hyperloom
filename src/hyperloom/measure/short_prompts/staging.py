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
import stat
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom_common.shared_storage import current_umask, make_shared_dirs, shared_parent
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
#: How the entry being built is opened below ``evalsets``, which other accounts of the group may write: never through
#: a link one of them put at its name.
_ENTRY_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_NEW_FILE = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
_BUILDING = stat.S_IRWXU
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
    a reader never sees half an entry. ``evalsets`` is made as
    :func:`~hyperloom_common.shared_storage.make_shared_dirs` makes it: group-shared (``2775``) below storage set up to
    be shared, so the next account of a run's group can stage another entry beside it, and as the umask leaves it
    anywhere else. Below shared storage the entry is readable by every account whatever this process's umask (``755``
    directories, ``644`` files: a runner sandbox reads it as an account of its own) and writable by none but its
    writer; anywhere else the umask decides its modes, as for a plain create, so a private cache stays private. The
    entry is built and put in place through ``evalsets`` as the walk of
    :func:`~hyperloom_common.shared_storage.open_directory` holds it, never through a link on the way or at its name.

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
    make_shared_dirs(final.parent)
    with shared_parent(final) as (evalsets, name, widen):
        modes = (_ENTRY_DIR_MODE, _ENTRY_FILE_MODE)
        dir_mode, file_mode = modes if widen else tuple(mode & ~current_umask() for mode in modes)
        incoming = f".incoming-{uuid.uuid4().hex}"
        os.mkdir(incoming, _BUILDING, dir_fd=evalsets)
        try:
            _build_entry(
                evalsets, incoming, {DATA_DIR: {PROMPTS_FILE: raw}, MANIFEST_NAME: manifest}, dir_mode, file_mode
            )
            try:
                os.rename(incoming, name, src_dir_fd=evalsets, dst_dir_fd=evalsets)
            except OSError:
                if not final.exists():
                    raise
        finally:
            _remove_entry(evalsets, incoming)
    return staged


def _build_entry(parent: int, name: str, tree: Mapping[str, Any], dir_mode: int, file_mode: int) -> None:
    """Fill the directory ``name`` of ``parent`` (opened without following a link) with ``tree`` (a file's bytes, or a
    directory's own tree, by name), each new file with ``file_mode`` and each directory, this one last, with
    ``dir_mode``, set by ``fchmod``."""
    directory = os.open(name, _ENTRY_DIRECTORY, dir_fd=parent)
    try:
        for entry, content in tree.items():
            if isinstance(content, bytes):
                _write_new(directory, entry, content, file_mode)
            else:
                os.mkdir(entry, _BUILDING, dir_fd=directory)
                _build_entry(directory, entry, content, dir_mode, file_mode)
        os.fchmod(directory, dir_mode)
    finally:
        os.close(directory)


def _write_new(directory: int, name: str, data: bytes, mode: int) -> None:
    """``data`` in a new file ``name`` of ``directory``, its ``mode`` set by ``fchmod``."""
    descriptor = os.open(name, _NEW_FILE, stat.S_IRUSR | stat.S_IWUSR, dir_fd=directory)
    with os.fdopen(descriptor, "wb") as handle:
        os.fchmod(handle.fileno(), mode)
        handle.write(data)


def _remove_entry(parent: int, name: str) -> None:
    """Remove what is left of the entry being built at ``name`` of ``parent`` (nothing once it was renamed into
    place), never through a link."""
    try:
        directory = os.open(name, _ENTRY_DIRECTORY, dir_fd=parent)
    except OSError:
        return
    try:
        for entry in os.listdir(directory):
            with suppress(OSError):
                if stat.S_ISDIR(os.stat(entry, dir_fd=directory, follow_symlinks=False).st_mode):
                    _remove_entry(directory, entry)
                else:
                    os.unlink(entry, dir_fd=directory)
    finally:
        os.close(directory)
    with suppress(OSError):
        os.rmdir(name, dir_fd=parent)


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
