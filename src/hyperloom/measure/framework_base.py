# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The framework's tuned tables as the session found them: one snapshot per session, one proof per test.

A local sandbox shares the image's framework install with every other local sandbox. aiter reads its tuned tables
(``aiter/configs``) from there, and when a variable lists several tables it merges them and writes the de-duplicated
rows back into each source file, so one test's merge changed what every later test measured. The toolbox takes a
snapshot of the tables when the session starts; every test then reads private copies of it (``framework_copies``) and
the shared files are only ever compared with the snapshot and, if anything changed them, put back.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.outside_facts import SGLANG_IMAGE_AITER_CONFIGS
from hyperloom.measure.framework_tables import TableVariable, read_variables

__all__ = [
    "DEFAULT_CONFIGS",
    "FRAMEWORK_BASE_FILE",
    "SESSION_SHA_ENV",
    "SNAPSHOT_ENV",
    "FrameworkBaseError",
    "Snapshot",
    "file_sha256",
    "files_digest",
    "framework_envs",
    "load_snapshot",
    "restore",
    "shared_view",
    "take_snapshot",
    "tree_files",
]

log = logging.getLogger(__name__)

#: Where the image keeps the tuned tables aiter reads and rewrites.
DEFAULT_CONFIGS = SGLANG_IMAGE_AITER_CONFIGS
#: Set on the runner template: the session's snapshot directory, and its digest (a test that finds another digest in the
#: snapshot began on a base that is not the session's).
SNAPSHOT_ENV = "HYPERLOOM_FRAMEWORK_SNAPSHOT"
SESSION_SHA_ENV = "HYPERLOOM_FRAMEWORK_SESSION_SHA256"
FRAMEWORK_BASE_FILE = "framework_base.json"
_MANIFEST = "snapshot.json"
_COPY = "configs"
_MODELS = "model_configs"
_SCHEMA = "hyperloom.framework-snapshot/v1"
_ABSENT = "absent"
_CHUNK = 1 << 20


class FrameworkBaseError(Exception):
    """The session's snapshot is missing or not one this code wrote."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def tree_files(root: Path) -> dict[str, str]:
    """Relative path and sha256 of every file below ``root``; empty when there is no such directory."""
    found: dict[str, str] = {}
    for current, directories, names in os.walk(root):
        directories.sort()
        for name in sorted(names):
            full = Path(current, name)
            if full.is_file():
                found[str(full.relative_to(root))] = file_sha256(full)
    return found


def files_digest(files: Mapping[str, str]) -> str:
    """One sha256 over names and content hashes, whatever order ``files`` lists them in."""
    digest = hashlib.sha256()
    for relative in sorted(files):
        digest.update(f"{relative}\0{files[relative]}\n".encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class Snapshot:
    """The tuned tables at session start: where they were taken from, their hashes and the variables that name them.

    ``model_order`` is the order the shared ``model_configs`` directory listed its files in, which is the order aiter
    merges them in when a variable is unset.
    """

    directory: Path
    source: Path
    files: Mapping[str, str]
    model_order: tuple[str, ...]
    variables: Mapping[str, TableVariable]

    @property
    def sha256(self) -> str:
        return files_digest(self.files)

    @property
    def copies(self) -> Path:
        """The pristine copy of every table, below the snapshot directory."""
        return self.directory / _COPY

    def to_document(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "source": str(self.source),
            "files": dict(self.files),
            "model_order": list(self.model_order),
            "variables": {name: {"default": v.default, "tuned": v.tuned} for name, v in self.variables.items()},
        }


def load_snapshot(directory: Path) -> Snapshot:
    """The snapshot in ``directory``.

    Raises:
        FrameworkBaseError: When it is absent or not readable as a snapshot.
    """
    try:
        document = json.loads((directory / _MANIFEST).read_text(encoding="utf-8"))
        if document["schema"] != _SCHEMA:
            raise FrameworkBaseError(f"{directory} holds a snapshot of schema {document['schema']!r}")
        return Snapshot(
            directory,
            Path(document["source"]),
            {str(k): str(v) for k, v in document["files"].items()},
            tuple(str(name) for name in document["model_order"]),
            {name: TableVariable(str(v["default"]), str(v["tuned"])) for name, v in document["variables"].items()},
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise FrameworkBaseError(f"no usable snapshot in {directory}: {type(exc).__name__}: {exc}") from exc


def _listed(models: Path) -> tuple[str, ...]:
    with contextlib.suppress(OSError), os.scandir(models) as entries:
        return tuple(entry.name for entry in entries if entry.is_file())
    return ()


def take_snapshot(source: Path, directory: Path) -> Snapshot:
    """Copy the tables below ``source`` into ``directory``, or keep the snapshot already there when it is intact.

    A restarted session finds its first snapshot again; one whose copies no longer match its own hashes is taken anew.
    """
    with contextlib.suppress(FrameworkBaseError):
        kept = load_snapshot(directory)
        if kept.source == source and tree_files(kept.copies) == dict(kept.files):
            return kept
    staging = directory.with_name(directory.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(source, staging / _COPY)
    snapshot = Snapshot(
        directory,
        source,
        tree_files(staging / _COPY),
        _listed(source / _MODELS),
        read_variables(source.parent / "jit" / "core.py"),
    )
    (staging / _MANIFEST).write_text(json.dumps(snapshot.to_document(), sort_keys=True), encoding="utf-8")
    shutil.rmtree(directory, ignore_errors=True)
    staging.rename(directory)
    return snapshot


def framework_envs(directory: Path, source: str | None = None) -> dict[str, str]:
    """The variables that make the runner of every test of the session read private copies of the tuned tables.

    Takes the snapshot into ``directory`` now. Empty on a machine without the tables (``source`` defaults to the
    image's); when the snapshot cannot be taken the directory is still named, and each test says in its result that it
    found none.
    """
    configs = Path(DEFAULT_CONFIGS if source is None else source)
    if not source and source is not None or not configs.is_dir():
        return {}
    try:
        return {SNAPSHOT_ENV: str(directory), SESSION_SHA_ENV: take_snapshot(configs, directory).sha256}
    except OSError as exc:
        log.warning("the framework tables cannot be snapshotted from %s: %s", configs, exc)
        return {SNAPSHOT_ENV: str(directory)}


def shared_view(snapshot: Snapshot) -> tuple[str, list[str]]:
    """The digest of the shared files the snapshot lists as they are now, and which of them differ from it."""
    now: dict[str, str] = {}
    for relative in snapshot.files:
        path = snapshot.source / relative
        now[relative] = file_sha256(path) if path.is_file() else _ABSENT
    return files_digest(now), [relative for relative, sha in now.items() if sha != snapshot.files[relative]]


def restore(snapshot: Snapshot, relatives: Iterable[str]) -> tuple[list[str], list[str]]:
    """Put the snapshot's copy of each file back in the shared tree; the files restored and the ones that could not be."""
    restored: list[str] = []
    failed: list[str] = []
    for relative in relatives:
        target = snapshot.source / relative
        staged = target.with_name(f".{target.name}.{os.getpid()}.hl-restore")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot.copies / relative, staged)
            os.replace(staged, target)
        except OSError:
            failed.append(relative)
            with contextlib.suppress(OSError):
                staged.unlink()
        else:
            restored.append(relative)
    return restored, failed
