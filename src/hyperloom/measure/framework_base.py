# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a test's framework install looked like when it started, as evidence the signed result carries.

A local sandbox gets a private view of its framework roots (``LocalTemplate.framework_roots``). The tables a framework
rewrites in place (aiter's tuned GEMM configs) are hashed once when the session starts and by the runner of each test
before anything runs; the runner lists both in ``framework_base.json``, so a test that began on another base than the
session's shows it in its signed result.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

__all__ = [
    "DEFAULT_FRAMEWORK_ROOTS",
    "DEFAULT_FRAMEWORK_WATCH",
    "FRAMEWORK_BASE_FILE",
    "FRAMEWORK_ROOTS_ENV",
    "FRAMEWORK_WATCH_ENV",
    "SESSION_DIGEST_ENV",
    "framework_base",
    "tree_digest",
    "write_framework_base",
]

#: The image's framework installs; the local runner template gives each one a sandbox sees a private view.
DEFAULT_FRAMEWORK_ROOTS = ("/sgl-workspace",)
#: The files and directories hashed: the tuned tables aiter reads at start and rewrites in place.
DEFAULT_FRAMEWORK_WATCH = ("/sgl-workspace/aiter/aiter/configs",)
#: Set by the local provider to the roots it made a private view of (``RESERVED_ENV_NAMES``).
FRAMEWORK_ROOTS_ENV = "HYPERLOOM_FRAMEWORK_ROOTS"
FRAMEWORK_WATCH_ENV = "HYPERLOOM_FRAMEWORK_WATCH"
SESSION_DIGEST_ENV = "HYPERLOOM_FRAMEWORK_SESSION_SHA256"
FRAMEWORK_BASE_FILE = "framework_base.json"
_SCHEMA = "hyperloom.framework-base/v1"
_CHUNK = 1 << 20


def _files(path: Path) -> Iterable[tuple[str, Path]]:
    if path.is_dir():
        for current, directories, names in os.walk(path):
            directories.sort()
            for name in sorted(names):
                full = Path(current, name)
                yield str(full.relative_to(path)), full
    elif path.is_file():
        yield "", path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(paths: Iterable[str]) -> str:
    """One sha256 over every file below ``paths`` (names and contents); a path that is not there counts as absent."""
    digest = hashlib.sha256()
    for raw in paths:
        digest.update(f"path {raw}\n".encode())
        for relative, full in _files(Path(raw)):
            digest.update(f"{relative}\0{_file_sha256(full)}\n".encode())
    return digest.hexdigest()


def framework_base(environ: Mapping[str, str]) -> dict[str, object]:
    """The framework base this process sees now: roots made private, what was hashed, and whether it is the session's."""
    roots = [part for part in environ.get(FRAMEWORK_ROOTS_ENV, "").split(os.pathsep) if part]
    watched = [part for part in environ.get(FRAMEWORK_WATCH_ENV, "").split(os.pathsep) if part]
    actual = tree_digest(watched) if watched else None
    session = environ.get(SESSION_DIGEST_ENV) or None
    return {
        "schema": _SCHEMA,
        "isolated_roots": roots,
        "watched": watched,
        "tree_sha256": actual,
        "session_sha256": session,
        "matches_session": None if actual is None or session is None else actual == session,
    }


def write_framework_base(files_dir: Path, base: Mapping[str, object]) -> None:
    """Write ``framework_base.json`` into the output files, so the manifest lists it and the signature covers it."""
    files_dir.mkdir(parents=True, exist_ok=True)
    (files_dir / FRAMEWORK_BASE_FILE).write_text(json.dumps(base, sort_keys=True), encoding="utf-8")
