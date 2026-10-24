# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cache files that name the directory they were written in: stored without it, given the reader's own again (design 4.21).

Triton lists the files of a compiled kernel in ``<hash>/__grp__<kernel>.json``. The file cache manager of Triton 3.0 and
3.2 writes ``{"child_paths": {name: absolute path}}`` and treats a listed path that does not exist as absent. Every test
writes its cache in a directory of its own, so a group copied from another test's directory lists files that are nowhere:
Triton finds no kernel, compiles it again and rewrites the same files, and the cache has given nothing. Current Triton
writes ``{"child_paths": [name, ...]}`` and joins each name to the group's own directory, so that form needs nothing taken
out and nothing put in; both are accepted.

The cache therefore keeps such a file in a form that holds no directory: each path is made relative to the cache directory
(:func:`portable`) before the bytes are hashed, staged and published, so the stored name and hash are the same wherever the
test ran. A reader copies the stored bytes and checks them against the hash in the snapshot as for every other file, and
only then writes the paths of its own directory into its private copy (:func:`localised`); that copy is no longer the bytes
the snapshot hashed, and nothing reads it back as the stored one (:func:`portable` of it is the stored bytes again, which is
how a test's changes to it are judged).

A group that cannot be given this form (a path outside the cache directory, a file the kind does not store, anything but
a group of files) is refused rather than altered: a missing group only costs the compile, a group with a path nobody
checked would send the compiler outside the cache.
"""

from __future__ import annotations

import json
import os
import posixpath
from pathlib import Path

from hyperloom.measure import fsafe
from hyperloom.measure.docs import DocumentError, canonical_json, parse_json_object
from hyperloom.measure.jit_kinds import CacheKind

__all__ = ["MAX_LOCATED_BYTES", "NotPortable", "localised", "portable", "read_located"]

#: A group file is a few lines per kernel; one larger than this is not one.
MAX_LOCATED_BYTES = 1 << 20
_CHILDREN = "child_paths"
_TOO_LARGE = "the file is larger than a group of files is"


class NotPortable(ValueError):
    """The file cannot be stored without the directory it names, or is not stored in that form."""


def read_located(fd: int) -> bytes:
    """The bytes of a file that names its directory, from an open descriptor; a larger file is not one."""
    data = bytearray()
    for chunk in fsafe.stream(fd):
        data += chunk
        if len(data) > MAX_LOCATED_BYTES:
            raise NotPortable(_TOO_LARGE)
    return bytes(data)


def portable(raw: bytes, root: Path, kind: CacheKind, relative: str) -> bytes:
    """The stored form of the group ``relative`` (its path below ``root``): every path relative to ``root``, in canonical JSON.

    A path that is already relative stays as it is. A path that is not below ``root`` (as the test spelled it, or as the
    file system resolves it), or that names a file the kind does not store, refuses the whole group.
    """
    children = _children(raw)
    if isinstance(children, list):
        return _document(_beside(children, kind, relative))
    spellings = {os.path.normpath(root), os.path.normpath(os.path.abspath(root)), os.path.realpath(root)}
    return _document({name: _below(name, target, spellings, kind) for name, target in children.items()})


def localised(raw: bytes, root: Path, kind: CacheKind, relative: str) -> bytes:
    """A stored group ``relative`` for a test whose cache directory is ``root``, spelled the way Triton writes it. Every
    stored path must be one the kind admits (relative, no ``..``) or the group is refused: the bytes are the cache's, not
    yet the test's."""
    children = _children(raw)
    if isinstance(children, list):
        return _document(_beside(children, kind, relative))
    for name, stored in children.items():
        if not kind.admits(stored):
            raise NotPortable(f"{name}: {stored!r} is not a file of the {kind.name} cache")
    return json.dumps({_CHILDREN: {name: f"{root}/{stored}" for name, stored in children.items()}}).encode("utf-8")


def _document(children: dict[str, str] | list[str]) -> bytes:
    return canonical_json({_CHILDREN: children}).encode("utf-8")


def _children(raw: bytes) -> dict[str, str] | list[str]:
    if len(raw) > MAX_LOCATED_BYTES:
        raise NotPortable(_TOO_LARGE)
    try:
        document = parse_json_object(raw, what="group")
    except DocumentError as exc:
        raise NotPortable(str(exc)) from None
    children = document.get(_CHILDREN)
    if set(document) != {_CHILDREN} or not isinstance(children, (dict, list)):
        raise NotPortable(f"the file is not a group: it is not an object with only {_CHILDREN}")
    members = children.items() if isinstance(children, dict) else ((item, item) for item in children)
    if not all(isinstance(name, str) and isinstance(target, str) for name, target in members):
        raise NotPortable(f"{_CHILDREN} must name files by strings")
    return children


def _beside(names: list[str], kind: CacheKind, relative: str) -> list[str]:
    """The names a group lists beside itself, sorted; each must be a file of the kind in the group's own directory."""
    directory = posixpath.dirname(relative)
    for name in names:
        if not kind.admits(f"{directory}/{name}"):
            raise NotPortable(f"{name}: not a file of the {kind.name} cache beside the group")
    return sorted(names)


def _below(name: str, target: str, roots: set[str], kind: CacheKind) -> str:
    stored = target if not os.path.isabs(target) else _strip(os.path.normpath(target), roots)
    if stored is None or not kind.admits(stored):
        raise NotPortable(f"{name}: the path is not one of the files the {kind.name} cache holds below its directory")
    return stored


def _strip(path: str, roots: set[str]) -> str | None:
    for root in roots:
        prefix = root.rstrip("/") + "/"
        if path.startswith(prefix):
            return path[len(prefix) :]
    return None
