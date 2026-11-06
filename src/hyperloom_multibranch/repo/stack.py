# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A component's patch stack applied across its source roots, as the experiment repository checks and rebases it.

The toolbox applies a stack under one prefix, the package's, but a patch to aiter also names files of its csrc, beside
the package. Each patch is split by the root its files lie in, and each root's part is applied under that root's
prefix to the tree the install keeps there. Roots share no file, so the parts apply as the whole patch would.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath

from hyperloom.contracts.recipe import ComponentPatches, PatchRef
from hyperloom.measure.compile import (
    BlobStore,
    ComponentLayout,
    PackageBase,
    PatchApplyError,
    apply_patch_stack,
    patch_targets,
)
from hyperloom_multibranch.contracts.sources import SourceRoot
from hyperloom_multibranch.repo.rebase import Files

# Hunk lines start with a space, "+", "-" or "\", so a line opening a file's section is always its header.
_FILE_SECTION = re.compile(r"^(?=diff --git )", re.MULTILINE)


class ScratchBlobs:
    """A blob store for reads, keeping what a trial application writes in memory instead."""

    def __init__(self, blobs: BlobStore) -> None:
        self.root = blobs.root
        self._blobs = blobs
        self._kept: dict[str, bytes] = {}

    def put(self, data: bytes) -> str:
        sha256 = hashlib.sha256(data).hexdigest()
        self._kept[sha256] = data
        return sha256

    def get(self, sha256: str) -> bytes:
        kept = self._kept.get(sha256)
        return self._blobs.get(sha256) if kept is None else kept


def stack_files(
    component: ComponentPatches, roots: Sequence[SourceRoot], blobs: BlobStore, paths: Collection[str]
) -> Files:
    """Every file a patch of ``component`` touches and each of ``paths`` as the stack leaves them, by path in the
    upstream repository; ``roots`` are the component's, its package first. Raises ``PatchApplyError`` when a patch
    does not apply or names a file outside every root."""
    scratch = ScratchBlobs(blobs)
    files = {path: _installed(roots, path) for path in paths}
    for root, stack, touched in _by_root(component, roots, scratch):
        files.update({f"{root.prefix}/{path}": _read(root.path / path) for path in touched})
        package = PackageBase(component.component, root.path, None, None)
        applied = apply_patch_stack(stack, ComponentLayout(component.component, root.prefix), package, scratch)
        files.update(
            {
                f"{root.prefix}/{entry.path}": None if entry.blob is None else scratch.get(entry.blob)
                for entry in applied.entries
            }
        )
    return files


def _by_root(
    component: ComponentPatches, roots: Sequence[SourceRoot], scratch: ScratchBlobs
) -> list[tuple[SourceRoot, ComponentPatches, list[str]]]:
    """Each root that a patch of ``component`` touches, with the stack of the patches' parts in it, kept in
    ``scratch`` under each patch's own path, and the root-relative files those parts touch."""
    parts: dict[str, list[PatchRef]] = {root.prefix: [] for root in roots}
    touched: dict[str, list[str]] = {root.prefix: [] for root in roots}
    for patch in component.patches:
        for prefix, text in _split(_decode(scratch.get(patch.sha256), patch.path), roots, patch.path).items():
            parts[prefix].append(PatchRef(patch.path, scratch.put(text.encode())))
            touched[prefix] += patch_targets(text, prefix, patch.path)
    return [
        (root, replace(component, patches=tuple(parts[root.prefix])), list(dict.fromkeys(touched[root.prefix])))
        for root in roots
        if parts[root.prefix]
    ]


def _split(text: str, roots: Sequence[SourceRoot], name: str) -> dict[str, str]:
    """``text``'s file sections by the prefix of the root each lies in. A section in no root, and a text with no
    section, go to the package's root, whose application refuses them as the toolbox would."""
    preamble, *sections = _FILE_SECTION.split(text)
    if not sections:
        return {roots[0].prefix: text}
    parts: dict[str, str] = {}
    for section in sections:
        prefix = _root_of(patch_targets(section, "", name)[0], roots)
        parts[prefix] = parts.get(prefix, preamble if not parts else "") + section
    return parts


def _root_of(path: str, roots: Sequence[SourceRoot]) -> str:
    inside = [root.prefix for root in roots if PurePosixPath(root.prefix) in PurePosixPath(path).parents]
    return inside[0] if inside else roots[0].prefix


def _installed(roots: Sequence[SourceRoot], path: str) -> bytes | None:
    for root in roots:
        if PurePosixPath(root.prefix) in PurePosixPath(path).parents:
            return _read(root.path / PurePosixPath(path).relative_to(root.prefix))
    return None


def _read(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


def _decode(data: bytes, name: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PatchApplyError(f"{name} is not UTF-8 text") from error


__all__ = ["ScratchBlobs", "stack_files"]
