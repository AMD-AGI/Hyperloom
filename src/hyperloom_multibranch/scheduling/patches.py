# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The patch and kernel files the session's recipes name, kept by content in the session's blob store.

A recipe names each file by a path and the sha256 of its bytes; the experiment repository keeps only the recipe, so
the bytes live in the blob store the toolbox's environment compiler reads them from. The scheduling side stores the
patches PRELUDE keeps and the patch resolver rewrites, reads a patch back for the resolver's prompt, and writes the
chosen commit's files out for the Recipe KB write-back.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from hyperloom.contracts.recipe import PATCH_ROOT, PatchRef, Recipe
from hyperloom.measure.compile import BlobStore

#: How many hex digits of a patch's sha256 its path carries; enough to keep two patches of a component apart.
PATCH_NAME_DIGITS = 16


class PatchStore:
    """Patch files by content in ``blobs``; a patch's path is ``patch/<component>/<sha256 prefix>.patch``, so the
    same text stored twice is one file."""

    def __init__(self, blobs: BlobStore) -> None:
        self._blobs = blobs

    def store(self, component: str, text: str) -> PatchRef:
        sha256 = self._blobs.put(text.encode())
        return PatchRef(f"{PATCH_ROOT}/{component}/{sha256[:PATCH_NAME_DIGITS]}.patch", sha256)

    def text(self, patch: PatchRef) -> str:
        return self._blobs.get(patch.sha256).decode()

    def checkout(self, recipe: Recipe, destination: Path) -> None:
        """Write every patch and kernel file ``recipe`` names under ``destination``, at the path it names it by."""

        files = [(patch.path, patch.sha256) for component in recipe.components for patch in component.patches]
        files.extend((kernel.ref, kernel.sha256) for kernel in recipe.kernels)
        for path, sha256 in files:
            target = destination.joinpath(*PurePosixPath(path).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self._blobs.get(sha256))


__all__ = ["PATCH_NAME_DIGITS", "PatchStore"]
