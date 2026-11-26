# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lay a directory out as the place a tree is imported from under its package's name.

A tree that holds a package's own files at its root -- a checkout of a snapshot of an installed package -- is imported
by the name of its directory, from the directory that holds it: the import root. A package may also find files by where
it sits (aiter looks for its device sources in ``aiter_meta`` beside itself), so what is installed beside the package
is linked into the import root too. A copy of the tree gets an import root of its own, laid out the same way.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


def link_beside(import_root: Path, beside: Mapping[str, Path]) -> None:
    """Link each directory of ``beside`` into ``import_root`` under the name it is keyed by."""
    import_root.mkdir(parents=True, exist_ok=True)
    for name, target in beside.items():
        (import_root / name).symlink_to(target, target_is_directory=True)


def linked_beside(import_root: Path) -> dict[str, Path]:
    """What :func:`link_beside` put into ``import_root``, for a copy's import root to be given the same."""
    return {entry.name: Path(os.readlink(entry)) for entry in sorted(import_root.iterdir()) if entry.is_symlink()}


def import_path_first(import_root: Path, inherited: str) -> str:
    """``PYTHONPATH`` with ``import_root`` ahead of the ``inherited`` one."""
    return os.pathsep.join(filter(None, (str(import_root), inherited)))


def import_name_of(tree: Path) -> str:
    """The name this process imports ``tree`` by: its directory's, when the directory holding it is first on the
    import path, else empty."""
    first = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
    return tree.name if first and Path(first).resolve() == tree.resolve().parent else ""


__all__ = ["import_name_of", "import_path_first", "link_beside", "linked_beside"]
