# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Make a private checkout of a snapshot of an installed package the package a campaign's processes import.

A repository the host snapshots from an installed package holds that package's files at its root, without the compiled
extension modules. Nothing imports a checkout of it by the package's name, so a driver that does loads the installed
copy and the campaign measures the unmodified original whatever it edits.

:func:`stand_in_for_served_package` completes such a checkout: the checkout is created under the package's own name
(:func:`~.worktree.operator_workspace`), the directory holding it becomes the import root forge-loop puts first on the
import path, the installed extension modules are linked into the checkout, and what is installed beside the package is
linked beside the checkout, where a package that finds files by its own location looks for them.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from kernelforge.utils.import_root import link_beside

#: The compiled extension modules a source snapshot leaves out. The snapshot's repository ignores them, so the links
#: never reach a patch.
_EXTENSION_GLOBS = ("*.so", "*.so.*")


def _link_extension_modules(workspace: Path, served_tree: Path) -> None:
    """Link each extension module of ``served_tree`` into ``workspace`` at the same relative path."""
    for pattern in _EXTENSION_GLOBS:
        for module in served_tree.rglob(pattern):
            link = workspace / module.relative_to(served_tree)
            if link.exists() or link.is_symlink():
                continue
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(module)


def stand_in_for_served_package(workspace: Path, served_tree: Path, installed_beside: Iterable[Path]) -> None:
    """Complete ``workspace``, a checkout named after the package served from ``served_tree``, as that package.

    ``installed_beside`` are the other trees the host serves from the directory ``served_tree`` is installed in; each
    is linked beside the checkout under its own name.
    """
    _link_extension_modules(workspace, served_tree)
    link_beside(workspace.parent, {tree.name: tree for tree in installed_beside})


__all__ = ["stand_in_for_served_package"]
