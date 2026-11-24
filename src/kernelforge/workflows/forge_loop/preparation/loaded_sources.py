# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which module each source file of a campaign is, and so which file a driver has to have loaded it from.

A driver that imports the kernel by its package's name loads whatever the interpreter finds first under that name.
When that is not the campaign workspace -- an installed copy, another checkout -- the driver times code the campaign
never edits, and every score the campaign records is about the original. The driver preflight's probe is given these
modules and reports the ones its process loaded from another file.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from kernelforge.utils.import_root import import_name_of


def declared_source_modules(workspace_dir: str, source_files: Iterable[str]) -> dict[str, str]:
    """``{module name: file}`` for each Python source file of the campaign inside ``workspace_dir``.

    The name is the one the file has when the workspace is what is imported: below the workspace's own name where the
    workspace is imported by it (:func:`~kernelforge.utils.import_root.import_name_of`), else from its root.
    """
    workspace = Path(workspace_dir).resolve()
    package = import_name_of(workspace)
    modules: dict[str, str] = {}
    for source in source_files:
        path = Path(source).resolve()
        if path.suffix != ".py" or workspace not in path.parents:
            continue
        parts = [*([package] if package else []), *path.relative_to(workspace).with_suffix("").parts]
        if parts[-1] == "__init__":
            parts.pop()
        if parts and all(part.isidentifier() for part in parts):
            modules[".".join(parts)] = str(path)
    return modules


__all__ = ["declared_source_modules"]
