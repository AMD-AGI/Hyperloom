# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where the server imports each package from, as its own interpreter answered for a round.

Every round the suite starts asks the serving interpreter where it imports the framework and the kernel libraries
from, and ``launch.json`` keeps the answer (``framework.import_roots``). That record is what names the trees a kernel
rewrite is written against, and what tells a patch that landed in a copy of a package the server does not import.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from ..actions.executors import read_round_launch

#: The integrate failure of a patch that wrote into a copy of a package the measured server does not import.
PATCH_OUTSIDE_SERVED_ROOTS = "patch_outside_served_roots"


def round_import_roots(workspace: str | Path | None) -> Mapping[str, str]:
    """Where the server of the round in ``workspace`` imported each probed package from; empty when it probed none."""
    launch = read_round_launch(Path(workspace)) if workspace else None
    return dict(launch.framework.import_roots) if launch is not None else {}


def _top_package(path: Path) -> Path | None:
    """The outermost package directory holding ``path``, or ``None`` for a file in no package."""
    top = None
    for parent in path.parents:
        if not (parent / "__init__.py").is_file():
            break
        top = parent
    return top


def unserved_writes(written: Iterable[str], roots: Mapping[str, str]) -> list[str]:
    """The written ``.py`` files of a probed package that the server imports from another directory."""
    unserved = []
    for raw in written:
        path = Path(raw).resolve()
        package = _top_package(path) if path.suffix == ".py" else None
        served = roots.get(package.name) if package is not None else None
        if served is not None and package != Path(served):
            unserved.append(str(path))
    return unserved


__all__ = ["PATCH_OUTSIDE_SERVED_ROOTS", "round_import_roots", "unserved_writes"]
