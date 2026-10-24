# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a component's editable sources lie: its installed package, and the source trees beside it that its runtime
compiles (aiter's csrc).

A patch names each file by its path in the component's upstream repository. The specialist's source base lays the
sources out at those paths, and the experiment repository applies a patch to the tree each path falls in.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from hyperloom.measure.compile import ComponentLayout


@dataclass(frozen=True)
class SourceLayout:
    """What of a component lies beyond its installed package's text files."""

    #: Directories inside the package, relative to it, that its runtime writes builds into.
    built: tuple[str, ...] = ()
    #: Source trees in the package's parent directory, by name: beside the package in the checkout it is installed
    #: from, and beside the package's upstream prefix in its repository.
    meta: tuple[str, ...] = ()


SOURCE_LAYOUTS: Mapping[str, SourceLayout] = {
    # A develop install compiles csrc from its checkout (aiter/jit/core.py).
    "aiter": SourceLayout(built=("jit/build", "jit/hyperloom_jit_backup", "jit/flydsl_cache"), meta=("csrc",)),
}


@dataclass(frozen=True)
class SourceRoot:
    """One tree of a component's sources: ``prefix``, its path in the upstream repository, and ``path``, where the
    install keeps it."""

    prefix: str
    path: Path


def meta_prefix(layout: ComponentLayout, meta: str) -> str:
    """The path of the meta tree ``meta`` in the upstream repository: beside the package's upstream prefix."""
    return (PurePosixPath(layout.upstream_prefix).parent / meta).as_posix()


def source_roots(layout: ComponentLayout, package: Path, source: SourceLayout) -> tuple[SourceRoot, ...]:
    """The package installed at ``package`` first, then each of ``source``'s meta trees beside it."""
    metas = (SourceRoot(meta_prefix(layout, meta), package.parent / meta) for meta in source.meta)
    return (SourceRoot(layout.upstream_prefix, package), *metas)


__all__ = ["SOURCE_LAYOUTS", "SourceLayout", "SourceRoot", "meta_prefix", "source_roots"]
