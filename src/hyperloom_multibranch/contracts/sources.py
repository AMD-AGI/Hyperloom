# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a component's editable sources lie: its installed package, the meta trees beside it that its runtime
compiles (aiter's ``csrc`` and ``hsa``), as the rule pack's component layout names them, and, on a host that builds
native code, the tree its native build compiles (sgl-kernel's C++), in a checkout of the commit the image installs.

A patch names each file by its path in the component's upstream repository. The specialist's source base lays the
sources out at those paths, and the experiment repository applies a patch to the tree each path falls in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.outside_facts.component_layout import META_BASE, LayoutSection
from hyperloom.measure.compile import ComponentLayout


@dataclass(frozen=True)
class SourceRoot:
    """One tree of a component's sources: ``prefix``, its path in the upstream repository, and ``path``, where the
    install keeps it."""

    prefix: str
    path: Path


@dataclass(frozen=True)
class NativeTree:
    """The tree a component's native build compiles, in a checkout of ``upstream_repo`` at ``upstream_ref``.

    A stack that changes it is built from that commit, so a patch to it is a diff against that checkout and names that
    commit.
    """

    root: SourceRoot
    upstream_repo: str
    upstream_ref: str


def meta_sections(layout: ComponentLayout) -> tuple[LayoutSection, ...]:
    """The layout's sections that lie over the meta trees beside the package, not over the package itself."""
    return tuple(section for section in layout.spec.sections if section.base == META_BASE)


def source_roots(layout: ComponentLayout, package: Path, native: NativeTree | None = None) -> tuple[SourceRoot, ...]:
    """The package installed at ``package`` first, then each meta tree beside it, in the checkout it is installed
    from, then ``native``'s tree.

    The native tree may hold the package's upstream path too; a path is in the first root that holds it, so the
    package is read as installed.
    """
    metas = (SourceRoot(section.prefix, package.parent / section.into) for section in meta_sections(layout))
    return (SourceRoot(layout.upstream_prefix, package), *metas, *((native.root,) if native else ()))


__all__ = ["NativeTree", "SourceRoot", "meta_sections", "source_roots"]
