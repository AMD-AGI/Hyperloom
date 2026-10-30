# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which stacks need a native build, and the key of each: the compile's side of #125 stage 4b.

:class:`NativePlanner` is the :class:`~hyperloom.measure.jit.native.needs.NativePlan` an environment compiler is given when
native builds are configured. A stack of a component that has a native recipe and changes a file its package layout
cannot lay (a ``.cu``, a header) is planned like a build runner plans it (:func:`~hyperloom.measure.runner.native.builder.
plan`): applied over the checkout of the component's pinned upstream commit, which yields the key, and its files inside
the installed package, which the farm lays like a package root's. A stack that changes only Python is left to the
package layout, as before. The planner keeps what it planned by key, so the build runner of a key can build it.

Refused here, at submission, naming the path: a change to the build definition (``CMakeLists.txt``, ``setup.py`` and the
other files of :data:`~hyperloom.measure.runner.native.recipes.NATIVE_BUILD_REFUSE`, and anything below a ``cmake/``
directory), which would run arbitrary code at build time; and a stack whose upstream commit has no checkout here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Protocol

from hyperloom.contracts.recipe import ComponentPatches
from hyperloom.measure.compile import (
    COMPONENT_LAYOUTS,
    BlobStore,
    ComponentLayout,
    EnvEntry,
    EnvRoot,
    PatchApplyError,
    UnsupportedComponentError,
    patch_targets,
)
from hyperloom.measure.jit.native.needs import NativeStack
from hyperloom.measure.jit.native.key import BuildInputs
from hyperloom.measure.jit.native.store import BuildInfraError
from hyperloom.measure.runner.native.builder import BuildMachine, BuildRequest, LaidTree, lay_tree, plan
from hyperloom.measure.runner.native.recipes import NATIVE_BUILD_REFUSE, NativeRecipe

__all__ = ["NativePlanner", "NativeSite", "Planned", "PlannedBuild"]

#: A directory whose files are build definition, whatever their names.
_BUILD_DEFINITION_DIRS = frozenset({"cmake"})


@dataclass(frozen=True)
class NativeSite:
    """What every build of a session shares: where the outputs run (``machine``), the recipe of each component that can
    be built, the directory of upstream checkouts (``<sources>/<component>/<upstream_ref>``) and the image's directory
    of installed packages (``site-packages``), which outputs equal to the image's are left out against."""

    machine: BuildMachine
    recipes: Mapping[str, NativeRecipe]
    sources: Path
    image_root: Path


class PlannedBuild(Protocol):
    """A planned build as a build runner takes it: :meth:`lay` lays its tree below ``work`` (nothing of the candidate
    runs then) for the child that compiles it."""

    def lay(self, blobs: BlobStore, work: Path) -> LaidTree: ...


@dataclass(frozen=True)
class Planned:
    """A planned build: the request a build runner compiles and the stack's changed files over the upstream tree."""

    request: BuildRequest
    root: EnvRoot

    def lay(self, blobs: BlobStore, work: Path) -> LaidTree:
        return lay_tree(self.request, self.root, blobs, work)


class NativePlanner:
    """The native build each stack needs (see the module); ``layouts`` are the package layouts that say which files are
    native (those the layout refuses)."""

    def __init__(
        self, site: NativeSite, blobs: BlobStore, layouts: Mapping[str, ComponentLayout] = COMPONENT_LAYOUTS
    ) -> None:
        self._site, self._blobs, self._layouts = site, blobs, layouts
        self._planned: dict[str, PlannedBuild] = {}

    @property
    def site(self) -> NativeSite:
        return self._site

    def __call__(self, component: ComponentPatches) -> NativeStack | None:
        recipe = self._site.recipes.get(component.component)
        layout = self._layouts.get(component.component)
        if recipe is None or layout is None or not self._changes_native(component, recipe, layout):
            return None
        upstream = self._site.sources / component.component / component.upstream_ref
        if not upstream.is_dir():
            raise UnsupportedComponentError(
                f"{component.component} at {component.upstream_ref} has no upstream checkout to build from; its native "
                "files cannot be measured until one is staged"
            )
        request = BuildRequest(recipe, component, upstream, self._site.image_root, self._site.machine)
        inputs, root = plan(request, self._blobs)
        self.remember(inputs, Planned(request, root))
        return NativeStack(inputs, _package_entries(root, recipe))

    def remember(self, inputs: BuildInputs, planned: PlannedBuild) -> None:
        """Keep ``planned`` under ``inputs``' key, for the build runner of the key (:meth:`planned`)."""
        self._planned[inputs.key] = planned

    def planned(self, inputs: BuildInputs) -> PlannedBuild:
        """What was planned under ``inputs``' key.

        Raises:
            BuildInfraError: Nothing was planned under it in this process (a restart since): the build is retried once
                the request is planned again, never built from nothing.
        """
        found = self._planned.get(inputs.key)
        if found is None:
            raise BuildInfraError(f"no stack of {inputs.component} was planned under {inputs.key} in this process")
        return found

    def _changes_native(self, component: ComponentPatches, recipe: NativeRecipe, layout: ComponentLayout) -> bool:
        """Whether the stack changes a file the package layout cannot lay; a build-definition change is refused."""
        native = False
        for ref in component.patches:
            text = self._blobs.get(ref.sha256).decode("utf-8", errors="replace")
            for relative in patch_targets(text, recipe.source_dir, ref.path):
                path = PurePosixPath(recipe.source_dir, relative).as_posix()
                _refuse_build_definition(path, relative, component.component, ref.path)
                native = native or layout.spec.refused_by(path) is not None
        return native


def _refuse_build_definition(path: str, relative: str, component: str, patch: str) -> None:
    parts = PurePosixPath(relative).parts
    named = next((pattern for pattern in NATIVE_BUILD_REFUSE if fnmatchcase(parts[-1], pattern)), None)
    directory = next((part for part in parts[:-1] if part in _BUILD_DEFINITION_DIRS), None)
    if named is not None or directory is not None:
        what = f"a build definition ({named})" if named else f"below the build definition directory {directory}/"
        raise PatchApplyError(
            f"{patch}: {path!r} is {what} of {component}; a native build does not take a change to how it is built"
        )


def _package_entries(root: EnvRoot, recipe: NativeRecipe) -> tuple[EnvEntry, ...]:
    """The stack's files inside the installed package, package-relative: the farm lays them over the image's package."""
    package = PurePosixPath(recipe.package_dir, recipe.import_name).parts
    return tuple(
        replace(entry, path=PurePosixPath(*parts[len(package) :]).as_posix())
        for entry in root.entries
        if (parts := PurePosixPath(entry.path).parts)[: len(package)] == package and len(parts) > len(package)
    )
