# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The native builds a compiled environment needs (#125 stage 4b), as the compile and the sheet name them.

A component whose stack changes native sources compiles to a ``runtime`` root: its stack's files inside the installed
package are laid like a package root's, and its native outputs come from a build in the native build store
(:mod:`hyperloom.measure.jit.native`). The compile names each such build by the inputs whose hash is its key
(:class:`BuildNeed`); the build is made while the test waits without a card, and the sheet names it once it is ready, by
its key and the SHA-256 of its signed manifest (:class:`BuildRef`, ``environment.builds``).

The compiled overlay's roots (:class:`EnvRoot`) and their entries (:class:`EnvEntry`) are defined here, below
:mod:`hyperloom.measure.compile`, because the builds named here are made from them.

aiter's touched modules (#125 stage 2b) are built the same way but stay a package root's: :class:`ModulePlan` says which
of them a CPU build makes (:class:`ModuleBuild`), and the sheet names that build on aiter's package root.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from hyperloom.contracts.recipe import DIGEST_PREFIX, ComponentPatches, Recipe, RecipeError

if TYPE_CHECKING:
    from hyperloom.measure.jit.native.key import BuildInputs

__all__ = [
    "BuildNeed",
    "BuildNotReady",
    "BuildRef",
    "EnvEntry",
    "EnvRoot",
    "ModuleBuild",
    "ModulePlan",
    "NativePlan",
    "NativeStack",
    "RuntimeSourcesConflict",
    "check_runtime_sources",
]


@dataclass(frozen=True)
class EnvEntry:
    """One file of a root that differs from the base: laid over it (``upsert``) or removed (``delete``)."""

    path: str
    op: Literal["upsert", "delete"]
    blob: str | None = None
    mode: str | None = None

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {"path": self.path, "op": self.op}
        if self.op == "upsert":
            document.update(blob=f"{DIGEST_PREFIX}{self.blob}", mode=self.mode)
        return document


@dataclass(frozen=True)
class EnvRoot:
    """A package in the image and everything that differs from it."""

    id: str
    import_name: str
    dist_name: str | None
    dist_version: str | None
    entries: tuple[EnvEntry, ...]
    kind: Literal["package", "runtime"] = "package"

    def to_document(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "import_name": self.import_name,
            "dist": {"name": self.dist_name, "version": self.dist_version},
            "entries": [entry.to_document() for entry in self.entries],
        }


@dataclass(frozen=True)
class NativeStack:
    """What a component's stack makes of its native sources: the key of the build that compiles them, and the stack's
    files inside the installed package (package-relative), which are laid like a package root's."""

    inputs: BuildInputs
    entries: tuple[EnvEntry, ...]


class NativePlan(Protocol):
    """The native build ``component``'s stack needs; ``None`` when it changes no native source (or the component has no
    native recipe), and a :class:`CompileError` when it changes one that cannot be built."""

    def __call__(self, component: ComponentPatches) -> NativeStack | None:
        """The native build ``component``'s stack needs, if any."""


@dataclass(frozen=True)
class ModuleBuild:
    """The CPU build of aiter's modules that a stack touches (#125 stage 2b): the inputs whose hash is the store's key,
    and the modules it makes, which the test's server then loads instead of building them on its card."""

    inputs: BuildInputs
    modules: frozenset[str]


class ModulePlan(Protocol):
    """Which of the touched modules ``candidates`` of aiter's stack (its ``roots``, the module config ``config`` they
    leave) a CPU build makes, ``None`` when it makes none; and how long building ``modules`` on CPU takes."""

    def __call__(
        self, roots: tuple[EnvRoot, ...], config: Mapping[str, Any], candidates: Collection[str]
    ) -> ModuleBuild | None:
        """The CPU build of the ``candidates`` it makes, if any."""

    def seconds(self, modules: Collection[str]) -> float:
        """How long building ``modules`` on CPU takes, in seconds."""


@dataclass(frozen=True)
class BuildNeed:
    """A ``runtime`` root's build: the root it belongs to and the inputs whose hash is the store's key."""

    root_id: str
    inputs: BuildInputs


@dataclass(frozen=True)
class BuildRef:
    """A ready build as the sheet names it (``environment.builds``): the root, the key and the SHA-256 of its signed
    manifest."""

    root_id: str
    key: str
    manifest_sha256: str

    def to_document(self) -> dict[str, str]:
        return {"root_id": self.root_id, "key": self.key, "manifest_sha256": self.manifest_sha256}


class BuildNotReady(RecipeError):
    """A sheet was asked for before every build its environment needs was ready."""


class RuntimeSourcesConflict(RecipeError):
    """A recipe names a runtime snapshot and native stacks: two sources of the same native code."""


def check_runtime_sources(recipe: Recipe, builds: tuple[BuildNeed, ...]) -> None:
    """Refuse a recipe with both a ``runtime_snapshot`` and native stacks: two sources of the same native code."""

    if builds and recipe.runtime_snapshot is not None:
        components = ", ".join(need.root_id for need in builds)
        raise RuntimeSourcesConflict(
            f"the recipe names the runtime snapshot {recipe.runtime_snapshot} and native stacks ({components}): "
            "the native code would have two sources; drop the snapshot or the native patches"
        )
