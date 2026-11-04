# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A recipe as the ``value`` columns of a Recipe KB record, and back.

CLOSE writes a session's final recipe to the Recipe KB; PRELUDE reads one back as a lead for the first specialists.

* ``config`` is the recipe's launch document: ``extra_server_args`` and ``extra_envs``, and ``remove_args`` when the
  launch removes options of the launch the framework resolves.
* ``patch``: component ``i`` is overlay stack ``i``. Its patches are listed as
  ``patch/overlays/<i:06d>/<j:02d>-<name>.patch``, the refs the KB orders replay by. Its provenance row carries the
  component, its upstream repository, its upstream commit as ``base_sha``, and each patch's own path and hash, so a
  component without patches survives the round trip.
* ``kernel``: artifacts are grouped by kind under the list each kind keeps its rows in. A row carries the artifact's
  metadata plus ``target``, ``artifact_path`` and ``sha256``.

The base image digest and runtime snapshot are not part of ``value``; they travel as record metadata.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import PurePosixPath

from hyperloom.contracts.recipe import (
    ComponentPatches,
    KernelArtifact,
    KernelKind,
    LaunchConfig,
    PatchRef,
    Recipe,
    RecipeError,
)
from hyperloom.orchestrator.knowledge.remote_recipe.models import CONFIG_SECTION, KERNEL_SECTION, PATCH_SECTION

#: The list each kernel sub-column keeps its rows under.
KERNEL_ROWS: dict[KernelKind, str] = {
    KernelKind.GEMM: "optimizations",
    KernelKind.FUSION: "items",
    KernelKind.REWRITE: "items",
}
_ROW_FIELDS = frozenset({"target", "artifact_path", "sha256"})
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def overlay_ref(stack: int, member: int, path: str) -> str:
    """The KB ref of patch ``member`` of overlay stack ``stack``, named after the patch file."""

    name = _UNSAFE_NAME.sub("-", PurePosixPath(path).stem).strip("._-") or "patch"
    return f"{PATCH_SECTION}/overlays/{stack:06d}/{member:02d}-{name}.patch"


def to_recipe_kb_value(recipe: Recipe) -> dict[str, object]:
    """The ``value`` of the Recipe KB record that publishes ``recipe``."""

    return {
        CONFIG_SECTION: recipe.launch.to_document(),
        PATCH_SECTION: _patch_column(recipe.components),
        KERNEL_SECTION: _kernel_column(recipe.kernels),
    }


def _patch_column(components: tuple[ComponentPatches, ...]) -> dict[str, object]:
    refs: list[str] = []
    provenance: list[dict[str, object]] = []
    for stack, component in enumerate(components):
        members = [
            {"ref": overlay_ref(stack, member, patch.path), "path": patch.path, "sha256": patch.sha256}
            for member, patch in enumerate(component.patches)
        ]
        refs.extend(str(row["ref"]) for row in members)
        provenance.append(
            {
                "stack_index": stack,
                "base_sha": component.upstream_ref,
                "component": component.component,
                "upstream_repo": component.upstream_repo,
                "members": members,
            }
        )
    return {"patches": refs, "provenance": provenance}


def _kernel_column(kernels: tuple[KernelArtifact, ...]) -> dict[str, object]:
    column: dict[str, dict[str, list[dict[str, object]]]] = {
        kind.value: {rows: []} for kind, rows in KERNEL_ROWS.items()
    }
    for artifact in kernels:
        clashing = sorted(_ROW_FIELDS & set(artifact.metadata))
        if clashing:
            raise RecipeError(f"the metadata of {artifact.ref} uses the row fields {', '.join(clashing)}")
        row = {**artifact.metadata, "target": artifact.target, "artifact_path": artifact.ref, "sha256": artifact.sha256}
        column[artifact.kind.value][KERNEL_ROWS[artifact.kind]].append(row)
    return column


def from_recipe_kb_value(
    value: Mapping[str, object], *, base_image_digest: str, runtime_snapshot: str | None = None
) -> Recipe:
    """The recipe a ``value`` written by :func:`to_recipe_kb_value` publishes; raises :class:`RecipeError` otherwise."""

    return Recipe(
        base_image_digest=base_image_digest,
        runtime_snapshot=runtime_snapshot,
        launch=LaunchConfig.from_document(value.get(CONFIG_SECTION)),
        components=_components(_column(value, PATCH_SECTION)),
        kernels=_kernels(_column(value, KERNEL_SECTION)),
    )


def _column(value: Mapping[str, object], section: str) -> Mapping[str, object]:
    column = value.get(section)
    if not isinstance(column, Mapping):
        raise RecipeError(f"the Recipe KB value has no {section} column")
    return column


def _components(column: Mapping[str, object]) -> tuple[ComponentPatches, ...]:
    rows = _rows(column.get("provenance"), "patch provenance")
    if [row.get("stack_index") for row in rows] != list(range(len(rows))):
        raise RecipeError("patch provenance must describe overlay stacks 0, 1, ... in order")
    components = tuple(
        ComponentPatches(
            component=row.get("component"),
            upstream_repo=row.get("upstream_repo"),
            upstream_ref=row.get("base_sha"),
            patches=tuple(PatchRef(member.get("path"), member.get("sha256")) for member in _rows(row.get("members"))),
        )
        for row in rows
    )
    listed = [str(member.get("ref")) for row in rows for member in _rows(row.get("members"))]
    if column.get("patches") != listed:
        raise RecipeError("the patch column lists other overlays than its provenance describes")
    return components


def _kernels(column: Mapping[str, object]) -> tuple[KernelArtifact, ...]:
    artifacts = []
    for kind, rows_key in KERNEL_ROWS.items():
        sub_column = column.get(kind.value, {})
        rows = sub_column.get(rows_key, []) if isinstance(sub_column, Mapping) else None
        for row in _rows(rows, f"kernel rows of {kind.value}"):
            metadata = {key: item for key, item in row.items() if key not in _ROW_FIELDS}
            artifacts.append(
                KernelArtifact(kind, row.get("target"), row.get("artifact_path"), row.get("sha256"), metadata)
            )
    return tuple(artifacts)


def _rows(value: object, what: str = "provenance members") -> list[Mapping[str, object]]:
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise RecipeError(f"{what} must be a list of objects, got {value!r}")
    return value


__all__ = ["KERNEL_ROWS", "from_recipe_kb_value", "overlay_ref", "to_recipe_kb_value"]
