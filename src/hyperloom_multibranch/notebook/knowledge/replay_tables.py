# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stack a replayed kernel table over the head's instead of replacing it (#109 item 7).

A replay that adds a kernel in a slot the head already fills would otherwise swap the head's table out. This module
asks the same questions a merge does: :func:`wider_table` when one table holds every row of the other, else
:func:`union_rows` when both can be one table. The result stays one artifact in one slot. When neither answer fits, the
swap is left as it was and the head's artifact is named in ``replaces``, so the specialist still sees the loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

from hyperloom.contracts.recipe import KernelArtifact, Recipe, RecipeDelta, RecipeError
from hyperloom.measure.compile import KernelLoadError, kernel_load_env, kernel_rows_only
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.repo.merge import TableBytes, TableStore, wider_table
from hyperloom_multibranch.toolbox.kernel_artifacts import (
    KernelRowsError,
    mint_kernel_artifact,
    union_rows,
)

log = logging.getLogger(__name__)

#: Reads a kernel table's bytes from the session's blob store.
TableReader = TableBytes
#: Keeps a united table's bytes in the session's blob store.
TableWriter = TableStore


@dataclass(frozen=True)
class TableStack:
    """What a replay needs to stack tables: the bytes behind each artifact, a store for a united table, and the aiter
    install :func:`union_rows` reads a shape key from (the image's when ``None``)."""

    kernel_table: TableReader
    store_table: TableWriter | None = None
    aiter_configs: Path | None = None


def stack_added_kernels(
    delta: RecipeDelta, head: Recipe, tables: TableStack | None
) -> tuple[RecipeDelta, tuple[KernelArtifact, ...]]:
    """``delta`` with each added kernel stacked over the head's table in the same slot, and the head artifacts still
    swapped out with loss.

    Without ``tables``, every conflicting add stays a swap and is listed in the second return value. With it, a wider
    table or a union replaces the swap when those primitives accept the pair; a pair they refuse keeps the swap and the
    listing.
    """

    if not delta.add_kernels:
        return delta, ()
    added: list[KernelArtifact] = []
    replaces: list[KernelArtifact] = []
    for artifact in delta.add_kernels:
        current = next((item for item in head.kernels if item.slot == artifact.slot), None)
        if current is None or current == artifact:
            added.append(artifact)
            continue
        stacked = _stack_one(artifact, current, tables) if tables is not None else None
        if stacked is current:
            continue
        if stacked is not None:
            added.append(stacked)
            continue
        added.append(artifact)
        replaces.append(current)
    return replace(delta, add_kernels=tuple(added)), tuple(replaces)


def _stack_one(
    wanted: KernelArtifact, current: KernelArtifact, tables: TableStack
) -> KernelArtifact | None:
    """The artifact that serves both tables, or ``None`` when the replay should keep swapping ``current`` out."""

    try:
        wider = wider_table(wanted, current, tables.kernel_table)
    except (KeyError, OSError, ValueError) as exc:
        log.warning("could not compare kernel tables for %s: %s", wanted.slot, exc)
        return None
    if wider is not None:
        return wider
    if tables.store_table is None:
        return None
    try:
        return _united(wanted, current, tables)
    except (RecipeError, KernelLoadError, KernelRowsError, RequestRejected, KeyError, OSError, ValueError) as exc:
        log.info("could not unite kernel tables for %s: %s", wanted.slot, exc)
        return None


def _united(theirs: KernelArtifact, ours: KernelArtifact, tables: TableStack) -> KernelArtifact:
    """Both tables' rows as one artifact of their slot, kept in the blob store."""

    assert tables.store_table is not None
    env, other = kernel_load_env(theirs), kernel_load_env(ours)
    if env != other:
        raise RecipeError(f"one table loads through {env}, the other through {other}")
    flags = kernel_rows_only(theirs), kernel_rows_only(ours)
    data = union_rows(
        tables.kernel_table(theirs),
        tables.kernel_table(ours),
        env,
        rows_only=flags,
        configs=tables.aiter_configs,
    )
    united = mint_kernel_artifact(theirs.kind, theirs.target, env, data, rows_only=flags[0])
    tables.store_table(data)
    return united


__all__ = ["TableReader", "TableStack", "TableWriter", "stack_added_kernels"]
