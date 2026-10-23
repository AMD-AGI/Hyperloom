# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stack a replayed kernel table over the head's instead of replacing it (#109 item 7).

A replay that adds a kernel in a slot the head already fills would otherwise swap the head's table out. Stacking asks
the same questions a merge does (wider table, else union), but the primitives live in ``repo`` / ``toolbox``, which
notebook must not import: ``toolbox`` already imports notebook. The session hands in :attr:`TableStack.stack_slot`,
built where those packages are legal to call (scheduling). When there is no slot function, or it returns ``None``, the
swap is left as it was and the head's artifact is named in ``replaces``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from hyperloom.contracts.recipe import KernelArtifact, Recipe, RecipeDelta

#: ``(wanted, current) -> stacked artifact``, or ``None`` when the pair cannot stack and the swap should stay.
StackSlot = Callable[[KernelArtifact, KernelArtifact], KernelArtifact | None]


@dataclass(frozen=True)
class TableStack:
    """How a replay stacks two kernel artifacts in one slot. ``stack_slot`` is built by the session from the merge's
    primitives; notebook never imports those packages itself."""

    stack_slot: StackSlot | None = None


def stack_added_kernels(
    delta: RecipeDelta, head: Recipe, tables: TableStack | None
) -> tuple[RecipeDelta, tuple[KernelArtifact, ...]]:
    """``delta`` with each added kernel stacked over the head's table in the same slot, and the head artifacts still
    swapped out with loss.

    Without a ``stack_slot``, every conflicting add stays a swap and is listed in the second return value. With it, a
    wider table or a union replaces the swap when the slot function accepts the pair; a pair it refuses keeps the swap
    and the listing.
    """

    if not delta.add_kernels:
        return delta, ()
    added: list[KernelArtifact] = []
    replaces: list[KernelArtifact] = []
    stack = tables.stack_slot if tables is not None else None
    for artifact in delta.add_kernels:
        current = next((item for item in head.kernels if item.slot == artifact.slot), None)
        if current is None or current == artifact:
            added.append(artifact)
            continue
        stacked = stack(artifact, current) if stack is not None else None
        if stacked is current:
            continue
        if stacked is not None:
            added.append(stacked)
            continue
        added.append(artifact)
        replaces.append(current)
    return replace(delta, add_kernels=tuple(added)), tuple(replaces)


__all__ = ["StackSlot", "TableStack", "stack_added_kernels"]
