# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stack a replayed kernel table over the head's instead of replacing it (#109 item 7).

A replay that adds a kernel where the head already has a table, in the same slot or on the same load variable (the
service reads one table per variable), would otherwise swap the head's table out. Stacking asks the same questions a
merge does (wider table, else union), but the primitives live in ``repo`` / ``toolbox``, which blackboard must not
import: ``toolbox`` already imports blackboard. The session hands in a :data:`StackSlot`, built where those packages are
legal to call (scheduling). When there is none, or it returns ``None``, the swap is left as it was and the head's
artifact is named in ``replaces``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from hyperloom.contracts.recipe import KernelArtifact, Recipe, RecipeDelta
from hyperloom.measure.compile import KernelLoadError, kernel_load_env

#: ``(wanted, current) -> stacked artifact``, or ``None`` when the pair cannot stack and the swap should stay.
StackSlot = Callable[[KernelArtifact, KernelArtifact], KernelArtifact | None]


def stack_added_kernels(
    delta: RecipeDelta, head: Recipe, stack_slot: StackSlot | None
) -> tuple[RecipeDelta, tuple[KernelArtifact, ...]]:
    """``delta`` with each added kernel stacked over the head's table it meets, and the head artifacts still swapped
    out with loss.

    Without ``stack_slot``, every conflicting add stays a swap and is listed in the second return value. With it, a
    wider table or a union replaces the swap when the slot function accepts the pair; a pair it refuses keeps the swap
    and the listing. A head table met on its load variable from another slot is removed in the same delta, so the
    replayed recipe sets the variable once.
    """

    if not delta.add_kernels:
        return delta, ()
    added: list[KernelArtifact] = []
    replaces: list[KernelArtifact] = []
    removed: list[str] = list(delta.remove_kernels)
    for artifact in delta.add_kernels:
        current = _occupant(head, artifact)
        if current is None or current == artifact:
            added.append(artifact)
            continue
        stacked = stack_slot(artifact, current) if stack_slot is not None else None
        if stacked is current:
            continue
        chosen = stacked if stacked is not None else artifact
        added.append(chosen)
        if stacked is None:
            replaces.append(current)
        if chosen.slot != current.slot and current.ref not in removed:
            removed.append(current.ref)
    return replace(delta, add_kernels=tuple(added), remove_kernels=tuple(removed)), tuple(replaces)


def _occupant(head: Recipe, artifact: KernelArtifact) -> KernelArtifact | None:
    """The head's table ``artifact`` meets, as a merge finds it (``repo.merge.kernel_occupant``): the one in its slot,
    else the one loading through the same variable, which no launch can set twice."""

    in_slot = next((item for item in head.kernels if item.slot == artifact.slot), None)
    env = _load_env(artifact)
    if in_slot is not None or env is None:
        return in_slot
    return next((item for item in head.kernels if _load_env(item) == env), None)


def _load_env(artifact: KernelArtifact) -> str | None:
    try:
        return kernel_load_env(artifact)
    except KernelLoadError:
        return None


__all__ = ["StackSlot", "stack_added_kernels"]
