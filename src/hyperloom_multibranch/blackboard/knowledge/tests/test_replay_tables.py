# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A replayed kernel table stacks over the head's: the table the slot function keeps stays in its slot, whichever side
of the replay held it."""

from __future__ import annotations

from hyperloom.contracts.recipe import KernelArtifact, KernelKind, LaunchConfig, Recipe, RecipeDelta, apply
from hyperloom_multibranch.blackboard.knowledge.replay_tables import stack_added_kernels

IMAGE = "sha256:" + "a" * 64
NARROW = KernelArtifact(KernelKind.GEMM, "bf16 4096x14336", "kernel/gemm/narrow.csv", "1" * 64)
WIDE = KernelArtifact(KernelKind.GEMM, "bf16 4096x14336", "kernel/gemm/wide.csv", "2" * 64)


def _wider(wanted: KernelArtifact, current: KernelArtifact) -> KernelArtifact:
    return current if current is WIDE else wanted


def test_a_replayed_table_leaves_the_heads_wider_table_in_its_slot() -> None:
    head = Recipe(IMAGE, LaunchConfig(), kernels=(WIDE,))
    swap = RecipeDelta(add_kernels=(NARROW,), remove_kernels=(WIDE.ref,))

    delta, replaces = stack_added_kernels(swap, head, _wider)

    assert (delta.add_kernels, delta.remove_kernels, replaces) == ((), (), ())
    assert apply(head, delta).kernels == (WIDE,)


def test_a_replayed_wider_table_takes_the_slot_and_without_a_slot_function_the_swap_is_named() -> None:
    head = Recipe(IMAGE, LaunchConfig(), kernels=(NARROW,))
    swap = RecipeDelta(add_kernels=(WIDE,), remove_kernels=(NARROW.ref,))

    stacked, stacked_replaces = stack_added_kernels(swap, head, _wider)
    unstacked, replaces = stack_added_kernels(swap, head, None)

    assert apply(head, stacked).kernels == (WIDE,) and stacked_replaces == ()
    assert unstacked == swap and replaces == (NARROW,)
