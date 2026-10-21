# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a change between two recipes names, which is what decides whether a not-stackable candidate comes back."""

from __future__ import annotations

from dataclasses import replace

from hyperloom.contracts.recipe import (
    ComponentPatches,
    KernelArtifact,
    KernelKind,
    LaunchConfig,
    PatchRef,
    Recipe,
)
from hyperloom_multibranch.scheduling.candidates import touched

IMAGE_DIGEST = "sha256:" + "b" * 64
UPSTREAM_REF = "a" * 40
UPSTREAM_REPO = "https://example.com/vllm.git"
PATCH = PatchRef("patch/vllm/first.diff", "1" * 64)
OTHER_PATCH = PatchRef("patch/vllm/second.diff", "2" * 64)
VLLM = ComponentPatches("vllm", UPSTREAM_REPO, UPSTREAM_REF, (PATCH,))
KERNEL = KernelArtifact(KernelKind.GEMM, "qkv_proj", "kernel/qkv.hsaco", "3" * 64)
BEFORE = Recipe(IMAGE_DIGEST, LaunchConfig("--max-num-seqs 64", {"HL_MODE": "fast"}), (VLLM,), (KERNEL,))


def test_a_changed_server_argument_is_named_by_its_flag_whatever_the_value() -> None:
    raised = replace(BEFORE, launch=LaunchConfig("--max-num-seqs 256", BEFORE.launch.envs))
    lowered = replace(BEFORE, launch=LaunchConfig("--max-num-seqs 32", BEFORE.launch.envs))

    assert touched(BEFORE, raised) == frozenset({"arg --max-num-seqs"})
    assert touched(BEFORE, lowered) == touched(BEFORE, raised)


def test_an_added_server_argument_is_named_by_its_flag() -> None:
    added = replace(BEFORE, launch=LaunchConfig("--max-num-seqs 64 --enable-chunked-prefill", BEFORE.launch.envs))

    assert touched(BEFORE, added) == frozenset({"arg --enable-chunked-prefill"})


def test_a_changed_environment_variable_is_named() -> None:
    changed = replace(BEFORE, launch=LaunchConfig(BEFORE.launch.server_args, {"HL_MODE": "slow"}))
    unset = replace(BEFORE, launch=LaunchConfig(BEFORE.launch.server_args, {}))

    assert touched(BEFORE, changed) == frozenset({"env HL_MODE"})
    assert touched(BEFORE, unset) == frozenset({"env HL_MODE"})


def test_a_component_is_named_whether_its_patch_stack_or_its_upstream_ref_moved() -> None:
    restacked = replace(BEFORE, components=(replace(VLLM, patches=(PATCH, OTHER_PATCH)),))
    rebased = replace(BEFORE, components=(replace(VLLM, upstream_ref="c" * 40),))
    dropped = replace(BEFORE, components=())

    assert touched(BEFORE, restacked) == frozenset({"component vllm"})
    assert touched(BEFORE, rebased) == frozenset({"component vllm"})
    assert touched(BEFORE, dropped) == frozenset({"component vllm"})


def test_a_kernel_artifact_is_named_by_its_kind_and_target() -> None:
    swapped = replace(BEFORE, kernels=(replace(KERNEL, ref="kernel/qkv2.hsaco", sha256="4" * 64),))
    added = replace(BEFORE, kernels=(KERNEL, KernelArtifact(KernelKind.FUSION, "rmsnorm", "kernel/n.hsaco", "5" * 64)))

    assert touched(BEFORE, swapped) == frozenset({"kernel gemm qkv_proj"})
    assert touched(BEFORE, added) == frozenset({"kernel fusion rmsnorm"})


def test_two_recipes_that_differ_in_nothing_name_nothing() -> None:
    assert touched(BEFORE, replace(BEFORE)) == frozenset()
