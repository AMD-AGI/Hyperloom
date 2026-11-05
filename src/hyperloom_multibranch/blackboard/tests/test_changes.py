# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A recipe delta in one line: config first, then components, then kernels, as apply takes them."""

from __future__ import annotations

from dataclasses import replace

from hyperloom.contracts.recipe import (
    ArgsMode,
    ComponentChange,
    ComponentPatches,
    ConfigDelta,
    KernelArtifact,
    KernelKind,
    LaunchConfig,
    PatchRef,
    Recipe,
    RecipeDelta,
)
from hyperloom_multibranch.blackboard.changes import NO_CHANGE, describe_change, same_recipe

SHA = "0" * 64


def test_an_empty_delta_changes_nothing() -> None:
    assert describe_change(RecipeDelta()) == NO_CHANGE


def test_the_config_part_names_the_launch_s_removals_when_it_gives_them_whole() -> None:
    removing = RecipeDelta(config=ConfigDelta(remove_args=("--enable-x",), launch_remove_args=("--enable-x",)))
    restoring = RecipeDelta(config=ConfigDelta(launch_remove_args=()))

    assert (describe_change(removing), describe_change(restoring)) == (
        "remove args --enable-x; resolved launch removes only --enable-x",
        "resolved launch removes nothing",
    )


def test_the_config_part_names_what_it_adds_removes_sets_and_unsets() -> None:
    config = ConfigDelta(
        server_args="--cuda-graph-max-bs 256", remove_args=("--enable-x",), envs={"B": "2", "A": "1"}, unset_envs=("C",)
    )

    assert describe_change(RecipeDelta(config=config)) == (
        "add args --cuda-graph-max-bs 256; remove args --enable-x; set env A=1, B=2; unset env C"
    )


def test_replacing_the_args_says_with_what() -> None:
    replaced = RecipeDelta(config=ConfigDelta(server_args="--tp 2", args_mode=ArgsMode.REPLACE))
    cleared = RecipeDelta(config=ConfigDelta(args_mode=ArgsMode.REPLACE))

    assert (describe_change(replaced), describe_change(cleared)) == (
        "replace args with --tp 2",
        "replace args with none",
    )


def test_components_and_kernels_are_named_by_their_files() -> None:
    delta = RecipeDelta(
        components=(
            ComponentChange(
                "aiter",
                upstream_ref="7eb34ead" + "1" * 32,
                upstream_repo="https://github.com/ROCm/aiter",
                add_patches=(PatchRef("patch/aiter/0001-tile.patch", SHA),),
            ),
            ComponentChange(
                "sglang", upstream_ref="2a3f7ac8" + "2" * 32, remove_patches=("patch/sglang/0001-fuse.patch",)
            ),
            ComponentChange("triton"),
        ),
        add_kernels=(KernelArtifact(KernelKind.GEMM, "qkv_proj", "kernel/gemm/tuned.csv", SHA),),
        remove_kernels=("kernel/fusion/old.so",),
    )

    assert describe_change(delta) == (
        "aiter: new at 7eb34ead, add 0001-tile.patch; sglang: move to 2a3f7ac8, drop 0001-fuse.patch; "
        "triton: unchanged; drop kernel old.so; add gemm kernel for qkv_proj"
    )


def test_two_recipes_are_the_same_whatever_the_order_of_their_flags_and_differ_on_any_change() -> None:
    patched = ComponentPatches(
        "sglang", "https://github.com/sgl-project/sglang", "2" * 40, (PatchRef("patch/sglang/0001-fuse.patch", SHA),)
    )
    tuned = KernelArtifact(KernelKind.GEMM, "qkv_proj", "kernel/gemm/tuned.csv", SHA)
    first = Recipe("sha256:" + "a" * 64, LaunchConfig("--kv-cache-dtype fp8 --tp 1", {"A": "1"}), (patched,), (tuned,))

    assert same_recipe(first, replace(first, launch=LaunchConfig("--tp 1 --kv-cache-dtype fp8", {"A": "1"})))
    assert not same_recipe(first, replace(first, launch=LaunchConfig("--kv-cache-dtype fp8 --tp 2", {"A": "1"})))
    assert not same_recipe(first, replace(first, launch=LaunchConfig("--kv-cache-dtype fp8 --tp 1")))
    assert not same_recipe(first, replace(first, components=(replace(patched, patches=()),)))
    assert not same_recipe(first, replace(first, kernels=()))
