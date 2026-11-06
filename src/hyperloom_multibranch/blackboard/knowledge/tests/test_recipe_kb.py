# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy

import pytest

from hyperloom.contracts.recipe import (
    ComponentPatches,
    KernelArtifact,
    KernelKind,
    LaunchConfig,
    PatchRef,
    Recipe,
    RecipeError,
)
from hyperloom.orchestrator.knowledge.remote_recipe import values as kb_values
from hyperloom_multibranch.blackboard.knowledge.recipe_kb import from_recipe_kb_value, overlay_ref, to_recipe_kb_value

IMAGE = "rocm/vllm@sha256:" + "c" * 64
SNAPSHOT = "sha256:" + "d" * 64


def _recipe() -> Recipe:
    return Recipe(
        base_image_digest=IMAGE,
        runtime_snapshot=SNAPSHOT,
        launch=LaunchConfig(
            server_args='--max-num-seqs 256 --compilation-config {"level":3}', envs={"VLLM_ROCM_USE_AITER": "1"}
        ),
        components=(
            ComponentPatches(
                "vllm",
                "https://github.com/vllm-project/vllm",
                "1" * 40,
                (
                    PatchRef("patch/vllm/0001-moe align.patch", "a" * 64),
                    PatchRef("patch/vllm/0002-attn.patch", "b" * 64),
                ),
            ),
            ComponentPatches("triton", "https://github.com/triton-lang/triton", "2" * 40),
            ComponentPatches(
                "aiter", "https://github.com/ROCm/aiter", "3" * 40, (PatchRef("patch/aiter/fix.diff", "f" * 64),)
            ),
        ),
        kernels=(
            KernelArtifact(KernelKind.GEMM, "bf16 4096x14336", "kernel/gemm/tuned.csv", "4" * 64, {"speedup": 1.2}),
            KernelArtifact(KernelKind.FUSION, "rmsnorm+quant", "kernel/fusion/rms.so", "5" * 64),
            KernelArtifact(KernelKind.REWRITE, "moe_align", "kernel/rewrite/moe.py", "6" * 64, {"op": "moe"}),
        ),
    )


def test_a_recipe_survives_the_round_trip_through_the_recipe_kb_columns() -> None:
    recipe = _recipe()

    value = to_recipe_kb_value(recipe)

    assert from_recipe_kb_value(value, base_image_digest=IMAGE, runtime_snapshot=SNAPSHOT) == recipe
    assert value["config"] == recipe.launch.to_document()


def test_the_columns_are_what_the_recipe_kb_itself_accepts() -> None:
    value = to_recipe_kb_value(_recipe())
    knowledge = {
        "knowledge_schema_version": kb_values.CURRENT_KNOWLEDGE_SCHEMA_VERSION,
        "record_kind": kb_values.RECORD_KIND_HYPERLOOM_RECIPE,
        "value": value,
    }

    kb_values._validate_patch_column(value)
    row = kb_values.knowledge_to_warm_recipe({"knowledge": knowledge})

    assert row["record_kind"] == kb_values.RECORD_KIND_HYPERLOOM_RECIPE
    assert value["patch"]["patches"] == [
        "patch/overlays/000000/00-0001-moe-align.patch",
        "patch/overlays/000000/01-0002-attn.patch",
        "patch/overlays/000002/00-fix.patch",
    ]
    assert [row["base_sha"] for row in value["patch"]["provenance"]] == ["1" * 40, "2" * 40, "3" * 40]
    assert value["kernel"]["gemm"]["optimizations"][0]["artifact_path"] == "kernel/gemm/tuned.csv"
    assert value["kernel"]["fusion"]["items"][0]["target"] == "rmsnorm+quant"


def test_artifact_metadata_cannot_take_the_row_fields() -> None:
    recipe = Recipe(
        base_image_digest=IMAGE,
        kernels=(KernelArtifact(KernelKind.GEMM, "t", "kernel/g.csv", "4" * 64, {"sha256": "x"}),),
    )

    with pytest.raises(RecipeError, match="row fields sha256"):
        to_recipe_kb_value(recipe)


def _broken(edit: str) -> dict[str, object]:
    value = copy.deepcopy(to_recipe_kb_value(_recipe()))
    if edit == "patches":
        value["patch"]["patches"].reverse()
    elif edit == "stacks":
        value["patch"]["provenance"].reverse()
    elif edit == "kernel":
        value["kernel"]["gemm"] = []
    else:
        del value["patch"]
    return value


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        ("patches", "other overlays"),
        ("stacks", "in order"),
        ("kernel", "kernel rows of gemm"),
        ("column", "no patch column"),
    ],
)
def test_a_value_this_module_did_not_write_is_refused(edit: str, message: str) -> None:
    with pytest.raises(RecipeError, match=message):
        from_recipe_kb_value(_broken(edit), base_image_digest=IMAGE)


def test_overlay_names_are_made_safe_and_never_empty() -> None:
    assert overlay_ref(3, 12, "patch/x/fix: the [bad] one!.patch") == "patch/overlays/000003/12-fix-the-bad-one.patch"
    assert overlay_ref(0, 0, "patch/x/__.patch") == "patch/overlays/000000/00-patch.patch"
