# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The prompt names the model directory so the author does not go looking for it."""

from __future__ import annotations

from kernelforge.fusion.author import build_author_prompt, build_multi_author_prompt

RECIPE = {
    "pattern": "llm:x",
    "description": "d",
    "env_flag": "X_FUSED",
    "source_file": "vllm/models/m/model.py",
    "source_hints": ["h"],
    "fusion_math": "a+b",
    "eager_reference_hint": "ref",
    "shapes": {"decode_batch": 16},
}

MODEL_DIR = "/shared_nfs/hyperloom/models/DeepSeek-V4-Flash"


def test_single_recipe_prompt_names_the_model_directory() -> None:
    prompt = build_author_prompt(RECIPE, framework="vllm", ab_hint="", model_path=MODEL_DIR)

    assert MODEL_DIR in prompt
    assert "find /" in prompt  # the reason it must not be searched for is stated


def test_multi_recipe_prompt_names_the_model_directory() -> None:
    prompt = build_multi_author_prompt(
        [RECIPE, dict(RECIPE, pattern="llm:y")],
        framework="vllm",
        ab_hint="",
        model_path=MODEL_DIR,
    )

    assert MODEL_DIR in prompt


def test_prompt_is_unchanged_when_the_path_is_unknown() -> None:
    prompt = build_author_prompt(RECIPE, framework="vllm", ab_hint="")

    assert "Model directory" not in prompt
