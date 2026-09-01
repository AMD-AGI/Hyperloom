# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The author must be told the arch the run is actually on.

Tile shapes, warp counts and intrinsics are chosen per ISA, so naming one chip
in the prompt while running on another asks the author to tune for hardware
that is not there.
"""

from __future__ import annotations

from kernelforge.fusion.author import _arch_phrase, build_author_prompt, build_multi_author_prompt
from kernelforge.fusion.gpu_arch import canon_arch

RECIPE = {
    "pattern_id": "residual_add_rmsnorm",
    "description": "Fuse residual add into RMSNorm",
    "env_flag": "QWEN3_FUSED",
    "source_file": "/site-packages/vllm/model_executor/models/qwen3.py",
    "source_hints": ["+ residual"],
    "fusion_math": "y = rmsnorm(x + residual)",
    "eager_reference_hint": "import RMSNorm",
    "shapes": {"hidden_size": 4096},
    "matched_categories": ["add", "rmsnorm"],
}


# --- arch normalization ---------------------------------------------------- #
def test_canon_arch_folds_marketing_names_and_rejects_unknown():
    assert canon_arch("gfx942") == "gfx942"
    assert canon_arch("GFX950") == "gfx950"
    assert canon_arch("MI300X") == "gfx942"
    assert canon_arch("mi355x") == "gfx950"
    assert canon_arch("AMD Instinct MI355X") == "gfx950"
    # Unresolvable arch must be empty: naming the wrong ISA is worse than
    # naming none at all.
    assert canon_arch("") == ""
    assert canon_arch("some-new-gpu") == ""


# --- prompt wording -------------------------------------------------------- #
def test_known_archs_get_their_marketing_name():
    assert _arch_phrase("gfx950") == "AMD MI355X (gfx950)"
    assert _arch_phrase("gfx942") == "AMD MI300X/MI325X (gfx942)"


def test_an_unrecognised_arch_is_still_named_exactly():
    assert _arch_phrase("gfx1201") == "an AMD GPU (gfx1201)"


def test_an_unknown_arch_says_nothing_rather_than_guessing():
    assert _arch_phrase("") == "an AMD ROCm GPU"
    assert "gfx" not in _arch_phrase("")


def test_the_prompt_names_the_arch_it_was_given():
    prompt = build_author_prompt(RECIPE, framework="vllm", ab_hint="hint", gpu_arch="gfx950")
    assert "MI355X (gfx950)" in prompt
    # Regression: this used to be hardcoded to another chip.
    assert "gfx942" not in prompt


def test_the_multi_prompt_names_the_arch_too():
    prompt = build_multi_author_prompt(
        [RECIPE, dict(RECIPE, pattern_id="qk_norm_rope", env_flag="QWEN3_FUSED_QK")],
        framework="vllm",
        ab_hint="hint",
        gpu_arch="gfx950",
    )
    assert "MI355X (gfx950)" in prompt
    assert "gfx942" not in prompt


def test_no_arch_leaves_no_false_target_in_the_prompt():
    prompt = build_author_prompt(RECIPE, framework="vllm", ab_hint="hint")
    assert "an AMD ROCm GPU" in prompt
    assert "gfx942" not in prompt
    assert "MI325X" not in prompt
