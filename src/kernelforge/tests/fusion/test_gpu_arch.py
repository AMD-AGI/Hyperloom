# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The author must be told the arch the run is actually on."""

from __future__ import annotations

from hyperloom.common.gpu_identity import AMD_GPU_BOARDS
from kernelforge.fusion.author import AuthorBrief, build_author_prompt, build_multi_author_prompt
from kernelforge.fusion.gpu_arch import arch_phrase, canon_arch

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
    assert canon_arch("R9700") == "gfx1201"
    assert canon_arch("AMD Radeon AI PRO R9700") == "gfx1201"
    # Unresolvable arch must be empty: naming the wrong ISA is worse than naming none at all.
    assert canon_arch("") == ""
    assert canon_arch("some-new-gpu") == ""


# --- prompt wording -------------------------------------------------------- #
def test_known_archs_get_their_marketing_name():
    assert arch_phrase("gfx950") == "AMD MI355X (gfx950)"
    assert arch_phrase("gfx942") == "AMD MI300X/MI308X/MI325X (gfx942)"
    assert arch_phrase("gfx1201") == "AMD R9700 (gfx1201)"


def test_an_unrecognised_arch_is_still_named_exactly():
    assert arch_phrase("gfx1100") == "an AMD GPU (gfx1100)"


def test_an_unknown_arch_says_nothing_rather_than_guessing():
    assert arch_phrase("") == "an AMD ROCm GPU"
    assert "gfx" not in arch_phrase("")


def test_the_prompt_names_the_arch_it_was_given():
    prompt = build_author_prompt(RECIPE, AuthorBrief(framework="vllm", ab_hint="hint", gpu_arch="gfx950"))
    assert "MI355X (gfx950)" in prompt
    # Regression: this used to be hardcoded to another chip.
    assert "gfx942" not in prompt


def test_the_multi_prompt_names_the_arch_too():
    prompt = build_multi_author_prompt(
        [RECIPE, dict(RECIPE, pattern_id="qk_norm_rope", env_flag="QWEN3_FUSED_QK")],
        AuthorBrief(framework="vllm", ab_hint="hint", gpu_arch="gfx950"),
    )
    assert "MI355X (gfx950)" in prompt
    assert "gfx942" not in prompt


def test_no_arch_leaves_no_false_target_in_the_prompt():
    prompt = build_author_prompt(RECIPE, AuthorBrief(framework="vllm", ab_hint="hint"))
    assert "an AMD ROCm GPU" in prompt
    assert "gfx942" not in prompt
    assert "MI325X" not in prompt


def test_every_board_and_its_marketing_name_resolve_to_its_arch():
    """The arch comes from the shared board table, so a new board resolves here in the commit that adds it."""
    for board in AMD_GPU_BOARDS.values():
        assert canon_arch(board.product) == board.gfx, board.product
        assert canon_arch(board.marketing_name) == board.gfx, board.marketing_name
        assert board.product.upper() in arch_phrase(board.gfx)
