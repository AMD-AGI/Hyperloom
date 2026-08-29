# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A discovered fusion's identity must not depend on how the model worded it.

Everything the KB key rests on -- the op categories, and whether the chain is
claimed as a framework compile pass -- used to be recovered by keyword-matching
the model's own prose. Measured on a real gateway, that made five runs against
one unchanged trace look up three different keys: a proposal that merely
mentioned writing to the KV cache picked up a ``copy`` category, and a reworded
one stopped matching the compile-pass keyword groups, which changed which
candidate ranked first.

These tests pin the fix: when the model declares its ops from a fixed
vocabulary, identity comes from that declaration and rewording cannot move it.
"""

from __future__ import annotations

import json

from kernelforge.fusion.discover import parse_discovered_recipes
from kernelforge.fusion.locate import PassState

SHAPES = {"batch": 16}
SOURCE = "/sp/vllm/model_executor/models/qwen3.py"


def _claimable(flag: str) -> PassState:
    """A compile pass that exists, is off, and is actually flippable.

    ``source="default"`` matters: a flag an optimization level pins is not
    claimable, because editing the PassConfig default would not change runtime
    behaviour.
    """
    return PassState(
        flag=flag,
        present=True,
        enabled=False,
        config_file="/sp/vllm/config.py",
        source="default",
    )


def _parse(payload: list[dict], *, framework: str = "vllm", probe=None):
    return parse_discovered_recipes(
        json.dumps(payload),
        model_type="qwen3",
        framework=framework,
        source_file=SOURCE,
        shapes=SHAPES,
        pass_probe=probe or _claimable,
        framework_root="",
    )


# The same fusion, described the way two different runs actually described it.
# ``qk_norm`` is a trait, not an op: declaring it under "ops" would be dropped
# silently, which would make the compile-pass assertion below vacuous.
TERSE = {
    "name": "attn_qk_norm_rope",
    "op_chain": "q_norm, k_norm, then rope",
    "fusion_math": "RMSNorm over head_dim for q and k, then rotary.",
    "ops": ["rmsnorm", "rope"],
    "traits": ["qk_norm"],
}
VERBOSE = {
    "name": "attn_qk_norm_rope_cache_prologue",
    "op_chain": "normalise q and k, apply the rotary embedding, then write into the KV cache",
    "fusion_math": (
        "Apply RMS normalisation to the query and key projections coming out of the "
        "qkv gemm, run the rotary transform, and copy the result into the paged KV cache."
    ),
    "ops": ["rmsnorm", "rope"],
    "traits": ["qk_norm"],
}


def test_rewording_one_proposal_does_not_change_its_categories():
    terse = _parse([TERSE])
    verbose = _parse([VERBOSE])
    assert terse and verbose
    assert terse[0].matched_categories == verbose[0].matched_categories, (
        "the verbose wording mentions the KV cache and the gemm; neither is part "
        "of the declared ops, so neither may enter the identity"
    )


def test_rewording_one_proposal_does_not_change_the_compile_pass_verdict():
    """Claiming a pass rewrites the pattern id, so a flip here moves the key.

    The proposal's own name is not asserted: it never reaches the key, because
    an ``llm:`` pattern hashes the category set instead.
    """
    terse = _parse([TERSE])
    verbose = _parse([VERBOSE])
    assert [r.candidate_kind for r in terse] == [r.candidate_kind for r in verbose]
    # Asserted absolutely, not just for agreement: two proposals that both fail
    # to reach the gate would agree too, and the test would prove nothing.
    assert terse[0].candidate_kind == "compile_pass"


def test_the_gate_still_sees_the_prose_when_traits_are_omitted():
    """``traits`` is optional, so a model will leave it out -- often.

    The compile-pass table keys on precision and variant words that live only in
    the prose. Dropping the prose the moment ``ops`` appears blinds the gate, and
    the run then hand-writes a kernel vLLM already ships.
    """

    def claimable(flag):
        return PassState(flag=flag, present=True, enabled=False, config_file="/sp/c.py", source="default")

    proposal = {
        "name": "norm_then_quant",
        "op_chain": "rmsnorm then fp8 quant scaled_mm",
        "fusion_math": "RMSNorm the hidden states, then quantize to fp8 for the scaled_mm.",
        "ops": ["rmsnorm"],
    }
    recipes = parse_discovered_recipes(
        json.dumps([proposal]),
        model_type="qwen3",
        framework="vllm",
        source_file="/sp/vllm/models/qwen3.py",
        shapes={},
        pass_probe=claimable,
    )
    assert recipes[0].candidate_kind == "compile_pass"


def test_declared_ops_outrank_the_prose():
    """The declaration is the identity; the prose is only description."""
    recipes = _parse(
        [
            {
                "name": "misleading_name_mentioning_moe_and_conv",
                "op_chain": "this sentence talks about gemm and attention in passing",
                "fusion_math": "and this one mentions layernorm and a memcpy",
                "ops": ["rmsnorm", "add"],
            }
        ],
        framework="sglang",
    )
    assert recipes[0].matched_categories == ["add", "rmsnorm"]


def test_an_undeclared_proposal_still_falls_back_to_the_prose():
    """Older prompts and models that ignore the field must keep working."""
    recipes = _parse(
        [
            {
                "name": "residual_add_rmsnorm",
                "op_chain": "add then rmsnorm",
                "fusion_math": "y = rmsnorm(x + residual)",
            }
        ],
        framework="sglang",
    )
    assert recipes[0].matched_categories == ["add", "rmsnorm"]


def test_junk_in_the_declaration_is_ignored_not_trusted():
    """A model inventing op names must not invent an identity segment with them."""
    recipes = _parse(
        [
            {
                "name": "f",
                "op_chain": "c",
                "fusion_math": "m",
                "ops": ["rmsnorm", "not_a_real_op", "", 7, "ROPE"],
            }
        ],
        framework="sglang",
    )
    # Case is normalised, unknown entries dropped, and the rest still identifies it.
    assert recipes[0].matched_categories == ["rmsnorm", "rope"]


def test_a_wholly_invalid_declaration_falls_back_rather_than_emptying_identity():
    """Dropping every entry must not leave the fusion with no identity at all."""
    recipes = _parse(
        [
            {
                "name": "residual_add_rmsnorm",
                "op_chain": "add then rmsnorm",
                "fusion_math": "y = rmsnorm(x + residual)",
                "ops": ["nonsense", "alsojunk"],
            }
        ],
        framework="sglang",
    )
    assert recipes[0].matched_categories == ["add", "rmsnorm"]


def test_traits_describe_the_kernel_without_moving_the_key():
    """Precision, variant and placement must not decide where a fusion is stored.

    A run that reads the same chain as fp8 rather than quantized, or is unsure
    whether it counts as attention, still has to find what the previous run
    stored. Over 20 measured runs ``attention`` was the term that flipped, and it
    separates nothing -- nearly every decode fusion sits beside attention.
    """
    plain = _parse([{**TERSE, "traits": []}], framework="sglang")
    adorned = _parse(
        [
            {
                **TERSE,
                "traits": ["attention", "qk_norm", "kvcache", "fp8", "quant"],
            }
        ],
        framework="sglang",
    )
    assert plain[0].matched_categories == adorned[0].matched_categories


def test_traits_still_reach_the_compile_pass_gate():
    """They are excluded from identity, not discarded: the gate keys on them."""
    claimed = _parse(
        [
            {
                "name": "qk_norm_then_rope",
                "op_chain": "c",
                "fusion_math": "m",
                "ops": ["rope"],
                "traits": ["qk_norm"],
            }
        ]
    )
    assert claimed[0].candidate_kind == "compile_pass", (
        "qk_norm + rope is a vLLM compile pass; the trait carries the half that no op category can express"
    )


def test_a_term_declared_in_the_wrong_field_is_ignored():
    """The split is only meaningful if each field rejects the other's terms."""
    recipes = _parse(
        [
            {
                "name": "f",
                "op_chain": "c",
                "fusion_math": "m",
                "ops": ["attention", "fp8", "mla"],  # traits, not ops
                "traits": ["rmsnorm", "add"],  # ops, not traits
            }
        ],
        framework="sglang",
    )
    # Nothing valid was declared in either field, so identity falls back to prose.
    assert recipes[0].matched_categories == []


def test_declaration_order_does_not_matter():
    """A set, not a sequence: the same ops in any order are the same fusion."""
    a = _parse([{**TERSE, "ops": ["rope", "qk_norm", "rmsnorm"]}], framework="sglang")
    b = _parse([{**TERSE, "ops": ["rmsnorm", "rope", "qk_norm"]}], framework="sglang")
    assert a[0].matched_categories == b[0].matched_categories
