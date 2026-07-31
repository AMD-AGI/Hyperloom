###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for :mod:`_bypass_classify.classify_kernel` taxonomy coverage."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from _bypass_classify import classify_kernel  # noqa: E402


# (device_kernel_name, expected_category) covering real sglang+aiter kernels.
_CATEGORY_CASES = [
    # Communication / collective.
    ("cross_device_reduce_2stage", "Communication"),
    ("sglang::outplace_all_reduce", "Communication"),
    ("all_reduce", "Communication"),
    ("allgather_vec", "Communication"),
    ("reg_all_gather", "Communication"),
    ("_ZN5aiter26cross_device_reduce_2stageIDF16bLi8ELb0EEEvPNS_8RankDataE", "Communication"),
    # Attention.
    ("_fwd_grouped_kernel_stage1", "SDPA"),
    ("_fwd_kernel_stage2", "SDPA"),
    ("_decode_grouped_att", "SDPA"),
    ("mla_decode", "SDPA"),
    ("paged_attention", "SDPA"),
    ("_score_kernel", "SDPA"),
    ("_combine_kernel", "SDPA"),
    ("kda_packed_decode", "SDPA"),
    # MoE.
    ("fused_moe", "MoE"),
    ("moe_gemm1", "MoE"),
    ("moe_gemm2", "MoE"),
    ("ck_moe", "MoE"),
    ("fmoe", "MoE"),
    ("moe_sorting", "MoE"),
    ("grouped_topk", "MoE"),
    ("moe_align", "MoE"),
    ("topk_softmax", "MoE"),
    ("biased_grouped_topk", "MoE"),
    # GEMM.
    ("gemm_a16w16", "GEMM"),
    ("gemm_a8w8", "GEMM"),
    ("hgemm_bf16", "GEMM"),
    ("flatmm", "GEMM"),
    ("Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT160x64x128", "GEMM"),
    ("opus_gemm", "GEMM"),
    # Normalization.
    ("add_rmsnorm", "Normalization"),
    ("rmsnorm", "Normalization"),
    ("layernorm", "Normalization"),
    ("rms_norm", "Normalization"),
    ("add_rmsnorm_quant_kernel", "Normalization"),
    # Elementwise.
    ("silu_and_mul", "Elementwise"),
    ("situ_and_mul", "Elementwise"),
    ("act_and_mul", "Elementwise"),
    ("elementwise_kernel", "Elementwise"),
    ("index_elementwise", "Elementwise"),
    ("copy_", "Elementwise"),
    ("_to_copy", "Elementwise"),
    ("void (anonymous namespace)::situ_and_mul_kernel", "Elementwise"),
    # Quantization.
    ("quant", "Quantization"),
    ("dynamic_per_token", "Quantization"),
    ("per_tensor_quant", "Quantization"),
    ("static_quant_fp8", "Quantization"),
]


@pytest.mark.parametrize("name,expected", _CATEGORY_CASES)
def test_category_mapping(name: str, expected: str) -> None:
    assert classify_kernel(name).category == expected, name


def test_communication_is_not_reusable() -> None:
    for name in [
        "cross_device_reduce_2stage",
        "sglang::outplace_all_reduce",
        "all_reduce",
        "allgather_vec",
        "reg_all_gather",
        "_ZN5aiter26cross_device_reduce_2stageIDF16bLi8ELb0EEEvPNS_8RankDataE",
    ]:
        kc = classify_kernel(name)
        assert kc.category == "Communication", name
        assert kc.reusable is False, name


def test_mangled_allreduce_via_op_name_fallback() -> None:
    # op_name attribution present: still Communication and not reusable.
    kc = classify_kernel("_ZN5aiter26unknown_mangledEv", op_name="sglang::outplace_all_reduce")
    assert kc.category == "Communication"
    assert kc.reusable is False


def test_moe_wins_over_gemm_substring() -> None:
    assert classify_kernel("moe_gemm1").category == "MoE"


def test_kda_packed_decode_is_reusable_sdpa() -> None:
    kc = classify_kernel("void (anonymous namespace)::kda_packed_decode_kernel<8, false>(x)")
    assert kc.category == "SDPA"
    assert kc.reusable is True
