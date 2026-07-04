###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the bypass device-kernel-name classifier (_bypass_classify)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bypass_classify import classify_kernel  # noqa: E402


def test_paged_attention_is_sdpa_reusable():
    kc = classify_kernel("paged_attention_v1_kernel")
    assert kc.category == "SDPA"
    assert kc.reusable is True
    assert kc.skip_reason == ""


def test_vendor_gemm_is_gemm_but_skipped():
    kc = classify_kernel("Cijk_Alik_Bljk_HHS_BH_MT128x128")
    assert kc.category == "GEMM"
    assert kc.reusable is False
    assert "vendor" in kc.skip_reason


def test_scaled_mm_gemm_unresolved_source_skipped():
    # scaled_mm matches GEMM but is neither a vendor binary nor a reusable
    # native kernel -> skipped with an unresolved-source reason.
    kc = classify_kernel("scaled_mm_kernel")
    assert kc.category == "GEMM"
    assert kc.reusable is False
    assert "source file not resolved" in kc.skip_reason


def test_reshape_and_cache_is_kvcachestore_reusable():
    kc = classify_kernel("reshape_and_cache_flash_kernel")
    assert kc.category == "KVCacheStore"
    assert kc.reusable is True


def test_rmsnorm_is_normalization_reusable():
    kc = classify_kernel("rms_norm_kernel")
    assert kc.category == "Normalization"
    assert kc.reusable is True


def test_quantization_high_priority_over_generic():
    kc = classify_kernel("per_tensor_quant_fp8")
    assert kc.category == "Quantization"
    assert kc.reusable is True


def test_gpu_cat_memcpy_override_forces_memcpy():
    # Even a GEMM-looking name is MemCpy when the Kineto cat says so.
    kc = classify_kernel("Cijk_whatever", gpu_cat="gpu_memcpy")
    assert kc.category == "MemCpy"
    assert kc.reusable is False


def test_memset_cat_override():
    kc = classify_kernel("anything", gpu_cat="gpu_memset")
    assert kc.category == "MemCpy"
    assert kc.reusable is False


def test_triton_fused_is_reusable_elementwise():
    kc = classify_kernel("triton_poi_fused_add_0")
    assert kc.category == "Elementwise"
    assert kc.reusable is True


def test_unknown_kernel_is_others_and_skipped():
    kc = classify_kernel("some_mystery_kernel_xyz")
    assert kc.category == "Others"
    assert kc.reusable is False
    assert kc.skip_reason == "source file not resolved"


def test_empty_name_is_others_skipped():
    kc = classify_kernel("")
    assert kc.category == "Others"
    assert kc.reusable is False


def test_moe_before_generic_gemm():
    kc = classify_kernel("kernel_moe_gemm_blockscale")
    assert kc.category == "MoE"


# ── DiT / diffusion kernels (M7b) ────────────────────────────────────────────


def test_dit_adaln_modulate_is_normalization_reusable():
    for name in ("FusedLnModulate", "triton_ada_ln_fwd", "scale_shift_modulate"):
        kc = classify_kernel(name)
        assert kc.category == "Normalization", name
        assert kc.reusable is True, name


def test_vae_conv_is_convolution():
    # Vendor conv library -> Convolution, not rewritable.
    kc = classify_kernel("miopenConvolutionForward")
    assert kc.category == "Convolution"
    assert kc.reusable is False
    # Named conv kernel still classifies as Convolution.
    assert classify_kernel("conv2d_nhwc_kernel").category == "Convolution"


def test_non_conv_miopen_kernel_is_not_convolution():
    # MIOpen/cuDNN also emit non-conv kernels; those must not be mislabeled
    # Convolution, but are still vendor (non-reusable).
    kc = classify_kernel("miopenBatchNormalizationForwardInference")
    assert kc.category != "Convolution"
    assert kc.reusable is False


def test_dit_rules_do_not_regress_text_gen_norm():
    # Adding modulate/conv rules must not steal existing rmsnorm classification.
    assert classify_kernel("rms_norm_kernel").category == "Normalization"
    assert classify_kernel("Cijk_Alik_Bljk_HHS").category == "GEMM"
    assert classify_kernel("paged_attention_v1").category == "SDPA"
