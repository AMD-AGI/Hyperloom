###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the bypass device-kernel-name classifier (_bypass_classify)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

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


def test_real_sana_dit_kernels_via_op_fallback():
    # Real Sana xDiT trace (2026-07-05): bare device names attn_fwd / naive_conv
    # are NOT matched by device rules, but their launching op names resolve them
    # via the op-name fallback (generalizes better than per-name device regexes).
    assert classify_kernel("attn_fwd").category == "Others"  # device name alone: unknown
    assert classify_kernel("attn_fwd", op_name="aten::_efficient_attention_forward").category == "SDPA"
    assert classify_kernel("naive_conv_ab_nonpacked_fwd_nchw_ushort").category == "Others"
    assert (
        classify_kernel("naive_conv_ab_nonpacked_fwd", op_name="aten::miopen_depthwise_convolution").category
        == "Convolution"
    )
    # The dominant Sana VAE kernel (MIOpen conv) is caught by the device rule.
    assert classify_kernel("miopenSp3AsmConv_v30_3_1_gfx9_fp32_f2x3_stride1").category == "Convolution"


def test_op_name_fallback_rules():
    # Op-name fallback maps stable aten/framework symbols when the device name
    # is unclassifiable.
    cases = {
        "aten::conv2d": "Convolution",
        "aten::scaled_dot_product_attention": "SDPA",
        "aten::native_layer_norm": "Normalization",
        "aten::rms_norm": "Normalization",
        "aten::addmm": "GEMM",
        "aten::mm": "GEMM",
        "aten::bmm": "GEMM",
        "aten::silu": "Elementwise",
        "aten::mul": "Elementwise",
    }
    for op, cat in cases.items():
        assert classify_kernel("some_unknown_device_kernel", op_name=op).category == cat, op


def test_op_fallback_only_on_others_and_needs_op():
    # A specific device category is NOT overridden by an unrelated op name.
    assert classify_kernel("paged_attention_v1", op_name="aten::mm").category == "SDPA"
    # Exception: a GEMM device IS overridden by a conv/attention op (vendor libs
    # lower those onto a Tensile GEMM); a non-lowered op leaves GEMM unchanged.
    assert classify_kernel("Cijk_gemm", op_name="aten::conv2d").category == "Convolution"
    assert classify_kernel("Cijk_gemm", op_name="aten::mul").category == "GEMM"
    # No op name -> unclassifiable device name stays Others.
    assert classify_kernel("totally_unknown_kernel_xyz").category == "Others"
    # An unrecognized op name -> stays Others.
    assert classify_kernel("totally_unknown_kernel_xyz", op_name="aten::weird_op").category == "Others"


def test_op_fallback_category_does_not_override_vendor_verdict():
    # op-fallback affects the CATEGORY only; the reusability verdict stays
    # device-name based. A cuDNN attention kernel has an unclassifiable device
    # name (Others) but is a vendor precompiled binary: the launching op resolves
    # the category to SDPA, yet it must remain non-reusable (vendor), not become
    # a rewritable-source candidate.
    name = "cudnn_mha_fwd_bf16"
    base = classify_kernel(name)
    assert base.category == "Others"
    assert base.reusable is False
    assert "vendor" in base.skip_reason

    resolved = classify_kernel(name, op_name="aten::scaled_dot_product_attention")
    assert resolved.category == "SDPA"      # category filled from the op fallback
    assert resolved.reusable is False       # but still a vendor binary
    assert "vendor" in resolved.skip_reason


def test_op_fallback_math_norm_is_not_normalization():
    # F2 guard: a bare math reduction op (aten::norm / linalg_vector_norm) must
    # NOT be mapped to Normalization — only explicit *_norm layer ops do. A bare
    # \bnorm\b op rule would have mis-mapped these.
    assert classify_kernel("unknown_reduce_kernel", op_name="aten::norm").category == "Others"
    assert classify_kernel("unknown_reduce_kernel", op_name="aten::linalg_vector_norm").category == "Others"
    # Explicit normalization layers still resolve via the fallback.
    assert classify_kernel("unknown_kernel", op_name="aten::native_layer_norm").category == "Normalization"
    assert classify_kernel("unknown_kernel", op_name="aten::group_norm").category == "Normalization"


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


# ── op-name overrides a GEMM-lowered device kernel (roofline correctness) ─────


def test_conv_op_overrides_gemm_lowered_device_name():
    # MIOpen lowers a 1x1 conv onto a Tensile GEMM (Cijk_) device kernel. The
    # device name alone classifies it GEMM; the launching op carries the true
    # primitive, so the category must be Convolution (for a correct conv
    # roofline) while reusability stays vendor (Cijk_ is a precompiled binary).
    kc = classify_kernel(
        "Cijk_Ailk_Bljk_BBS_BH_MT64x256x32_MI16x16x16x1", op_name="aten::miopen_convolution"
    )
    assert kc.category == "Convolution"
    assert kc.reusable is False
    assert "vendor" in kc.skip_reason


def test_attention_op_overrides_gemm_lowered_device_name():
    # An attention op lowered onto a GEMM device kernel -> SDPA (same principle).
    kc = classify_kernel("Cijk_Ailk_Bljk_foo", op_name="aten::_efficient_attention_forward")
    assert kc.category == "SDPA"


def test_genuine_gemm_op_is_not_overridden():
    # A real GEMM (op aten::mm) on a Cijk_ device kernel must STAY GEMM: the
    # override only applies to conv/attention primitives vendors lower to GEMM.
    assert classify_kernel("Cijk_Alik_Bljk_HHS", op_name="aten::mm").category == "GEMM"
    assert classify_kernel("Cijk_Alik_Bljk_HHS", op_name="aten::addmm").category == "GEMM"


def test_gemm_override_only_applies_to_gemm_device_category():
    # A conv op on an already-Convolution device name stays Convolution (no-op),
    # and a conv op does not upgrade a non-GEMM device category (e.g. Elementwise).
    assert classify_kernel("naive_conv_fwd", op_name="aten::miopen_convolution").category == "Convolution"
    assert classify_kernel("silu_kernel", op_name="aten::miopen_convolution").category == "Elementwise"
