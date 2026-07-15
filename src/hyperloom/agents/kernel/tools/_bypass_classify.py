###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Device-kernel-name classification for the bypass analysis backend.

A compact kernel-name taxonomy; the primary categorization signal for the bypass
route because it has full coverage even when Kineto op-correlation is broken by
cudagraph/torch.compile replay.

Two outputs per kernel name:
  * ``category``: coarse perf category aligned with the labels downstream and
    the golden reports use (SDPA / GEMM / Normalization / Convolution /
    Quantization / KVCacheStore / Elementwise / MemCpy / MoE / Others).
  * ``reusable``: whether the kernel is a rewritable native-source kernel
    (True) versus a vendor precompiled binary or an unresolved kernel (False),
    plus a ``skip_reason`` for the False case. Mirrors the golden
    ``summary.json`` routed-vs-skipped semantics.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# (compiled_pattern, category, priority) — higher priority wins on conflict.
_RULES: list[tuple[re.Pattern, str, int]] = [
    # MemCpy (highest; also detected via Kineto cat).
    (re.compile(r"(?i)memcpy|memset"), "MemCpy", 30),
    # MoE (specific, before generic GEMM/Elementwise).
    (re.compile(r"(?i)swiglu|fmoe|kernel_moe_gemm|MoeGemmBlockScale"), "MoE", 25),
    (re.compile(r"(?i)moe_sorting|moe_align|topk_softmax|topkGatingSoftmax|routing"), "MoE", 22),
    # Attention / SDPA.
    (re.compile(r"(?i)paged_attention|PagedAttention"), "SDPA", 20),
    (re.compile(r"(?i)attention_[23]d|unified_attention"), "SDPA", 20),
    (re.compile(r"(?i)flash_attn|flash_fwd|fmha"), "SDPA", 20),
    (re.compile(r"(?i)_fwd_kernel|reduce_segments"), "SDPA", 18),
    # KV cache store.
    (re.compile(r"(?i)reshape_and_cache|concat_and_cache"), "KVCacheStore", 20),
    # Normalization.
    (re.compile(r"(?i)rmsnorm|rms_norm|layer_norm|layernorm|l2norm"), "Normalization", 20),
    # DiT adaptive layernorm (adaLN / modulate) — before generic norm.
    (re.compile(r"(?i)fusedlnmodulate|ada_?ln|modulate|scale_shift"), "Normalization", 19),
    (re.compile(r"(?i)add_rmsnorm|fused.*mean.*rsqrt|rsqrt.*mean"), "Normalization", 18),
    # Convolution. Require a conv context for miopen/cudnn (they also emit
    # norm/pooling kernels that must not be mislabeled as Convolution).
    (re.compile(r"(?i)conv2d|conv_2d|conv_fwd|conv_bwd|convolution|miopen.*conv|cudnn.*conv"), "Convolution", 16),
    # Rotary embedding -> elementwise family.
    (re.compile(r"(?i)rotary|\brope\b"), "Elementwise", 18),
    # Quantization (fp8/fp4 scale/quant kernels).
    (re.compile(r"(?i)per_tensor_quant|per_token.*quant|dynamic.*quant|scaled_quant|data_to_scale|initializeScale"), "Quantization", 18),
    (re.compile(r"(?i)\bquant\b|quantize"), "Quantization", 6),
    # Activation / elementwise.
    (re.compile(r"(?i)silu|swish|\bgelu\b|act_and_mul"), "Elementwise", 15),
    (re.compile(r"(?i)embedding|gather_kernel|vectorized_gather"), "Elementwise", 14),
    (re.compile(r"(?i)elementwise|CatArrayBatchedCopy|direct_copy_kernel|_to_copy\b|FillFunctor"), "Elementwise", 10),
    # GEMM (vendor + generic).
    (re.compile(r"(?i)Cijk_|wvSplitK|splitKreduce|hipblaslt|rocblas|cublas|nvjet"), "GEMM", 12),
    (re.compile(r"(?i)kernel_gemm_xdl_cshuffle|_gemm_a\d+w\d+|_gemm_a16_w16"), "GEMM", 12),
    (re.compile(r"(?i)scaled_mm|\bgemm\b|matmul|\bbmm\b"), "GEMM", 8),
    # Triton / generic catch-alls (lowest).
    (re.compile(r"(?i)triton_poi_fused|triton_red_fused|triton_per_fused"), "Elementwise", 3),
    (re.compile(r"(?i)\bnorm\b"), "Normalization", 1),
    (re.compile(r"(?i)rocprim|hipcub|DeviceScan|DeviceRadixSort|DeviceReduce"), "Elementwise", 1),
]

# Vendor precompiled kernels: rankable but not rewritable.
_VENDOR_BINARY_RE = re.compile(
    r"(?i)Cijk_|wvSplitK|splitKreduce|hipblaslt|rocblas|cublas|nvjet_tst|miopen|cudnn"
)
# Native-source kernels that are rewritable (triton / aiter / CK / vLLM native).
_REUSABLE_RE = re.compile(
    r"(?i)triton_|^_fwd_kernel|aiter|ck_tile|paged_attention|reshape_and_cache|"
    r"rmsnorm|rms_norm|add_rmsnorm|silu|per_tensor_quant|scaled_quant|data_to_scale|initializeScale|"
    r"fusedlnmodulate|ada_?ln|modulate|scale_shift"
)


# Launching-op-name -> category fallback, consulted only on the ``Others``
# fallthrough. ORDER IS PRIORITY: ``_classify_by_op`` returns the FIRST matching
# row, so rows MUST stay ordered most-specific-first. Norm keeps only explicit
# ``*_norm`` names so a bare ``\bnorm\b`` does not mis-map ``aten::norm``.
_OP_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)convolution|conv[123]d|conv_transpose|_convolution"), "Convolution"),
    (re.compile(r"(?i)scaled_dot_product_attention|efficient_attention|flash_attention|"
                r"memory_efficient_attention|\bsdpa\b|_attention_forward|multi_head_attention"), "SDPA"),
    (re.compile(r"(?i)layer_norm|layernorm|rms_norm|rmsnorm|group_norm|groupnorm|batch_norm"), "Normalization"),
    (re.compile(r"(?i)reshape_and_cache|concat_and_cache"), "KVCacheStore"),
    (re.compile(r"(?i)\bmoe\b|fused_moe|topk|routing|expert"), "MoE"),
    (re.compile(r"(?i)quantize|\bquant\b|per_tensor|per_token"), "Quantization"),
    (re.compile(r"(?i)addmm|baddbmm|\bbmm\b|\bmm\b|matmul|\blinear\b|\bgemm\b|scaled_mm"), "GEMM"),
    (re.compile(r"(?i)silu|swish|\bgelu\b|\brelu\b|sigmoid|\btanh\b|gelu_and_mul|act_and_mul"), "Elementwise"),
    (re.compile(r"(?i)rotary|\brope\b|embedding"), "Elementwise"),
    (re.compile(r"(?i)elementwise|\bmul\b|\badd\b|\bsub\b|\bdiv\b|\bcopy_?\b|\bcat\b|concat|"
                r"index_select|slice|\bview\b|reshape|transpose|permute|fill|clamp|to_copy"), "Elementwise"),
]


# High-level primitives that vendor libraries lower onto a Tensile GEMM device
# kernel. When the device name is GEMM but the launching op is one of these, the
# op name overrides so the roofline uses the right FLOP model.
_GEMM_LOWERED_OPS = frozenset({"Convolution", "SDPA"})


def _classify_by_op(op_name: str) -> str:
    """Return a category from a launching op name, or ``""`` on no match.

    Returns the FIRST matching :data:`_OP_RULES` row.

    Args:
        op_name: Resolved launching op name (e.g. ``aten::miopen_convolution``).

    Returns:
        The matched category, or ``""`` when no op rule applies.
    """
    n = op_name or ""
    for pat, cat in _OP_RULES:
        if pat.search(n):
            return cat
    return ""


class KernelClass(NamedTuple):
    """Classification result for one device-kernel name."""

    category: str
    reusable: bool
    skip_reason: str


def classify_kernel(name: str, *, gpu_cat: str = "", op_name: str = "") -> KernelClass:
    """Classify a device-kernel name into a category + reusability verdict.

    Args:
        name: The device (GPU) kernel name from the trace.
        gpu_cat: The Kineto ``cat`` of the event (``gpu_memcpy`` / ``gpu_memset``
            force the ``MemCpy`` category regardless of name).
        op_name: Optional launching op name (from Kineto correlation). Used only
            as a category fallback when the device name is unclassifiable
            (``Others``); the reusability verdict stays device-name based.

    Returns:
        A :class:`KernelClass` with category, reusable flag, and a skip_reason
        (empty when reusable).
    """
    if gpu_cat in ("gpu_memcpy", "gpu_memset"):
        return KernelClass("MemCpy", False, "device memcpy/memset (not a rewritable kernel)")
    n = name or ""
    category = "Others"
    best_prio = -1
    for pat, cat, prio in _RULES:
        if prio > best_prio and pat.search(n):
            category, best_prio = cat, prio

    # Op-name signal: fallback when the device name is ``Others``, and a
    # Convolution/SDPA op overrides a GEMM device classification.
    if op_name:
        op_cat = _classify_by_op(op_name)
        if category == "Others":
            category = op_cat or "Others"
        elif category == "GEMM" and op_cat in _GEMM_LOWERED_OPS:
            category = op_cat

    if category == "MemCpy":
        return KernelClass("MemCpy", False, "device memcpy/memset (not a rewritable kernel)")
    if _VENDOR_BINARY_RE.search(n):
        return KernelClass(category, False, "vendor backend library (precompiled binary, no rewritable source)")
    if _REUSABLE_RE.search(n):
        return KernelClass(category, True, "")
    return KernelClass(category, False, "source file not resolved")
