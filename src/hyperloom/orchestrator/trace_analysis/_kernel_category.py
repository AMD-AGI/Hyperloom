"""Canonical kernel-category display vocabulary for rendered reports."""

from __future__ import annotations

#: Normalized-key -> canonical display label. Keys are lower-cased with
#: ``[ /-]`` collapsed to ``_`` before lookup (see :func:`canonical_category`).
_CANONICAL: dict[str, str] = {
    # GEMM / matmul family.
    "gemm": "GEMM",
    "grouped_gemm": "GEMM",
    "groupedgemm": "GEMM",
    "groupedgemm_fwd": "GEMM",
    "groupedgemm_bwd": "GEMM",
    "matmul": "GEMM",
    "bmm": "GEMM",
    # Attention / SDPA.
    "sdpa": "SDPA",
    "sdpa_fwd": "SDPA",
    "sdpa_bwd": "SDPA",
    "attention": "SDPA",
    "inferenceattention": "SDPA",
    # Mixture of experts.
    "moe": "MoE",
    "moe_fused": "MoE",
    "moe_unfused": "MoE",
    "moe_aux": "MoE",
    # Elementwise.
    "elementwise": "Elementwise",
    # Normalization.
    "normalization": "Normalization",
    "norm": "Normalization",
    "norm_fwd": "Normalization",
    "norm_bwd": "Normalization",
    "layernorm": "Normalization",
    "rmsnorm": "Normalization",
    # Convolution.
    "convolution": "Convolution",
    "conv_fwd": "Convolution",
    "conv_bwd": "Convolution",
    # Quantization.
    "quantization": "Quantization",
    "quant": "Quantization",
    # KV-cache store.
    "kvcachestore": "KVCacheStore",
    # Reduction.
    "reduce": "Reduction",
    "reduction": "Reduction",
    # Communication.
    "communication": "Communication",
    "customcollective": "Communication",
    # Framework buckets.
    "triton": "Triton",
    "flydsl": "FlyDSL",
    "memcpy": "MemCpy",
    # Catch-all.
    "other": "Other",
    "others": "Other",
    "cpu_idle": "Other",
    "unknown": "Other",
}


def canonical_category(raw: str | None) -> str | None:
    """Map a route-native category string to the canonical display label."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    key = s.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    return _CANONICAL.get(key, s)
