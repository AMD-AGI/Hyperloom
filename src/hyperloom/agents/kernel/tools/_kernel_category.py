"""Canonical kernel-category display vocabulary shared across trace routes.

The two trace-analysis backends (bypass and TraceLens) carry route-native
category taxonomies that overlap on common kernels but diverge at the margins
and in casing. This module maps BOTH vocabularies onto ONE canonical display
label so ``analysis.md`` shows a consistent category regardless of route.

Display-only: this does NOT rewrite the ``kernel_category`` /
``tracelens_category`` fields in ``kernel_candidates.json`` / ``kernel_roofline.json``,
which downstream GEAK skill-routing consumes with its own taxonomy.

Kept dependency-free (stdlib only) so both routes can import it freely.
"""

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
    """Map a route-native category string to the canonical display label.

    Args:
        raw: A route-native category (e.g. ``"gemm"``, ``"Others"``, ``"norm"``).

    Returns:
        The canonical display label, or ``None`` when ``raw`` is empty/None, or
        the original (stripped) value when it maps to nothing known (so a novel
        category is surfaced verbatim rather than silently dropped).
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    key = s.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    return _CANONICAL.get(key, s)
