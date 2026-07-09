"""Canonical kernel-category display vocabulary shared across trace routes.

The two trace-analysis backends carry route-native category taxonomies:
  - bypass (``_bypass_classify``): ``GEMM / SDPA / Elementwise / Normalization /
    Convolution / Quantization / KVCacheStore / MoE / MemCpy / Others``.
  - TraceLens deterministic (GEAK labels via ``normalize_upstream_category``):
    ``GEMM / SDPA / Elementwise / Reduction / LayerNorm / Convolution / MoE /
    Communication / Triton / FlyDSL / Other``.

They overlap on the common kernels (GEMM/SDPA/Elementwise/Convolution/MoE) but
diverge at the margins (``Others`` vs ``Other``; ``Normalization`` vs
``LayerNorm``) and in casing when a raw upstream value leaks through. This module
maps BOTH vocabularies onto ONE canonical display label so the human-facing
``analysis.md`` shows a consistent category regardless of which route produced it.

Display-only: this deliberately does NOT rewrite the ``kernel_category`` /
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
    # Normalization (unify bypass "Normalization" + GEAK "LayerNorm").
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
    # Catch-all (unify bypass "Others" + GEAK "Other").
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
