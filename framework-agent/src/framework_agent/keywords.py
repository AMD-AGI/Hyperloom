# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Smart keyword extraction from free-form gap descriptions. Pure-Python.

Returns a sorted, deduplicated list of: (1) words matching the curated
ROCm/LLM technical-term whitelist, (2) CamelCase identifiers (e.g.
AsyncLLMEngine), (3) fallback to the first few 3+ letter words.
"""

from __future__ import annotations

import re
from typing import Sequence


_TECHNICAL_TERMS = frozenset({
    "gemm", "moe", "attention", "allreduce", "fp8", "fp16", "bf16", "int8", "int4",
    "quantization", "quantize", "triton", "ck", "composable_kernel", "aiter",
    "cudagraph", "cuda_graph", "flashattention", "flash_attn", "flash-attn",
    "paged_attention", "pagedattention", "rope", "rotary", "kv_cache", "kvcache",
    "speculative", "spec_decode", "tensor_parallel", "tp", "pipeline_parallel", "pp",
    "fused", "fusion", "kernel", "hipify", "rocm", "hip", "nccl", "rccl",
    "allgather", "reducescatter", "reduce_scatter", "all_reduce",
    "prefill", "decode", "batching", "continuous_batching", "chunked_prefill",
    "vllm", "sglang", "atom", "trtllm", "tensorrt", "lora", "qlora", "awq", "gptq",
    "marlin", "w4a16", "w8a8", "smoothquant", "activation_order",
    "custom_all_reduce", "custom_ar", "radix", "scheduler",
    # atom-specific surfaces: keeps search relevance on the atom axis
    # (MTP / EP / aiter routing) instead of collapsing to generic "moe".
    "mtp", "ep", "moe_ep", "dp_attention", "dp", "kv_cache_dtype",
    "torch_profiler_dir",
    # GPU hardware codenames (lowercase; gap is lowercased before lookup).
    # Critical for relevance ranking so a gap on ``mi300x`` scopes to
    # AMD-validated PRs instead of e.g. an SM90 (Hopper) MoE kernel.
    # AMD CDNA accelerators (MI200/300/350 families + gfx IDs + uarch labels):
    "mi200", "mi210", "mi250", "mi250x",
    "mi300", "mi300a", "mi300x", "mi325x",
    "mi350x", "mi355x",
    "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950",
    "cdna", "cdna2", "cdna3", "cdna4",
    # NVIDIA datacenter accelerators (Ampere -> Blackwell + SM IDs + uarch labels):
    "a100", "h100", "h200", "b100", "b200",
    "sm80", "sm86", "sm89", "sm90", "sm100",
    "ampere", "hopper", "blackwell",
})


def extract_keywords(description: str) -> list[str]:
    """Extract a sorted, deduplicated keyword list from a gap description.

    Whitelist hits first, then CamelCase identifiers; if nothing matched,
    fall back to the first five 3+ letter words so callers get some signal.

    Args:
        description: The gap description text to mine.

    Returns:
        A sorted, deduplicated list of keywords.
    """
    tokens = set(re.findall(r"[a-z][a-z0-9_]+", description.lower()))
    keywords = tokens & _TECHNICAL_TERMS
    camel = re.findall(r"[A-Z][a-z]+(?:[A-Z][a-z]+)+", description)
    for word in camel:
        keywords.add(word.lower())
    if not keywords:
        words = re.findall(r"\b[a-z]{3,}\b", description.lower())
        keywords = set(words[:5])
    return sorted(keywords)


def score_title_against_keywords(title: str, keywords: Sequence[str]) -> int:
    """Count keywords overlapping the title's tokens, to rerank candidate PRs.

    Lowercase, snake_case-aware token split mirrors :func:`extract_keywords`.

    Args:
        title: The PR title to score.
        keywords: Keywords to match against the title.

    Returns:
        The overlap count; ``0`` for an empty title or keyword list.
    """
    if not keywords or not title:
        return 0
    title_tokens = set(re.findall(r"[a-z][a-z0-9_]+", title.lower()))
    kw_set = {k.lower() for k in keywords}
    return len(title_tokens & kw_set)


# Anti-correlation table: when ``gap_keyword`` (key) is in the gap, any
# anti-set token in a PR title is evidence the PR is on the wrong axis (model
# family / GPU vendor / precision regime) and is demoted. Activation is gated
# on the gap keyword's presence, so a gap without ``dense`` won't penalize a
# generic ``moe`` PR. Bug-driven (session f219629b, Qwen3-32B dense/bf16/mi300x
# picked an MoE/NVIDIA-H20 PR via positive-only overlap); axes below cover that.
_ANTI_KEYWORDS: dict[str, frozenset[str]] = {
    # Model architecture: dense Transformers vs Mixture-of-Experts.
    "dense":   frozenset({"moe", "mega_moe", "deepseek", "mixtral", "expert", "ep"}),
    # GPU vendor / uarch: AMD CDNA vs NVIDIA datacenter accelerators.
    "mi300x":  frozenset({"h100", "h200", "h20", "b100", "b200",
                          "sm80", "sm86", "sm89", "sm90", "sm100",
                          "ampere", "hopper", "blackwell", "nvidia"}),
    "mi300":   frozenset({"h100", "h200", "h20", "b100", "b200",
                          "sm80", "sm86", "sm89", "sm90", "sm100",
                          "ampere", "hopper", "blackwell", "nvidia"}),
    "mi250":   frozenset({"h100", "h200", "sm90", "nvidia", "hopper"}),
    "mi250x":  frozenset({"h100", "h200", "sm90", "nvidia", "hopper"}),
    "mi200":   frozenset({"h100", "h200", "sm90", "nvidia", "hopper"}),
    "mi350x":  frozenset({"h100", "h200", "b100", "b200",
                          "sm90", "sm100", "nvidia", "blackwell"}),
    "cdna":    frozenset({"sm80", "sm86", "sm89", "sm90", "sm100",
                          "ampere", "hopper", "blackwell", "nvidia"}),
    "cdna3":   frozenset({"sm90", "sm100", "hopper", "blackwell", "nvidia"}),
    "cdna4":   frozenset({"sm100", "blackwell", "nvidia"}),
    "rocm":    frozenset({"cuda", "cudnn", "cublas", "tensorrt"}),
    # Reverse direction: NVIDIA-gap PRs containing AMD-only signals.
    "h100":    frozenset({"mi200", "mi210", "mi250", "mi250x",
                          "mi300", "mi300x", "mi325x", "mi350x",
                          "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950",
                          "cdna", "cdna2", "cdna3", "cdna4", "rocm"}),
    "h200":    frozenset({"mi200", "mi210", "mi250", "mi250x",
                          "mi300", "mi300x", "mi325x", "mi350x",
                          "gfx940", "gfx942", "gfx950", "cdna3", "cdna4", "rocm"}),
    "sm90":    frozenset({"mi300", "mi300x", "mi350x",
                          "gfx940", "gfx942", "gfx950", "cdna3", "cdna4", "rocm"}),
    "hopper":  frozenset({"mi300", "mi300x", "mi350x", "cdna3", "cdna4", "rocm"}),
    # Quantization regime: full-precision bf16/fp16 vs low-bit / quant PRs.
    "bf16":    frozenset({"awq", "gptq", "fp8", "fp4", "int4", "int8",
                          "w4a16", "w8a8", "smoothquant", "marlin"}),
    "fp16":    frozenset({"fp8", "fp4", "int4", "awq", "gptq"}),
    "fp8":     frozenset({"bf16", "fp16"}),
}


def score_title_with_anti_signal(
    title: str,
    keywords: Sequence[str],
    *,
    anti_penalty: float = 2.0,
) -> float:
    """Rank a PR title by ``max(0, positive - anti_penalty * anti)``.

    ``positive`` = keyword tokens in the title; ``anti`` = title tokens in the
    anti-set of any active gap keyword (only :data:`_ANTI_KEYWORDS` entries
    activate). A default ``anti_penalty`` of 2.0 means one anti hit erases two
    positive hits, so a wrong-axis PR ranks below any single correct-axis hit.

    Args:
        title: The PR title to score.
        keywords: Active gap keywords driving positive and anti matches.
        anti_penalty: Weight applied per anti-signal hit.

    Returns:
        The clamped score (>= 0.0); callers can drop ``score == 0`` PRs.
    """
    if not keywords or not title:
        return 0.0
    title_tokens = set(re.findall(r"[a-z][a-z0-9_]+", title.lower()))
    kw_set = {k.lower() for k in keywords}
    positive = len(title_tokens & kw_set)
    anti = 0
    for k in kw_set:
        anti_set = _ANTI_KEYWORDS.get(k)
        if anti_set:
            anti += len(title_tokens & anti_set)
    return max(0.0, float(positive) - anti_penalty * float(anti))


__all__ = [
    "extract_keywords",
    "score_title_against_keywords",
    "score_title_with_anti_signal",
]
