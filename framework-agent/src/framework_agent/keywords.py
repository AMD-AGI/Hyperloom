"""Smart keyword extraction from free-form gap descriptions.

Adapted from Arbor's `_TECHNICAL_TERMS` + `_extract_keywords` helpers
(TBO/src/arbor/github_search.py). Pure-Python, zero deps.

Returns a sorted, deduplicated list of:
  1. words matching the curated ROCm/LLM technical-term whitelist,
  2. CamelCase identifiers (e.g. AsyncLLMEngine, RadixCache),
  3. fallback to the first few 3+ letter words when nothing else matched.
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
    "vllm", "sglang", "trtllm", "tensorrt", "lora", "qlora", "awq", "gptq",
    "marlin", "w4a16", "w8a8", "smoothquant", "activation_order",
    "custom_all_reduce", "custom_ar", "radix", "scheduler",
    # GPU hardware codenames. Critical for relevance ranking: a gap such as
    # "improve sglang bf16 throughput on mi300x" must keep ``mi300x`` so the
    # downstream primus_cortex / github search can scope to AMD-validated PRs
    # instead of e.g. a freshly-merged SM90 (NVIDIA Hopper) MoE kernel. Listed
    # in lowercase since extract_keywords() lowercases the gap before lookup.
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

    Whitelist hits (60+ ROCm/LLM terms) come first; CamelCase identifiers
    such as ``AsyncLLMEngine`` are added next; if nothing else matched
    we fall back to the first five 3+ letter words so callers still get
    *some* signal instead of an empty list.
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
    """Count how many of the given keywords overlap with the title's tokens.

    Used by the primus_cortex dispatcher to rerank candidate PRs returned
    by service-side search so the most gap-relevant titles come first.
    Lowercase, snake_case-aware token split mirrors :func:`extract_keywords`
    so a keyword like ``fp8`` matches both ``fp8`` and ``fp8_moe``.

    Returns 0 for an empty title or an empty keywords list. Pure-Python,
    no regex compile cache needed because titles are short.
    """
    if not keywords or not title:
        return 0
    title_tokens = set(re.findall(r"[a-z][a-z0-9_]+", title.lower()))
    kw_set = {k.lower() for k in keywords}
    return len(title_tokens & kw_set)


__all__ = ["extract_keywords", "score_title_against_keywords"]
