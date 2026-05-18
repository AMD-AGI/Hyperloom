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


__all__ = ["extract_keywords"]
