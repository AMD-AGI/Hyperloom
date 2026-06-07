# Copyright Advanced Micro Devices, Inc. All rights reserved.

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
    "vllm", "sglang", "atom", "trtllm", "tensorrt", "lora", "qlora", "awq", "gptq",
    "marlin", "w4a16", "w8a8", "smoothquant", "activation_order",
    "custom_all_reduce", "custom_ar", "radix", "scheduler",
    # atom-specific surfaces: PR titles in the ROCm/ATOM repo tend to
    # mention these together; listing them here keeps the
    # primus_cortex / github search relevance on the atom-shaped axis
    # (MTP / EP / aiter routing) instead of collapsing to generic
    # "moe" matches that surface unrelated PRs.
    "mtp", "ep", "moe_ep", "dp_attention", "dp", "kv_cache_dtype",
    "torch_profiler_dir",
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


# Anti-correlation table: when ``gap_keyword`` (key) is present in the gap,
# any token in its anti-set appearing in a PR title is treated as strong
# evidence that the PR is on the *wrong* axis (different model family,
# different GPU vendor, different precision regime) and should be demoted.
#
# Activation is gated on the gap keyword being present, so PRs are only
# penalized when the gap explicitly carries the orthogonal signal — a gap
# without ``dense`` will not penalize a generic ``moe`` PR.
#
# Bug-driven: session f219629b on Qwen-Qwen3-32B (dense, bf16, mi300x)
# had fa pick PR:25769 ("Enable MegaMoE for NextN with TP attn A2A
# scatter padding") because positive-only overlap matched ``throughput``
# while the PR's MoE / NVIDIA-H20 signals were ignored. Anti pairs below
# cover the three orthogonal axes that surfaced in that case.
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
    """Rank a PR title by positive overlap minus anti-correlated penalty.

    Computes ``max(0, positive - anti_penalty * anti)`` where:

    * ``positive`` is the number of ``keywords`` tokens that also appear
      in ``title`` (same as :func:`score_title_against_keywords`);
    * ``anti`` is the number of title tokens that appear in the anti-set
      of any active gap keyword (only keywords present in
      :data:`_ANTI_KEYWORDS` activate the lookup).

    The default ``anti_penalty`` of 2.0 means a single anti hit erases
    two positive hits. Empirically tuned so a PR whose title squarely
    targets the opposite axis (e.g. ``MegaMoE`` when the gap calls for
    ``dense``) ranks below any PR that scores even one positive hit on
    the correct axis. Returns a float so partial-penalty experiments
    stay possible; ``sorted(..., key=score, reverse=True)`` is stable
    under float keys, so ties preserve upstream order.

    The result is clamped to ``0.0`` rather than going negative so the
    ordering remains intuitive ("zero relevance" is a floor); rerank
    callers that want to *drop* anti-heavy PRs can post-filter on
    ``score == 0`` themselves.
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
