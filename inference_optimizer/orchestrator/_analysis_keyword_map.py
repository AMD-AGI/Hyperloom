"""Roofline-v2 N22: analysis.md keyword → variant requirement mapping.

Bridges the gap left by N20-A (which only renders the catalogue and
hopes the LLM picks the right variants). Empirically the LLM under-
includes — today's Qwen3-30B-A3B N20c session is the worked example:
analysis.md said "host-bound, use GPU graphs or torch.compile",
LLM picked cuda_graph_max_bs_* and decode_steps_* but SKIPPED
torch_compile_on. This module gives the PolicyGate the means to
detect that miss and surface an advisory (non-blocking) so:

* coordinator can record the miss into shared_state.last_proposal_advice
  → the next-tick orchestration prompt sees it and the LLM corrects
  on the follow-up cheap-action propose
* the audit log retains a permanent trail of which variants the
  analysis flagged vs which the LLM actually proposed

Design properties

* **Advisory, not blocking**: N20-A keeps LLM agency (the LLM may have
  a legitimate reason to skip a keyword-implied variant, e.g. budget
  pressure or the keyword is mentioned in passing rather than as a
  P1 finding). The gate emits advice, doesn't deny.
* **Case-insensitive substring match**: simplest contract that
  handles "torch.compile" / "torch_compile" / "Torch Compile" /
  "torch.compile()" uniformly.
* **Available-variants intersection**: a keyword may imply 4 variants
  but only 2 of them live in the current framework's registered grid
  (e.g. SGLang vs vLLM); the gate only requires the ones available.
* **No false positive on absent keyword**: a keyword that doesn't
  appear in analysis.md generates no advice — the LLM is free.

Keyword map is curated, not auto-generated:

The simpler alternative ("scan analysis.md for any registered
variant's `note` substring") was rejected because notes are
internal scheduling tags (`tier1_attention`, `cuda_graph`, ...)
that almost never appear verbatim in TraceLens output. The
curated map below tracks the actual VOCABULARY TraceLens uses
when emitting analysis prose / system findings — synced against
real R1 / Qwen3 / DeepSeek-V4 reports rather than against the
executor's internal tag scheme.
"""
from __future__ import annotations

import re
from typing import Iterable


# Curated map: analysis.md substring (case-insensitive) -> required
# variant names. Mapping is intentionally inclusive on the variant side
# (a keyword like "cuda graph" maps to the WHOLE cuda_graph_* family)
# so the LLM has the full sweep available; the executor's grid
# de-duplication via `explore_search.tested` fingerprint will skip
# re-running variants already in current_best.
ANALYSIS_KEYWORD_TO_VARIANTS: dict[str, tuple[str, ...]] = {
    # ---------- Compile / capture ----------
    "torch.compile":          ("torch_compile_on",),
    "torch_compile":          ("torch_compile_on",),
    "torch compile":          ("torch_compile_on",),
    "torch-compile":          ("torch_compile_on",),
    "compile":                ("torch_compile_on",),
    "cuda graph":             ("cuda_graph_max_bs_8", "cuda_graph_max_bs_16",
                                "cuda_graph_max_bs_32", "cuda_graph_max_bs_64"),
    "cuda graphs":            ("cuda_graph_max_bs_8", "cuda_graph_max_bs_16",
                                "cuda_graph_max_bs_32", "cuda_graph_max_bs_64"),
    "cuda_graph":             ("cuda_graph_max_bs_8", "cuda_graph_max_bs_16",
                                "cuda_graph_max_bs_32", "cuda_graph_max_bs_64"),
    "gpu graph":              ("cuda_graph_max_bs_8", "cuda_graph_max_bs_16",
                                "cuda_graph_max_bs_32", "cuda_graph_max_bs_64"),
    "gpu graphs":             ("cuda_graph_max_bs_8", "cuda_graph_max_bs_16",
                                "cuda_graph_max_bs_32", "cuda_graph_max_bs_64"),
    "graph capture":          ("cuda_graph_max_bs_8", "cuda_graph_max_bs_16",
                                "cuda_graph_max_bs_32", "cuda_graph_max_bs_64"),
    "cudagraph":              ("cuda_graph_max_bs_8", "cuda_graph_max_bs_16",
                                "cuda_graph_max_bs_32", "cuda_graph_max_bs_64"),
    # ---------- KV cache / memory ----------
    "kv cache":               ("mem_fraction_0_85", "mem_fraction_0_90",
                                "mem_fraction_0_80"),
    "kv-cache":               ("mem_fraction_0_85", "mem_fraction_0_90",
                                "mem_fraction_0_80"),
    "kv_cache":               ("mem_fraction_0_85", "mem_fraction_0_90",
                                "mem_fraction_0_80"),
    "memory pressure":        ("mem_fraction_0_85", "mem_fraction_0_90",
                                "mem_fraction_0_80"),
    "memory fraction":        ("mem_fraction_0_85", "mem_fraction_0_90"),
    # ---------- Attention ----------
    "attention backend":      ("attn_aiter", "attn_triton", "decode_aiter"),
    "attention-backend":      ("attn_aiter", "attn_triton", "decode_aiter"),
    "flash attention":        ("attn_aiter", "attn_triton"),
    "flashattention":         ("attn_aiter", "attn_triton"),
    "flashattn":              ("attn_aiter", "attn_triton"),
    "aiter":                  ("attn_aiter", "decode_aiter", "moe_aiter"),
    # ---------- Collectives / comm ----------
    "allreduce":              ("custom_ar",),
    "all-reduce":             ("custom_ar",),
    "all_reduce":             ("custom_ar",),
    "all reduce":             ("custom_ar",),
    "collective communication": ("custom_ar",),
    "nccl":                   ("custom_ar",),
    "rccl":                   ("custom_ar",),
    # ---------- MoE ----------
    "moe":                    ("moe_aiter", "enable_fused_moe"),
    "mixture of experts":     ("moe_aiter", "enable_fused_moe"),
    "expert routing":         ("moe_aiter",),
    "expert dispatch":        ("moe_aiter", "enable_fused_moe"),
    "fused moe":              ("enable_fused_moe",),
    "moe backend":            ("moe_aiter",),
    # ---------- Decode / prefill / scheduling ----------
    "long decode":            ("decode_steps_16", "decode_steps_32",
                                "decode_steps_8"),
    "decode chain":           ("decode_steps_16", "decode_steps_32"),
    "decode bound":           ("decode_steps_16", "decode_steps_32", "decode_aiter"),
    "decode-bound":           ("decode_steps_16", "decode_steps_32", "decode_aiter"),
    "long prompt":            ("chunked_prefill_32k", "chunked_prefill_64k",
                                "chunked_prefill_128k", "max_prefill_tokens_32k",
                                "max_prefill_tokens_64k"),
    "long prompts":           ("chunked_prefill_32k", "chunked_prefill_64k",
                                "chunked_prefill_128k", "max_prefill_tokens_32k",
                                "max_prefill_tokens_64k"),
    "prefill":                ("chunked_prefill_32k", "chunked_prefill_64k",
                                "max_prefill_tokens_32k", "max_prefill_tokens_64k"),
    "chunked prefill":        ("chunked_prefill_32k", "chunked_prefill_64k",
                                "chunked_prefill_128k"),
    "queue depth":            ("max_running_requests_128", "max_running_requests_256",
                                "sched_lpm", "sched_dfs"),
    "scheduling":             ("sched_lpm", "sched_dfs",
                                "max_running_requests_128", "max_running_requests_256"),
    "schedule":               ("sched_lpm", "sched_dfs"),
    "concurrency":            ("max_running_requests_128", "max_running_requests_256"),
    "request batching":       ("max_running_requests_128", "max_running_requests_256"),
    "batching":               ("max_running_requests_128", "max_running_requests_256",
                                "cuda_graph_max_bs_64", "cuda_graph_max_bs_32"),
    "raise effective batch":  ("max_running_requests_256", "cuda_graph_max_bs_64",
                                "decode_steps_32"),
    "raise concurrency":      ("max_running_requests_256", "max_running_requests_128"),
    # ---------- Host bound / idle / sync ----------
    "host-side":              ("torch_compile_on", "cuda_graph_max_bs_64",
                                "cuda_graph_max_bs_32", "decode_steps_32",
                                "decode_steps_16"),
    "host side":              ("torch_compile_on", "cuda_graph_max_bs_64",
                                "cuda_graph_max_bs_32", "decode_steps_32",
                                "decode_steps_16"),
    "host-bound":             ("torch_compile_on", "cuda_graph_max_bs_64",
                                "cuda_graph_max_bs_32", "decode_steps_32"),
    "host bound":             ("torch_compile_on", "cuda_graph_max_bs_64",
                                "cuda_graph_max_bs_32", "decode_steps_32"),
    "host overhead":          ("torch_compile_on", "cuda_graph_max_bs_64",
                                "decode_steps_32"),
    "host stall":             ("torch_compile_on", "cuda_graph_max_bs_64",
                                "decode_steps_32"),
    "blocking sync":          ("decode_steps_16", "decode_steps_32"),
    "cpu sync":               ("decode_steps_16", "decode_steps_32",
                                "cuda_graph_max_bs_64"),
    "cpu overhead":           ("torch_compile_on", "cuda_graph_max_bs_64",
                                "decode_steps_32"),
    "per-launch":             ("cuda_graph_max_bs_64", "cuda_graph_max_bs_32",
                                "torch_compile_on"),
    "per launch":             ("cuda_graph_max_bs_64", "cuda_graph_max_bs_32",
                                "torch_compile_on"),
    "amortize":               ("cuda_graph_max_bs_64", "torch_compile_on"),
    "gpu idle":               ("torch_compile_on", "cuda_graph_max_bs_64",
                                "decode_steps_32"),
    "gpu underutil":          ("torch_compile_on", "cuda_graph_max_bs_64",
                                "decode_steps_32"),  # matches "underutilized"/"underutilization"
    "underutilization":       ("torch_compile_on", "cuda_graph_max_bs_64",
                                "decode_steps_32"),
    # ---------- Fusion / overlap ----------
    "kernel fusion":          ("enable_fused_moe", "enable_mixed"),
    "fusion":                 ("enable_fused_moe", "enable_mixed"),
    "overlap":                ("sglang_multi_stream_overlap",),
    "multi-stream":           ("sglang_multi_stream_overlap",),
    "multi stream":           ("sglang_multi_stream_overlap",),
    # ---------- Radix cache ----------
    "radix cache":            ("disable_radix_cache",),
    "radix-cache":            ("disable_radix_cache",),
    "prefix cache":           ("disable_radix_cache",),
}


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace; preserves substring semantics."""
    return re.sub(r"\s+", " ", text.lower())


def extract_required_variants_from_analysis(
    analysis_md_text: str,
    available_variants: Iterable[str],
) -> tuple[list[str], list[tuple[str, tuple[str, ...]]]]:
    """Scan analysis.md for keyword matches and return:

    1. A sorted list of variant names that the analysis implies, narrowed
       to ``available_variants`` (so SGLang-only variants don't surface
       on a vLLM run and vice versa).
    2. A list of ``(keyword, variants_tuple)`` for every match found,
       useful for the advice message showing the LLM exactly which
       keyword triggered which variant.

    The text is normalized (lowercase + whitespace collapsed) before
    matching so `"torch.compile()"`, `"Torch Compile"`, multi-line
    `"torch\\ncompile"` all match the `"torch.compile"` / `"torch
    compile"` keys.

    Empty / None text -> ([], []).
    """
    if not analysis_md_text:
        return [], []
    norm = _normalize(analysis_md_text)
    available = set(available_variants or [])
    required: set[str] = set()
    matches: list[tuple[str, tuple[str, ...]]] = []
    seen_keys: set[str] = set()
    for key, variants in ANALYSIS_KEYWORD_TO_VARIANTS.items():
        if key in seen_keys:
            continue
        if key in norm:
            seen_keys.add(key)
            # Narrow to available (per-framework) variants
            narrowed = tuple(v for v in variants if v in available)
            if not narrowed:
                continue
            required.update(narrowed)
            matches.append((key, narrowed))
    return sorted(required), matches


def format_missing_variants_advice(
    proposed_variants: list[str],
    required_variants: list[str],
    matches: list[tuple[str, tuple[str, ...]]],
    *,
    action_name: str,
) -> str | None:
    """Build the operator-facing advice string for a propose with
    keyword-implied variants that aren't in the LLM's variants list.

    Returns None when there is nothing to advise (all required variants
    are already included, or the analysis didn't mention any tracked
    keyword).
    """
    if not required_variants:
        return None
    proposed_set = set(proposed_variants or [])
    missing = [v for v in required_variants if v not in proposed_set]
    if not missing:
        return None
    # Build a per-keyword breakdown so the LLM sees WHY each missing
    # variant is implied: "torch.compile -> [torch_compile_on]"
    trigger_lines = []
    for key, variants in matches:
        triggered_missing = [v for v in variants if v in missing]
        if triggered_missing:
            trigger_lines.append(
                f"  - analysis.md keyword {key!r} -> "
                f"missing variant(s): {sorted(triggered_missing)}"
            )
    body = "\n".join(trigger_lines) if trigger_lines else (
        f"  - missing variants: {sorted(missing)}"
    )
    return (
        f"[N22 advisory] action={action_name!r} accepted, but the analysis.md "
        f"snapshot the proposal is based on flags variants you didn't include "
        f"in `params.variants`. The proposal is NOT denied (LLM keeps "
        f"agency), but if the first round doesn't find sufficient gain, "
        f"consider extending the variants list with these analysis-implied "
        f"candidates on the next propose:\n"
        f"{body}\n"
        f"(Override: the keyword map lives in "
        f"inference_optimizer/orchestrator/_analysis_keyword_map.py — "
        f"trim it if false positives become common for a given workload.)"
    )


__all__ = [
    "ANALYSIS_KEYWORD_TO_VARIANTS",
    "extract_required_variants_from_analysis",
    "format_missing_variants_advice",
]
