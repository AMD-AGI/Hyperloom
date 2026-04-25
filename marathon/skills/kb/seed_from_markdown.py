#!/usr/bin/env python3
"""Seed the KB from the existing KNOWLEDGE-BASE.md.

Parses the markdown into structured JSONL entries. Run once:
    python3 seed_from_markdown.py
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from kb_schema import new_entry

KB_MD = SCRIPT_DIR.parent / "KNOWLEDGE-BASE.md"
KB_JSONL = SCRIPT_DIR / "entries.jsonl"
SOURCE = "seed-from-knowledge-base-md"


def seed():
    if KB_JSONL.exists() and KB_JSONL.stat().st_size > 0:
        print(f"KB already has entries ({KB_JSONL}). Use --force to overwrite.")
        if "--force" not in sys.argv:
            return

    entries = []

    # --- Summary takeaways ---
    for i, (lesson, tags) in enumerate([
        (
            "torch.compile is a prerequisite for large GEAK wins — with torch.compile, "
            "GEAK reached up to +14.72%; without it, gains were <=1.76%",
            ["torch.compile", "GEAK", "prerequisite"],
        ),
        (
            "GEAK E2E gain depends on concurrency (CONC) — sweet spot around CONC=4; "
            "at high concurrency, gains are diluted by the pipeline",
            ["GEAK", "concurrency", "CONC"],
        ),
        (
            "Server parameter tuning is often more effective than GEAK — "
            "e.g. Kimi vLLM +84%, DSR1 +13.9%",
            ["server-params", "tuning", "GEAK-comparison"],
        ),
        (
            "Already highly tuned models (gpt-oss) had little headroom — "
            "GPU utilization ~94.7%, 95%+ vendor kernels",
            ["vendor-kernels", "diminishing-returns"],
        ),
        (
            "Backend switches + scheduling modes outperform parameter sweeps — "
            "GLM-5-FP8: backend switches gave +16.2% combined vs <1% from any single parameter",
            ["backend-exploration", "scheduling", "super-linear-synergy"],
        ),
    ]):
        entries.append(new_entry(
            category="lesson",
            action=f"cross-model takeaway #{i+1}",
            lesson=lesson,
            tags=tags,
            confidence=0.95,
            source=SOURCE,
            context="Validated across 6 models, 2026-03-23",
        ))

    # --- DeepSeek-R1-0528 ---
    entries.append(new_entry(
        category="architecture_constraint",
        model="DeepSeek-R1-0528",
        gpu="MI355X",
        framework="sglang",
        action="torch.compile on MoE+MLA+FP8",
        lesson="torch.compile INCOMPATIBLE: MLA + FP8 causes CUDA graph capture failure. "
               "Must run without torch.compile.",
        tags=["torch.compile", "MLA", "FP8", "incompatible"],
        result={"status": "INCOMPATIBLE"},
        confidence=0.99,
        source=SOURCE,
        context="Validated 2026-03-21/22. Error: get_heuristic_kernel_mla q_type:fp8",
    ))
    entries.append(new_entry(
        category="kernel_optimization",
        model="DeepSeek-R1-0528",
        gpu="MI355X",
        framework="sglang",
        action="GEAK on aiter vendor kernels (gemm_a8w8_blockscale, fused_rms_fp8)",
        lesson="GEAK 0% E2E on vendor aiter kernels. Micro +44-127% but E2E -19.9% "
               "due to register pressure. Vendor kernels already AMD-engineer-optimized.",
        tags=["GEAK", "vendor-kernel", "aiter", "register-pressure"],
        result={"gain_pct": 0.0, "status": "REVERT"},
        confidence=0.99,
        source=SOURCE,
        context="Validated 2026-03-21. Controlled test. AST patching required.",
    ))
    entries.append(new_entry(
        category="server_params",
        model="DeepSeek-R1-0528",
        gpu="MI355X",
        framework="sglang",
        action="num-continuous-decode-steps 4→8",
        lesson="decode-steps 8 gives +13.9% on DSR1 (controlled pair: 2229→2539 tok/s). "
               "Highly model-dependent — gpt-oss TP=8 showed 0%.",
        tags=["decode-steps", "server-params"],
        result={"tput_per_gpu_before": 278.6, "tput_per_gpu_after": 317.4,
                "gain_pct": 13.9, "status": "KEEP"},
        confidence=0.95,
        source=SOURCE,
        context="Validated 2026-03-22. Controlled pair test, same session, cuda-graph-max-bs=64",
    ))

    # --- Qwen3-30B-A3B ---
    entries.append(new_entry(
        category="kernel_optimization",
        model="Qwen3-30B-A3B",
        gpu="MI355X",
        framework="sglang",
        action="GEAK RMSNorm single-pass via torch.compile + Inductor cache patching",
        lesson="Single-pass RMSNorm eliminates 50% memory reads. +14.72% at CONC=4. "
               "CONC-dependent: -5.9% at CONC=1 (L2 cache), ~0% at CONC=16 (GPU saturated). "
               "Average +2.20% across 9 configs.",
        tags=["GEAK", "RMSNorm", "torch.compile", "Inductor", "single-pass"],
        result={"tput_per_gpu_before": 596.51, "tput_per_gpu_after": 684.29,
                "gain_pct": 14.72, "status": "KEEP"},
        confidence=0.95,
        source=SOURCE,
        context="Validated 2026-03-23. CONC=4, ISL=1024, OSL=256, TP=1. "
                "Must patch STANDALONE files not graph modules.",
    ))
    entries.append(new_entry(
        category="framework_comparison",
        model="Qwen3-30B-A3B",
        gpu="MI355X",
        framework="",
        action="SGLang vs vLLM GEAK RMSNorm comparison",
        lesson="SGLang +14.7% from GEAK RMSNorm but vLLM ~0%. "
               "vLLM uses more C++ kernels so RMSNorm is smaller share. "
               "vLLM Inductor level=3 already compresses headroom.",
        tags=["SGLang", "vLLM", "GEAK", "RMSNorm", "framework-comparison"],
        result={},
        confidence=0.9,
        source=SOURCE,
    ))

    # --- Kimi-K2.5 ---
    entries.append(new_entry(
        category="architecture_constraint",
        model="Kimi-K2.5",
        gpu="MI355X",
        framework="sglang",
        action="Attention backend configuration for MoE+MLA TP=8",
        lesson="MUST use split backends: --decode-attention-backend triton "
               "--prefill-attention-backend aiter. Unified aiter fails (8 heads/partition < 16 min). "
               "MUST set SGLANG_ROCM_FUSED_DECODE_MLA=0. FP8 KV cache crashes.",
        tags=["MLA", "attention-backend", "split-backend", "TP8"],
        result={},
        confidence=0.99,
        source=SOURCE,
        context="Validated 2026-03-23. Config discovery from SGLang test suite saved 30+ min.",
    ))
    entries.append(new_entry(
        category="benchmark_methodology",
        model="Kimi-K2.5",
        gpu="MI355X",
        framework="sglang",
        action="Invalid +40.4% GEAK claim due to benchmark parameter drift",
        lesson="Baseline CONC=64, GEAK test CONC=128 (omitted --max-concurrency). "
               "Also decode-steps 4→8, mem-fraction 0.8→0.85. Fair A/B: only +0.81%. "
               "TPOT increased 172→234ms (smoking gun: batching not kernel speed).",
        tags=["benchmark-fairness", "pitfall", "concurrency-mismatch"],
        result={"gain_pct": 0.81, "status": "KEEP"},
        confidence=0.99,
        source=SOURCE,
        context="Controlled A/B with matched params. Previous +40.4% was INVALID.",
    ))
    entries.append(new_entry(
        category="server_params",
        model="Kimi-K2.5",
        gpu="MI355X",
        framework="vllm",
        action="gpu-memory-utilization 0.85→0.90 + max-num-seqs 256",
        lesson="vLLM server param tuning gave +84% (141→259 tok/s). "
               "Much more impactful than GEAK (+1.76%). "
               "P0 recommendation: switch to SGLang (324 > 264 tok/s).",
        tags=["server-params", "vLLM", "memory-utilization"],
        result={"tput_per_gpu_before": 35.25, "tput_per_gpu_after": 64.75,
                "gain_pct": 84.0, "status": "KEEP"},
        confidence=0.95,
        source=SOURCE,
        context="Validated 2026-03-23. TP=4, CONC=64",
    ))

    # --- gpt-oss-120b ---
    entries.append(new_entry(
        category="architecture_constraint",
        model="gpt-oss-120b",
        gpu="MI355X",
        framework="sglang",
        action="torch.compile + SWA incompatibility",
        lesson="SWA memory pool doesn't support CUDA graph capture with torch.compile. "
               "aiter attention backend also NOT SUPPORTED (requires triton/trtllm_mha). "
               "FP8 KV cache incompatible with SWA.",
        tags=["torch.compile", "SWA", "incompatible"],
        result={"status": "INCOMPATIBLE"},
        confidence=0.99,
        source=SOURCE,
    ))
    entries.append(new_entry(
        category="server_params",
        model="gpt-oss-120b",
        gpu="MI355X",
        framework="sglang",
        action="cuda-graph-max-bs 4→16",
        lesson="CUDA graph coverage expansion: +35% at CONC=4 (793→1073 tok/s). "
               "Most impactful single parameter when misconfigured. "
               "Rule: always set cuda-graph-max-bs >= max CONC.",
        tags=["cuda-graph", "server-params", "coverage"],
        result={"gain_pct": 35.0, "status": "KEEP"},
        confidence=0.99,
        source=SOURCE,
        context="Validated 2026-03-22. TP=1, ISL=1024, OSL=256, decode-steps=8 baseline",
    ))

    # --- GLM-5-FP8 ---
    entries.append(new_entry(
        category="backend_exploration",
        model="GLM-5-FP8",
        gpu="MI355X",
        framework="sglang",
        action="--nsa-decode-backend aiter (tilelang→aiter CK)",
        lesson="Switches NSA decode kernel from tilelang to aiter CK. "
               "+3.1% individually. Combined with mixed-chunk: +16.2% (super-linear synergy).",
        tags=["NSA", "aiter", "decode-backend", "CK"],
        result={"gain_pct": 3.1, "status": "KEEP"},
        confidence=0.95,
        source=SOURCE,
        context="Validated 2026-03-26. TP=4, CONC=64",
    ))
    entries.append(new_entry(
        category="backend_exploration",
        model="GLM-5-FP8",
        gpu="MI355X",
        framework="sglang",
        action="--enable-mixed-chunk scheduling",
        lesson="Overlaps prefill/decode in same forward batch. +2.9% individually. "
               "Combined with aiter NSA decode: +16.2%. "
               "More tokens per forward × faster per-token = compounding.",
        tags=["mixed-chunk", "scheduling", "prefill-decode-overlap"],
        result={"gain_pct": 2.9, "status": "KEEP"},
        confidence=0.95,
        source=SOURCE,
        context="Validated 2026-03-26. TP=4, CONC=64",
    ))
    entries.append(new_entry(
        category="backend_exploration",
        model="GLM-5-FP8",
        gpu="MI355X",
        framework="sglang",
        action="Combined: aiter NSA decode + mixed-chunk scheduling",
        lesson="Super-linear synergy: 3.1% + 2.9% individually → +16.2% combined. "
               "Mixed-chunk feeds more tokens; aiter processes each faster. "
               "Always test winners together — gains are NOT additive.",
        tags=["synergy", "combined-backends", "super-linear"],
        result={"tput_per_gpu_before": 351, "tput_per_gpu_after": 408,
                "gain_pct": 16.2, "status": "KEEP"},
        confidence=0.95,
        source=SOURCE,
        context="Validated 2026-03-26. TP=4, CONC=64. 1403→1630 total tok/s",
    ))
    entries.append(new_entry(
        category="target_comparison",
        model="GLM-5-FP8",
        gpu="MI355X",
        framework="sglang",
        action="MI355X vs NVIDIA B200 throughput comparison",
        lesson="B200: ~660 tok/s/GPU. MI355X optimized: ~408 tok/s/GPU (TP=4). "
               "Gap: ~38%. B200 uses TRT-LLM NSA, FlashInfer MoE, FP8 KV, DeepGEMM. "
               "FP8 KV cache fails on MI355X NSA path — blocks a key NVIDIA technique.",
        tags=["NVIDIA", "B200", "target", "gap-analysis"],
        result={"tput_per_gpu_before": 408, "target_tput_per_gpu": 660},
        confidence=0.9,
        source=SOURCE,
        context="From NVIDIA_B200_COMPARISON.md and AGGRESSIVE_OPTIMIZATION_PLAN.md",
    ))

    # --- Pitfalls ---
    for pitfall, tags_list in [
        ("Always include --max-concurrency $CONC in benchmark commands. "
         "Omitting it sends ALL prompts at once, inflating throughput.",
         ["benchmark-fairness", "concurrency"]),
        ("Save baseline server config to file and source it for all re-tests. "
         "Server param drift (decode-steps, mem-fraction) invalidates comparisons.",
         ["benchmark-fairness", "config-drift"]),
        ("GEAK agent may write output to input file path instead of output dir. "
         "Always check geak_get_outputs and download from correct path.",
         ["GEAK", "output-path"]),
        ("Always filter traces before kernel analysis. Raw traces are 349MB (97% python_function). "
         "Filtered traces are ~5MB.",
         ["profiling", "trace-filtering"]),
        ("MUST use Python AST for aiter source patching. Naive function-end detection "
         "deletes module-level variables (make_kernel_repr), causing NameError.",
         ["GEAK", "patching", "AST", "aiter"]),
    ]:
        entries.append(new_entry(
            category="pitfall",
            action="operational pitfall",
            lesson=pitfall,
            tags=tags_list,
            confidence=0.99,
            source=SOURCE,
        ))

    # Write
    with open(KB_JSONL, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Seeded {len(entries)} entries into {KB_JSONL}")


if __name__ == "__main__":
    seed()
