#!/usr/bin/env python3
"""Seed the training optimization KB from validated lessons in the SKILL.md.

Run once to populate entries.jsonl with known lessons from prior optimization runs.
Will not overwrite existing entries unless --force is passed.
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from kb_schema import new_entry

KB_FILE = SCRIPT_DIR / "entries.jsonl"

SEED_ENTRIES = [
    {
        "category": "fusion_flags",
        "model": "GPT-OSS-20B",
        "action": "moe_permute_fusion=true",
        "lesson": "Largest single win (+1.19%) for MoE models. Replaces generic PyTorch MoE dispatch kernels (CatArrayBatchedCopy, scatter_gather, indexFuncLargeIndex) with fused Triton permute kernels.",
        "tags": ["moe", "permute", "fusion", "triton", "gpt-oss"],
        "result": {"gain_pct": 1.19, "status": "KEEP"},
        "context": "Validated on 8xMI355X, EP=8, BF16, 13265->13107 ms/iter. Same GBS=512.",
    },
    {
        "category": "fusion_flags",
        "model": "GPT-OSS-20B",
        "action": "gradient_accumulation_fusion=true",
        "lesson": "Reliable +0.46%. Fuses wgrad GEMM with optimizer accumulation. No downside observed.",
        "tags": ["grad_accum", "fusion", "gpt-oss"],
        "result": {"gain_pct": 0.46, "status": "KEEP"},
        "context": "Validated on 8xMI355X, EP=8, BF16. Same GBS=512.",
    },
    {
        "category": "fusion_flags",
        "model": "GPT-OSS-20B",
        "action": "moe_use_fused_router_with_aux_score=true",
        "lesson": "Neutral perf but reduces kernel count. Free cleanup, no regression.",
        "tags": ["moe", "router", "fusion", "gpt-oss"],
        "result": {"gain_pct": 0.0, "status": "KEEP"},
        "context": "Validated on 8xMI355X. Reduces kernel launch overhead slightly.",
    },
    {
        "category": "pitfall",
        "model": "GPT-OSS-20B",
        "action": "use_turbo_grouped_mlp=true with wide FFN dims",
        "lesson": "Fused SwiGLU kernel has suboptimal tile config for non-square shapes (e.g., ffn_hidden_size>8192). Causes regression.",
        "tags": ["moe", "turbo", "regression", "gpt-oss"],
        "result": {"gain_pct": -1.0, "status": "DISCARD"},
        "context": "GPT-OSS 20B has ffn_hidden_size=10944. Regressed ~1%.",
    },
    {
        "category": "pitfall",
        "model": "",
        "action": "sink_sliding_window=N with Triton attention backend",
        "lesson": "Triton attention backend does NOT support sliding window. Crashes with ValueError.",
        "tags": ["attention", "sliding_window", "crash"],
        "result": {"status": "CRASH"},
        "context": "Any model using use_sink_attention=true with sink_sliding_window>0.",
    },
    {
        "category": "pitfall",
        "model": "",
        "action": "PYTORCH_TUNABLEOP_ENABLED=1 during benchmarking",
        "lesson": "GEMM autotuning takes >30 min on MoE models with many unique shapes. Only useful as offline pre-tuning pass.",
        "tags": ["tunableop", "gemm", "slow"],
        "result": {"status": "DISCARD"},
        "context": "GPT-OSS 20B has 32 experts × multiple GEMM shapes.",
    },
    {
        "category": "attention_backend",
        "model": "GPT-OSS-20B",
        "action": "use_sink_attention=true (Triton) vs aiter",
        "lesson": "For GPT-OSS 20B (64 Q-heads, 8 KV-heads, head_dim=128, seq=4096), Triton attention was faster than aiter v3. Always measure — don't assume vendor is faster.",
        "tags": ["attention", "triton", "aiter", "gpt-oss"],
        "result": {"status": "KEEP"},
        "context": "PrimusTurbo with use_sink_attention=true on MI355X.",
    },
    {
        "category": "pitfall",
        "model": "",
        "action": "aiter deterministic=True on gfx950",
        "lesson": "aiter flash_attn_func defaults to deterministic=True. On gfx950 with seqlen>256, this DISABLES fmha_v3_bwd and falls back to mha_bwd (2.7x slower backward). PrimusTurbo defaults to deterministic=False.",
        "tags": ["aiter", "deterministic", "backward", "critical"],
        "result": {"status": "DISCARD"},
        "context": "Affects all models on MI355X with aiter attention.",
    },
    {
        "category": "architecture_constraint",
        "model": "",
        "action": "hipBLASLt GEMMs (Cijk_* kernels)",
        "lesson": "Vendor BLAS dominates GPU time (60-70%). Default kernel selection is near-optimal. Cannot be improved through code changes. Gains come from reducing everything else.",
        "tags": ["gemm", "hipblaslt", "vendor"],
        "result": {},
        "context": "Applies to all models on MI355X/MI300X.",
    },
    {
        "category": "benchmark_methodology",
        "model": "",
        "action": "GBS must remain constant across all optimization attempts",
        "lesson": "Global batch size is IMMUTABLE. Changing GBS changes work per iteration, making ms/iter comparisons meaningless. Always verify GBS in training log before KEEP.",
        "tags": ["gbs", "methodology", "critical"],
        "result": {},
        "context": "Core constraint for all training optimization.",
    },
    {
        "category": "benchmark_methodology",
        "model": "",
        "action": "Use iterations 6-10 for timing, skip 1-5 warmup",
        "lesson": "First 5 iterations include JIT compilation, NCCL init, and cache warmup. Iterations 6-10 are representative of steady-state training.",
        "tags": ["warmup", "methodology"],
        "result": {},
        "context": "Standard protocol for Primus/Megatron training benchmarks.",
    },
    {
        "category": "pitfall",
        "model": "",
        "action": "Port conflicts after killing training runs",
        "lesson": "Master port (default 29500) may stay bound after kill. Increment port for next run (29501, 29502, ...). Always pkill -9 -f 'primus/cli/main.py' before retrying.",
        "tags": ["port", "torchrun", "crash"],
        "result": {},
        "context": "Common issue on multi-GPU nodes.",
    },
    {
        "category": "pitfall",
        "model": "",
        "action": "MockGPTDataset C++ helpers not compiled",
        "lesson": "Mock data generation fails with RuntimeError about building helpers. Fix: make -C /workspace/Primus/third_party/Megatron-LM/megatron/core/datasets",
        "tags": ["mock_data", "compilation", "setup"],
        "result": {},
        "context": "Required one-time setup step for Primus/Megatron.",
    },
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing entries")
    args = parser.parse_args()

    if KB_FILE.exists() and not args.force:
        existing = KB_FILE.read_text().strip()
        if existing:
            print(f"KB already has entries ({len(existing.splitlines())} lines). Use --force to overwrite.")
            return

    entries = []
    for seed in SEED_ENTRIES:
        entry = new_entry(**seed)
        entries.append(json.dumps(entry, ensure_ascii=False))

    KB_FILE.write_text("\n".join(entries) + "\n")
    print(f"Seeded {len(entries)} entries into {KB_FILE}")


if __name__ == "__main__":
    main()
