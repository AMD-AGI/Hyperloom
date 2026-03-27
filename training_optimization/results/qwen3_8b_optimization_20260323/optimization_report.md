# Qwen3 8B Optimization Report — torchtune on MI355X 8-GPU

**Date:** 2026-03-23
**Platform:** 8× AMD Instinct MI355X (gfx950, CDNA4, 288 GiB HBM per GPU)
**ROCm:** 7.2.26015, PyTorch 2.10.0a0+git449b176
**Framework:** torchtune (dev build, PyTorch-native post-training library)
**Model:** Qwen3 8B Instruct (36 layers, 32 heads, 8 KV heads, 4096 hidden, bf16)
**Recipe:** `full_finetune_distributed` (FSDP)
**Dataset:** alpaca_cleaned (51,760 samples)
**Time budget:** ~1 hour wall clock

---

## Executive Summary

| Metric | Baseline | Optimized | Delta |
|---|---|---|---|
| **tokens/sec/gpu** | 862.6 | 18,859.9 | **+18,000** |
| **Speedup** | — | — | **+2,086%** (~21.9×) |
| Total attempts | — | 19 | |
| Kept | — | 8 | |
| Crashes | — | 4 | |

**Baseline:** Stock torchtune `qwen3/8B_full` config — batch_size=2, gradient_accumulation_steps=8, unpacked variable-length sequences, activation checkpointing enabled, compile disabled, FSDP FULL_SHARD.

**Optimized config (attempt 14, best):**
```yaml
compile: True
dataset.packed: True
tokenizer.max_seq_len: 4096
fsdp_reshard_after_forward: False   # SHARD_GRAD_OP
batch_size: 8
gradient_accumulation_steps: 1
enable_activation_checkpointing: False
```

---

## What Worked (cumulative progression)

| # | tok/s/gpu | vs Baseline | vs Previous | Description |
|---|-----------|-------------|-------------|-------------|
| 0 | 862.6 | — | — | **Baseline** (stock config) |
| 1 | 1,396.8 | +61.9% | +61.9% | `compile=True` — fuses elementwise ops, reduces kernel launch overhead |
| 2 | 7,546.1 | +774.8% | +440.3% | + `packed=True` (seq_len=2048) — eliminates padding waste |
| 3 | 9,255.4 | +972.9% | +22.6% | + `fsdp_reshard_after_forward=False` — SHARD_GRAD_OP, halves all-gather ops |
| 5 | 11,370.0 | +1,218.1% | +22.8% | + `batch_size=4, GA=4` — larger micro-batch amortizes FSDP communication |
| 6 | 13,206.5 | +1,431.0% | +16.2% | + `batch_size=8, GA=2` — same trend |
| 7 | 14,239.6 | +1,550.7% | +7.8% | + `batch_size=16, GA=1` — maximum batch/minimum GA |
| 10 | 17,142.8 | +1,887.3% | +20.4% | + `enable_activation_checkpointing=False` — eliminates recomputation |
| **14** | **18,859.9** | **+2,086.3%** | **+10.0%** | + `seq_len=4096` — longer sequences amortize per-token overhead |

### What Didn't Work

| # | Result | Description |
|---|--------|-------------|
| 4 | -0.5% | `NCCL_ALGO=Ring` — no improvement on intra-node 8-GPU topology |
| 9 | crash | `optimizer_in_bwd` — incompatible with compiled optimizer step |
| 11 | OOM | `batch_size=32` without activation checkpointing |
| 12 | +0.06% | `CUDA_DEVICE_MAX_CONNECTIONS=2` — negligible |
| 13 | OOM | `seq_len=4096, batch_size=16` without AC |
| 15 | -15.1% | `seq_len=4096, batch_size=16` with AC — AC overhead dominates |
| 16 | OOM | `seq_len=4096, batch_size=12` without AC |
| 17 | -2.8% | `seq_len=4096, batch_size=10` — slightly worse than batch_size=8 |
| 18 | -0.6% | `optimizer_in_bwd` with selective compile — slight regression |
| 19 | -1.7% | `seq_len=8192, batch_size=4` — smaller batch offsets longer-sequence gains |

---

## Profile Analysis

### Baseline Kernel Breakdown (rank 0, 2 profiled steps)

| Category | % GPU Time | Key Kernels |
|----------|-----------|-------------|
| **NCCL communication** | **45.3%** | `ncclDevKernel_Generic_1` (all-gather + reduce-scatter for FSDP) |
| **Elementwise ops** | ~20% | RMSNorm, activations, type conversions (`elementwise_kernel_manual_unroll`, `vectorized_elementwise_kernel`) |
| **GEMM** | ~8% | hipBLASLt kernels (`Cijk_Ailk_Bjlk_*`, `Cijk_Alik_Bljk_*`) — vendor-optimal |
| **bf16↔fp32 copies** | ~4% | Type conversion overhead |
| **Memory ops** | ~5% | `split_with_sizes_copy`, `chunk_cat`, `multi_tensor_apply` |

The baseline was **communication-dominated**: an 8B model on 8× MI355X GPUs with only batch_size=2 and gradient_accumulation_steps=8 meant each micro-forward/backward processed very little compute relative to FSDP all-gather/reduce-scatter overhead.

### Optimization Strategy Rationale

1. **`torch.compile`** (→ +62%): Fused the ~20% elementwise ops into fewer, larger kernels; reduced Python overhead and kernel launch latency.
2. **`packed=True`** (→ +440%): The unpacked baseline wasted ~85% of sequence positions on padding (avg alpaca sequence ~200 tokens vs batch max). Packing to 2048-length sequences eliminated this waste entirely.
3. **`fsdp_reshard_after_forward=False`** (→ +23%): Switched from FULL_SHARD to SHARD_GRAD_OP, keeping parameters in memory between forward and backward to eliminate all-gather during backward.
4. **Increased batch_size / decreased GA** (→ +54% cumulative): Larger micro-batches mean more compute per FSDP communication round, improving the compute-to-communication ratio.
5. **Disabled activation checkpointing** (→ +20%): With 288 GiB HBM per GPU and SHARD_GRAD_OP already keeping full parameters, we had ample memory to hold activations. This eliminated the recomputation cost.
6. **`seq_len=4096`** (→ +10%): Longer sequences increase compute density (quadratic attention), better amortizing per-token and communication overhead.

---

## Memory Analysis

| Config | Peak GPU Memory |
|--------|----------------|
| Baseline (bs=2, GA=8, AC=True, FULL_SHARD) | ~14.3 GiB / 288 GiB |
| Optimized (bs=8, GA=1, AC=False, SHARD_GRAD_OP, seq=4096) | ~283 GiB / 288 GiB |

The baseline was severely under-utilizing the available 288 GiB HBM per GPU. The optimized config uses nearly all available memory, which is the correct operating point for maximum throughput.

---

## All Attempts

| # | tok/s/gpu | Speedup | Status | Description |
|---|-----------|---------|--------|-------------|
| 0 | 862.6 | 0.0% | baseline | Stock torchtune qwen3/8B_full config |
| 1 | 1,396.8 | +61.9% | keep | compile=True |
| 2 | 7,546.1 | +774.8% | keep | compile + packed=True (seq=2048) |
| 3 | 9,255.4 | +972.9% | keep | + fsdp_reshard_after_forward=False |
| 4 | 9,211.5 | +967.8% | discard | + NCCL_ALGO=Ring |
| 5 | 11,370.0 | +1,218.1% | keep | + batch_size=4 GA=4 |
| 6 | 13,206.5 | +1,431.0% | keep | + batch_size=8 GA=2 |
| 7 | 14,239.6 | +1,550.7% | keep | + batch_size=16 GA=1 |
| 8 | 14,751.5 | +1,610.1% | keep | + batch_size=32 GA=1 |
| 9 | -1 | — | crash | optimizer_in_bwd (conflict with compiled opt step) |
| 10 | 17,142.8 | +1,887.3% | keep | + no activation checkpointing (bs=16) |
| 11 | -1 | — | crash | batch_size=32 noAC (OOM) |
| 12 | 17,153.0 | +1,888.5% | discard | + CUDA_DEVICE_MAX_CONNECTIONS=2 |
| 13 | -1 | — | crash | seq=4096 bs=16 noAC (OOM) |
| 14 | **18,859.9** | **+2,086.3%** | **keep** | **BEST: seq=4096 bs=8 noAC** |
| 15 | 16,010.6 | +1,756.0% | discard | seq=4096 bs=16 AC=True |
| 16 | -1 | — | crash | seq=4096 bs=12 noAC (OOM) |
| 17 | 18,324.6 | +2,024.3% | discard | seq=4096 bs=10 noAC |
| 18 | 18,755.8 | +2,074.3% | discard | optimizer_in_bwd + selective compile |
| 19 | 18,545.1 | +2,049.8% | discard | seq=8192 bs=4 noAC |

---

## Methodology

6 rounds of optimization:
1. **torch.compile** — single biggest per-change win (+62%)
2. **Data efficiency** — packed sequences to eliminate padding waste (+440%)
3. **FSDP strategy** — SHARD_GRAD_OP to reduce communication (+23%)
4. **Batch size / GA tuning** — maximize compute per communication round (+54% cumulative)
5. **Memory management** — disable AC when memory permits (+20%)
6. **Sequence length tuning** — longer sequences for better compute density (+10%)

---

## Artifacts

| Artifact | Location |
|----------|----------|
| Results TSV | `/shared_nfs/nehaprakriya/results/qwen3_8b_optimization_20260323/results.tsv` |
| Baseline log | `/shared_nfs/nehaprakriya/results/qwen3_8b_optimization_20260323/baseline.log` |
| Baseline trace | `/shared_nfs/nehaprakriya/results/qwen3_8b_optimization_20260323/traces/iteration_8/` |
| Baseline config | `/shared_nfs/nehaprakriya/results/qwen3_8b_optimization_20260323/configs/baseline.yaml` |
| All attempt logs | `/shared_nfs/nehaprakriya/results/qwen3_8b_optimization_20260323/logs/` |
| This report | `/shared_nfs/nehaprakriya/results/qwen3_8b_optimization_20260323/optimization_report.md` |

---

## Reproducibility

**Baseline:**
```bash
/opt/venv/bin/tune run --nnodes 1 --nproc_per_node 8 full_finetune_distributed \
    --config /shared_nfs/nehaprakriya/results/qwen3_8b_optimization_20260323/configs/baseline.yaml
```

**Optimized:**
```bash
/opt/venv/bin/tune run --nnodes 1 --nproc_per_node 8 full_finetune_distributed \
    --config /shared_nfs/nehaprakriya/results/qwen3_8b_optimization_20260323/configs/baseline.yaml \
    compile=True \
    dataset.packed=True \
    tokenizer.max_seq_len=4096 \
    fsdp_reshard_after_forward=False \
    batch_size=8 \
    gradient_accumulation_steps=1 \
    enable_activation_checkpointing=False
```

---

## Key Takeaways for Cross-Framework Generalizability

1. **Profile first, optimize what matters:** The baseline was 45% NCCL communication — knowing this drove every subsequent decision (reduce communication rounds, increase compute per round).
2. **Data efficiency is often the biggest lever:** Packed sequences turned padding waste into useful compute — a purely algorithmic improvement with zero hardware cost.
3. **torch.compile is a universal win on modern PyTorch:** 62% from a single flag change, applicable to any PyTorch model.
4. **Match memory utilization to hardware:** The baseline used 14 GiB of 288 GiB available — a 20× underutilization that directly limited throughput.
5. **FSDP sharding strategy matters more than NCCL tuning:** Switching FULL_SHARD → SHARD_GRAD_OP gave 23%, while NCCL env vars gave <1%.
6. **Optimization is framework-specific but methodology is universal:** The same THINK → TRY → MEASURE → DECIDE loop that worked on Primus/Megatron (gpt-oss +4.7%) works on torchtune, with different levers producing even larger gains.

---

*Generated by workload-optimization agent on Sun Mar 23 17:45:00 UTC 2026*
