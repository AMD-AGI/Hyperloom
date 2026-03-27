# GLM-5-FP8 Optimization Report
## 8x AMD MI355X (gfx950) — SGLang v0.5.9-dev

**Date**: March 26, 2026  
**Model**: `zai-org/GLM-5-FP8` (MoE+MLA+NSA, 78 layers, 256+1 experts, FP8 blockscale)  
**Hardware**: 8x AMD MI355X GPUs, 288 GB HBM3e each  
**Benchmark**: ISL=1024, OSL=1024, random_range_ratio=0.8

---

## Executive Summary

Starting from a baseline of **1403 tok/s** (TP=4, single instance), we achieved:

| Configuration | Total tok/s | Per-GPU tok/s | Improvement | TPOT |
|---|---|---|---|---|
| Baseline (TP=4) | 1,403 | 351 | — | 84.4 ms |
| **Optimized (TP=4)** | **1,630** | **408** | **+16.2%** | **72.8 ms** |
| Optimized + higher conc (TP=4, conc=128) | 2,206 | 551 | +57.2% | 107.4 ms |
| Previous DP=2/TP=4 (unoptimized) | 2,794 | 349 | +99.1% | 84.9 ms |
| **Projected DP=2/TP=4 (optimized)** | **~3,244** | **~406** | **~+131%** | **~73 ms** |

The per-GPU throughput improved from 351 → 408 tok/s (**+16.2%**), which scales with DP.
Projected DP=2/TP=4 with optimizations: **~3,244 tok/s** (vs 2,794 unoptimized = **+16% over prior best**).

---

## Winning Optimizations

### 1. NSA Decode Backend: aiter (+3.1% standalone → critical in combo)
```
--nsa-decode-backend aiter
```
Switched Native Sparse Attention decode from `tilelang` to `aiter` (AMD's Composable Kernel library). The aiter implementation is better optimized for MI355X decode-time attention patterns.

### 2. Mixed Chunk Scheduling (+2.9% standalone → critical in combo)
```
--enable-mixed-chunk
```
Enables prefill/decode batches to share a single forward pass. Instead of strictly alternating between prefill and decode phases, the scheduler can mix decode tokens into prefill batches, reducing GPU idle time between phases.

### 3. Synergistic Combination (+16.2% combined)
The two optimizations interact synergistically — the combined gain (+16.2%) far exceeds the sum of individual gains (+6.0%). Mixed-chunk scheduling feeds more tokens into each forward pass, and the faster aiter NSA decode processes them more efficiently, creating a compounding effect.

### 4. Kernel Tuning (foundational)
- **Dense GEMM tuning**: 40 shapes tuned via aiter's a8w8_blockscale tuner (16 MLP + 24 attention projection shapes)
- **Fused MoE tuning**: 11 shapes tuned for GLM-5's unique MoE configuration (257 experts, topk=9, FP8 blockscale)
- **FP8 bypass removal**: Patched aiter's `fused_moe.py` to use tuned kernels for small batches (previously bypassed for `token*topk <= 128`)

---

## Full Results Table (TP=4, CONC=64 unless noted)

| # | Experiment | Total tok/s | Δ vs baseline | TPOT (ms) | TTFT (ms) | Notes |
|---|---|---|---|---|---|---|
| 1 | **combined_nsa_mixed_ds16** | **1,630.2** | **+16.2%** | **72.8** | **1,757** | **Best config** |
| 2 | **combined_ds8** | **1,629.9** | **+16.1%** | **72.8** | **1,744** | **Decode steps don't matter much** |
| 3 | tuned_nsa_aiter_ds16 | 1,446.4 | +3.1% | 82.1 | 1,786 | NSA aiter alone |
| 4 | tuned_kitchen_sink_ds16 | 1,444.3 | +2.9% | 82.3 | 1,779 | AR fusion + fused-moe + mixed |
| 5 | tuned_mixedchunk_ds16 | 1,443.6 | +2.9% | 82.3 | 1,773 | Mixed-chunk alone |
| 6 | tuned_allreduce_fusedmoe_ds16 | 1,414.7 | +0.8% | 84.1 | 1,744 | AR fusion + fused-moe flag |
| 7 | tuned_fusedmoe_ds64 | 1,413.6 | +0.7% | 84.2 | 1,750 | Higher decode steps |
| 8 | tuned_nccl32ch_ds16 | 1,413.4 | +0.7% | 84.2 | 1,746 | NCCL_MIN_NCHANNELS=32 |
| 9 | tuned_moe_triton_ds16 | 1,411.0 | +0.5% | 84.3 | 1,796 | Triton MoE backend |
| 10 | tuned_fusedmoe_ds16 | 1,408.5 | +0.4% | 84.4 | 1,794 | Tuned GEMMs baseline |
| 11 | tuned_mem90_ds16 | 1,408.0 | +0.3% | 84.4 | 1,845 | mem-fraction-static=0.90 |
| 12 | aiter_allreduce_fusion | 1,406.8 | +0.2% | 84.6 | 1,808 | AR fusion alone |
| 13 | **baseline (ds=8)** | **1,403.4** | **—** | **84.4** | **—** | **Reference** |
| 14 | fused_decode_mla | 1,396.3 | -0.5% | 85.2 | 1,835 | FUSED_DECODE_MLA=1 (worse) |

### Higher Concurrency Alternative (TP=4, CONC=128)

We found an alternative configuration that pushes raw throughput significantly higher by doubling the max concurrency from 64 to 128 concurrent requests (`--num-continuous-decode-steps 32 --cuda-graph-max-bs 128`). This hit **2,206 tok/s on TP=4 alone** (+57.2% over baseline), without any DP scaling.

However, this is a **throughput-for-latency tradeoff**, not a fundamental per-request improvement:

| Metric | Best config (conc=64) | High-conc config (conc=128) |
|---|---|---|
| Total tok/s | 1,630 | 2,206 |
| TPOT (inter-token latency) | **72.8 ms** | 107.4 ms (+47%) |
| TTFT (time to first token) | **1,757 ms** | 2,911 ms (+66%) |
| p99 TTFT | **7,040 ms** | 13,578 ms (+93%) |

The server processes more requests simultaneously (128 vs 64 in-flight), so aggregate throughput increases, but each individual user sees noticeably worse latency — tokens arrive 47% slower, and time-to-first-token nearly doubles. The p99 tail latency on TTFT is particularly bad at 13.6 seconds.

Our optimized conc=64 config (NSA aiter + mixed-chunk) achieves better per-request performance with lower latency, and projects to **~3,244 tok/s with DP=2** — exceeding the conc=128 single-instance number while maintaining the lower latency profile. The high-conc config is a valid option for batch/offline workloads where latency is not a concern.

### DP=2/TP=4 Results (from earlier session, unoptimized)

| Experiment | Total tok/s | Per-GPU tok/s | TPOT (ms) | TTFT (ms) |
|---|---|---|---|---|
| dp2_tp4_ds16_fusedmoe_conc128 | 2,794.4 | 349.3 | 84.9 | 1,635 |
| dp2_tp4_ds16_aiter_fp8_conc128 | 2,787.4 | 348.4 | 85.3 | 1,644 |
| dp2_tp4_ds8_conc128 | 2,768.5 | 346.1 | 85.7 | 1,703 |

---

## DP=2 Projection

Previous DP=2 scaling factor: 2,794 / 1,403 = **1.99x** (near-linear).

Applying to our optimized TP=4 number:
- **1,630 × 1.99 ≈ 3,244 tok/s** projected for DP=2/TP=4 with all optimizations
- This is **+16.1%** over the previous best DP=2 number (2,794 tok/s)
- Per-GPU throughput maintained at ~406 tok/s

---

## What Didn't Work

| Optimization | Result | Why |
|---|---|---|
| `--enable-fused-moe-sum-all-reduce` | No effect | Flag only affects Triton MoE path; aiter handles topk reduction internally |
| `SGLANG_ROCM_FUSED_DECODE_MLA=1` | -0.5% | MLA fused decode kernel slightly worse on MI355X |
| `--enable-aiter-allreduce-fusion` | +0.2% | Fuses AR with RMSNorm — too small to matter alone |
| `mem-fraction-static=0.90` | +0.3% | 0.85 already sufficient for 64 concurrency |
| `moe-runner-backend=triton` | +0.5% | Triton and aiter CK essentially equivalent |
| `NCCL_MIN_NCHANNELS=32` | +0.7% | Marginal improvement from more NCCL channels |
| `--enable-mscclpp` | N/A | Only supports world_size=[8,16], not TP=4 |
| Piecewise CUDA graphs | N/A | Disabled on ROCm (HIP) — no compute/comm overlap |
| SBO/TBO overlap | N/A | Requires FlashInfer/DeepGemm backends, not available on ROCm |

---

## Key Bottleneck Analysis

From initial profiling:
- **49.2%** GPU idle time (waiting for communication)
- **44.9%** NCCL/RCCL all-reduce communication
- **~6%** actual compute

The model has 78 layers × 2 all-reduces/layer = **156 all-reduces per forward pass**.
All-reduce is using `AiterCustomAllreduce` (AMD's fast shared-memory path), not vanilla NCCL.
The `QuickAllReduce` with INT4 quantization is also available for smaller messages.

The communication is the fundamental bottleneck. The +16.2% improvement came from **scheduling** (mixed-chunk) and **kernel speed** (aiter NSA), not from reducing communication.

---

## Optimizations Applied (Code Changes)

### 1. Aiter Fused MoE bypass removal
**File**: `/sgl-workspace/aiter/aiter/fused_moe.py` line 785  
Removed the bypass that skipped tuned MoE kernels for small batches with FP8 blockscale (`token*topk <= 128`). Now all batch sizes use the tuned 2-stage pipeline.

### 2. Aiter GEMM tuning configs
**Files**: `/sgl-workspace/aiter/aiter/configs/a8w8_blockscale_tuned_gemm.csv`, `tuned_fmoe.csv`  
Added 40 dense GEMM shapes and 11 fused MoE shapes specific to GLM-5-FP8's architecture.

---

## Recommended Server Launch Command

```bash
python3 -m sglang.launch_server \
    --model-path zai-org/GLM-5-FP8 \
    --tensor-parallel-size 4 \
    --nsa-prefill-backend tilelang \
    --nsa-decode-backend aiter \
    --enable-mixed-chunk \
    --num-continuous-decode-steps 16 \
    --cuda-graph-max-bs 64 \
    --disable-radix-cache \
    --mem-fraction-static 0.85 \
    --trust-remote-code \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}'
```

For DP=2 (8 GPUs), use the InferenceX launch script with `--dp 2 --tp 4` and the above server args.

---

## Files & Artifacts

| File | Description |
|---|---|
| `RESUME_STATE.md` | Full state for resuming optimization |
| `ALL_RESULTS.csv` | All results in CSV format |
| `rapid_experiment.sh` | Single experiment runner |
| `batch_v3.sh` | First batch of experiments |
| `batch_combined.sh` | Combined winner sweep |
| `results_v3/` | All benchmark JSON results |
| `glm5_tuned_gemm.csv` | Dense GEMM tuned shapes |
| `glm5_tuned_fmoe.csv` | Fused MoE tuned shapes |

All artifacts at: `/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/glm5_optimization/`
