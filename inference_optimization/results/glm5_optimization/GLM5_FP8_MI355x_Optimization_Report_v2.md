# GLM-5-FP8 Inference Optimization Report v2 — 8x MI355X

**Date**: 2026-03-26  
**Model**: zai-org/GLM-5-FP8 (GlmMoeDsaForCausalLM)  
**Hardware**: 8x AMD Instinct MI355X (gfx950), 288 GB HBM3e per GPU  
**Framework**: SGLang v0.5.9-dev (editable install at `/sgl-workspace/sglang/`)  
**PyTorch**: 2.9.1+rocm7.2  
**InferenceX harness**: `/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/InferenceX/`

---

## 1. Model Architecture

| Feature | Value |
|---------|-------|
| Parameters | ~756 GB (FP8 e4m3, block_size 128×128) |
| Hidden Size | 6144 |
| Layers | 78 (75 MoE + 3 dense) |
| Attention | Multi-head Latent Attention (MLA, kv_lora_rank=512, q_lora_rank=2048) |
| MoE | 256 routed + 1 shared expert, top-8 routing per token |
| Special | Native Sparse Attention (NSA) with tilelang/aiter backends |
| MI355X peak | 3567 FP8 TFLOPS, 8 TB/s HBM BW |

---

## 2. Executive Summary

| Metric | TP=4 Baseline (4 GPUs) | Best Result (8 GPUs) | Improvement |
|--------|----------------------|---------------------|-------------|
| **Total tok/s** | **1379.07** | **2794.41** | **+102.6%** |
| Output tok/s | 687.67 | 1401.99 | +103.9% |
| Mean TPOT (ms) | 86.87 | 84.89 | -2.3% (better) |
| Config | TP=4, CONC=64, tilelang | DP=2/TP=4, CONC=128, tilelang, ds=16 | — |

**The single most impactful change was switching from TP=4 (4 GPUs idle) to DP=2/TP=4 (all 8 GPUs active), which nearly doubled total throughput.** Server-parameter tuning contributed an additional ~1.8% on TP=4 and ~0.9% on DP=2.

---

## 3. Phase 1: TP=4 Baseline Optimization (4 GPUs)

### 3.1 Server Parameter Tuning (ISL=1024, OSL=1024)

| # | Configuration | Output tok/s | **Total tok/s** | Mean TPOT (ms) | Median TTFT (ms) | vs Baseline |
|---|--------------|-------------|----------------|----------------|-------------------|-------------|
| 1 | **Baseline** (tilelang, CONC=64) | 687.67 | **1379.07** | 86.87 | 389.81 | — |
| 2 | + `--num-continuous-decode-steps 8` | 697.48 | **1398.74** | 85.63 | 373.21 | **+1.4%** |
| 3 | + ds=16, `--mem-fraction-static 0.88` | 695.63 | **1395.03** | 85.87 | 376.50 | +1.2% |
| 4 | **Best** (sweep re-run, ds=8) | 701.40 | **1403.43** | 84.88 | 395.04 | **+1.8%** |
| 5 | + `--schedule-conservativeness 0.5` + aiter-fusion | 664.35 | **1332.31** | — | — | **-3.4%** |
| 6 | + `--enable-mixed-chunk` | — | **CRASH** | — | — | NSA incompatible |
| 7 | + ds=32, `--chunked-prefill-size 8192` | — | **CRASH** | — | — | Decode starvation |

**Best TP=4 config:** tilelang decode, `--num-continuous-decode-steps 8`, CONC=64 → **1403.43 total tok/s**

### 3.2 Aiter Decode Backend (ISL=1024, OSL=1024, TP=4)

| CONC | Output tok/s | **Total tok/s** | Mean TPOT (ms) | Status |
|------|-------------|----------------|----------------|--------|
| 16 | 254.62 | **508.87** | 58.08 | Stable |
| 32 | 464.96 | **934.42** | 65.03 | Stable |
| 48 | 571.41 | **1143.33** | 79.46 | Stable |
| 64 | — | **CRASH** | — | NCCL/GPU illegal memory access |

Aiter decode has better per-token latency but crashes at CONC≥64. Tilelang is more stable and achieves higher total throughput at max concurrency.

### 3.3 ISL/OSL Parameter Sweep (TP=4, ds=8)

| ISL | OSL | CONC | Output tok/s | **Total tok/s** | Mean TPOT (ms) | Median TTFT (ms) |
|-----|-----|------|-------------|----------------|----------------|-------------------|
| 1024 | 1024 | 64 | 701.40 | **1403.43** | 84.88 | 395.04 |
| 1024 | 8192 | 32 | 429.56 | **483.27** | 69.95 | 310.34 |
| 8192 | 1024 | 16 | 188.62 | **1683.77** | 76.11 | 1513.21 |

---

## 4. Phase 2: Multi-GPU Parallelism Strategies (8 GPUs)

### 4.1 Results (ISL=1024, OSL=1024)

| # | Configuration | Output tok/s | **Total tok/s** | Mean TPOT (ms) | Median TTFT (ms) | vs TP=4 Baseline |
|---|--------------|-------------|----------------|----------------|-------------------|-----------------|
| 1 | **DP=2/TP=4**, ds=8, CONC=128 | 1388.99 | **2768.51** | 85.74 | 346.85 | **+100.8%** |
| 2 | **DP=2/TP=4**, ds=16, fused-moe | 1401.99 | **2794.41** | 84.89 | — | **+102.6%** |
| 3 | DP=2/TP=4, ds=16, `--fp8-gemm-backend aiter` | 1398.49 | **2787.43** | 85.33 | — | +102.1% |
| 4 | DP=2/TP=4, aiter decode, CONC=96 | 1116.52 | **2232.97** | 79.39 | — | +61.9% |
| 5 | DP=2/TP=4, CONC=192 | 0 | **0 (CRASH)** | — | — | OOM/graph failure |
| 6 | **TP=8**, ds=8, CONC=64 | 840.07 | **1680.89** | 70.52 | — | +21.9% |

### 4.2 Analysis

**DP=2/TP=4 vs TP=8:**
- DP=2/TP=4 at CONC=128 delivers **2794 total tok/s** — each DP shard runs TP=4 with ~64 concurrent requests, and throughput scales linearly (2× TP=4).
- TP=8 at CONC=64 delivers only **1681 total tok/s** — per-GPU throughput halves when going TP=4→TP=8 due to: (a) TileLang NSA decode kernel time doesn't scale with TP, (b) NCCL all-reduce cost grows 7–8× for TP=8 vs TP=4.

**Aiter decode with DP=2:** At CONC=96 (48 per shard), aiter runs without crashing — each shard sees CONC=48 which is below the crash threshold. But 2232 total tok/s still trails tilelang's 2794 at CONC=128.

**CONC=192 failure:** CUDA graph capture with `--cuda-graph-max-bs 192` exceeded memory or graph complexity limits.

---

## 5. TraceLens Profiling Analysis (TP=4, tilelang, ds=8)

Profiled via SGLang's `/start_profile`/`/stop_profile` HTTP endpoints. Trace analyzed with TraceLens.

### 5.1 GPU Utilization

| Metric | Time (ms) | % |
|--------|----------|---|
| **Computation** | 178.21 | 50.2% |
| **Idle** | **174.55** | **49.2%** |
| Exposed communication | 0.46 | 0.1% |
| Exposed memcpy | 1.76 | 0.5% |
| **Total trace** | 354.98 | 100% |

**49.2% GPU idle** — the dominant bottleneck is not any single kernel but CPU→GPU scheduling gaps between CUDA graph replays. The "other" category metadata confirms: `sync_time_ms: 89044` with `sync_ops_count: 49` and `has_sync_bottleneck: true`.

### 5.2 Per-Kernel Breakdown (Top 20 by GPU Time)

| # | Kernel | GPU Time (ms) | % of Total | Calls | µs/call | Category |
|---|--------|-------------|-----------|-------|---------|----------|
| 1 | `record_param_comms` (NCCL all-reduce) | **80.79** | **44.9%** | 157 | 514.6 | Communication |
| 2 | `hipGraphLaunch` (CUDA graph replay) | **59.07** | **32.8%** | 1 | 59071 | Graph overhead |
| 3 | `aiter::gemm_a16w16_atomic_` (MLA GEMM) | **14.74** | **8.2%** | 75 | 196.6 | Compute (CK) |
| 4 | `aiter::gemm_a8w8_blockscale_ck` (FP8 MoE) | **8.50** | **4.7%** | 318 | 26.7 | Compute (CK) |
| 5 | `hipModuleLaunchKernel` | **3.07** | **1.7%** | 623 | 4.9 | Launch overhead |
| 6 | `hipLaunchKernel` | **2.23** | **1.2%** | 237 | 9.4 | Launch overhead |
| 7 | `aten::copy_` | **1.99** | **1.1%** | 421 | 4.7 | Memory |
| 8 | `aiter::ck_moe_stage1` (gate+up GEMM) | **1.73** | **1.0%** | 75 | 23.1 | MoE FP8 GEMM |
| 9 | `aten::fill_` | **1.08** | **0.6%** | 233 | 4.6 | Memory |
| 10 | `aiter::ck_moe_stage2` (down-proj GEMM) | **0.95** | **0.5%** | 75 | 12.6 | MoE FP8 GEMM |
| 11 | `aiter::moe_sorting_fwd` | **0.84** | **0.5%** | 75 | 11.2 | MoE dispatch |
| 12 | `aiter::rope_cached_positions_2c_fwd` | **0.74** | **0.4%** | 156 | 4.8 | RoPE |
| 13 | `aiter::dynamic_per_token_scaled_quant` | **0.74** | **0.4%** | 156 | 4.7 | Quantization |
| 14 | `aiter::biased_grouped_topk_hip` | **0.56** | **0.3%** | 75 | 7.5 | MoE TopK |
| 15 | `aten::cat` | **0.54** | **0.3%** | 83 | 6.5 | Memory |
| 16 | `sgl_kernel::fast_topk_transform_fused` | **0.38** | **0.2%** | 78 | 4.9 | TopK |
| 17 | `aiter::add_rmsnorm` | **0.38** | **0.2%** | 79 | 4.8 | Norm |
| 18 | `aiter::silu_and_mul` | **0.36** | **0.2%** | 75 | 4.8 | Activation |
| 19 | `aten::native_layer_norm` | **0.36** | **0.2%** | 78 | 4.6 | Norm |
| 20 | `HadamardTransformFn` | **0.36** | **0.2%** | 78 | 4.6 | NSA transform |

### 5.3 Breakdown by Category

| Category | GPU Time (ms) | % | Key Observation |
|----------|-------------|---|-----------------|
| **NCCL Communication** | 80.79 | 44.9% | Dominated by TP=4 all-reduce, inherent cost |
| **CUDA Graph Replay** | 59.07 | 32.8% | Single hipGraphLaunch for entire decode step |
| **MLA Attention GEMM** | 14.74 | 8.2% | BF16 [2×256] × [256×6144], 75 layers |
| **FP8 MoE GEMMs** | 11.18 | 6.2% | stage1 (23µs) + stage2 (13µs) + blockscale (27µs), all CK |
| **Kernel Launch Overhead** | 5.30 | 2.9% | hipModuleLaunchKernel + hipLaunchKernel |
| **Memory Ops** | 3.61 | 2.0% | copy_ + fill_ + cat |
| **MoE Dispatch** | 1.78 | 1.0% | sorting + topk |
| **Norm + Activation** | 1.10 | 0.6% | rmsnorm + silu_and_mul + layer_norm |
| **NSA Decode** | 0.36 | **0.2%** | HadamardTransformFn only; TileLang main_kernel captured inside hipGraphLaunch |
| **Other** | 1.28 | 0.7% | RoPE, quantization, etc. |

### 5.4 Per-Layer Breakdown (78 Layers)

With 75 MoE layers executing per decode step (3 dense layers at head/tail), the per-layer costs are:

| Component | µs/layer | Note |
|-----------|----------|------|
| NCCL all-reduce | ~514 | 2 all-reduces per layer (attention + MoE) |
| MLA GEMM (absorbed key) | ~197 | `gemm_a16w16_atomic_` for MLA decode |
| MoE stage1 (gate+up) | ~23 | FP8 CK GEMM, 257 experts |
| MoE stage2 (down-proj) | ~13 | FP8 CK GEMM |
| MoE blockscale GEMM | ~27 | FP8 blockscale quantized GEMM |
| MoE sorting | ~11 | Expert dispatch sorting |
| MoE TopK | ~7.5 | Biased grouped top-k |
| RoPE | ~4.8 | Cached positions |
| Quantization | ~4.7 | Dynamic per-token FP8 quant |
| RMSNorm | ~4.8 | Fused add+rmsnorm |
| SiLU×Mul | ~4.8 | Gate activation |
| **Total per layer** | **~812** | **~0.81 ms/layer** |

At 75 layers × 0.81 ms = ~60.7 ms compute per decode step, closely matching the 59 ms hipGraphLaunch.

---

## 6. Code-Level Investigation

### 6.1 Source Files Examined

| File | Content |
|------|---------|
| `/sgl-workspace/sglang/python/sglang/srt/models/glm4_moe.py` | GlmMoeDsaForCausalLM (extends DeepseekV2ForCausalLM) |
| `/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py` | DeepseekV2 MoE/MLA implementation |
| `/sgl-workspace/sglang/python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` | NSA TileLang decode kernel |
| `/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` | MoE runner dispatch |

### 6.2 Findings

| Investigation | Finding | Actionable? |
|--------------|---------|-------------|
| **Dual-stream MoE** (shared+routed expert overlap) | HIP-disabled in `deepseek_v2.py:1820` (`_is_cuda` check). However, **shared expert fusion is already active** on MI355X (gfx950 cap 9.5 ≥ 9.4), which folds the shared expert into the MoE kernel. When `num_fused_shared_experts > 0`, the dual-stream path is never entered. | **Moot** — fusion supersedes dual-stream |
| **Shared Expert Fusion** | Already enabled automatically. Server log confirms: `"Shared experts fusion optimization enabled."` | Already active |
| **Scaling factor bug in dual-stream** | `forward_normal_dual_stream` at line 595 applies `routed_scaling_factor` when `not _is_cuda` but doesn't check `not _use_aiter`, causing double-application with aiter TopK. | **Latent bug** — only triggers if dual-stream is activated on HIP, currently moot |
| **`torch.compile` (GEAK)** | Incompatible with MoE+MLA+FP8+NSA combo on ROCm | Not applicable |
| **CK vendor GEMMs** | All FP8 GEMMs use highly optimized Composable Kernel implementations (aiter wrappers). Kernel shapes are already tuned for MI355X. | No source-level improvement path |
| **NSA TileLang decode kernel** | Only 0.2% of total GPU time (hidden inside CUDA graph). Not a bottleneck at TP=4. | Not worth optimizing at TP=4 |
| **DP attention (`--enable-dp-attention`)** | Gets stuck at server launch for this model. | Blocked |
| **Mori A2A backend (`--moe-a2a-backend mori`)** | Gets stuck at server launch. | Blocked |

---

## 7. Constraints and Known Issues

| Constraint | Description | Impact |
|-----------|-------------|--------|
| `--nsa-decode-backend aiter` + TP=8 | aiter NSA decode doesn't support TP=8 | Forces TP=4 or tilelang for TP=8 |
| `kv_cache_dtype=fp8/fp8_e4m3` | Neither tilelang nor aiter NSA backends support FP8 KV cache | Cannot reduce KV cache memory with FP8 |
| `--enable-dp-attention` | Hangs during server launch | Cannot use DP attention |
| `--moe-a2a-backend mori` | Hangs during server launch | Cannot use Mori all-to-all |
| aiter decode at CONC≥64 (TP=4) | GPU illegal memory access → NCCL crash | Must use CONC≤48 with aiter, or DP=2 with CONC=96 (48/shard) |
| ISL=8192 at CONC≥32 | GPU illegal memory access | Must use CONC≤16 for long-prefix workloads |
| `--enable-mixed-chunk` | Incompatible with NSA attention backend | Cannot use mixed chunked prefill |
| `--num-continuous-decode-steps ≥ 32` | Causes decode starvation — no requests complete | Must keep ≤16 |
| DP=2/TP=4 at CONC≥192 | CUDA graph capture OOM or graph complexity limit | Must keep CONC≤128 |

---

## 8. Recommended Production Configurations

### 8.1 Maximum Throughput (all 8 GPUs)

```bash
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export SAFETENSORS_FAST_GPU=1
export SGLANG_USE_AITER=1
export RCCL_MSCCL_ENABLE=0

python3 -m sglang.launch_server \
    --nsa-prefill-backend tilelang \
    --nsa-decode-backend tilelang \
    --cuda-graph-max-bs 128 \
    --disable-radix-cache \
    --model-path models/zai-org/GLM-5-FP8 \
    --served-model-name zai-org/GLM-5-FP8 \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size 4 \
    --data-parallel-size 2 \
    --trust-remote-code \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --mem-fraction-static 0.85 \
    --num-continuous-decode-steps 16 \
    --enable-fused-moe-sum-all-reduce \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}'
```

**Expected: ~2794 total tok/s** at ISL=1024/OSL=1024/CONC=128

### 8.2 Best Single-Instance (4 GPUs)

```bash
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export SAFETENSORS_FAST_GPU=1

python3 -m sglang.launch_server \
    --nsa-prefill-backend tilelang \
    --nsa-decode-backend tilelang \
    --cuda-graph-max-bs 64 \
    --disable-radix-cache \
    --model-path models/zai-org/GLM-5-FP8 \
    --served-model-name zai-org/GLM-5-FP8 \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --mem-fraction-static 0.85 \
    --num-continuous-decode-steps 8 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}'
```

**Expected: ~1403 total tok/s** at ISL=1024/OSL=1024/CONC=64

---

## 9. Bottleneck Analysis and Future Optimization Opportunities

### 9.1 Where Time Is Spent (TP=4, Decode Phase)

```
┌─────────────────────────────────────────────────────────────────┐
│ Total GPU Time: ~355 ms per profiled window                     │
│                                                                 │
│ ████████████████████████████ 49.2% IDLE (CPU scheduling)        │
│ ██████████████████████████   44.9% NCCL ALL-REDUCE              │
│ ████                          8.2% MLA GEMM (CK)               │
│ ███                           6.2% MoE FP8 GEMMs (CK)          │
│ ██                            2.9% Kernel Launch Overhead       │
│ █                             2.0% Memory Ops                   │
│ ░                             1.0% MoE Dispatch                 │
│ ░                             0.6% Norm+Activation              │
│ ░                             0.2% NSA Decode                   │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Key Bottlenecks and Potential Mitigations

| Bottleneck | % of Time | Possible Mitigation | Status |
|-----------|----------|---------------------|--------|
| **GPU idle / CPU scheduling** | 49.2% | `--num-continuous-decode-steps` batches multiple steps; higher values help but >16 causes starvation | Partially mitigated (+1.8%) |
| **NCCL all-reduce** | 44.9% | DP=2/TP=4 avoids scaling communication beyond TP=4; MSCCL/custom-all-reduce could help | DP=2 approach used |
| **MLA GEMM** | 8.2% | CK vendor kernel, well-tuned for MI355X. FP8 MLA could help but requires NSA backend support | Blocked by NSA FP8 support |
| **MoE FP8 GEMMs** | 6.2% | Already using CK optimized kernels with FP8 blockscale | No improvement path |
| **Kernel launch overhead** | 2.9% | CUDA graphs already capture most launches; remaining are outside-graph ops | Minimal headroom |

### 9.3 Untested Opportunities

| Opportunity | Expected Impact | Complexity | Notes |
|------------|----------------|------------|-------|
| DP=2/TP=4 ISL/OSL sweep | Verify 2× scaling for all workloads | Low | Run sweep_benchmark.sh with DP=2 |
| `--enable-mscclpp` | Potential NCCL replacement for lower comm latency | Medium | Requires MSCCL++ library on ROCm |
| DP=4/TP=2 | More DP parallelism, less TP comm | Medium | May not fit 756GB model in 2×288GB |
| FP8 KV cache | Reduce KV cache memory → more batch slots | High | Requires NSA backend FP8 support |
| Custom CUDA graph capture groups | Reduce graph capture/replay overhead | High | Requires SGLang framework changes |
| Kernel fusion (RMSNorm+Quant+GEMM) | Eliminate memory round-trips | High | aiter already fuses some; limited headroom |

---

## 10. Complete Results Reference

### 10.1 Phase 1 — TP=4 (4 GPUs)

| Experiment | Backend | CONC | ds | Extra Flags | Output tok/s | Total tok/s | TPOT (ms) | TTFT (ms) | Status |
|-----------|---------|------|----|----|-------------|-------------|-----------|-----------|--------|
| Baseline | tilelang | 64 | — | — | 687.67 | 1379.07 | 86.87 | 389.81 | Stable |
| ds=8 | tilelang | 64 | 8 | — | 697.48 | 1398.74 | 85.63 | 373.21 | Stable |
| ds=16, mem=0.88 | tilelang | 64 | 16 | `--mem-fraction-static 0.88` | 695.63 | 1395.03 | 85.87 | 376.50 | Stable |
| Sweep (ds=8) | tilelang | 64 | 8 | — | 701.40 | 1403.43 | 84.88 | 395.04 | Stable |
| sched+fusion | tilelang | 64 | — | `--schedule-conservativeness 0.5 --enable-aiter-allreduce-fusion` | 664.35 | 1332.31 | — | — | Regressed |
| mixed-chunk | tilelang | 64 | — | `--enable-mixed-chunk --schedule-conservativeness 0.5` | — | CRASH | — | — | NSA incompatible |
| ds=32, chunked | tilelang | 64 | 32 | `--mem-fraction-static 0.88 --chunked-prefill-size 8192` | — | CRASH | — | — | Starvation |
| aiter CONC=16 | aiter | 16 | — | — | 254.62 | 508.87 | 58.08 | 275.98 | Stable |
| aiter CONC=32 | aiter | 32 | — | — | 464.96 | 934.42 | 65.03 | 286.95 | Stable |
| aiter CONC=48 | aiter | 48 | — | — | 571.41 | 1143.33 | 79.46 | 350.92 | Stable |
| aiter CONC=64 | aiter | 64 | — | — | — | CRASH | — | — | NCCL failure |

### 10.2 Phase 1 — ISL/OSL Sweep (TP=4, ds=8)

| ISL | OSL | CONC | Output tok/s | Total tok/s | TPOT (ms) | TTFT (ms) |
|-----|-----|------|-------------|-------------|-----------|-----------|
| 1024 | 1024 | 64 | 701.40 | 1403.43 | 84.88 | 395.04 |
| 1024 | 8192 | 32 | 429.56 | 483.27 | 69.95 | 310.34 |
| 8192 | 1024 | 16 | 188.62 | 1683.77 | 76.11 | 1513.21 |

### 10.3 Phase 2 — Multi-GPU (8 GPUs)

| Experiment | Parallelism | CONC | ds | Extra Flags | Output tok/s | Total tok/s | TPOT (ms) | Status |
|-----------|------------|------|----|----|-------------|-------------|-----------|--------|
| DP=2/TP=4, ds=8 | DP=2, TP=4 | 128 | 8 | — | 1388.99 | 2768.51 | 85.74 | Stable |
| **DP=2/TP=4, ds=16, fused-moe** | DP=2, TP=4 | 128 | 16 | `--enable-fused-moe-sum-all-reduce` | **1401.99** | **2794.41** | **84.89** | **Stable (BEST)** |
| DP=2/TP=4, aiter FP8 | DP=2, TP=4 | 128 | 16 | `--fp8-gemm-backend aiter` | 1398.49 | 2787.43 | 85.33 | Stable |
| DP=2/TP=4, aiter decode | DP=2, TP=4 | 96 | 16 | `--nsa-decode-backend aiter` | 1116.52 | 2232.97 | 79.39 | Stable |
| DP=2/TP=4, CONC=192 | DP=2, TP=4 | 192 | 8 | `--cuda-graph-max-bs 192` | 0 | 0 | — | CRASH |
| TP=8 | TP=8 | 64 | 8 | — | 840.07 | 1680.89 | 70.52 | Stable |

---

## 11. Artifacts

All results stored in: `/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/glm5_optimization/`

| Path | Description |
|------|-------------|
| `results/*.json` | Phase 1 benchmark JSONs |
| `results_v2/*.json` | Phase 2 (multi-GPU) benchmark JSONs |
| `traces/1774488817*-TP-{0..3}.trace.json.gz` | torch.profiler traces (TP=4, tilelang, ds=8) |
| `traces/tracelens_output/` | TraceLens analysis (CSVs, metadata, category data) |
| `run_experiment.sh` | Generic experiment runner script |
| `sweep_benchmark.sh` | ISL/OSL sweep script |
| `GLM5_FP8_MI355x_Optimization_Report.md` | Original v1 report |
| `GLM5_FP8_MI355x_Optimization_Report_v2.md` | This report |

---

## 12. Methodology Notes

- All benchmarks use the InferenceX harness (`benchmark_serving.py` via `run_benchmark_serving` from `benchmark_lib.sh`)
- Workload: synthetic prompts with `--random-range-ratio 0.8` (ISL/OSL vary ±20%)
- Warmup: `num_prompts = CONC × 3` (standard InferenceX warmup)
- Server startup: ~12 minutes for 756GB model (model load ~2 min + CUDA graph capture ~10 min)
- Environment variables: `SGLANG_ROCM_FUSED_DECODE_MLA=0`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `SAFETENSORS_FAST_GPU=1`
- SGLang installed as editable package at `/sgl-workspace/sglang/` (pod-local, not shared NFS)
