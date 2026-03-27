# GLM-5-FP8 Inference Optimization Report — 8x MI355X

**Date**: 2026-03-26  
**Model**: zai-org/GLM-5-FP8 (GlmMoeDsaForCausalLM)  
**Hardware**: 8x AMD Instinct MI355X (gfx950), 256GB HBM3e per GPU  
**Framework**: SGLang (via InferenceX benchmark harness)  
**Tensor Parallelism**: TP=4  

---

## 1. Model Architecture Summary

| Feature | Value |
|---------|-------|
| Parameters | ~756 GB (FP8 e4m3, block_size 128x128) |
| Hidden Size | 6144 |
| Layers | 78 |
| Attention Heads | 64 |
| Attention Type | Multi-head Latent Attention (MLA, kv_lora_rank=512, q_lora_rank=2048) |
| MoE Experts | 256 routed + 1 shared |
| Experts per Token | 8 |
| Special Feature | Native Sparse Attention (NSA) |

---

## 2. Optimization Results Summary

### Best Configuration
```
--nsa-prefill-backend tilelang
--nsa-decode-backend tilelang
--num-continuous-decode-steps 8
--mem-fraction-static 0.85
--disable-radix-cache
--cuda-graph-max-bs 64
--tensor-parallel-size 4
```

Environment:
```
SGLANG_ROCM_FUSED_DECODE_MLA=0
ROCM_QUICK_REDUCE_QUANTIZATION=INT4
SAFETENSORS_FAST_GPU=1
```

### Parameter Sweep Results (Optimized Config)

| ISL | OSL | CONC | Output tok/s | Total tok/s | TPOT (ms) | Median ITL (ms) |
|-----|-----|------|-------------|-------------|-----------|-----------------|
| 1024 | 1024 | 64 | 701.40 | **1403.43** | 84.41 | 77.61 |
| 1024 | 8192 | 32 | 429.56 | **483.27** | — | — |
| 8192 | 1024 | 16 | 188.62 | **1683.77** | 76.11 | 61.57 |

### ISL=1024/OSL=1024 Tuning Progression

| Configuration | Output tok/s | Total tok/s | Change vs Baseline |
|--------------|-------------|-------------|-------------------|
| Baseline (tilelang, CONC=64) | 687.67 | 1379.07 | — |
| + decode-steps=8 | 697.48 | 1398.74 | **+1.4%** |
| + decode-steps=16, mem=0.88 | 695.63 | 1395.03 | +1.2% |
| **Best (sweep re-run, ds=8)** | **701.40** | **1403.43** | **+1.8%** |
| + sched=0.5, aiter-allreduce-fusion | 664.35 | 1332.31 | -3.4% (regressed) |
| + mixed-chunk + sched=0.5 | — | CRASH | Incompatible with NSA |
| + decode-steps=32, mem=0.88, chunked=8192 | — | CRASH | Decode-starvation |

### Aiter Decode Backend Analysis (ISL=1024/OSL=1024)

| Config | Output tok/s | Total tok/s | TPOT (ms) | Status |
|--------|-------------|-------------|-----------|--------|
| aiter, CONC=16 | 254.62 | 508.87 | 58.08 | Stable |
| aiter, CONC=32 | 464.96 | 934.42 | 65.03 | Stable |
| aiter, CONC=48 | 571.41 | 1143.33 | 79.46 | Stable |
| aiter, CONC=64 | — | CRASH | — | NCCL failure |

Conclusion: `tilelang` is more stable and achieves higher throughput at CONC=64 (1403 vs 1143 at CONC=48 for aiter). `aiter` has better per-token latency (TPOT) at lower concurrency but hits stability walls at CONC>=64 due to NCCL/memory issues.

---

## 3. TraceLens Profiling Analysis

Profiled with `torch.profiler` via SGLang's HTTP `/start_profile`/`/stop_profile` endpoints with `tilelang` backend, TP=4, decode-steps=8.

### GPU Utilization
- **Computation**: 50.2%
- **Communication (exposed)**: 0.13%
- **Memory copy (exposed)**: 0.50%
- **Idle**: **49.2%**

### GPU Kernel Time Breakdown (Total: ~181 ms per decode step)

| Category | GPU Time (ms) | % | Details |
|----------|-------------|---|---------|
| **NCCL All-Reduce** | 80.79 | 44.6% | `record_param_comms` — TP all-reduce across 4 GPUs |
| **CK FP8 GEMMs** (MoE weights) | ~14.6 | ~8.0% | `_gemm_a16_w16_atomic_kernel` (MLA attn GEMM, 194µs×75) |
| **aiter cross-device reduce** | 8.06 | 4.5% | `cross_device_reduce_1stage` (custom all-reduce) |
| **CK FP8 GEMMs** (linear layers) | ~12.0 | ~6.6% | Various `ck::kernel_gemm_xdl_cshuffle_v3` instances |
| **MoE Fused** | 6.43 | 3.5% | Fused MoE dispatch + combine |
| **aiter FP8 block-scale GEMM** | ~6.0 | ~3.3% | `aiter::gemm_a8w8_blockscale_ck` |
| **Elementwise** | 3.06 | 1.7% | Activation functions, scaling |
| **MLA decode** | 2.51 | 1.4% | `_deepgemm_fp8_paged_mqa_logits` |
| **NSA TileLang decode** | 1.96 | 1.1% | `main_kernel` (sparse attention) |
| **Normalization** | 0.79 | 0.4% | RMSNorm |

### Key Findings

1. **The primary bottleneck is NOT any single kernel but GPU idle time (49.2%).** This is caused by CPU scheduling overhead between CUDA graph launches. `--num-continuous-decode-steps 8` reduced this overhead by batching 8 decode steps before re-scheduling.

2. **Communication is dominant at 49.1% of GPU active time** (NCCL all-reduce + aiter cross-device reduce). This is inherent to TP=4 and cannot be reduced without changing the parallelism strategy.

3. **NSA TileLang decode is only 1.1% of GPU time** — contrary to the user's initial hypothesis, the NSA decode kernel is NOT the bottleneck in the overall pipeline. It becomes more significant at higher TP sizes (where TP=8 halves throughput) but at TP=4 it's well-hidden.

4. **CK vendor GEMMs (FP8) account for ~18% of GPU time** — these are highly optimized Composable Kernel implementations and cannot be meaningfully improved via source code changes.

5. **Shared Expert Fusion is already active on MI355X** (`gfx950` capability 9.5 >= 9.4 threshold), which means the shared expert is fused directly into the MoE kernel. The dual-stream optimization (overlapping shared + routed experts on separate streams) does NOT activate when fusion is enabled.

---

## 4. Code-Level Investigation

### Examined Source Files
- `/sgl-workspace/sglang/python/sglang/srt/models/glm4_moe.py` — GlmMoeDsaForCausalLM model
- `/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py` — Base DeepseekV2 model with MoE/MLA
- `/sgl-workspace/sglang/python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` — NSA TileLang decode kernels

### Findings

| Optimization | Status | Impact |
|-------------|--------|--------|
| **Dual-stream MoE (shared+routed overlap)** | Already bypassed — shared expert fusion is active | N/A |
| **Shared Expert Fusion** | Already enabled automatically on MI355X (gfx950 >= gfx942) | Active |
| **FP8 Block-scale GEMM (CK)** | Vendor-optimized, no source-level improvement path | N/A |
| **NSA TileLang decode** | Only 1.1% of GPU time, not worth optimizing at TP=4 | Negligible |
| **`torch.compile` (GEAK)** | Incompatible with MoE+MLA+FP8 architecture | Skipped |
| **`--enable-mixed-chunk`** | Incompatible with NSA attention backend | Crashed |
| **`--enable-aiter-allreduce-fusion`** | Regressed throughput by 3.4% | Not recommended |

### Bug Found and Fixed (reverted as moot)
In `deepseek_v2.py`'s `forward_normal_dual_stream` method, the `routed_scaling_factor` was applied when `not _is_cuda` without checking `not _use_aiter`. This would cause double-application of the scaling factor when running HIP+aiter with dual-stream enabled. However, since shared expert fusion is active (making dual-stream unused), this bug doesn't manifest in practice.

---

## 5. Constraints and Limitations

| Constraint | Description |
|-----------|-------------|
| **`--nsa-decode-backend aiter` + TP=8** | Not supported (aiter limitation) |
| **`kv_cache_dtype=fp8/fp8_e4m3`** | Not supported with either tilelang or aiter NSA backends |
| **`--enable-dp-attention`** | Gets stuck during server launch |
| **`--moe-a2a-backend mori`** | Gets stuck during server launch |
| **`aiter` at CONC>=64** | NCCL/memory crash (ISL=1024, OSL=1024) |
| **ISL=8192, CONC>=32** | GPU illegal memory access |
| **`--enable-mixed-chunk`** | Incompatible with NSA backend |
| **`--num-continuous-decode-steps >= 32`** | Causes decode starvation, all requests fail |

---

## 6. Recommended Production Configuration

```bash
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export SAFETENSORS_FAST_GPU=1

python3 -m sglang.launch_server \
    --nsa-prefill-backend tilelang \
    --nsa-decode-backend tilelang \
    --cuda-graph-max-bs $CONC \
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

---

## 7. Artifacts

All results stored in: `/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/glm5_optimization/`

| Path | Description |
|------|-------------|
| `results/*.json` | Benchmark result JSON files |
| `traces/` | torch.profiler traces (TP0-TP3) |
| `traces/tracelens_output/` | TraceLens analysis CSVs and metadata |
| `sweep_benchmark.sh` | Parameter sweep script |
| `GLM5_FP8_MI355x_Optimization_Report.md` | This report |
