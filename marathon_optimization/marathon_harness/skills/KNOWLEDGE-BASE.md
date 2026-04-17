---
name: inference-optimization-knowledge-base
description: |
  Shared knowledge base for Sprint and Marathon inference optimization skills.
  This KB is shared between sprint-inference-optimization and marathon-inference-optimization.
  Changes made here should be synced to the counterpart using sync_kb.sh.
---

# Inference Optimization Knowledge Base

## Optimization Results Summary (All Models)

| Model | Framework | TP | torch.compile | GEAK E2E Gain | Server Tuning Gain | Peak Output tok/s | Report |
|-------|-----------|---:|:-------------:|:-------------:|:------------------:|------------------:|--------|
| Qwen3-30B-A3B | SGLang | 1 | ✅ | **+27.6%** @ CONC=4 (full multi-round loop through `mm_0`) | N/A | ~751 (CONC=4 opt); ~1859 (CONC=16 baseline peak) | `optimization_report.md` |
| Qwen3-30B-A3B | SGLang | 8 | ✅ | **+27.8%** @ CONC=4 (full loop + topkGatingSoftmax HIP) | N/A | — | — |
| Qwen3-30B-A3B | vLLM | 1 | ✅ | ~0% (±1%) | N/A | 4209 (CONC=64) | `optimization_report_qwen3_30b_vllm.md` |
| Kimi-K2.5 | SGLang | 8 | ❌ | **+0.81%** | +2.4% (decode-steps) | 509 (CONC=128) | `optimization_report_kimi_k25.md` |
| Kimi-K2.5 | vLLM | 4 | ❌ | **+1.76%** | **+84%** (gpu-mem/seqs) | 264 (CONC=64) | `kimi-k25-vllm-optimization-report.md` |
| DeepSeek-R1-0528 | SGLang | 8 | ❌ | 0% (reverted) | +13.9% (decode-steps) | 3192 (CONC=64) | `optimization_report_dsr1_0528.md` |
| gpt-oss-120b | SGLang | 8 | ❌ | SKIPPED | 0% (±0.5%) | 9825 (CONC=64) | `gpt-oss-120b_optimization_report.md` |
| **GLM-5-FP8** | **SGLang** | **4** | **❌** | **N/A** | **+16.2% (backends + scheduling)** | **1630 (CONC=64)** | **`OPTIMIZATION_REPORT.md`** |

**Key takeaways:**

1. **Full multi-round loop is mandatory** — Qwen3-30B-A3B: RMSNorm-only ~+15% vs full loop **+27.6% (TP=1, through `mm_0`)** / **+27.8% (TP=8, incl. HIP)**. Extra gain came from `triton_tem_fused_mm_0` (~22% GPU) that was previously skipped. **Always run the full loop; do not stop after the first kernel.**
2. **Full torch.compile enables the largest GEAK wins** — with full compile, GEAK reached up to +27.6%; without full compile, GEAK on vendor kernels yields ≤ 1.76%. **Selective submodule compilation (Strategy C) is an untested middle ground** on most MoE+MLA models — it can generate Inductor Triton targets for norm/activation ops even when full torch.compile is incompatible.
3. **GEAK end-to-end gain depends on concurrency (CONC)** — sweet spot around CONC=4; at high concurrency, gains are diluted by the pipeline.
4. **Server parameter tuning is often more effective than GEAK** — e.g. Kimi vLLM +84%, DSR1 +13.9%.
5. **Already highly tuned models (gpt-oss) had little headroom** — GPU utilization ~94.7%, 95%+ vendor kernels.
6. **GEAK supports HIP kernels** — needs full `.cu` source in task input (not path-only). `topkGatingSoftmax`: use SGLang image + explicit source dir in prompt; test harness creation can fail with minimal input.
7. **Framework choice dominates for some stacks** — Kimi-K2.5: SGLang ~324 vs vLLM ~141 tok/s (~2.3× gap).
8. **Backend switches + scheduling modes outperform parameter sweeps** — GLM-5-FP8: backend switches gave +16.2% combined vs <1% from any single parameter change. **Always explore backends before sweeping parameters.**
9. **Combination synergies can be super-linear** — GLM-5-FP8: two +3% backend switches combined for +16.2%. Always test winners together.

*(All models, summary validated 2026-03-26.)*

## Server Parameter Reference

### SGLang Server Parameters (MI355X)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--attention-backend aiter` | Required | AMD aiter MLA/GQA fusion kernels |
| `--chunked-prefill-size 196608` | InferenceX default | Larger = better prefill throughput |
| `--mem-fraction-static 0.8` | 0.8 for DSR1 | Fraction of GPU memory for KV cache |
| `--disable-radix-cache` | InferenceX default | For random-input benchmarks |
| `--num-continuous-decode-steps 4` | InferenceX default | Batches decode steps |
| `--kv-cache-dtype fp8_e4m3` | Recommended | Halves KV cache memory |
| `--cuda-graph-max-bs N` | Set to max CONC | CUDA graph captures for batch 1..N |

### vLLM Server Parameters (MI355X)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--gpu-memory-utilization 0.85` | Script default | Fraction of GPU memory for model + KV cache |
| `--max-model-len N` | Auto from config | Cap max context length (reduces KV cache memory) |
| `--max-num-seqs N` | 256 | Max concurrent sequences (increase for throughput) |
| `--disable-log-stats` | Script default | Reduce log noise during benchmark |
| `--kv-cache-dtype fp8_e4m3` | Optional | FP8 KV cache, halves memory |
| `--enforce-eager` | Off | Disable torch.compile + CUDA graph |
| `--max-num-batched-tokens N` | 8192 | Similar to SGLang's chunked-prefill-size |
| `--compilation-config level=N` | 3 | 0=off, 3=full compile+cudagraph |
| `--block-size N` | 16 | MLA models require `--block-size 1` |

### vLLM Version Compatibility

**Validated 2026-03-23:**

| Flag | v0.9+ | v0.17+ | Notes |
|------|-------|--------|-------|
| `--enforce-eager` | ✅ | ✅ | Stable across versions |
| `--compilation-config level=N` | ❌ | ✅ | v0.9 uses `--enforce-eager` to disable |
| `--max-num-seqs` | ✅ | ✅ | Stable, but default changed |
| `--num-scheduler-steps` | ✅ | ❌ Removed | Use vLLM's internal scheduler |
| `--max-seq-len-to-capture` | ✅ | ❌ Removed | CUDA graph capture is automatic |
| `--enable-chunked-prefill` | Bool flag | ❌ Changed | Now always enabled, use `--max-num-batched-tokens` |
| `--block-size` | ✅ | ✅ | MLA models need `1` |
| `--gpu-memory-utilization` | ✅ | ✅ | Stable |

**Always check `vllm serve --help`** for the installed version's supported flags before launching.

### SGLang Environment Variables

| Variable | Value | Effect |
|----------|-------|--------|
| `SGLANG_USE_AITER=1` | Required | Enables aiter backend |
| `RCCL_MSCCL_ENABLE=0` | Recommended | Disables unstable MSCCL |
| `ROCM_QUICK_REDUCE_QUANTIZATION=INT4` | Recommended | Faster collectives |
| `SGLANG_TORCH_PROFILER_DIR=<path>` | For profiling only | Must set BEFORE server launch |

### Server Parameter Tuning Results

**Validated 2026-03-22:**

| Parameter | DSR1 (TP=8, CONC=64) | gpt-oss-120b (TP=8, CONC=64) | gpt-oss-120b (TP=1, CONC=4) | Recommendation |
|-----------|----------------------|-----------------------------|---------------------------|----------------|
| `--num-continuous-decode-steps 8` | **+13.9%** (controlled pair: 2229→2539) | ±0.5% (no effect) | **+9.7%** (725→795) | Model-dependent; always test |
| `--num-continuous-decode-steps 12` | Not tested | +0.1% | Not tested | Usually negligible |
| `--num-continuous-decode-steps 16` | Not tested | Not tested | Not tested | — |
| `--cuda-graph-max-bs` | N/A (already 64) | N/A (already 64) | **+35% at CONC=4** (was 4→16) | **Always match CONC** |
| `--enable-mixed-chunk` | Not tested | Not tested | Not tested | Promising for scheduling |

**Key insight 1**: TPOT often remains constant across decode-steps values. The throughput difference comes from **scheduling efficiency**, not kernel performance. But the effect is **highly model-dependent** — gpt-oss TP=8 showed 0% while DSR1 showed +13.9%.

**Key insight 2**: `--cuda-graph-max-bs` is the most impactful parameter when it's misconfigured. If max-bs < actual decode batch size, EVERY decode step re-launches ALL kernels individually. CUDA graph collapses this into 1 launch. The impact is largest for models with many small kernels.

## Model-Specific Knowledge

### DeepSeek-R1-0528

*(Validated 2026-03-21/22 on DeepSeek-R1-0528.)*

- 671B MoE with Multi-head Latent Attention (MLA), 256 experts, 8 active/token
- FP8 weights (block-scaled per_128x128), hidden_size=7168, 61 layers
- **torch.compile INCOMPATIBLE**: `--enable-torch-compile` causes CUDA graph capture failure: `get_heuristic_kernel_mla: cannot get heuristic kernel! q_type:fp8 kv_type:byte`. MLA + FP8 is not supported in torch.compile mode. Must run without torch.compile.
- **GPU breakdown (without torch.compile)**: 57.4% vendor CK/aiter C++, 42.6% aiter Triton kernels. GPU utilization only 73.2% (26.4% idle — scheduling overhead).
- **Kernel optimization — OPEN OPPORTUNITY (13% GPU time in aiter Triton)**:
  - `_gemm_a8w8_blockscale_kernel` (10.6% GPU): GEAK Strategy B micro-benchmark showed **+44%~+127%**, but E2E **-19.9%** due to register pressure (excessive register usage reduced occupancy) → REVERTED. The micro-benchmark gain proves the kernel IS suboptimal. The revert was caused by a specific, fixable problem (register pressure), NOT because the kernel is "already optimal."
  - `_fused_rms_fp8_group_quant_kernel` (3.6% GPU): GEAK Strategy B micro-benchmark 0% → DISCARDED.
  - **What was NOT tried**: (1) Register-constrained optimization (passing `num_warps`/`waves_per_eu` limits to prevent spilling), (2) OOB agents (Claude/Codex) with register-pressure awareness, (3) Strategy C selective compile, (4) aiter GEMM shape-level tuning via `gemm_a8w8_blockscale_tune.py`, (5) Verifying whether all DSR1 GEMM shapes (N,K with hidden_size=7168) have specialized gfx950 configs or fall back to generic defaults.
  - **Shape tuning gap**: The generic config (`gfx950-GEMM-A8W8_BLOCKSCALE.json`) has a single `"any"` entry (128×128×128 blocks). Specialized N-K configs exist for some shapes but NOT verified for all DSR1 projection shapes. Shapes without specialized configs use the generic default — almost certainly suboptimal.
  - **Tags**: `kernel-opt-open`, `register-pressure-fixable`, `shape-tuning-untested`, `strategy-c-untested`, `oob-untested`
- **Strategy B patching pitfall**: aiter source files have module-level variable definitions (`make_kernel_repr(...)`) between kernel functions. Naive function-end detection (by indentation) deletes these definitions, causing `NameError` at import. **MUST use Python AST** (`ast.parse` + `node.end_lineno`) for precise function boundary detection.
- **Server parameter tuning**: Controlled pair test (param_grid, same session, cuda-graph-max-bs=64): decode-steps 4→8 produced **+13.9%** (2229→2539 tok/s). The baseline 1813 tok/s was from an earlier session; comparing 1813→2537 cross-session overstates the isolated decode-steps effect.
- **Lesson**: GEAK Strategy B alone (unconstrained register optimization of vendor Triton) is insufficient for these kernels. The +44-127% micro-benchmark proves optimization potential EXISTS — the integration failed due to register pressure, not lack of headroom. **Next steps**: (1) OOB agents with register-constraint instructions, (2) aiter shape-level GEMM tuning for DSR1 shapes, (3) Strategy C selective compile. Do NOT treat this as "kernel-opt exhausted."
- **Strategy C (selective compile) NOT TESTED**: The results above only cover Strategy B (vendor Triton, unconstrained GEAK). Selective submodule compilation — compiling RMSNorm/SiLU/GELU individually via `torch.compile(module, fullgraph=False)` — was never attempted. OOB agents (Claude/Codex) were also never tried on these kernels. Both remain open optimization paths.
- SGLang is 9-10x faster than vLLM for this model on MI355X
- Recommended params: `--attention-backend aiter --chunked-prefill-size 196608 --max-prefill-tokens 196608 --kv-cache-dtype fp8_e4m3 --num-continuous-decode-steps 4`
- Baseline: 1813.09 tok/s at TP=8, CONC=64, ISL=1024, OSL=256
- Best output throughput: 3191.83 tok/s (CONC=64, ISL=1024, OSL=1024)
- Best total throughput: 15580.23 tok/s (CONC=64, ISL=8192, OSL=1024)
- **Report**: `optimization_report_dsr1_0528.md`

### Qwen3-30B-A3B

*(Validated 2026-03-26 on MI355X. Full multi-round loop supersedes 2026-03-23 RMSNorm-only headline.)*

#### Full multi-round optimization loop: +27.6% (TP=1) / +27.8% (TP=8)

**Round 1 — RMSNorm (Inductor Triton):** dual-pass → single-pass → **+15.2%** E2E @ CONC=4 vs baseline in the 2026-03-26 fair A/B (588.11 → 677.72 tok/s). *(Older 2026-03-23 sweep: +14.72% @ CONC=4, 596→684 tok/s.)*

**Round 2 — `triton_tem_fused_mm_0` (GEMM):** largest non-vendor kernel (~22% GPU); GEAK stacked on Round 1 → **+27.6%** cumulative @ CONC=4 vs baseline — matches summary **SGLang TP=1** row (endpoint after RMSNorm + `mm_0`).

**Round 3 — `topkGatingSoftmax` (HIP):** GEAK HIP on aiter C++; micro +2–4%, E2E +0.2% on top of Round 2 → **+27.8%** cumulative — matches summary **SGLang TP=8** row (full stack including HIP; reproduce at TP=8 when validating that endpoint).

**Takeaway:** Always run the **full multi-round loop** on all top non-vendor kernels. Do not stop after RMSNorm or skip `mm_0` because a prior GEMM attempt crashed.

**Cumulative stack (CONC=4, ISL=1024, OSL=256, fair A/B, same server + benchmark params):**

| # | Kernel | GPU% | E2E tok/s | Cumulative gain | TPOT (ms) | Status |
|---|--------|------|-----------|-----------------|-----------|--------|
| 0 | Baseline | — | 588.11 | — | 6.57 | baseline |
| 1 | + RMSNorm (5-ptr+3-ptr) | 5.9% | 677.72 | **+15.2%** | 5.67 | **KEEP** |
| 2 | + triton_tem_fused_mm_0 | 22.4% | 750.54 | **+27.6%** | 5.10 | **KEEP** |
| 3 | + triton_tem_fused_mm_1 | 7.3% | CRASH | — | — | **DISCARD** (GPU memory access fault) |
| 4 | + topkGatingSoftmax (HIP) | 4.8% | 751.47 | **+27.8%** | 5.08 | **KEEP** |
| 5 | triton_poi_fused_rope | 3.0% | — | — | — | **DISCARD** (GEAK no output after long run) |

*(Step table: one CONC=4 fair A/B chain; **+27.6%** / **+27.8%** align with summary **TP=1** / **TP=8** rows.)*

**Key insight:** `mm_0` was skipped in an earlier run due to “GEMM crashed” prior experience; the full loop recovered **~+10.7%** on top of RMSNorm alone.

**TPOT:** Baseline 6.57 ms → optimized 5.10 ms (**−22.4%**) — confirms kernel speedup, not batching artifacts.

#### torch.compile + Inductor + GEAK: workflow

1. **Enable torch.compile**: `--enable-torch-compile` — generates Inductor Triton kernels
2. **Profile in torch.compile mode**: list **all** kernels above ~3% GPU time (not only RMSNorm)
3. **Rank non-vendor kernels** and submit **top candidates in parallel** (RMSNorm, `mm_0`, `mm_1`, `topkGatingSoftmax`, etc.)
4. **Patch + benchmark each**; keep/revert by E2E + TPOT direction
5. **HIP kernels:** use SGLang GEAK image, full `.cu` in `files[].content`, prompt with exact dir (e.g. `/sgl-workspace/aiter/csrc/kernels/`) — no `find /` on NFS

**Why this works when direct kernel replacement doesn't:**

- Without torch.compile: SGLang uses aiter C++ kernels (not Triton) → GEAK has no Inductor targets
- With torch.compile: Inductor replaces many ops with Triton → GEAK can optimize → patch Inductor cache; HIP paths patch aiter sources

**Estimated end-to-end time (full loop):** ~45–60 minutes (vs ~30 min for RMSNorm-only path).

#### RMSNorm-only CONC sweep (2026-03-23 — superseded for headline %; still useful for CONC sensitivity)

| CONC | Baseline | RMSNorm-only optimized | E2E gain | TPOT change |
|------|----------|------------------------|----------|-------------|
| 1 | 224 tok/s | 211 tok/s | **−5.9%** | +6.4% |
| 4 | 597 tok/s | 684 tok/s | **+14.7%** | −13.1% |
| 16 | 1720 tok/s | 1700 tok/s | **−1.2%** | +1.4% |

Average across 9 configs (3 CONC × 3 ISL/OSL): **+2.20%**. Full-loop headline % uses CONC=4 fair A/B with baseline 588.11 tok/s (see table above).

#### Model specifics

- Architecture: Qwen3MoeForCausalLM, 128 experts, 8 active/token, hidden_size=2048, 48 layers
- Model size: ~30GB BF16 — fits on a single MI355X (TP=1)
- **torch.compile COMPATIBLE** — enables Inductor Triton + GEAK
- **GEAK targets (full loop):** RMSNorm single-pass; **triton_tem_fused_mm_0**; `mm_1` (reverted — crash); **topkGatingSoftmax** HIP

**SGLang results (torch.compile, full loop, CONC=4):**

- Baseline: 588.11 tok/s (ISL=1024, OSL=256)
- Full stack: 751.47 tok/s (**+27.8%** vs baseline); **+27.6%** after step 2 (`mm_0`) before final HIP delta
- Legacy RMSNorm-only @ CONC=4: 596.51 → 684.29 tok/s (**+14.72%**); peak ~1859 tok/s @ CONC=16 (high-CONC baseline-style)

**vLLM results (TP=1, vLLM v0.17.0 + torch.compile level 3):**

- Baseline CONC=4 (ISL=1024, OSL=256): ~534 tok/s
- RMSNorm GEAK: **~0% (±1%)** — within GPU-to-GPU measurement noise
- Inter-GPU variation ~±1% (GPU 0: 537.9, GPU 1: 534.4, GPU 4: 533.5 tok/s)
- Best throughput: 4209.90 tok/s @ CONC=64, ISL=1024, OSL=1024

**Why SGLang shows large gains but vLLM ~0% on RMSNorm-only tests?**

1. vLLM uses more C++ kernels (topkGating and act_and_mul each ~4–5%); RMSNorm only ~7.6% GPU time
2. vLLM Inductor level=3 full compilation already compresses headroom
3. SGLang's Triton pipeline concentrates more time in optimizable Inductor kernels

**TraceLens profile (SGLang):** GPU 98.2% compute. Top kernels: triton_tem_fused_mm 21.4%, CK MoE 26.3%, paged_attention 8.2%, RMSNorm 6.0%

- **Reports**: `optimization_report.md` (SGLang), `optimization_report_qwen3_30b_vllm.md` (vLLM)

### Kimi-K2.5

*(Validated 2026-03-23 on MI355X.)*

- Multimodal VLM: `KimiK25ForConditionalGeneration`, text backbone is **DeepseekV3ForCausalLM**
- MoE: 384 experts, 8 active/token; MLA: kv_lora_rank=512, q_lora_rank=1536
- INT4 compressed-tensors quantization (MoE experts only; attention + shared experts = BF16)
- 61 layers, hidden_size=7168, 64 attention heads, ~555GB model files
- **torch.compile INCOMPATIBLE**: MoE+MLA+INT4 is incompatible with torch.compile; no Inductor Triton kernels
- **Requires split attention backends**: `--decode-attention-backend triton --prefill-attention-backend aiter`
- **MUST set** `SGLANG_ROCM_FUSED_DECODE_MLA=0` (Docker default is 1, causes crash with triton decode)
- **Do NOT use** `--kv-cache-dtype fp8_e4m3` (MLA ASM kernel assertion failure: `q_scale.has_value()`)
- **Do NOT use** unified `--attention-backend aiter` (TP=8 → 8 heads/partition, aiter requires ≥16)
- **Do NOT use** unified `--attention-backend triton` with `SGLANG_ROCM_FUSED_DECODE_MLA=1` (ForwardMetadata unpack TypeError)
- **MLA head count constraint**: aiter MLA ASM kernel only supports `num_head_qo ∈ {16, 128}` or `(16*N, seqlen=1)` where N∈[2,8). With 64 heads and TP=8 → 8 heads/partition → incompatible. TP=4 works (16 heads) but is 4-5x slower due to doubled expert weights per GPU.
- **Config discovery saved 30+ minutes**: SGLang test suite at `/sgl-workspace/sglang/test/registered/amd/` contains validated launch configs for this model. Always check Step 0 in Phase 0 before manual attempts.

**SGLang results (TP=8, CONC=64):**

- Baseline: 323.95 tok/s (ISL=1024, OSL=256)
- GEAK fair A/B: **+0.81% throughput, -2.04% TPOT** (3 kernels patched: fused_moe GROUP_ALIGNED, GEMM software pipelining, attention pointer hoisting)
- Server tuning: decode-steps=8 → +2.4%
- Best throughput: 509.11 tok/s @ CONC=128 (with tuning)

**vLLM results (TP=4, vLLM v0.17.0, block-size=1 MLA constraint):**

- Baseline (correct parameters): 141 tok/s — significantly slower than SGLang
- Server param tuning (gpu-mem=0.90, max-num-seqs=256): **+84%** (141→259 tok/s)
- GEAK MoE WNA16 kernel: **+1.76%** (259→263.56 tok/s), TPOT -2.06%
- Total optimization: **+86.9%** (141→263.56) — mostly from server tuning; GEAK contributed 1.76%
- **P0 recommendation**: switch to SGLang (323.95 tok/s > 263.56 tok/s; SGLang is clearly faster)

**WARNING: Initial "+40.4%" SGLang GEAK claim was INVALID** — benchmark used different concurrency (128 vs 64) and server params. Fair A/B test only +0.81%.

**Lesson**: For MoE+MLA models with INT4 quantization and without torch.compile, GEAK has very limited impact (<2%). Server parameter tuning (+84% on vLLM) and framework selection (SGLang vs vLLM) are far more impactful.

- **Validated SGLang config:**
  ```bash
  export SGLANG_ROCM_FUSED_DECODE_MLA=0
  export SGLANG_USE_AITER=1
  python3 -m sglang.launch_server \
    --model-path $MODEL --tp 8 --trust-remote-code \
    --decode-attention-backend triton --prefill-attention-backend aiter \
    --mem-fraction-static 0.8 --cuda-graph-max-bs 64 \
    --num-continuous-decode-steps 4 --disable-radix-cache \
    --host 0.0.0.0 --port 8888
  ```
- **Validated vLLM config:**
  ```bash
  export VLLM_ROCM_USE_AITER=0
  export AITER_ENABLE_VSKIP=0
  export NCCL_MIN_NCHANNELS=112
  vllm serve $MODEL --tensor-parallel-size 4 --trust-remote-code \
    --max-model-len 4096 --block-size 1 \
    --gpu-memory-utilization 0.90 --max-num-seqs 256 --port 8000
  ```
- **Reports**: `kimi-k25-vllm-optimization-report.md`, `optimization_report_kimi_k25.md`

### gpt-oss-120b

*(Validated 2026-03-22.)*

- 117B params (5.1B active), MoE 128 experts, 4 active/token, GQA 64 heads / 8 KV heads
- MXFP4 quantization (MoE weights), BF16 (attention/embedding), 36 layers
- Sliding Window Attention (alternating `sliding_attention` / `full_attention` layers)
- **torch.compile INCOMPATIBLE**: SWA memory pool (`swa_memory_pool.py`) doesn't support CUDA graph capture with torch.compile. Error: `AttributeError: type object 'weakref.ProxyType' has no attribute '__torch_dispatch__'`
- **aiter attention backend NOT SUPPORTED**: gpt_oss requires `triton`, `trtllm_mha`, `fa3`, `fa4`, or `ascend` attention backend
- **FP8 KV cache INCOMPATIBLE** with SWA: CUDA graph capture fails with `PassManager::run failed`
- **GPU breakdown (TP=1)**: hipBLASLt 31.5%, CK-tile MoE 29.5%, PyTorch C++ elementwise 19.9%, aiter C++ 11.0%, Triton attention 6.8%
- **GPU utilization**: 94.9% (very high — little scheduling overhead at TP=1)
- **GEAK result: -17.6% REVERT**: Attention kernel pre-scale + precompute optimizations increased register pressure → lower occupancy. These Triton kernels are already MI3xx-tuned (waves_per_eu=1, kpack=2).
- **Best optimization path**: Server params + CUDA graph coverage, NOT kernel optimization

**Optimized config (TP=1, 3 cumulative optimizations):**

```bash
--num-continuous-decode-steps 8 \  # +9.7% isolated (725→795)
--cuda-graph-max-bs 16 \           # +35% isolated at CONC=4 (793→1073, measured on decode-steps=8 baseline)
# + num_warps=2 in decode_attention.py grouped kernel: ~+3% isolated
```

**NOTE on TP=8 (formal optimization report):** gpt-oss-120b at TP=8 showed **0% improvement** from any tuning — GPU utilization already 94.7%, decode-steps ±0.5%, no GEAK candidates (88.9% GPU time in CUDA graph, only 0.14% in Triton). Peak throughput: 9825 tok/s (CONC=64, ISL=1024, OSL=1024). The TP=1 results below are from preliminary experiments with different methodology. **Report**: `gpt-oss-120b_optimization_report.md`

**TP=1 concurrency sweep (decode-steps=8, cuda-graph-max-bs=16, ISL=1024, OSL=256):**

| CONC | Output tok/s | TPOT (ms) |
|------|-------------|-----------|
| 1 | 344 | 12.54 |
| 4 | 793 | 17.10 |
| 8 | 986 | 23.72 |
| 16 | 1745 | 23.59 |

**Validated CUDA graph coverage (TP=1, ISL=1024, OSL=256, decode-steps=8):**

| Config | CONC=4 | CONC=8 | CONC=16 |
|--------|--------|--------|---------|
| `cuda-graph-max-bs=4` | 793 tok/s | 986 tok/s | 1745 tok/s |
| `cuda-graph-max-bs=16` | **1073 tok/s (+35%)** | **1231 tok/s (+25%)** | **1917 tok/s (+10%)** |

**Note**: These baselines (793/986/1745) already include `--num-continuous-decode-steps 8`. The +35% is the **isolated** CUDA graph effect.

## Benchmark Fairness Case Study

### Kimi-K2.5: invalid "+40.4%" GEAK claim

**CRITICAL — validated 2026-03-23 on Kimi-K2.5**

**This is the most common source of false optimization results.** On Kimi-K2.5, a reported "+40.4% GEAK improvement" was invalidated because the GEAK benchmark accidentally used different server params AND benchmark params:

| What changed | Baseline | GEAK test | Impact |
|-------------|----------|-----------|--------|
| `--max-concurrency` | 64 | **None (=128)** | **Dominant factor** — doubles batching |
| `--num-continuous-decode-steps` | 4 | **8** | +2.4% (already validated) |
| `--mem-fraction-static` | 0.8 | **0.85** | Larger KV cache → bigger batch |
| `--disable-radix-cache` | Yes | **No** | Minor for random inputs |

**Smoking gun**: TPOT went from 172ms → 234ms (+36%). If kernels were truly faster, TPOT should DECREASE at the same concurrency. TPOT increased because the batch size effectively doubled.

**Rules to prevent this:**

1. **Save baseline server config to a file** (`/tmp/baseline_server_config.sh`) and source it when relaunching for GEAK testing
2. **ALWAYS include `--max-concurrency $CONC`** in benchmark commands — omitting it sends ALL num_prompts at once
3. **ALWAYS use `--num-prompts $((CONC * 3))`** consistently
4. **Verify TPOT direction**: True kernel improvement → TPOT decreases (or stays flat). TPOT increase = throughput from batching, not kernels
5. **Log both server config AND benchmark command** for every result JSON

**Template for fair A/B comparison:**

```bash
# Save baseline config (do this in Phase 2)
cat > /tmp/baseline_config.sh << 'EOF'
SERVER_ARGS="--decode-attention-backend triton --prefill-attention-backend aiter \
  --mem-fraction-static 0.8 --cuda-graph-max-bs 64 \
  --num-continuous-decode-steps 4 --disable-radix-cache"
BENCH_ARGS="--max-concurrency 64 --num-prompts 192 \
  --random-input-len 1024 --random-output-len 256"
EOF

# Reuse for ALL subsequent benchmarks (Phase 8, Phase 10 baseline comparison)
source /tmp/baseline_config.sh
```

## Common Pitfalls

**During skills execution (validated 2026-03-26):**

| Pitfall | What happened | Prevention |
|---------|--------------|------------|
| **Concurrency mismatch** | GEAK benchmark omitted `--max-concurrency`, sending 128 requests at once vs baseline's 64 | Always include `--max-concurrency $CONC` |
| **Server param drift** | GEAK server used decode-steps=8 while baseline used 4 | Save baseline config to file, source it for all re-tests |
| **GEAK output path confusion** | GEAK agent wrote output to input file path instead of output dir | Always check `geak_get_outputs` and download from the correct path |
| **benchmark_serving.py args** | Used `--output-file` (wrong) instead of `--save-result --result-dir --result-filename` | Check InferenceX script's `--help` first |
| **InferenceX path wrong** | Used `/shared_nfs/limou/InferenceX/` instead of user's path | Always use `$INFERENCEX_PATH` from Phase 1 setup |
| **Trace file too large** | Raw trace 349MB, 97% python_function events | Always filter before analysis (see Trace Size and Filtering in SKILL.md) |
| **Kernel analysis skipped** | Said "analyzed trace" but didn't actually parse kernel events | Must parse filtered trace JSON and build kernel breakdown table |
| **GEAK input: comments not code** | Submitted `.cu` with only comments/path references, GEAK had no source to optimize | Always embed full source in `files[].content` |
| **GEAK: wrong image** | Default ROCm image lacks framework code and headers; paths in GEAK prompt don't exist | Pass framework image (`GEAK_IMAGE_SGLANG` or `GEAK_IMAGE_VLLM`) for all kernel types |
| **GEAK HIP: `find /` on NFS** | GEAK agent ran `find \| grep` to locate source, hung ~35 min on NFS | Prompt must specify exact source dir and say "Do NOT search filesystem with find / or grep -r /" |
| **Skipped full loop** | Only optimized RMSNorm from prior experience, missed `mm_0` (~22% GPU) | **IRON RULE:** run full multi-round loop on all top-5 non-vendor candidates |
| **"GEAK 0%" treated as "kernel-opt exhausted"** | KB said "GEAK 0%" on DSR1 Triton GEMMs → agent skipped 13% GPU time as "already tuned". GEAK actually showed +44-127% micro but -19.9% E2E (register pressure). OOB agents were never tried. | **NEVER** equate one backend's failure with kernel-opt exhaustion. Check tags: `oob-untested`, `register-pressure-fixable`. Submit to OOB agents with failure diagnosis. |
| **"Already tuned" assumption without verification** | Agent read KB "already highly optimized by AMD engineers" and skipped GEMM shape tuning. Never verified whether DSR1's specific shapes had specialized gfx950 configs or fell back to generic defaults. | Always verify shape-level tuning: check `aiter/ops/triton/configs/gemm/` for model-specific N-K configs. Run `gemm_a8w8_blockscale_tune.py` for untuned shapes. |

### Kernel-Opt Strategy Guide

**CRITICAL: A GEAK failure ≠ "kernel-opt exhausted."** GEAK is ONE backend using ONE
strategy. OOB agents (Claude, Codex) are **equal or better** than GEAK for kernel
optimization, especially when: (a) register pressure caused GEAK's E2E regression (OOB
agents can be instructed to constrain registers), (b) multi-file analysis is needed
(Strategy D/E), (c) Triton kernel fusion is needed (Strategy F). A GEAK failure is
information about GEAK's approach, not about the kernel's optimization potential.

When KB entries report prior kernel-opt results, verify WHICH strategy AND backend were tested:

| Strategy | Target | Best backend | Typical yield | Status |
|----------|--------|:------------:|:-------------:|--------|
| **A** (full torch.compile) | Inductor Triton kernels | GEAK, OOB, LLM | +14-27% | Validated on dense models |
| **B** (no compile, unconstrained) | Framework/vendor Triton | GEAK, OOB, LLM | ≤1.76% on vendor (GEAK); **OOB untested** | GEAK validated; OOB gap |
| **B'** (no compile, register-constrained) | Same as B + explicit register/occupancy limits | **OOB preferred** | Est. +2-10% | NEW — addresses DSR1 register-pressure failure |
| **C** (selective compile) | Norm/activation → Inductor | GEAK, OOB | Untested on most MoE | Gap: `strategy-c-untested` tag |
| **D** (call stack patching) | aiter dispatch, FP8 bypass | **OOB Claude/Codex** | +1-5% est. | Validated on GLM-5 (FP8 bypass) |
| **E** (framework scheduling) | Batch scheduler, overlap | **OOB Claude** | +5-16% est. | Validated on GLM-5 (mixed-chunk) |
| **F** (kernel fusion) | Multi-kernel sequences | GEAK + OOB | +2-5% est. | Identified on gpt-oss-120b |
| **G** (GEMM shape tuning) | aiter Triton GEMM configs | aiter tuning tool | +1-5% est. | Validated on GLM-5; untested on DSR1 |

**When prior kernel-opt result was "0%" or "reverted":**
1. Check if the failure was **strategy-specific** (e.g., register pressure) — try with constraints
2. Check if **OOB agents were tried** — if not, they are an independent opportunity
3. Check if **shape-level GEMM tuning** was run — untuned shapes use generic configs
4. Look for tags: `kernel-opt-open`, `oob-untested`, `shape-tuning-untested`, `register-pressure-fixable`
5. Do NOT zero kernel-opt scores until ALL applicable strategies AND backends have been tested

### Call Stack Optimization Targets

Known optimization surfaces in the call stack above vendor kernels:

**aiter dispatch layer** (`/sgl-workspace/aiter/aiter/ops/`):
- `fused_moe.py` — kernel variant selection, FP8 bypass logic, config lookup
- `gemm.py` — GEMM dispatch, CSV config lookup, shape-specific routing
- Python wrappers that translate framework calls → CK/hipBLAS invocations

**Framework scheduling layer** (SGLang `srt/managers/`):
- `schedule_batch.py` — batch formation, prefill/decode mixing
- `tp_worker.py` — tensor-parallel worker, forward pass orchestration
- `tp_worker_overlap_thread.py` — compute/communication overlap
- Key flags: `--enable-mixed-chunk`, `--num-continuous-decode-steps`

**Communication layer** (NCCL/RCCL + custom):
- AiterCustomAllreduce — shared-memory fast path, INT4 quantization
- NCCL channel tuning — `NCCL_MIN_NCHANNELS`, topology awareness
- Piecewise CUDA graphs — currently disabled on ROCm (`disable_piecewise_cuda_graph = True`)

**Per-model call stack breakdown (from profiling):**

| Model | T1 Triton | T2 aiter/CK | T3 Scheduling | T4 Comm | T5 Compiled |
|-------|:---------:|:-----------:|:-------------:|:-------:|:-----------:|
| Qwen3-30B (compile) | 35% | 5% | 5% | 5% | 50% |
| DSR1-0528 (no compile) | 3% | 70% | 5% | 15% | 7% |
| gpt-oss-120b | 7% | 30% | 5% | 5% | 53% |
| GLM-5-FP8 | ~3% | ~25% | 5% | 45% | ~22% |
| Kimi-K2.5 | ~5% | ~55% | 5% | 15% | ~20% |

These breakdowns show that T2+T3+T4 represent 30-85% of GPU time on non-compile
models — a massive optimization surface that was previously ignored.

## Per-Layer Kernel Sequence Analysis

*(Validated 2026-03-22 on gpt-oss-120b.)*

**When neither torch.compile nor GEAK can optimize individual kernels, analyze the kernel SEQUENCE to find fusion opportunities.** Extract the per-layer kernel timeline from the trace:

```python
# Sort kernel events by timestamp, find repeating patterns
kernels_timeline = sorted(
    [(e['ts'], e['dur'], e['name'][:80]) for e in trace['traceEvents'] if e.get('cat') == 'kernel'],
    key=lambda x: x[0]
)
# Print ~30 consecutive kernels to see one decoder layer pattern
for i, (ts, dur, name) in enumerate(kernels_timeline[len(kernels_timeline)//2:][:30]):
    print(f'[{i:3d}] {dur:6.1f}us  {name}')
```

**gpt-oss-120b per-layer kernel sequence (22 kernels, ~170us decode):**

```
# Attention phase
[ 0] GEMM: QKV proj                 15us  (vendor)
[ 1] rotary_embedding                4us  (aiter C++)
[ 2] index_elementwise × 3           4us  ← KV cache ops, FUSIBLE
[ 3] elementwise × 3                 4us  ← type casts, FUSIBLE
[ 4] _fwd_grouped_kernel_stage1      6us  (Triton GQA)
[ 5] _fwd_kernel_stage2              5us  (Triton reduce)
# Post-attention
[ 6] GEMM: O proj                   15us  (vendor)
[ 7] add_rmsnorm_quant               5us  (aiter, already fused)
# MoE phase
[ 8] GEMM: Router                   18us  (vendor)
[ 9] topkGatingSoftmax               5us  (aiter C++)
[10] Fill + elementwise              8us  ← MoE output init, FUSIBLE
[11] MoeSorting                      5us  (CK vendor)
[12] MoeFlatmm × 2                  45us  (CK vendor)
[13] elementwise                     5us  ← routing weight scale, FUSIBLE
[14] add_rmsnorm_quant               5us  (aiter, already fused)
```

**Identified fusion opportunities (total ~5% E2E potential):**

1. **KV cache ops** (items 2-3): 6 small kernels between QKV proj and attention = 24us/layer. aiter has `fused_qkv_split_qk_rope.py` as base.
2. **MoE routing prep** (items 9-10): topkGatingSoftmax + Fill + cast = 13us/layer. aiter has `moe_routing_sigmoid_top1_fused.py` for top-1 (needs extension for top-k).
3. **MoE output scaling** (item 13): routing weight multiply after MoE GEMM2 = 5us/layer. Could fuse into MoE epilogue.

## Attention Kernel Parameter Tuning

*(Validated 2026-03-22 on gpt-oss-120b.)*

When GEAK structural rewrite is not applicable (e.g., vendor binary kernels), **Triton launch parameter tuning** can still help:

| Parameter | Original | Tuned | Effect |
|-----------|----------|-------|--------|
| `num_warps` (grouped GQA stage1) | 4 | **2** | +4.1% at CONC=4 (more registers/warp for kv_group=8) |

**When to try num_warps=2**: GQA decode with small head_dim (≤64) and moderate kv_group_num (4-16). Fewer warps = more registers per warp = better for register-heavy attention kernels with BLOCK_H × BLOCK_DV accumulator.

**How to apply**: Edit `_decode_grouped_att_m_fwd()` in `sglang/srt/layers/attention/triton_ops/decode_attention.py`, change `num_warps=4` → `num_warps=2`. Clear `__pycache__` and restart.

---

## GLM-5-FP8 (MoE + MLA + NSA, 8x MI355X)

*(Validated 2026-03-26. This is the flagship example of backend exploration outperforming parameter sweeps.)*

### Model Architecture
- **78 layers**, 256 routed + 1 shared expert, topk=9, FP8 blockscale (per_1x128)
- **Native Sparse Attention (NSA)** — unique attention mechanism with separate prefill/decode backends
- **MLA (Multi-head Latent Attention)** — compressed KV projections
- hidden_size=6144, moe_intermediate_size=2048, intermediate_size=12288
- On TP=4: 156 all-reduces per forward pass (2 per layer × 78 layers)

### Profiling Breakdown (TP=4, baseline)
- **49.2%** GPU idle time (communication bubbles)
- **44.9%** NCCL/RCCL all-reduce communication
- **~6%** actual compute
- Communication uses `AiterCustomAllreduce` (AMD shared-memory fast path), not vanilla NCCL
- `QuickAllReduce` with INT4 quantization also available as complement

### Backend Exploration Results (the key insight)

| Optimization | Type | Individual Gain | Notes |
|---|---|---|---|
| `--nsa-decode-backend aiter` | Backend switch | **+3.1%** | Switches NSA decode kernel from tilelang to aiter CK |
| `--enable-mixed-chunk` | Scheduling mode | **+2.9%** | Overlaps prefill/decode in same forward batch |
| **Combined** | **Both together** | **+16.2%** | **Super-linear synergy** |

### Parameter Sweep Results (for comparison — much smaller gains)

| Parameter | Gain | Notes |
|---|---|---|
| `--num-continuous-decode-steps 64` | +0.7% | Higher decode steps, marginal |
| `NCCL_MIN_NCHANNELS=32` | +0.7% | More NCCL channels, marginal |
| `--moe-runner-backend triton` | +0.5% | Triton MoE ≈ aiter CK |
| `--enable-fused-moe-sum-all-reduce` | 0% | **No effect on ROCm aiter path** |
| `--mem-fraction-static 0.90` | +0.3% | 0.85 already sufficient |
| `--enable-aiter-allreduce-fusion` | +0.2% | Fuses AR with RMSNorm, too small |
| `SGLANG_ROCM_FUSED_DECODE_MLA=1` | -0.5% | Slightly worse |

### Why the synergy is super-linear

`--enable-mixed-chunk` changes scheduling: instead of strict prefill-then-decode phases, decode tokens get mixed into prefill batches. This means **more tokens per forward pass** during decode. `--nsa-decode-backend aiter` switches to a faster NSA decode kernel. Together: more tokens × faster per-token processing = compounding effect. Each optimization amplifies the other.

### Kernel Tuning Details

**Dense GEMM tuning** (aiter a8w8_blockscale):
- Tuned 40 shapes: M=[1,2,4,...,128], N/K combinations for MLP (6144×6144, 6144×3072) and attention projections (6144×4096, 2624×6144, 128×6144)
- Tool: `/sgl-workspace/aiter/csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py`
- Output merged into: `/sgl-workspace/aiter/aiter/configs/a8w8_blockscale_tuned_gemm.csv`
- Direct impact: <1% (only 3 dense layers out of 78 MoE layers)

**Fused MoE tuning** (aiter FMoE):
- Tuned 11 shapes: token=[1..1024], model_dim=6144, inter_dim=512 (2048/TP4), expert=257, topk=9
- QuantType: per_1x128 (FP8 blockscale), fp8_e4m3fn
- Tool: `/sgl-workspace/aiter/csrc/ck_fused_moe/fmoe_tune.py`
- Output merged into: `/sgl-workspace/aiter/aiter/configs/tuned_fmoe.csv`

**FP8 bypass removal** (code change):
- File: `/sgl-workspace/aiter/aiter/fused_moe.py` line 785
- Original: `if problem_type == bypass_type and (token * topk) <= 128: return False` — skipped tuned kernels for small FP8 blockscale batches
- Fix: `def use_cfg(): return True` — always use tuned kernels

### What doesn't work on ROCm/MI355X

| Feature | Status | Why |
|---|---|---|
| `--enable-mscclpp` | **Not applicable** | `PyMscclppCommunicator` only supports world_size=[8,16], not TP=4 |
| Piecewise CUDA graphs | **Disabled** | `disable_piecewise_cuda_graph = True` when `is_hip()` — no compute/comm overlap |
| SBO/TBO overlap | **Not applicable** | Requires FlashInfer/DeepGemm backends (CUDA-only) |
| `--enable-fused-moe-sum-all-reduce` | **No effect** | The aiter `fused_moe` CK path handles topk reduction internally; this flag only affects the Triton MoE path (confirmed from code in `fused_moe_triton/fused_moe.py`) |
| `--enable-torch-compile` | **Not tested** | MoE+MLA+NSA likely incompatible (same as DSR1) |

### Recommended Launch Config

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

Environment variables:
```bash
export SGLANG_USE_AITER=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export SAFETENSORS_FAST_GPU=1
```

### Results Summary

| Config | Total tok/s | Per-GPU | TPOT | Improvement |
|---|---|---|---|---|
| Baseline (TP=4, ds=8, conc=64) | 1,403 | 351 | 84.4 ms | — |
| Optimized (TP=4, conc=64) | **1,630** | **408** | **72.8 ms** | **+16.2%** |
| High-conc (TP=4, conc=128) | 2,206 | 551 | 107.4 ms | +57.2% (latency tradeoff) |
| Previous DP=2/TP=4 (unopt) | 2,794 | 349 | 84.9 ms | +99.1% (DP scaling) |
| Projected DP=2/TP=4 (opt) | ~3,244 | ~406 | ~73 ms | ~+131% |

Artifacts: `inference_optimization/results/glm5_optimization/`

---

## Dream System — Cross-Run Learning

The dream system consolidates learnings across optimization runs to build institutional
knowledge that improves future runs. Inspired by Claude Code's Auto Dream protocol.

### How It Works

```
Run N completes → Dream consolidation → KB updated → Run N+1 starts with richer KB
                                                       ↓
                                                   KB warm-up injects model playbook
                                                       ↓
                                                   Agent starts with prior knowledge
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| Dream protocol | `actions/dream.md` | 4-phase consolidation (Orient, Gather, Consolidate, Prune) |
| Consolidation engine | `scripts/dream.py` | Programmatic: group entries, detect contradictions, merge duplicates |
| Playbook generator | `scripts/kb_summary.py` | Model-specific playbook for prompt injection at KB warm-up |
| Checkpoint | `scripts/checkpoint.py` | State save/load with atomic writes for crash recovery |

### Cross-Model Patterns Discovered

These patterns emerged from consolidating KB entries across multiple model runs:

1. **Backend switches outperform parameter sweeps** — validated on GLM-5-FP8 (+16.2%
   backends vs <1% params), Kimi-K2.5 (SGLang 2.3x faster than vLLM). Always explore
   backends first.

2. **Combination synergies can be super-linear** — GLM-5-FP8: two +3% backends → +16.2%
   combined. Always test winners together.

3. **torch.compile viability determines optimization strategy** — Compile-compatible
   models get Strategy A (Inductor Triton → GEAK, +14-27%). Non-compile models need
   Strategies B-F (call stack optimization, framework tuning, fusion).

4. **Vendor kernels have deep Python call stacks** — even when CK/aiter compiled
   kernels dominate GPU time (60-70%), the Python dispatch logic above them is
   optimizable. Strategy D (dispatch patching) and Strategy E (scheduling) target this.

5. **CUDA graph coverage is the most impactful single parameter** — gpt-oss-120b:
   cuda-graph-max-bs change → +35% at CONC=4. Always match to actual decode batch size.

### Model Playbooks

Generated automatically by `scripts/kb_summary.py` and stored in `kb/playbooks/`.
The agent injects the relevant playbook at KB warm-up (Step 4) for immediate context.

```bash
# Generate playbook for a model
python3 $SKILL_ROOT/scripts/kb_summary.py \
    --model "DeepSeek-R1-0528" \
    --kb-path "$SKILL_ROOT/kb/entries.jsonl" \
    --output "$SKILL_ROOT/kb/playbooks/deepseek_r1_0528.md"
```
