# Inference Optimization Report — Qwen3-14B on MI355X

## Summary

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| **Output Throughput** | 490.93 tok/s | 510.04 tok/s | **+3.9%** |
| **TPOT** | 7.78 ms | 7.46 ms | **-4.1%** |
| **TTFT** | 97.94 ms | 97.28 ms | -0.7% |

## Environment

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-14B (dense, Qwen3ForCausalLM) |
| Hardware | 1x AMD Instinct MI355X (gfx950, CDNA4, 288GB HBM3e) |
| Framework | SGLang 0.5.9.dev20260324 |
| TP | 1 |
| torch.compile | Enabled (Inductor backend) |
| CUDA Graph | max-bs=4 |
| mem-fraction-static | 0.6 |
| Benchmark | ISL=1024, OSL=256, CONC=4, 12 prompts |

## Strategy: Dense Model + torch.compile (Strategy A)

Qwen3-14B is a dense transformer (no MoE, no MLA, no SWA). torch.compile works correctly, generating Inductor Triton kernels that are captured inside piecewise CUDA graphs. This is the highest-yield optimization path.

## Kernel Optimization (Parallel GEAK + LLM Race)

### Candidates Identified

From Inductor cache scan: **21 unique standalone kernels**, 2 high-priority dual-loop RMSNorm candidates:

| Kernel | Type | Files | xnumel | r0_numel | Priority |
|--------|------|-------|--------|----------|----------|
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 3-ptr RMSNorm | 8 | 1-4 | 5120 | HIGH |
| `triton_red_fused__to_copy_add_gemm_a16w16_mean_mul_pow_rsqrt_0` | 5-ptr fused residual+RMSNorm | 4 | 1-4 | 5120 | HIGH |

### Parallel Race Results

**Strategy: Submit to GEAK + 3 LLM models simultaneously.**

#### 5-ptr Fused Residual+RMSNorm

| Backend | Time | Single-pass? | Name OK? | Sig OK? | Verdict |
|---------|------|-------------|----------|---------|---------|
| claude-opus-4-6 | 34.0s | YES (0 loops) | YES | YES | **PASS** |
| claude-opus-4.5 | 24.2s | YES (0 loops) | YES | YES | **PASS** |
| gpt-4.1 | 7.9s | YES (0 loops) | YES | NO (changed signature) | FAIL |
| GEAK | >15 min | N/A | N/A | N/A | STUCK (pod scheduling) |

**Winner: claude-opus-4-6** (first passing result with correct signature)

#### 3-ptr RMSNorm

| Backend | Time | Single-pass? | Verdict |
|---------|------|-------------|---------|
| claude-opus-4-6 | 18.6s | YES (0 loops) | **PASS** |

### Optimization Applied

**Dual-loop elimination**: Both RMSNorm variants had two sequential loops where the second loop redundantly re-loaded data already available in registers. Since R0_BLOCK (8192) >= r0_numel (5120), the loop executes exactly once, so all computation was collapsed into a single straight-line block with zero loops.

**Impact**: 12 standalone files patched (4 for 5-ptr + 8 for 3-ptr), covering all batch-size variants used in the CUDA graph.

## Concurrency Sweep (Optimized, ISL=1024, OSL=256)

| CONC | Throughput (tok/s) | TPOT (ms) | TTFT (ms) |
|------|-------------------|-----------|-----------|
| 1 | 140.80 | 6.97 | 38.01 |
| 2 | 264.01 | 7.36 | 59.64 |
| **4** | **509.26** | **7.48** | **100.22** |
| 8 | 600.06 | 12.65 | 185.05 |

Note: CONC=8 exceeds `cuda-graph-max-bs=4`, causing decode steps >4 to skip CUDA graph. Setting `--cuda-graph-max-bs 8` would likely improve the CONC=8 numbers.

## ISL/OSL Sweep (Optimized, CONC=4)

| ISL | OSL | Throughput (tok/s) | TPOT (ms) | TTFT (ms) |
|-----|-----|-------------------|-----------|-----------|
| 512 | 128 | 511.65 | 7.38 | 58.50 |
| 1024 | 256 | 509.41 | 7.47 | 100.30 |
| 2048 | 512 | 508.59 | 7.55 | 165.29 |

Throughput is stable (~510 tok/s) across ISL/OSL — workload is decode-bound at CONC=4. TTFT increases linearly with ISL (expected for prefill).

## Key Findings

1. **Parallel GEAK+LLM race validated**: LLM returned optimized kernels in 7-34 seconds while GEAK was stuck in pod scheduling. The race strategy ensures we never wait on a single backend.

2. **LLM model diversity matters**: gpt-4.1 was fastest (7.9s) but changed the function signature (known failure mode). claude-opus-4-6 and claude-opus-4.5 both passed, confirming multi-model diversity catches per-model blind spots.

3. **Single-pass RMSNorm is the key optimization**: Eliminating redundant memory loads in the dual-loop RMSNorm pattern yields ~4% E2E throughput improvement. This is a consistent win on dense models where RMSNorm runs 48x per forward pass (once per layer).

4. **CUDA graph coverage is critical**: At CONC=4 (matching cuda-graph-max-bs), throughput nearly doubles from CONC=2 (264 -> 509 tok/s), demonstrating the importance of CUDA graph coverage.

## Files

| File | Description |
|------|-------------|
| `baseline_sglang_tp1_conc4_isl1024_osl256.json` | Baseline benchmark results |
| `optimized_sglang_tp1_conc4_isl1024_osl256.json` | Optimized benchmark results |
| `optimized_confirm.json` | Confirmation run (24 prompts) |
| `llm_claude-opus-4-6_5ptr.py` | Winning LLM kernel (5-ptr) |
| `llm_opus46_3ptr.py` | Winning LLM kernel (3-ptr) |
| `sweep_conc*_isl*_osl*.json` | Parameter sweep results |

## GEAK Task Reference

| Task ID | Status | Notes |
|---------|--------|-------|
| `8791b1fe-ec42-4361-8e49-06a0ce6804f4` | STUCK (running) | Pod scheduling failure on control-plane-prod |
