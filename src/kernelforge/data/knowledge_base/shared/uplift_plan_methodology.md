# Uplift Plan Methodology

Extracted from `inference-testing/docs/uplift_plan/{overview,1_kernel_extraction,2_kernel_analysis}.md`.
The vLLM-on-AMD performance team's standard workflow for kernel-level uplift.

## The loop

1. Profile → PyTorch trace (`*.pt.trace.json[.gz]`)
2. Extract top kernels per phase (prefill, decode) via `uplift-plan` CLI
3. Amdahl-prioritize by total time contribution
4. Fuse / retile / roofline-check the top contributors
5. Reprofile → compare
6. Track in the shared Excel template (baseline → per-optimization OTPS column)

```
uplift-plan \
  --trace-path /path/to/trace.pt.trace.json.gz \
  --model Qwen/Qwen3-8B \
  --input-sequence-length 1024 \
  --output-sequence-length 27 \
  --output-dir ./uplift_results
```

Outputs `uplift_plan.md`, `prefill_kernels.csv`, `decode_kernels.csv`,
`config.json`.

## Amdahl is the only prioritization that matters

```
Total Time = duration_avg × occurrences
Speedup    = 1 / ((1 - P) + P/S)
```

Example: GEMM at 42% of decode time (P=0.42), locally 10% faster (S=1.10) →
E2E speedup = 1 / (0.58 + 0.42/1.10) = **1.041** (4.1%).

A 10× kernel-level win on a 1% contributor = 0.9% E2E. A 10% win on a 42%
contributor = 4.1% E2E. **Always rank by P before investing.**

## Prefill vs decode split for serving

Typical LLM with many output tokens:
- Prefill: 5–10% of E2E
- Decode: 90–95% of E2E

Conclusion: **focus decode.** Prefill GEMMs are compute-bound, decode GEMMs
are memory-bound — those are genuinely different optimization problems.

## Fusion pattern catalog

| Pattern | Example | Win mechanism |
|---|---|---|
| Elementwise chain | `fused_add` → `fused_mul` → `fused_view` | Eliminate launch overhead + HBM round-trips |
| Compute + norm | `gemm` → `fused_add_rmsnorm` | RMSNorm into GEMM epilogue; kill intermediate buffer |
| Attention | `paged_attention_QKV` → `paged_attention_reduce` | FlashAttention-style — usually already done, verify |
| Comm + compute | `cross_device_reduce_2stage` → `fused_add_rmsnorm` | Overlap AR with RMSNorm; pipelining |

Viability checklist before proposing a fusion:
- [ ] Data dependency: kernels touch related tensors
- [ ] Sequential: one finishes before the next starts
- [ ] Small individual durations (launch overhead dominates)
- [ ] Compatible ops (not hard-to-fuse semantics)
- [ ] Backend has a fusion primitive (CK, CK-Tile, Triton)

## Kernel name decoding

**CK (Composable Kernel)**:
`Cijk_Alik_Bljk_BBS_BH_..._MT16x16x256_..._WG16_4_2_LDSB1_MIWT1_4`
- `Cijk_Alik_Bljk` → matmul op pattern
- `MT16x16x256` → thread tile (M=16, N=16, K=256)
- `WG16_4_2` → workgroup config
- `LDSB0/LDSB1` → single vs double buffered LDS (double is usually faster)
- `MIWT1_4` → MFMA waves per tile

**Triton** (torch-inductor generated):
`triton_poi_fused_add_cat_index_select_mul_split_sub_unsqueeze_0`
- `poi` = pointwise → usually memory-bound
- `fused_*` list = ops rolled together; many ops = good fusion already
- trailing `_0` = unique id

**vLLM**:
`void vllm::reshape_and_cache_flash_kernel<__hip_bfloat16, ...>`
- namespace tells you the subsystem: `paged_attention_*`, `reshape_and_cache_*`,
  `cross_device_reduce_*`

## Red flags

- **High variance** (`dur_avg=10000, dur_median=1000`) — instability; re-run.
- **`cudaMemcpy` in hot path** — sync transfer; must be async or eliminated.
- **Same kernel twice at same timestamp** — redundant compute.
- **Many sub-5µs kernels with high count** — launch overhead dominates;
  strong fusion candidate.

## Efficiency thresholds (per the doc)
- ≥80%: good
- 60–80%: acceptable
- <60%: escalate (doc names Amir's team)

## Hardware gap analysis (MI355X vs B200)

Template tracks side-by-side OTPS:
```
Item         MI355X OTPS   B200 OTPS   MI355X vs B200
Current         2638.13      3982.88        0.662
Target          3917.91      3982.88        0.984
```

Gap = (B200 - MI355X) / B200 = 33.8%. Decompose into: GEMM efficiency,
HBM BW utilization, launch overhead, comm efficiency. Set realistic
incremental targets — not "match B200 tomorrow."

## Workshop materials (SharePoint, auth-walled)
- 2025-12-16: Uplift workshop Part 1 (instructions), Part 2 (tooling)
- 2026-02-11: MoE + GEMM roofline, advanced topics

Referenced but not readable from this box; linked in `overview.md`.
