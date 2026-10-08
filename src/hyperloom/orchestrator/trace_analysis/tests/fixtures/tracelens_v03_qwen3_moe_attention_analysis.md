# Qwen3 MoE LLM - MI355X Analysis

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Time | 1759.77 ms |
| Compute % | 23.30% |
| Idle % | 0.25% |
| Exposed Communication % | 0.00% |
| Top Bottleneck Category | InferenceAttention (impact_score 2.2) |

---

## Compute Kernel Optimizations

Findings from per-category kernel analysis (GEMM, SDPA, elementwise, etc.).

### Top Operations

| Rank | Category | Time (ms) | % of Compute Time | Ops | Potential improvement (time, E2E %) |
|------|----------|-----------|-------------------|-----|-------------------------------------|
| 1 | InferenceAttention | 45.862 | 11.19% | 1 | ~33.1 ms (1.9%) |

<!-- impact-begin kind=p_item category=inferenceattention low=1.88 mid=2.2 high=2.51 -->
### P1: Paged Attention forward kernel is memory-bound at 3.69% of HBM roofline (vLLM)

**Insight**: vLLM paged-attention forward at 3.69% HBM utilization.
**Action**: Tune GQA broadcast pattern in `_fwd_kernel`.
**Impact**: ~33.1-44.2 ms savings (1.9-2.5% of E2E).
<!-- impact-end -->

---

## Detailed Analysis

### Compute Kernel Insights

<a id="detailed-analysis-compute-p1"></a>
<!-- reasoning-candidate tier=compute rank=1 -->
#### 🔴 P1: Paged Attention forward kernel is memory-bound at 3.69% of HBM roofline

**Identification:** Single `vllm::unified_attention_with_output` operation accounting for 45.862 ms of GPU kernel time (2.61% of E2E). GQA pattern with 8:1 KV reuse, memory-bound at only 3.69% of the 8 TB/s HBM roofline. (source: `inferenceattention_metrics.json`)

**Data:**

| Operation |  Args  |            Kernel Path                  | Time (ms) | %E2E | Count |FLOPS/Byte| Efficiency | Bound | Dominant Kernel | Workload | Attention Pattern |
|-----------|--------|-----------------------------------------|-----------|------|-------|----------|------------|-------|-----------------|----------|-------------------|
| vllm::unified_attention_with_output | (1087,32,128) bf16<br>(1087,4,128) bf16<br>(1087,4,128) bf16<br>(1087,32,128) bf16<br>(0,) bf16 | vllm/model_executor/models/qwen3_moe.py(475): forward | 45.862 | 2.61 | 48 | 37.93 | 3.69% of 8.0 TB/s | memory-bound | `_fwd_kernel` (93.61%) | unknown | GQA (8:1) |

**Reasoning for Slowdown:** GQA 8:1 KV reuse pattern leaves HBM bandwidth underutilized at only 3.69% of the 8 TB/s peak.

**Resolution:** Restructure `_fwd_kernel` to load each KV head once per group of 8 query heads instead of reloading per query.

**Impact estimate:**
<!-- impact-begin kind=detail_estimate low=1.88 high=2.51 -->
- Low end impact_score: 1.88
- High end impact_score: 2.51
<!-- impact-end -->

---

## Appendix

(omitted for fixture compactness)
