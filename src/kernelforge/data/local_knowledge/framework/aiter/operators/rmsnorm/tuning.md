---
title: rmsnorm — tuning (bandwidth-bound row reduction on CDNA3/4)
kind: technique
operator: rmsnorm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/triton/normalization/rmsnorm.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# rmsnorm — tuning

RMSNorm is **bandwidth-bound**: the achievable ceiling is `(read x + write y) / HBM_BW`. On MI300X
(~5.3 TB/s) a bf16 `[16384, 8192]` norm moves ~0.5 GB → ~95 µs is the floor. Tuning = **get to that
floor**, i.e. saturate bandwidth and don't waste it. Three regimes, three strategies.

## 1. The decision: single-pass vs two-pass (set by N)
The aiter Triton impl keys the whole strategy off whether the row fits an LDS-sized block:
```python
def block_size(x):  return min(65536 // x.element_size(), next_power_of_2(x.shape[1]))   # 64KB / elt
def use_blocked(x): return x.shape[1] > block_size(x)                                     # N > block
```
- **N ≤ block** (bf16: N ≤ 32768 — i.e. *all* real hidden dims 4096/5120/8192): **single-pass**, whole
  row in registers/LDS, one reduction. This is the common case.
- **N > block**: **two-pass blocked** — pass 1 streams the row accumulating `Σx²` (fp32), compute
  `rsqrt`, pass 2 re-streams to normalize. Costs a 2nd read of `x`; only needed for exotic N.

## 2. Grid: persistent (decode) vs row-per-program (prefill)
```python
def num_programs(x): return min(x.shape[0], get_num_sms())   # persistent: ≤ 304 on MI300X
grid = (num_programs(x),)
# kernel: for row in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2): ...  # persistent loop
```
- **prefill (M ≥ 1024 rows)**: rows alone fill the chip → simplest is one program per row; the persistent
  loop also works (each CU sweeps a strided set of rows).
- **decode (M ≤ 256 rows)**: `num_programs = min(M, num_sms)` under-fills the 304 CUs / ≥1024-WG target,
  so RMSNorm decode is **pure latency** — there is no work to hide it. Keep the kernel minimal; this is
  the strongest argument to **fuse** it away (into residual-add / quant) rather than tune it.

## 3. Vectorized loads — the actual lever
Bandwidth comes from wide, coalesced, aligned access:
- **Triton**: `BLOCK_SIZE = next_pow2(N)` so consecutive lanes hit consecutive addresses → backend emits
  `global_load_dwordx4` (128-bit). `num_warps = min(max(BLOCK_SIZE//256,1), 8)` (aiter/Triton-tutorial
  heuristic) — memory-bound, so **2–4 warps** is the sweet spot, not 8 (8 → VGPR spill, 3–5× slower).
- **HIP/vLLM**: read `x` as `float4`/`__half2`×N — the vLLM #22602 vectorization (aligned vector I/O +
  shared-mem row cache) cut a `[16384,1024]` fp16 norm from **105.9 µs → 42.6 µs** (~2.5×) by traffic
  reduction alone (NVIDIA-measured in the PR; same principle on CDNA).
- Use `cache_modifier=".cg"` on the `x` load (aiter does): bypass L1, since the row is read once and
  never reused — frees L1 for weights.

## 4. The knob table
| knob | range | rmsnorm setting | why |
|---|---|---|---|
| `num_warps` | 1,2,**4**,8 | 2–4 (`min(max(BLOCK//256,1),8)`) | memory-bound; 8 spills VGPRs |
| `num_stages` | **1**,2 | 2 (block pipelining), 1 (single-pass) | overlaps loads, not compute |
| `BLOCK_SIZE` | next_pow2(N) | full row if ≤ 32768 (bf16) | full wave64 reduce, 128-bit loads |
| grid | row-per-prog / persistent | `min(M, num_sms)` | fill 304 CUs without over-launch |
| `waves_per_eu` | 0,2,3,4 | 3–4 | norm is VGPR-light → push occupancy |
| `cache_modifier` | `.cg` on x | `.cg` | x read-once, don't pollute L1 |

## 5. The fp32-accumulate cost (don't skip it)
`Σx²` **must** accumulate in fp32 (`x.to(tl.float32)`); the load is bf16 but the reduce is fp32. This is
free on bandwidth (the convert is in-register) and mandatory for accuracy at N≥4096 — see
[numerics.md](numerics.md).

## 6. When tuning is pointless → fuse
If isolated RMSNorm is already at ~90% of `2·bytes/BW`, there is nothing left. The win is structural:
fold the residual-add (fused_add_rmsnorm) and the downstream fp8 quant (fused_norm_quant) into
the same single read+write — see [fusion.md](fusion.md).
- **PTPC-FP8** (per-token-per-channel) fused quant is **up to 2.5× vs naive** on MI300X.
- vLLM **ActivationFusionPass +8% throughput**; the `rocm_aiter_fusion` RMSNorm+quant pass stitches the norm
  into the fp8 GEMM input. ⚠ Inductor torch-op quant can now **auto-fuse some patterns**, so the standalone
  RMSNorm+quant / SiLU+quant passes are **obsolete except custom-op cases** (attention, collectives, sub-byte
  quant) — verify the pass is still doing work in the compiled graph before relying on it.

## Sources
- block_size / use_blocked / num_programs / persistent loop / `.cg` loads / num_warps heuristic: `aiter/ops/triton/normalization/rmsnorm.py` (+ `aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py`).
- Memory-bound num_warps=2/4, next_pow2 block, 128-bit loads: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html.
- ≥1024 grid / 304 CUs / 5.3 TB/s HBM: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html.
- Vectorized RMSNorm 2.5× (traffic reduction): https://github.com/vllm-project/vllm/pull/22602.
- PTPC-FP8 up to 2.5× vs naive (MI300X): https://blog.vllm.ai/2025/02/24/ptpc-fp8-rocm.html.
- vLLM Inductor fusion passes (ActivationFusionPass +8%; torch-op quant auto-fuse obsoletes some passes except custom-op): https://docs.vllm.ai/en/latest/design/fusions/.
