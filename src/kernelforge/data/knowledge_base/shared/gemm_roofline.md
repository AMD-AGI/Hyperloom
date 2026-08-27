# GEMM Roofline Analysis

Extracted from `inference-testing/docs/uplift_plan/roofline/gemms.md`
(N. Holmberg, M. Hartikainen).

## The formula

For `C[M,N] = A[M,K] × B[K,N]`:

```
total_flops    = 2 × M × N × K
bytes_accessed = (M*K + K*N + M*N) × dtype_size     # same-dtype approx

compute_time   = total_flops / peak_tflops
memory_time    = bytes_accessed / peak_bandwidth
roofline_time  = max(compute_time, memory_time)
efficiency     = roofline_time / measured_time × 100%
```

If `compute_time > memory_time` → compute-bound. Else → memory-bound.
Efficiency above 100% is nonsense; below 80% warrants tuning.

## Hardware table (MAF, not vendor peak)

**Use Max-Achievable FLOPs (~50% of peak)** for the roofline denominator, not
the vendor peak FLOPS number. This is the realistic upper bound; citing peak
makes every kernel look bad. (Ref: AMD ROCm blog on MAF vs peak.)

| GPU    | HBM BW   | BF16 MAF (TFLOPs) | FP8 MAF (TFLOPs) |
|--------|----------|-------------------|------------------|
| MI300X | 5.3 TB/s | ~650              | ~1300            |
| MI325X | 6.0 TB/s | ~650              | ~1300            |
| MI355X | 8.0 TB/s | ~1260             | ~2500            |

MI355X doubles both BW and compute vs MI325X — the compute/memory ratio stays
roughly constant, so the **same kernel shape hits the same bound on both**;
memory-bound stays memory-bound, compute-bound stays compute-bound.

## Worked examples (Llama-70B QKV on MI325X)

**Prefill, seq=2048** — `[2048, 8192] × [8192, 24576]`:
```
FLOPs  = 825 GFLOPs    bytes = 504 MB (bf16)
compute_time = 825 GFLOPs / 650 TFLOPs = 1269 μs
memory_time  = 504 MB    / 6.0 TB/s   = 84 μs
→ compute-bound
```

**Decode, batch=32** — `[32, 8192] × [8192, 24576]`:
```
FLOPs  = 12.9 GFLOPs   bytes = 404 MB (bf16)
compute_time = 12.9 GFLOPs / 650 TFLOPs = 0.02 μs
memory_time  = 404 MB     / 6.0 TB/s   = 67 μs
→ memory-bound (by a factor of 3000x)
```

Decode GEMMs are **always** memory-bound — weight bytes dominate, FLOPs are
trivial. This is why vLLM serving is dominated by HBM traffic.

## Recommended workflow

1. Run inference with varying concurrency/seqlen.
2. Collect traces; extract GEMM shapes + durations.
3. Rank by total E2E time contribution (not single-kernel time).
4. Compute roofline efficiency for the top contributors only.
5. File tuning tickets for shapes below 80% efficiency OR underperforming vs
   NVIDIA equivalents.

**Do not roofline every GEMM in a trace** — decode has thousands of shapes,
most contribute < 0.1% to E2E. Amdahl-prioritize first.

## Efficiency target
- ≥80%: good, leave it
- 60–80%: investigate tuning
- <60%: file a ticket

## Gotcha: bytes formula is an approximation
The `(M*K + K*N + M*N) × dtype` formula omits bias, output accumulator dtype
differences, and any scale/quant overhead. Accurate for first-order roofline;
for fp8-with-fp32-scales workloads the scale bytes can matter at small M (see
MoE roofline analysis).
