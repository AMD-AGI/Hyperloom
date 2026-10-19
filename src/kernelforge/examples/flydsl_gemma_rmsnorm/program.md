# Program: optimize Gemma RMSNorm (FlyDSL)

**GPU**: gfx950 (AMD Instinct MI355X) — adjust `--gpu-target` for other hardware
**Backend**: flydsl

## Objective

Optimize `gemma_rmsnorm` in `rmsnorm_kernel.py` for maximum throughput while
keeping the result numerically correct. The loop gates correctness on an SNR
threshold (40 dB) against a Torch oracle before it ever benchmarks a change.

This is a real hot kernel: Gemma-4-26B-A4B-it RMSNorm at shape `(64, 2816)`, BF16.

## What the kernel does

```text
inv_rms = rsqrt(mean(x^2, dim=-1) + 1e-6)     # fp32 reduction per row
out     = x * inv_rms * (1 + weight)          # Gemma's (1 + weight) scale
```

Note the `(1 + weight)` form — this is the Gemma variant, not plain RMSNorm. The
shipped baseline is eager Torch: correct, but it materializes full-size fp32
temporaries and launches a separate kernel per elementwise step.

## Optimization ideas (not prescriptions — measure everything)

- The whole working set is ~720 KiB, so this is memory- and launch-bound. Fusing
  everything into one kernel launch is likely the dominant win.
- `hidden = 2816 = 64 x 44 = 256 x 11`. That factorization decides how cleanly a
  row maps onto a wave or a workgroup, and whether a cross-wave LDS reduction is
  needed at all. Try one wave per row versus one workgroup per row.
- Reduce the sum of squares in fp32 (required for the stability stage) but try
  vectorized 64-bit / 128-bit loads for the bf16 data.
- The second pass needs `x` again: keeping it in registers between the reduction
  and the scaling avoids re-reading it from HBM, at the cost of register pressure.
  Measure both.
- At this size the CUDA/HIP graph replay floor may dominate; if timings stop
  moving, say so rather than chasing noise.

## Modification rules

1. Keep the public `gemma_rmsnorm(x, weight, out, stream_handle=...)` signature
   unchanged — the driver imports it. Write the result into the caller's `out`.
2. **Honor `stream_handle`.** FlyDSL launches on the stream you give it. The
   driver passes the currently active stream, which under graph capture is the
   private capture stream. Launch with
   `stream=fx.Stream(stream_handle)` (falling back to the current stream when the
   handle is `None`). Launching on the default/NULL stream produces an EMPTY graph
   whose replay looks impossibly fast; the harness detects this and rejects it.
3. Keep the kernel in FlyDSL; do not rewrite it in Triton, HIP, or CUDA.
4. Do NOT import or call AITER — the driver asserts it was never loaded.
5. Do NOT edit `driver.py` or `graph_harness.py` — they are the measurement
   oracle and timing harness (the loop blocks edits to them). Optimize the
   kernel, not the measurement.
6. Stay replay-safe: no host syncs on device values (no `.item()` / `.cpu()` /
   data-dependent Python branches). Cache any JIT-compiled module across calls so
   compilation happens during warmup, not inside graph capture.
7. Build/run/verify your change yourself before finishing; the loop then runs a
   canonical correctness + benchmark pass and keeps the change only if it is
   correct AND faster than the current best.
