# Program: optimize fused residual-add + Gemma RMSNorm (HIP)

**GPU**: gfx950 (AMD Instinct MI355X) — adjust `--gpu-target` for other hardware
**Backend**: hip

## Objective

Optimize `fused_add_rmsnorm` in `fused_add_rmsnorm_kernel.py` for maximum
throughput while keeping the result numerically correct. The loop gates
correctness on an SNR threshold (35 dB) against a Torch oracle before it ever
benchmarks a change.

This is a real hot kernel: Gemma-4-26B-A4B-it fused add + RMSNorm at shape
`(64, 2816)`, BF16.

## What the kernel does

```text
summed       = x + residual
inv_rms      = rsqrt(mean(summed^2, dim=-1) + 1e-6)   # fp32 reduction per row
out          = summed * inv_rms * (1 + weight)        # Gemma's (1 + weight) scale
residual_out = summed                                 # next block consumes this
```

Note the `(1 + weight)` form — this is the Gemma variant, not plain RMSNorm. The
shipped baseline is eager Torch: correct, but it materializes full-size fp32
temporaries and launches a separate kernel per elementwise step.

## Optimization ideas (not prescriptions — measure everything)

- Working set is ~1.4 MiB total (2 reads, 2 writes of a 64x2816 bf16 tensor), so
  this is memory- and launch-bound. Getting it down to ONE kernel launch that
  reads `x`/`residual` once and writes both outputs once is the main prize.
- `hidden = 2816 = 64 x 44 = 256 x 11`. That factorization decides how cleanly a
  row maps onto a wave or a workgroup, and whether a cross-wave LDS reduction is
  needed. Try one wave per row versus one workgroup per row.
- Reduce the sum of squares in fp32 (required for the stability stage) but use
  vectorized loads for the bf16 data — e.g. `__hip_bfloat162` / `short4`-style
  128-bit accesses, given 2816 is divisible by 8.
- The `summed` values are needed twice (reduction, then scaling). Holding them in
  registers avoids a second HBM read at the cost of register pressure — measure
  both that and the LDS-staging alternative.
- Tune block size and `__launch_bounds__` / waves-per-EU for occupancy.

## Modification rules

1. Keep the public `fused_add_rmsnorm(x, residual, weight, out, residual_out)`
   signature unchanged — the driver imports it. Write results into the caller's
   `out` and `residual_out` buffers.
2. **Treat `x` and `residual` as read-only.** Production code updates `residual`
   in place, but the benchmark replays one captured invocation many times on the
   same memory; an in-place update would compound across replays and invalidate
   the measurement. That is why `residual_out` exists.
3. Implement the compute as a real HIP kernel (JIT-compiled, e.g. via
   `torch.utils.cpp_extension.load_inline`), not as a pure-Torch composition and
   not in Triton.
4. **Compile once, at import or first call, and cache the module at module scope.**
   Compilation cannot happen inside graph capture; the harness warms up before
   capturing, so a cached first-call compile is fine.
5. Do NOT import or call AITER — the driver asserts it was never loaded.
6. Do NOT edit `driver.py` or `graph_harness.py` — they are the measurement
   oracle and timing harness (the loop blocks edits to them). Optimize the
   kernel, not the measurement.
7. Stay replay-safe: no host syncs on device values (no `.item()` / `.cpu()` /
   data-dependent Python branches). Launch on the current stream
   (`c10::hip::getCurrentHIPStream()` / the stream Torch gives you) so the work is
   recorded into the graph.
8. Build/run/verify your change yourself before finishing; the loop then runs a
   canonical correctness + benchmark pass and keeps the change only if it is
   correct AND faster than the current best.
