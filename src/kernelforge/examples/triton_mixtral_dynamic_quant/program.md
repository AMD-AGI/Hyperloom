# Program: optimize dynamic per-tensor FP8 quantization (Triton)

**GPU**: gfx950 (AMD Instinct MI355X) — adjust `--gpu-target` for other hardware
**Backend**: triton

## Objective

Optimize `dynamic_quant_fp8` in `quant_kernel.py` for maximum throughput while
keeping the result numerically correct. The loop gates correctness on an SNR
threshold (35 dB) against a Torch oracle before it ever benchmarks a change.

This is a real hot kernel: Mixtral-8x7B-Instruct-v0.1 activation quantization at
shape `(64, 4096)`, BF16 in, FP8 E4M3FN out.

## What the kernel does

```text
amax  = max(|x|)                      # over the ENTIRE tensor (one scalar)
scale = amax / 448.0                  # 448 = finfo(float8_e4m3fn).max
out   = clamp(x / scale, -448, 448) -> fp8
```

Outputs are `out` (fp8, same shape as `x`) and `scale` (one fp32 value). The
shipped baseline is eager Torch: correct, but it materializes several full-size
fp32 temporaries and launches one kernel per elementwise step.

## Optimization ideas (not prescriptions — measure everything)

- The op moves only ~512 KiB in / 256 KiB out, so it is memory- and
  launch-bound. Cutting the number of launches and the number of passes over `x`
  is likely worth more than arithmetic tuning.
- `amax` is a *global* reduction, so a single pass cannot know the scale before
  quantizing. Consider a two-stage reduction (per-block partials, then a final
  reduce) versus atomics, and measure which wins at this size.
- Tune `BLOCK_SIZE` / `num_warps` and vectorized loads for the 4096-wide rows.
- Watch out for anything that needs a per-call memset: it costs an extra launch
  and can break graph replay.

## Modification rules

1. Keep the public `dynamic_quant_fp8(x, out, scale)` signature unchanged — the
   driver imports it. Write results into the caller's `out` / `scale` buffers.
2. Keep the kernel in Triton; do not rewrite it in another language.
3. Do NOT import or call AITER — the driver asserts it was never loaded.
4. Do NOT edit `driver.py` or `graph_harness.py` — they are the measurement
   oracle and timing harness (the loop blocks edits to them). Optimize the
   kernel, not the measurement.
5. Stay replay-safe: no host syncs on device values (no `.item()` /
   `.cpu()` / data-dependent Python branches), because one invocation is captured
   into a CUDA/HIP graph and replayed. Internal scratch allocations are fine.
6. Divide by `scale` rather than multiplying by its reciprocal if you want to
   match the oracle bit-for-bit; the SNR gate has slack either way.
7. Build/run/verify your change yourself before finishing; the loop then runs a
   canonical correctness + benchmark pass and keeps the change only if it is
   correct AND faster than the current best.
