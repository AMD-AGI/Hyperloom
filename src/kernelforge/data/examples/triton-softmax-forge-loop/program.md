# Program: optimize the Triton fused softmax kernel

**GPU**: gfx950 (AMD Instinct) — adjust `--gpu-target` to your hardware
**Backend**: triton

## Objective

Optimize `softmax` in `softmax_kernel.py` for maximum throughput on the target
GPU while keeping the result numerically correct. The loop gates correctness on
an SNR threshold (30 dB) before it ever benchmarks a change.

## What the kernel does

Row-wise softmax over the last dimension of a 2D `(rows, cols)` fp16 tensor,
with an fp32-stable max-subtraction and reduction. The baseline launches with a
deliberately conservative `num_warps=1`.

## Optimization ideas (not prescriptions — measure everything)

- Tune the launch config: `num_warps`, and `num_stages` to pipeline the load.
- Revisit `BLOCK_SIZE` relative to the row width and warp size.
- Improve the memory-access pattern / vectorized loads for wide rows.

## Modification rules

1. Keep the public `softmax(x)` signature unchanged — the driver imports it.
2. Keep the kernel in Triton; do not rewrite it in another language.
3. Do NOT edit `driver.py` — it is the measurement oracle (the loop blocks edits
   to it). Optimize the kernel, not the measurement.
4. Build/run/verify your change yourself before finishing; the loop then runs a
   canonical correctness + benchmark pass and keeps the change only if it is
   correct AND faster than the current best.
