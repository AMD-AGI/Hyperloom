# Program: optimize the FlyDSL softmax kernel

**GPU**: gfx950 (AMD Instinct) — adjust `--gpu-target` to your hardware
**Backend**: flydsl

## Objective

Optimize `build_softmax_module` in `softmax_kernel.py` for maximum throughput on
the target GPU while keeping the result numerically correct. The loop gates
correctness on an SNR threshold (30 dB) before it ever benchmarks a change.

## What the kernel does

Row-wise softmax over the last dimension of a 2D `(M, N)` tensor, with an
fp32-stable max-subtraction and `exp2(x * log2e)` for fast exponentiation. The
kernel builder specialises per `(M, N, dtype)`. The baseline runs the **scalar
generic path**; the vectorised buffer-load/store fast path is gated off.

## Optimization ideas (not prescriptions — measure everything)

- Enable / fix the vectorised fast path (`buffer_load/store`, `VEC_WIDTH`) for the
  `N % tile_cols == 0` case instead of the scalar `copy_atom_call` path.
- Tune `BLOCK_THREADS` and the register-buffering strategy against the row width.
- Improve the wave/block reduction (shuffle width, LDS traffic in `block_reduce`).
- Tune AMD knobs from the FlyDSL knowledge cards (WAVES_PER_EU, DMA) per the
  measured PMC bottleneck.

## Modification rules

1. Keep the public `build_softmax_module(M, N, dtype_str)` builder — the driver
   imports it — and keep the returned `launch_fn(A, C, m_in, stream=...)`
   accepting and honoring the `stream` kwarg. The driver passes the CUDA-graph
   capture stream through it; a kernel that ignores `stream` and launches on the
   default stream is NOT captured and its benchmark is meaningless.
2. Keep the kernel in FlyDSL; do not rewrite it in HIP, CUDA, or Triton.
3. Do NOT edit `driver.py` or `graph_harness.py` — they are the measurement
   oracle + timing harness (the loop blocks edits to them). Optimize the kernel,
   not the measurement.
4. Build/run/verify your change yourself before finishing; the loop then runs a
   canonical correctness + benchmark pass and keeps the change only if it is
   correct AND faster than the current best.
