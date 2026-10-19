# Program: optimize the Gluon fused softmax kernel

**GPU**: gfx950 (AMD Instinct CDNA4) — adjust `--gpu-target` to your hardware
**Backend**: gluon

## Objective

Optimize `softmax` in `softmax_kernel.py` for maximum throughput on the target
GPU while keeping the result numerically correct. The loop gates correctness on
an SNR threshold (30 dB) before it ever benchmarks a change.

## What the kernel does

Row-wise softmax over the last dimension of a 2D `(rows, cols)` fp16 tensor,
with an fp32-stable max-subtraction and reduction, one program per row.

It is written in **Gluon**, Triton's low-level dialect — same `@…jit`, same
launch surface, same JIT cache, same lowering. The difference that matters here:
the tile **layout** is an explicit object in the source rather than something
the compiler picks, so the distribution of a row's elements over registers,
lanes and warps is a degree of freedom you can measure.

The baseline is deliberately a **v0**: correct, layout stated, nothing else.

## Optimization ideas (not prescriptions — measure everything)

Roughly in the order the AMD Gluon ladder takes them:

- **The blocked layout.** `size_per_thread` is the elements each thread owns per
  load; at 1 there is no vectorization at all. On a 1D tile the entire space of
  blocked layouts is this one number times the wavefront times the warp count,
  so it is the clearest axis in the file. Sweep it in both directions.
- **`num_warps`**, and its interaction with the above — they jointly have to
  tile `BLOCK_SIZE`, so they are coupled and should be swept together rather
  than one at a time.
- **AMD buffer ops.** `gl.amd.cdna4.buffer_load` addresses global memory as a
  scalar base plus an offset tensor, moving bounds handling into the buffer
  descriptor instead of into masked-load branching.
- **Work per program.** One row per program leaves the small case at the launch
  floor; more rows per program is a structural change worth measuring.

Read the `languages/gluon/` knowledge cards before reaching for anything
lower-level than this — in particular `skills/optimize/gluon_levers/overview.md`
for whether a rung is worth its session, and `.../forge_integration.md` for the
version traps and how to shape a change so a KEEP can carry it.

Note that the two scored cases behave differently: the narrow one is close to
the launch floor and the wide one is not. The score is the equal-weight mean of
per-case speedups, so a change that only helps the wide case still moves it.

## Modification rules

1. Keep the public `softmax(x)` signature unchanged — the driver imports it.
2. Keep the kernel in the Triton/Gluon toolchain.
3. Do NOT edit `driver.py` or `graph_harness.py` — they are the measurement
   surface (the loop blocks edits to them). Optimize the kernel, not the
   measurement. Note the benchmark runs under HIP graph capture, so the kernel
   must stay capture-safe: no host syncs and no host-side branching on a device
   value in the steady state.
4. Prefer changing a tracked file over creating a new one — a KEEP commits
   tracked edits, and a new file needs `--commit-new-path` on the campaign.
5. Build/run/verify your change yourself before finishing; the loop then runs a
   canonical correctness + benchmark pass and keeps the change only if it is
   correct AND measurably faster than the current best.
