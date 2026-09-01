# Task: rewrite Triton softmax → FlyDSL, then optimize

## Operator
Row-wise softmax over the last dim of a 2D tensor:
`y[i, :] = softmax(x[i, :])`, numerically stabilized by subtracting the row max.

The source of truth is the Triton kernel in `softmax.py` (public entry
`softmax(x)`). Your FlyDSL port must reproduce its output within the SNR gate.

## What you must produce
A FlyDSL kernel in `kernel.py` exposing the factory the driver imports:

```python
build_softmax_module(M, N, dtype_str) -> launch_fn
launch_fn(A, C, m_rows, stream=fx.Stream(...))   # C = softmax(A) row-wise
```

- `A` and `C` are 2D `(M, N)` tensors of dtype `dtype_str` ("f32" / "f16" / "bf16").
- `m_rows` is the row count `M`.
- The launcher MUST accept a `stream` kwarg and launch on THAT stream — the
  driver passes the active (CUDA-graph capture) stream. A kernel that ignores it
  and launches on the default stream is not captured and mis-benchmarks.

The exact call convention is defined by `driver.py` (embedded read-only in your
task prompt). Match it exactly.

## Rules
- Implement in **FlyDSL only** (`import flydsl...`). Do NOT compute the result with
  Triton / torch / HIP — that defeats the rewrite.
- Edit ONLY `kernel.py`. `softmax.py`, `driver.py`, and `graph_harness.py` are the
  reference/measurement contract and are protected (edits are blocked).
- Keep the `build_softmax_module` factory name and the launch signature stable.
- Consult the FlyDSL knowledge (operator cards, API docs, examples) before
  writing — work from the docs, not from memory. A three-pass register-buffered
  row reduction (max → exp+sum → normalize) with a block/wave reduction maps
  cleanly onto FlyDSL; `exp2(x * log2e)` gives a fast, accurate exp.

## Phases
1. **PORT** (correctness-only): make `kernel.py` correct vs the Triton oracle
   (SNR ≥ 30 dB across the driver's full correctness suite).
2. **OPTIMIZE**: forge-loop then tunes the correct FlyDSL kernel for speed
   (block size, vectorized loads/stores, warp count, memory access) while keeping
   it correct. Measure every idea against the graph-timed baseline; keep only wins.
