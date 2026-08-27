---
title: splitk_streamk_gemm on triton — SOTA card
kind: sota_card
operator: splitk_streamk_gemm
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: sota
updated: 2026-06-05
sources:
  - https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
  - https://arxiv.org/abs/2301.03598
---

# splitk_streamk_gemm × triton

## TL;DR
> Triton is the **most ergonomic place to do split-K / stream-K** on AMD: `SPLIT_K` is a kernel arg and
> stream-K is a persistent-kernel pattern in the official tutorial. Best choice for CU-underutilized
> shapes you author yourself; library GEMMs (hipBLASLt/CK) also do this internally for covered shapes.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Triton split-K matmul (`SPLIT_K` arg, atomic or workspace reduce) | https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html | gfx942/950; bf16, fp8 | no first-party number reproduced; win is shape-dependent vs dense | large-K, small M·N |
| Triton stream-K persistent matmul | tutorial above + arXiv:2301.03598 | gfx942/950 | balances tiles across CUs, removes wave-quant tail | tiles ≈ CU-count remainder |

## Config space / knobs
- `SPLIT_K`, `BLOCK_M/N/K`, `NUM_SMS`(≈ CU count: 304 MI300X / 256 MI350X), reduction mode (atomic vs
  workspace), `num_warps`, `num_stages`, `waves_per_eu`, `matrix_instr_nonkdim` (16 vs 32). See
  [tuning.md](tuning.md).

## Numerics / parity
- fp32 accumulate; atomic reduce = non-deterministic, workspace = deterministic → [numerics.md](numerics.md).

## Integration (rebind seam)
- Overlay the triton GEMM module the framework calls (or aiter's triton GEMM path); verify the kernel +
  autotune key engage in a rocprof trace.

## Pitfalls & anti-patterns
- Enabling split-K on already CU-saturated shapes → slower (extra reduction). Gate on tile-count heuristic.
- bf16 atomic-add precision loss — accumulate/atomic in fp32.

## How to verify
- A/B split/stream on vs off vs dense, dense fp32 oracle ([numerics.md](numerics.md)).

## Alternatives / cross-links
ck · hipblaslt · asm · hip · [overview.md](overview.md)

## Sources
- Triton persistent/stream-K tutorial: https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
- Stream-K paper: https://arxiv.org/abs/2301.03598
