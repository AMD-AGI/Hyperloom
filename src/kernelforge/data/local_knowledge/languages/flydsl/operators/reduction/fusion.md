---
title: reduction — fusion
kind: operator_overview
operator: reduction
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16]
regimes: [both]
updated: 2026-06-08
---

# reduction — fusion

## TL;DR
A reduction loads a whole tensor to produce a tiny output, so the load is free real estate — fold the elementwise that **produces** the reduce input (`Σx²`) or **consumes** its output (`x·rsqrt(...)`) into one pass (this is exactly how RMSNorm/LayerNorm/softmax are built). Keep the input loaded once; never materialize `x²`/`|x|` to HBM; stop fusing when it spills VGPRs.

A reduction reads a whole tensor to produce a tiny output, so the load is "free real estate": fold the
elementwise that *produces* the reduce input, or the elementwise that *consumes* the reduce output, into
the same kernel. This is exactly how RMSNorm/LayerNorm/softmax are built — they are **fused
reduce+elementwise** kernels, not a reduce followed by a separate pass.

## The patterns

| fusion | example | saves | where |
|---|---|---|---|
| **elementwise → reduce** | `sum(x*x)`, `sum(abs(x))`, `max(clamp(x))` | the intermediate write+read | norm Σx² done in the load pass |
| **reduce → elementwise** | `x * rsqrt(mean(x²)+eps)` (RMSNorm) | re-loading x for the scale | RMSNorm/LayerNorm |
| **reduce + reduce (one load)** | mean **and** var (Welford) | a second full read | LayerNorm |
| **reduce → broadcast → elementwise** | softmax: `max`, then `exp(x-max)`, then `sum`, then `/sum` | two extra passes over the row | softmax (online) |
| **GEMM → reduce** | logits → `max`/`sum` for sampling | reading logits twice | lm_head + sampling |

## How it gets fused
- **Inductor (automatic)**: fuses a pointwise chain *and a trailing reduction* into one Triton kernel —
  the canonical free win on `torch.compile`. A `sum(x*x)` becomes one kernel that squares in registers and
  wave-reduces. See [`../../backends/pytorch_inductor/overview.md`](../../backends/pytorch_inductor/overview.md).
- **Triton (manual)**: load the row once, apply the elementwise, call `tl.sum`/`tl.max` (wave reduce),
  then the post-reduce elementwise, store once. This is the softmax/RMSNorm template in
  [`../../languages/triton_amd/patterns.md`](../../languages/triton_amd/patterns.md) §5.
- **aiter (library-fused)**: `fused_add_rmsnorm`, fused norm+quant — the reduce never appears as a
  separate kernel on the serving path. See [`../../backends/aiter/overview.md`](../../backends/aiter/overview.md).

## The keep-it-in-one-pass rule
The reduce input should be loaded **once**. If you find yourself writing the squared/clamped tensor to
HBM just to reduce it next, fuse. The norm operators are the worked examples:
[`../rmsnorm/overview.md`](../rmsnorm/overview.md), [`../layernorm/overview.md`](../layernorm/overview.md),
[`../softmax/overview.md`](../softmax/overview.md), [`../fused_add_rmsnorm/overview.md`](../fused_add_rmsnorm/overview.md).

## Anti-patterns
- Materializing `x²`/`|x|` to HBM before a `sum` (the thing fusion kills).
- A two-pass mean+var instead of Welford → an extra full read of a big `[tokens, hidden]` tensor.
- Fusing so much pre-reduce elementwise that the kernel **spills** (then split). Fusion helps until VGPR
  pressure cuts occupancy.
- Atomic-split reduce fused into a parity-critical path — the nondeterministic order can flip downstream
  argmax; keep that reduce deterministic ([numerics.md](numerics.md)).

## Verify
rocprof: a fused reduce shows **one** kernel reading the input tensor once; an unfused chain shows the
intermediate write + re-read. `TORCH_COMPILE_DEBUG=1` → `output_code.py` for the Inductor path.
