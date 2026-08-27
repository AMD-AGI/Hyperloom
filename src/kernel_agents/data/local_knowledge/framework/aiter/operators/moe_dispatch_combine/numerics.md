---
title: moe_dispatch_combine — numerics
kind: technique
operator: moe_dispatch_combine
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3, fp4_e2m1]
regimes: [prefill, decode]
updated: 2026-07-29
sources:
  - ROCm/mori@35e2effb6:docs/MORI-EP-GUIDE.md
  - ROCm/mori@35e2effb6:python/mori/ops/dispatch_combine.py
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
  - https://www.lmsys.org/blog/2026-05-28-mori/
---

# moe_dispatch_combine — numerics & parity

## What changes the numbers
Dispatch/combine **moves** data and does **one reduction** (combine gathers k expert outputs per token and
multiplies the routing weight). Two numeric levers: the **on-the-wire quant** of dispatch/combine, and the
**reduction order/dtype** of combine.

## fp8/fp4 dispatch, quantized combine
- **Dispatch quant** (`quant_type="fp8_direct_cast"`; E4M3FNUZ on gfx942, OCP on gfx950) quantizes tokens
  on the wire and returns per-token scales — this introduces **quant error before the expert GEMM**. It is
  a standard recipe and generally accuracy-safe for inference, but it **is** an accuracy gate (re-run a
  small eval when enabling).
- **Combine is bf16 by default**, even when dispatch is fp8: the gather+weight-multiply runs in bf16/fp32
  to avoid compounding quant error on the reduction. `EpDispatchCombineConfig.quant_type` also now exposes
  two *combine*-side blockwise codecs — `fp8_blockwise` (per-128-element FP32 scale by default, tunable via
  `MORI_FP8_COMBINE_SCALE_DIM`) and `fp4_blockwise` (**gfx950-only**: it hard-asserts at construction time
  if the detected arch isn't gfx950, since it needs the OCP FP4 conversion instructions). LMSYS's MI355X
  writeup reports `fp8_blockwise` combine at ~2% higher accuracy than direct-cast fp8, at a small latency
  cost (~736 µs vs ~907 µs BF16 reference in their micro-benchmark — see tuning.md).
- fnuz vs OCP: gfx942 is **fnuz** (exponent bias off-by-one vs OCP) — a token quantized in the wrong fp8
  dialect is off by exactly 2×. Match the dialect to the arch.

## Combine reduction order
Combine sums the k expert contributions (and any shared-expert contribution) per token. The **order**
differs from a dense reference and across backends (MoRI vs DeepEP vs a torch reference), so:
- expect small bf16 differences; gate with **greedy/temp=0 parity over ≥10 prompts**, not byte match.
- the routing **weight** multiplied in combine must be the **unbiased** routing weight (DeepSeek uses bias
  only for *selection*) — a common bug is carrying the biased score into combine.

## Where the multiply lands (and why it matters)
The routed-weight multiply can live in: the router (don't, under EP), **stage-1 of the grouped GEMM**
(`doweight_stage1`), **stage-2 epilogue** (`MulRoutedWeight1`), or **combine** (MoRI-EP's prob-mult). Each
choice changes the rounding point. Keep it consistent with the reference and in bf16/fp32, not fp8.

## Static-shape / cap correctness traps
- HIP-graph capture forces **static** tensor sizes, but EP token counts are **dynamic**. Padding to a fixed
  `max_num_inp_token_per_rank` must use a value (0 / a sentinel) that **does not** contribute to the combine
  reduction — a non-zero pad token leaks into an expert's output. Verify the pad is masked.
- `EpDispatchCombineConfig.max_total_recv_tokens` (default `0` = uncapped) derives the max tokens a rank
  can receive; if the actual received count exceeds the derived limit, the kernel **asserts** — this is a
  correctness/availability trap, not just a perf knob, on skewed routing.

## Verification recipe
1. Isolated: dispatch→(identity expert)→combine round-trip must reconstruct the input within the dispatch
   dtype's tolerance (catches a broken inverse map / wrong `get_dispatch_src_token_pos`).
2. Full MoE layer vs a torch reference (greedy) after switching all2all backend or enabling fp8/fp4 dispatch.
3. eval (e.g. gsm8k) when enabling quantized dispatch/combine — it's a quant gate, not just a kernel-error
   check.

## Sources
- fp8/fp4 dispatch, blockwise combine codecs, `max_total_recv_tokens`, layouts: `ROCm/mori@35e2effb6:docs/MORI-EP-GUIDE.md`, `python/mori/ops/dispatch_combine.py` (`EpDispatchCombineConfig`, `EpDispatchCombineQuantType`).
- shared-expert fusion preserves numerics, prob-mult in combine: https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
- fp8_blockwise ~2% higher accuracy than direct-cast; ~736 µs vs ~907 µs BF16 combine micro-benchmark: https://www.lmsys.org/blog/2026-05-28-mori/
- fnuz vs OCP fp8 off-by-2×: scaled_quant_gemm numerics; CDNA3/4 ISA.
