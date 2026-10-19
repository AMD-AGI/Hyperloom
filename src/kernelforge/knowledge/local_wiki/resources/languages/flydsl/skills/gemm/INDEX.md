---
title: FlyDSL dense GEMM on gfx950 — package map
kind: index
scope: languages/flydsl/skills/gemm
updated: 2026-10-19
pinned_source: AMD-AGI/Primus-Turbo@7615a5a2 (primus_turbo/flydsl/{gemm,utils}); ROCm/FlyDSL@dd8373349 (v0.3.2)
---

# FlyDSL dense GEMM — package map

How to write and optimize a dense GEMM in FlyDSL on gfx950 (MI350/MI355X): fp8 (per-tensor and per-row
scales), MXFP4, MXFP8 and bf16, layouts NT/NN/TN.

**The exemplar is Primus-Turbo's FlyDSL GEMM** (`primus_turbo/flydsl/gemm` and `utils` at main `7615a5a2`).
These are the fastest FlyDSL GEMMs we have measured on gfx950. Their comments and commit messages record
why each choice was made, and the cards cite them by file and line. FlyDSL's own `kernels/gemm` are
described in `flydsl_gemm_mainloop.md` §8, for when you call them as a library.

Numbers marked *(field)* in these cards were measured on MI355X during FlyDSL kernel work. They give
direction and order of magnitude; they are not targets for your shape.

## Start here

| You are… | Read |
|---|---|
| starting work on a GEMM, or deciding what to do next | `flydsl_gemm_workflow.md` |
| about to choose a tile, split-K or boundary body, or wanting the ceiling of an idea | `flydsl_gemm_grid_pricing.md` |
| designing or editing the K loop (8-wave pipeline vs 4-wave whole-loop asm, waits, registers) | `flydsl_gemm_mainloop.md` |
| working on LDS, swizzle, global→LDS DMA, descriptors, transpose reads, MX scales | `flydsl_gemm_datapath.md` |
| ordering tiles across XCDs, writing scheduling hints, mapping waves | `flydsl_gemm_schedule_and_locality.md` |
| working on the C store, beta=1, or a fused epilogue | `flydsl_gemm_epilogue.md` |
| putting several configurations behind one entry point; launch overhead; CUDA graphs | `flydsl_gemm_autotune.md` |
| finding out where the time goes before changing anything | `flydsl_gemm_attribution.md` |
| writing or trusting a correctness gate | `flydsl_gemm_correctness.md` |
| measuring an A/B, setting a noise floor, closing a campaign | `flydsl_gemm_measurement.md` |
| debugging a wrong result | `../bottleneck/debug-flydsl-kernel.md` |

## Kernel map: Primus-Turbo main `7615a5a2`, dense

Paths are under `primus_turbo/flydsl/`.

| File | dtype / layout | Loop design | Entry point |
|---|---|---|---|
| `gemm/gemm_fp8_kernel.py` | fp8 E4M3/E5M2, per-tensor scale; NT, NN, TN; bf16/fp16 out; any K | 8-wave pipeline raced against the 4-wave whole loop per shape | `gemm_fp8_tensorwise_flydsl_kernel` (:3448) |
| `gemm/gemm_mxfp4_kernel.py` | MXFP4 (E8M0 per 32) NT; fused GLU/dGLU + MXFP4 quant variants | 4-wave whole loop, VGPR-direct scales, folded C store | `gemm_mxfp4_flydsl_kernel` (:4351); `preshuffle_mxfp4_scales` (:4877) |
| `gemm/gemm_mxfp8_kernel.py` | MXFP8 NT | 8-wave pipeline, packed scales, tail-round split-K | `gemm_mxfp8_flydsl_kernel` (:985) |
| `gemm/gemm_bf16_kernel.py` | bf16/fp16 NT/NN/TN tile bodies (Turbo main has no dense bf16 FlyDSL entry point; these tiles serve its other kernels) | 8-wave, 16x16x32 MFMA, transpose reads | `gemm_bf16_tile` (:721) |
| `utils/gemm_helper.py` | shared | G2S/S2R loaders, MFMA wrappers, store classes, XCD remaps, timing | — |
| `utils/gemm_epilogue_helper.py` | shared | fused activation and quantization epilogues | — |
| `utils/prims.py` | shared | `readfirstlane`, unsigned index math, wave scans | — |

Constants in these files (candidate tables, wait counts, adoption margins, L2 and burst thresholds) were
tuned on specific model shapes on MI355X. Reuse the mechanism and re-derive the number.

## FlyDSL versions

Primus-Turbo main pins flydsl 0.3.4.1; this knowledge base's API docs are vendored from 0.3.2; Primus-Turbo
before `7615a5a2` used 0.2.4. Turbo main `7615a5a2` runs unmodified on 0.3.2 *(field, 2026-10-19: fp8 and MXFP4
8192³ bit-identical to 0.3.4.1)*. The differences that matter for GEMM code:

| | 0.2.4 | 0.3.2 | 0.3.4.1 |
|---|---|---|---|
| raw buffer helpers (`create_buffer_resource`, …) | `flydsl.expr.buffer_ops` | not in the wheel (`kernels/common/buffer_ops.py` in the repo); Turbo vendors its own copy in `utils/buffer_ops.py` | same as 0.3.2 |
| kernel attributes through `value_attrs` | `"passthrough"` kept, `"llvm.passthrough"` **silently ignored** | `"passthrough"` **silently dropped**, `"llvm.passthrough"` kept | same as 0.3.2 |
| `rocdl.s_waitcnt` | raw bitfield only | raw bitfield, or `s_waitcnt(vmcnt=…, lgkmcnt=…)` | same as 0.3.2 |
| `rocdl.asyncmark()` / `wait_asyncmark(n)` with `cdna4.BufferLoadAsyncLDS*` | absent | present | present |
| `aux` (cache policy) on `rocdl.raw_ptr_buffer_*` wrappers | — | a non-`int` value is silently replaced by `None` | same |
| closure-captured Python dict in the JIT cache key | — | not in the key | not in the key |

The attribute rows were checked by reading `.amdhsa_next_free_vgpr` in the ISA on MI355X; the others by
reading each tag's source. See `flydsl_gemm_mainloop.md` §4 and `../bottleneck/debug-flydsl-kernel.md` § What
the cache key sees.

## Cards

| Card | One line |
|---|---|
| `flydsl_gemm_workflow.md` | order of operations, reading the evidence, one change per round, escalation |
| `flydsl_gemm_grid_pricing.md` | dispatch rounds, last-round fill, split-K variants, alternative tiles, persistent tiles, fixed per-tile cost |
| `flydsl_gemm_mainloop.md` | 8-wave pipeline vs 4-wave whole loop, inline-asm register plan and constraints, MFMA atom, K tail |
| `flydsl_gemm_datapath.md` | LDS layout and swizzle, DMA and M0, i64 descriptors, transpose reads, row alignment, MX scales |
| `flydsl_gemm_schedule_and_locality.md` | wave placement, XCD/L2 tile order, scheduling hints, wait strategy |
| `flydsl_gemm_epilogue.md` | 64 B write requests, store classes, overlap with the last MFMAs, cache policy, beta=1, fusion |
| `flydsl_gemm_autotune.md` | first-call race, never-regress margins, cache keys, launch path, CUDA graphs |
| `flydsl_gemm_attribution.md` | regime probes, subtractive and additive arms, energy at the power cap, wave-cycle budget |
| `flydsl_gemm_correctness.md` | gates that reach the changed path, races, waits, bounds, register classes |
| `flydsl_gemm_measurement.md` | rulers, arm order, warm-up, noise floors, statistics, closing a campaign |
