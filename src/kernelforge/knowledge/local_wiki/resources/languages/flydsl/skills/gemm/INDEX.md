---
title: FlyDSL dense GEMM on gfx950 — package map
kind: index
scope: languages/flydsl/skills/gemm
updated: 2026-10-19
pinned_source: ROCm/FlyDSL@dd8373349 (v0.3.2; kernels/gemm, .claude/skills/gemm-optimization); AMD-AGI/Primus-Turbo@7615a5a2 (primus_turbo/flydsl/{gemm,utils})
---

# FlyDSL dense GEMM — package map

How to write and optimize a dense GEMM in FlyDSL on gfx950 (MI350/MI355X): fp8 (per-tensor and per-row
scales), MXFP4, MXFP8, bf16/fp16 and int8, layouts NT/NN/TN.

**Three sources, used together.**
- **FlyDSL's own GEMM guide and kernels.** `../optimize/gemm-optimization.md` walks through how a FlyDSL GEMM is
  built, on `preshuffle_gemm`: tiling, the 2-stage LDS ping-pong, the XOR swizzle, `hot_loop_scheduler`, direct
  and CShuffle epilogues, B preshuffle, register budget. `../optimize/lds-optimization.md` and
  `../optimize/prefetch-data-load.md` cover the general LDS and prefetch transformations. The kernels are in
  FlyDSL's `kernels/gemm/` (map below).
- **Primus-Turbo's FlyDSL GEMMs** (`primus_turbo/flydsl/{gemm,utils}` at main `7615a5a2`), built for training
  shapes (large M). They add NN/TN layouts, any K, a whole-loop inline-asm design, a per-shape race and fused
  epilogues. Their comments record why each choice was made, and the cards cite them by file and line.
- **Measured FlyDSL GEMM work**, marked *(field)*: what moved wall time and what did not.

The cards in this package start from what FlyDSL's guide already says and add the rest. Read the guide first if
you have not built a FlyDSL GEMM before.

**Small M (decode, M ≤ 256)** is in `flydsl_gemm_small_m.md`: the kernel there is FlyDSL's `gemm_a16w16_gfx950`,
as shipped in aiter. Primus-Turbo does not target small M.

Numbers marked *(field)* are measured values. Use them for direction and order of magnitude, not as targets.

## Start here

| You are… | Read |
|---|---|
| building your first FlyDSL GEMM, or learning how the pieces fit | `../optimize/gemm-optimization.md`, then `flydsl_gemm_workflow.md` |
| starting work on a GEMM, or deciding what to do next | `flydsl_gemm_workflow.md` |
| about to choose a tile, split-K or boundary body, or wanting the ceiling of an idea | `flydsl_gemm_grid_pricing.md` |
| working on a GEMM with M ≤ 256 (decode, speculative verify), or aiter's FlyDSL a16w16 tuned rows | `flydsl_gemm_small_m.md` |
| designing or editing the K loop (2-stage ping-pong, 8-wave pipeline, 4-wave designs, waits, registers) | `flydsl_gemm_mainloop.md` |
| working on LDS, swizzle, global→LDS DMA, B preshuffle, descriptors, transpose reads, MX scales | `flydsl_gemm_datapath.md` (with `../optimize/lds-optimization.md`) |
| ordering tiles across XCDs, writing scheduling hints, mapping waves | `flydsl_gemm_schedule_and_locality.md` |
| working on the C store, CShuffle, beta=1, or a fused epilogue | `flydsl_gemm_epilogue.md` |
| putting several configurations behind one entry point; launch overhead; CUDA graphs | `flydsl_gemm_autotune.md` |
| finding out where the time goes before changing anything | `flydsl_gemm_attribution.md` (with `../profile/kernel-trace-analysis/SKILL.md`) |
| writing or trusting a correctness gate | `flydsl_gemm_correctness.md` |
| measuring an A/B, setting a noise floor, deciding which of a series of changes paid | `flydsl_gemm_measurement.md` |
| debugging a wrong result | `../bottleneck/debug-flydsl-kernel.md` |

## Kernel map

### FlyDSL v0.3.2, `kernels/gemm/`

| File | dtype / layout | Loop design | Entry point |
|---|---|---|---|
| `preshuffle_gemm.py` | fp8, int8, fp16, bf16; per-row × per-column scales; B preshuffled; optional bias + ReLU/SiLU/GELU | 4 waves along N; A through a 2-stage LDS ping-pong, B straight to VGPR; `hot_loop_scheduler` hints; optional async A copy (gfx950) | `compile_preshuffle_gemm` (:108) |
| `gemm_a16w16_gfx950.py` | bf16, fp16 | `m×n×k` waves, A and B by async DMA, `stages` deep, CShuffle store, split-K by atomics | `gemm_a16w16` (:1283) |
| `fp8_gemm_8wave.py` | fp8, row-wise scales; NT; B plain or preshuffled | 8 waves (512 threads), 8 LDS buffers (A/B × cur/next × two halves) | `compile_fp8_gemm_8w` (:148) |
| `fp8_gemm_4wave.py` | fp8, row-wise scales; NT; B plain or preshuffled | 4 waves 2×2, each a 128×128 quadrant; accumulators pinned in AGPR by per-MFMA inline asm; XCD remap | `compile_fp8_gemm_4w` (:162) |
| `fp4_gemm_4wave.py` | MXFP4 (E8M0 per 32); NT | the fp8 4-wave structure with the scaled MFMA | `compile_fp4_gemm_4w` (:634) |
| `mxfp4_preshuffle.py` | A MXFP4/MXFP6/MXFP8 × B MXFP4 (preshuffled weights and scales) | A by double-buffered async DMA into LDS, scaled 16x16x128 MFMA | `launch_gemm` (:58) |

Shared helpers are in `kernels/common/mma/` (`mfma_preshuffle_pipeline.py`, `mfma_epilogues.py`). None of these
handles a K tail: K must be a multiple of the K tile. Calling them through aiter, and their argument space, is in
`../optimize/flydsl_levers/flydsl_kernel_library.md` and `../optimize/flydsl_levers/flydsl_knob_space.md`.

### Primus-Turbo main `7615a5a2`, `primus_turbo/flydsl/`

| File | dtype / layout | Loop design | Entry point |
|---|---|---|---|
| `gemm/gemm_fp8_kernel.py` | fp8 E4M3/E5M2, per-tensor scale; NT, NN, TN; bf16/fp16 out; any K | 8-wave pipeline raced against the 4-wave whole loop per shape | `gemm_fp8_tensorwise_flydsl_kernel` (:3448) |
| `gemm/gemm_mxfp4_kernel.py` | MXFP4 (E8M0 per 32) NT; fused GLU/dGLU + MXFP4 quant variants | 4-wave whole loop, VGPR-direct scales, folded C store | `gemm_mxfp4_flydsl_kernel` (:4351); `preshuffle_mxfp4_scales` (:4877) |
| `gemm/gemm_mxfp8_kernel.py` | MXFP8 NT | 8-wave pipeline, packed scales, tail-round split-K | `gemm_mxfp8_flydsl_kernel` (:985) |
| `gemm/gemm_bf16_kernel.py` | bf16/fp16 NT/NN/TN tile bodies (no dense bf16 entry point; these tiles serve other Turbo kernels) | 8-wave, 16x16x32 MFMA, transpose reads | `gemm_bf16_tile` (:721) |
| `utils/gemm_helper.py` | shared | G2S/S2R loaders, MFMA wrappers, store classes, XCD remaps, timing | — |
| `utils/gemm_epilogue_helper.py` | shared | fused activation and quantization epilogues | — |
| `utils/prims.py` | shared | `readfirstlane`, unsigned index math, wave scans | — |

Constants in both code bases (tile and preload tables, wait counts, adoption margins, L2 and burst thresholds)
were tuned on specific shapes. Reuse the mechanism and re-derive the number.

## FlyDSL version

All API statements in this package are for FlyDSL v0.3.2. Turbo main `7615a5a2` runs unmodified on it. The
raw buffer helpers (`create_buffer_resource`, …) are not in the wheel: they live in `kernels/common/buffer_ops.py`
in the FlyDSL repo, and Turbo vendors its own copy in `utils/buffer_ops.py`. Kernel attributes passed through
`value_attrs`, such as `amdgpu-agpr-alloc`, must go under `"llvm.passthrough"`; a bare `"passthrough"` key is
silently dropped. `rocdl.s_waitcnt` takes a raw bitfield or the keyword form `s_waitcnt(vmcnt=…, lgkmcnt=…)`.
`rocdl.asyncmark()` and `wait_asyncmark(n)` exist and pair with `cdna4.BufferLoadAsyncLDS*`. A non-`int` `aux`
(cache policy) on the `rocdl.raw_ptr_buffer_*` wrappers is silently replaced by `None`. A closure-captured Python
dict is not in the JIT cache key. See `flydsl_gemm_mainloop.md` §4 and `../bottleneck/debug-flydsl-kernel.md`
§ What the cache key sees.

## Cards

| Card | One line |
|---|---|
| `flydsl_gemm_workflow.md` | order of operations, reading the evidence, one change per round, escalation |
| `flydsl_gemm_grid_pricing.md` | dispatch rounds, last-round fill, split-K variants, alternative tiles, persistent tiles, fixed per-tile cost |
| `flydsl_gemm_mainloop.md` | 2-stage ping-pong, 8-wave pipeline, 4-wave designs, inline-asm register plan and constraints, MFMA atom, K tail |
| `flydsl_gemm_datapath.md` | LDS layout and swizzle, DMA and M0, i64 descriptors, transpose reads, row alignment, MX scales |
| `flydsl_gemm_schedule_and_locality.md` | wave placement, XCD/L2 tile order, scheduling hints, wait strategy |
| `flydsl_gemm_epilogue.md` | 64 B write requests, store classes, overlap with the last MFMAs, cache policy, beta=1, fusion |
| `flydsl_gemm_autotune.md` | first-call race, never-regress margins, cache keys, launch path, CUDA graphs |
| `flydsl_gemm_attribution.md` | regime probes, subtractive and additive arms, energy at the power cap, wave-cycle budget |
| `flydsl_gemm_correctness.md` | gates that reach the changed path, races, waits, bounds, register classes |
| `flydsl_gemm_measurement.md` | rulers, arm order, warm-up, noise floors, statistics, closing a series of changes |
| `flydsl_gemm_small_m.md` | decode-shape GEMM: cold rulers, dispatch rounds, split-K reduction cost, tile/stage/cache-policy choices, aiter tuned rows |
