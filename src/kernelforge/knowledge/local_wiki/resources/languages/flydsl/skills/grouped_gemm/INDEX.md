---
title: FlyDSL grouped GEMM on gfx950 — package map
kind: index
scope: languages/flydsl/skills/grouped_gemm
updated: 2026-10-27
pinned_source: AMD-AGI/Primus-Turbo@ca1fdd67 (primus_turbo/flydsl/{grouped_gemm,utils}); ROCm/FlyDSL@dd8373349 (v0.3.2; kernels/moe)
---

# FlyDSL grouped GEMM — package map

How to write and optimize a grouped GEMM in FlyDSL on gfx950 (MI350/MI355X): one launch that runs G independent
GEMMs whose sizes differ per group, as in a mixture-of-experts layer. The body is training: the forward pass
(NT), dgrad (NN, or NT on a transposed weight) and variable-K wgrad (TN, one contraction length per group), in
bf16, fp8 with per-tensor scales, MXFP8 and MXFP4, with GLU/dGLU epilogue fusion. Inference MoE expert GEMMs
(gathered by sorted token ids) have one card of their own.

A grouped GEMM is a dense GEMM plus a group dimension. The main loop, MFMA atoms, LDS layout, DMA, scheduling
hints, epilogue basics and the measurement protocol are the same as for dense GEMM and live in `../gemm/`; the
cards here cover what the group dimension adds: mapping tiles to groups, ordering them under token skew, the
per-tile fixed cost that small groups expose, variable-K wgrad, scales and quantization per group, the host
path, and the gates and rulers that a skewed distribution needs.

**Three sources, used together.**
- **Primus-Turbo's FlyDSL grouped GEMMs** (`primus_turbo/flydsl/grouped_gemm/` at main `ca1fdd67`), built for
  training. They keep group metadata on the device, launch a static upper-bound grid, find each tile's group
  on the device (a lane-resident prefix table in the MXFP4 kernels and in the fp8 kernels above eight groups),
  order tiles across XCDs, split deep variable-K groups with a deterministic reduce, and race most
  configurations on balanced synthetic data. The cards cite them by file and line as techniques; Turbo pins a
  newer FlyDSL, and every API named in the cards exists in v0.3.2.
- **FlyDSL's own MoE GEMMs** (`kernels/moe/` in v0.3.2): sorted-token, per-M-block expert ids, persistent
  stage 2. FlyDSL ships no variable-K grouped GEMM and no wgrad.
- **Measured grouped-GEMM work**, marked *(field)*: what moved wall time and what did not.

- FlyDSL facts are stated for v0.3.2.
- Numbers marked *(field)* are measured values. Read them as magnitudes, not targets.
- Signed percentages of wall time, throughput or a score are speed changes (+ = faster) unless the sentence says
  faster or slower; a sign next to a named quantity (cycles, sclk, a counter, bytes, registers) is the change of
  that quantity.
- Occupancy is stated as waves per SIMD or workgroups per CU, explicitly.

## Start here

| You are… | Read |
|---|---|
| starting on a grouped GEMM, or deciding what to try next | `flydsl_grouped_workflow.md` |
| mapping tiles to groups: grid size, group lookup, empty groups, 64-bit offsets | `flydsl_grouped_group_mapping.md` |
| ordering tiles across XCDs and groups, handling token skew, choosing persistent or not | `flydsl_grouped_scheduling.md` |
| choosing a tile, occupancy or boundary body per pass; pricing small groups | `flydsl_grouped_tile_pricing.md` |
| working on wgrad with a per-group contraction length, split-K, beta=1 accumulation | `flydsl_grouped_wgrad.md` |
| adding fp8, MXFP8 or MXFP4: scale preshuffle and packing, quantization, stores, GLU fusion | `flydsl_grouped_lowprec.md` |
| launching without host syncs, CUDA graphs, caching, autotuning across distributions | `flydsl_grouped_host_autotune.md` |
| benchmarking grouped kernels: shapes, distributions, deployment conditions, noise | `flydsl_grouped_measurement.md` (with `../gemm/flydsl_gemm_measurement.md`) |
| writing or trusting a correctness gate for grouped kernels | `flydsl_grouped_correctness.md` (with `../gemm/flydsl_gemm_correctness.md`) |
| working on an inference MoE expert GEMM (sorted token ids, decode) | `flydsl_grouped_moe_inference.md` |
| working on the main loop, LDS, DMA, MFMA atoms or the epilogue itself | `../gemm/INDEX.md` |

## Kernel map

### Primus-Turbo main `ca1fdd67`, `primus_turbo/flydsl/grouped_gemm/`

| File | dtype / passes | Entry points |
|---|---|---|
| `grouped_gemm_bf16_kernel.py` | bf16/fp16: forward NT, dgrad NN, variable-K wgrad | `grouped_gemm_bf16_nt_flydsl_kernel` (:720), `grouped_gemm_bf16_nn_flydsl_kernel` (:888), `grouped_gemm_bf16_variable_k_flydsl_kernel` (:533) |
| `grouped_gemm_fp8_kernel.py` | fp8 E4M3/E5M2 per-tensor scale: forward NT, dgrad NN, variable-K wgrad | `grouped_gemm_fp8_tensorwise_flydsl_kernel` (:2628), `grouped_gemm_fp8_variable_k_tensorwise_flydsl_kernel` (:4980) |
| `grouped_gemm_mxfp8_kernel.py` | MXFP8 (E8M0 per 32): forward and dgrad as NT, variable-K wgrad | `grouped_gemm_mxfp8_flydsl_kernel` (:1239), `grouped_gemm_mxfp8_variable_k_flydsl_kernel` (:1835) |
| `grouped_gemm_mxfp4_kernel.py` | MXFP4: forward and dgrad as NT, variable-K wgrad | `grouped_gemm_mxfp4_flydsl_kernel` (:1589), `grouped_gemm_mxfp4_variable_k_flydsl_kernel` (:2154) |
| `grouped_gemm_{fp8,mxfp8,mxfp4}_glu_kernel.py` | GLU (forward) and dGLU (dgrad): fp8 and MXFP4 fuse them into the GEMM epilogue; MXFP8 runs the GEMM, then a separate GLU + dual-quantization kernel. The MX variants also quantize the output | e.g. `grouped_gemm_fp8_tensorwise_epi_glu_flydsl_kernel` (fp8 :138), `grouped_gemm_mxfp4_epi_dglu_quant_flydsl_kernel` (mxfp4 :314) |
| `../utils/gemm_helper.py` | shared | group scans and lookups, XCD remaps, super-tile decode, 64-bit descriptor bases |

The bf16 kernels reuse Turbo's dense bf16 tile bodies; the fp8 and MX kernels share their main-loop design with
Turbo's dense kernels (`../gemm/INDEX.md` § Kernel map).

### FlyDSL v0.3.2, `kernels/moe/`

| Path | What it is | Entry points |
|---|---|---|
| `moe_gemm_2stage/` | fp8 / int8 / int4-weight MoE GEMM, stage 2 atomic or split + reduce | `compile_moe_gemm1` (`gemm1.py:541`), `compile_moe_gemm2` (`gemm2.py:595`), `compile_moe_reduction` (`moe_reduce.py:164`) |
| `mxfp_moe/` | MXFP4 / a8w4 MoE GEMM for gfx950, persistent non-atomic stage 2 | `compile_gemm1_a4w4_port` (`gemm1.py:680`), `compile_gemm2_a4w4_port` (`gemm2.py:78`), host `flydsl_mxfp4_gemm1/2` (`host.py:87`, `:168`) |
| `moe_2stage_a16wmix/` | bf16 activations × MXFP4 / int4 / bf16 weights | `compile_gemm1_a16w4_port` (`gemm1.py:708`), `compile_gemm2_a16w4_port` (`gemm2.py:508`) |
| `moe_sorting_kernel.py` | token sorting by expert, tile-M as the unit | `moe_sorting_flydsl` (:2070) |

`kernels/mega_moe/` fuses dispatch, both GEMMs and combine across GPUs; it is outside this package.

Constants in both code bases (tile tables, band widths, group-count limits, adoption margins) were tuned on
specific shapes and distributions. Reuse the mechanism and re-derive the number.

## Cards

| Card | One line |
|---|---|
| `flydsl_grouped_workflow.md` | what the group dimension changes, working order per pass, what is usually dead |
| `flydsl_grouped_group_mapping.md` | device-side group metadata, upper-bound grids, tile→group lookups and their cost, empty groups, 64-bit bases |
| `flydsl_grouped_scheduling.md` | XCD and L2 order across groups, where the skew lives, longest-first, persistent vs one tile per workgroup |
| `flydsl_grouped_tile_pricing.md` | tile and occupancy per pass, boundary and tail bodies, the per-tile fixed cost of small groups |
| `flydsl_grouped_wgrad.md` | variable-K loops, scale timing, store-bound short groups, deterministic split-K, beta=1 accumulation |
| `flydsl_grouped_lowprec.md` | fp8/MX scale preshuffle and packing, quantization share, K padding, store forms, GLU/dGLU fusion |
| `flydsl_grouped_host_autotune.md` | no host syncs, graph vs eager launch, caching, races on balanced data, beta and ranking |
| `flydsl_grouped_measurement.md` | shapes and distributions, deployment conditions, FLOP accounting, per-cell noise |
| `flydsl_grouped_correctness.md` | skewed and empty groups in the gate, out-of-bounds reads, MX probes, determinism vs correctness |
| `flydsl_grouped_moe_inference.md` | sorted-token MoE expert GEMMs in FlyDSL: work mapping, persistent stage 2, cache policy, decode |
