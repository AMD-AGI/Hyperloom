---
title: MFMA scheduling (matrix-core issue, AGPR, latency hiding)
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, fp4_e2m1]
regimes: [prefill, training, both]
updated: 2026-06-05
---

# MFMA scheduling

## TL;DR
The matrix cores execute `v_mfma_*` instructions that read A/B from VGPRs and accumulate into **AGPRs**.
Throughput comes from (a) choosing the right MFMA shape — **`mfma_16x16x16` usually beats
`mfma_32x32x8`** on MI300X for LLM N/K because it needs fewer AGPRs and keeps occupancy up; (b) issuing
MFMAs back-to-back so the systolic pipeline never drains; and (c) hiding `v_mfma` latency behind
independent MFMAs and the next tile's `ds_read`/`global_load`. The C write-out should use
**`OPTIMIZE_EPILOGUE=1`** to avoid the **512-byte Tagram (CShuffle) hotspot**. See
`[[hardware/shared/matrix_core_mfma_smfmac.md]]`, `[[hardware/cdna3_mi300/matrix_core.md]]`,
`[[languages/asm_mfma/]]` and `[[operators/dense_gemm/tuning.md]]`.

## CDNA wave-scheduling prior (ping-pong vs wave-specialization)
**Do not copy the NVIDIA producer-consumer model on CDNA.** HipKittens (Stanford Hazy Research,
arXiv 2511.08083, Nov 2025) found that **NVIDIA-style wave specialization underperforms on
CDNA3/CDNA4**: AMD's static register allocation gives each wave a fixed slice of the 512-VGPR file,
so dedicating waves as "producers" starves them of registers and the kernel tops out at only
**~80% of peak BF16 GEMM**. There is no hardware register reallocation / warp-group specialization
escape hatch like Hopper's.

The CDNA-correct schedule is one of two **symmetric, all-waves-compute** patterns (both originate
from HipKittens and were adopted into AMD's own CDNA4 GEMM blogs):
- **8-wave ping-pong** — 8 waves alternate (ping-pong) between MFMA-issue and memory phases so the
  matrix core is always fed while the other half loads. Robust default for FP8 GEMM. AMD's HIP/C++
  8-wave ping-pong FP8 GEMM hits **3204 TFLOPS** (M=N=K=8192, MI355X, ROCm 7.1.0) — *surpassing*
  hipBLASLt (3130) **with no assembly** (HK's own 8-wave FP8 is 3222 TFLOPS in 48 LoC).
- **4-wave interleave** — **one wave per SIMD**, so each wave owns the full **512-VGPR** budget; a
  **128×128 tile** per wave, with load/MFMA phases interleaved in the instruction stream. The
  robustness/perf successor to ping-pong: **no `#pragma unroll` tuning**, consistent across ROCm
  releases (HK 4-wave FP8 reaches **3327 TFLOPS** in 183 LoC).

Practical prior: on gfx942/gfx950, reach for 8-wave ping-pong or 4-wave interleave (all waves
compute, full register budget, no producer/consumer split). See `[[languages/hipkittens]]`,
`[[operators/dense_gemm]]`, `[[operators/scaled_quant_gemm/tuning.md]]`.

## Concepts (the hardware)
- **MFMA instruction**: `D = A·B + C`, one instruction per wave processes a fixed M×N×K block.
  Common CDNA3 shapes: `mfma_16x16x16` and `mfma_32x32x8` (bf16/fp16); fp8 variants pack 2× K;
  CDNA4 adds block-scaled MX `mfma` for fp8/fp6/fp4 (`[[hardware/cdna4_mi350/matrix_core_blockscale.md]]`).
- **AGPR accumulators**: the running `C` tile lives in AGPRs. A `32x32` instruction's accumulator is
  larger than `16x16`'s, so it consumes more of the 512-register budget and drops occupancy
  (`[[optimization/occupancy_and_registers.md]]`).
- **Systolic latency**: each `v_mfma` has multi-cycle latency; consecutive *independent* MFMAs pipeline
  (one issued per cadence while prior ones are in flight). A dependent MFMA (same accumulator next K
  step) must wait for the previous result — so the loop is unrolled across multiple accumulator tiles
  to keep independent work in flight.

## 16×16 vs 32×32 — how to choose
| | `mfma_16x16x16` | `mfma_32x32x8` |
|---|---|---|
| AGPR pressure | lower | higher |
| occupancy | higher (more waves fit) | lower |
| register-tile flexibility | finer (good for skinny N/K) | coarser |
| best for | **most LLM GEMMs on MI300X** (N/K of attention/MLP) | very large square tiles where the larger op amortizes overhead |

Default to **16×16** (triton `matrix_instr_nonkdim=16`); only switch to 32×32 if a tune shows it wins
for a specific large square shape. This is the validated MI300X default
(`[[operators/dense_gemm/tuning.md]]`).

## Scheduling levers
- **Keep MFMAs back-to-back**: structure the K-loop so the matrix core issues an MFMA every cadence
  with no stall between them. Compiler/scheduler hints and unrolling (triton `num_stages`, manual
  unroll in CK/asm) keep independent accumulator tiles in flight.
- **Multiple accumulator tiles**: split the C tile into several AGPR sub-tiles so the next K-step's MFMA
  on tile *j* fills the latency of tile *i* — classic latency hiding within the matrix core.
- **Overlap with LDS/global**: while MFMAs run on the current LDS tile, prefetch the next tile via
  `global_load_lds` / double-buffer (`[[optimization/memory_pipelining.md]]`,
  `[[optimization/lds_and_bank_conflicts.md]]`).
- **Operand staging / `b_preshuffle`**: pre-permute B so the `ds_read` feeding MFMA is conflict-free
  and aligned to the MFMA lane layout (`[[operators/dense_gemm/tuning.md]]`).

## Epilogue: the 512B Tagram hotspot
The default CShuffle epilogue routes the accumulator→C write through a small **512-byte Tagram**
staging path that becomes a serialization hotspot on write-heavy epilogues. **`OPTIMIZE_EPILOGUE=1`**
bypasses that path (writes the C tile more directly), recovering epilogue bandwidth — a standard MI300X
default for hand and library GEMM. See `[[operators/gemm_epilogue_fused/overview.md]]`.

## Pitfalls
- Defaulting to `32x32` from CUDA habit — it usually *lowers* MI300X throughput via occupancy loss.
- Single accumulator tile + dependent K chain ⇒ pipeline drains every step (MFMA latency exposed).
- Leaving the default epilogue on write-bound shapes (Tagram serialization).
- Feeding MFMA from conflicted LDS reads (`ds_read` stalls starve the matrix core).
- Mixing accumulate dtype assumptions — MFMA accumulates in fp32; see `[[optimization/numerical_stability.md]]`.

## Verify
- Omniperf: matrix-core busy / `MFMA` issue rate, `VALUBusy`, `ds_read` stall cycles
  (`[[profiling/]]`). Target high MFMA busy with low `ds`/mem stall.
- ISA dump: confirm `v_mfma_*_16x16x16`, count AGPRs, check the K-loop is unrolled with multiple
  accumulators (`[[languages/triton_amd/isa_verify.md]]`).
- A/B: `matrix_instr_nonkdim ∈ {16,32}` and `OPTIMIZE_EPILOGUE ∈ {0,1}`; keep fastest.
