---
title: MFMA scheduling — shape choice, wave pattern, keeping the matrix core fed
kind: lever
lever: mfma_sched
gens: [gfx950]
bottleneck: compute-bound
updated: 2026-08-28
---

# MFMA scheduling

## Route here when
- `lever_bottleneck_class.md` said **compute-bound** (MFMA busy high, HBM low).
- MFMA busy is **below ~60%** while the kernel is nominally compute-bound — the matrix core is
  starving, not saturated.
- You are writing a GEMM or fused-attention inner loop from scratch.

**Skip this lever if** the kernel is bandwidth-bound. Feeding the matrix core faster does nothing when
the bottleneck is bytes.

## gfx950 constants

| Fact | Value |
|---|---|
| Matrix cores | 4/CU × 256 CU = **1024** |
| Per-CU matrix throughput | **2× CDNA3** (4096 FP16 FLOPs/cycle/core) |
| Peaks | FP16/BF16 **2.5 PF** · FP8 **5 PF** · FP6/FP4 **10 PF** |
| Accumulate | FP32 / INT32, always |
| FP8 encoding | **OCP** (E4M3FN, E5M2) — *not* FNUZ |
| TF32 | **removed** — fall back to BF16 or FP32 |

Shapes and their per-lane register cost (wave64: `A = M·K/64`, `B = K·N/64`, `C = M·N/64`):

| Instruction | A/lane | B/lane | **C/lane** |
|---|---:|---:|---:|
| `v_mfma_f32_16x16x32_f16` / `_bf16` | 8 | 8 | **4** |
| `v_mfma_f32_32x32x16_f16` / `_bf16` | 8 | 8 | **16** |
| `v_mfma_f32_16x16x128_f8f6f4` | 32 | 32 | **4** |
| `v_mfma_f32_32x32x64_f8f6f4` | 32 | 32 | **16** |
| `v_mfma_scale_f32_*_f8f6f4` | +1 scale reg each | | block-scaled MXFP |
| `v_mfma_i32_16x16x64_i8` | 16 | 16 | **4** |

## Decision 1: 16×16 over 32×32 (default)

**The 4× C-register difference is the whole argument.** Both shapes reach the same peak; 32×32 carries
16 accumulator registers per lane against 16×16's 4. That extra pressure comes straight out of the
512-register budget and drops occupancy (`lever_occupancy.md`).

Set `matrix_instr_nonkdim=16` (Triton) or pick the 16×16 intrinsic directly. Only move to 32×32 if a
measured sweep wins on a specific large square shape — and then verify the register count did not
push you over a tier boundary.

## Decision 2: the wave pattern — do not port the NVIDIA model

**NVIDIA-style producer/consumer wave specialization underperforms on CDNA.** AMD's register allocation
is static: each wave gets a fixed slice of the 512-register file, so dedicating waves as "producers"
starves them of registers, and the kernel tops out around **~80% of peak BF16 GEMM**. There is no
warp-group specialization escape hatch like Hopper's.

Use one of two **symmetric, all-waves-compute** patterns instead (both from HipKittens,
arXiv 2511.08083, since adopted into AMD's own CDNA4 GEMM material):

| Pattern | Shape | When |
|---|---|---|
| **8-wave ping-pong** | 8 waves alternate MFMA-issue and memory phases so the core is always fed | robust default, esp. FP8 GEMM |
| **4-wave interleave** | **one wave per SIMD** → each wave owns the full 512-register budget; 128×128 tile per wave; load/MFMA interleaved in the instruction stream | the successor: no `#pragma unroll` tuning, stable across ROCm releases |

Reference points on MI355X / ROCm 7.1.0, M=N=K=8192 FP8: AMD's HIP 8-wave ping-pong reaches
**3204 TFLOPS** — beating hipBLASLt (3130) **with no assembly**. HipKittens' 8-wave is 3222 TFLOPS in
48 LoC; its 4-wave interleave reaches **3327 TFLOPS** in 183 LoC.

## Decision 3: keep the pipeline from draining

A `v_mfma` has multi-cycle latency. Consecutive **independent** MFMAs pipeline; a **dependent** one
(same accumulator, next K step) stalls until the previous result lands.

- **Split C into multiple accumulator sub-tiles** so the MFMA on tile *j* fills the latency of tile *i*.
  A single accumulator with a dependent K-chain drains the pipeline every step — this is the most
  common cause of low MFMA busy on an otherwise well-written kernel.
- **Unroll the K-loop** enough to keep those independent MFMAs in flight (Triton `num_stages`, manual
  unroll in CK/asm).
- **Overlap with the next tile's load**: while MFMAs consume LDS buffer 0, stage buffer 1 via 128-bit
  `global_load_lds` (`lever_prefetch.md`).
- **Feed from conflict-free LDS.** A conflicted `ds_read` starves the core no matter how good the
  schedule is — and gfx950 has **64 banks**, so any 32-bank swizzle you inherited is wrong
  (`lever_lds_banks.md`).

## gfx950-specific wins

- **Read-with-transpose `ds` loads** — transpose the B operand for free on the LDS read, removing an
  explicit transpose pass.
- **Block-scaled MXFP8/6/4** via `v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4` (ROCm 7.0+): 32-element
  blocks sharing one E8M0 scale. FP6 runs at the **FP4 rate** — if the task tolerates FP6, it is free
  relative to FP4.
- **`OPTIMIZE_EPILOGUE=1`** — writes the C tile in its native MFMA register layout, skipping the LDS
  reblock and the 512 B Tagram staging path that serializes write-heavy epilogues. Standard default;
  verify store coalescing on your shape since the global store may be less coalesced.

## Verify

| Check | How | Pass |
|---|---|---|
| Shape actually emitted | ISA dump: grep `v_mfma_` | the 16×16 form you asked for |
| Pipeline full | `rocprof-compute` matrix-core busy | high MFMA busy, low `ds`/mem stall |
| Multiple accumulators | ISA: count distinct accumulator regs in the K-loop | >1 |
| Cycle counts / eligibility | `amd_matrix_instruction_calculator --architecture cdna4 --instruction <name> --detail-instruction` | authoritative over any table, including this one |
| A/B | `matrix_instr_nonkdim ∈ {16,32}` × `OPTIMIZE_EPILOGUE ∈ {0,1}` | keep fastest |

## Expected magnitude
Fixing a drained pipeline (single → multiple accumulators): **often 1.5–2×**. 32×32 → 16×16 on an
occupancy-limited kernel: **10–30%**. `OPTIMIZE_EPILOGUE` on a write-heavy shape: **5–15%**.
Wave-pattern rework (specialized → symmetric): closes the gap from ~80% to ~95%+ of the library.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Chose 32×32 "because bigger" | CUDA habit | it is not faster; it costs 4× the C registers |
| MFMA busy low, no spills | dependent K-chain, single accumulator | split C into sub-tiles, unroll |
| MFMA busy low, `ds` stalls high | LDS bank conflicts on 64 banks | `lever_lds_banks.md` |
| ~80% of peak ceiling, well-tuned otherwise | producer/consumer wave split | switch to 8-wave ping-pong or 4-wave interleave |
| FP8 results wrong | fed FNUZ bits to an OCP MFMA | `lever_numerics.md` — re-cast, never bit-copy |
| Accuracy drift over long K | down-converted the accumulator in-loop | keep FP32 through the K-loop |

## Deeper
`hardware/mi350_matrix_core.md` (the model, capability list, instruction table, scaled MFMA) ·
`lever_occupancy.md` · `lever_prefetch.md`
