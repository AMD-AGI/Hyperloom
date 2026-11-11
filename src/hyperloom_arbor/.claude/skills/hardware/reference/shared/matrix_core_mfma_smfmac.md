---
title: Matrix Cores — MFMA / SMFMAC / scaled-MFMA (CDNA cross-gen)
kind: hardware
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp64, fp32, tf32, bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp8_e4m3, fp8_e5m2, fp6_e2m3, fp6_e3m2, fp4_e2m1, mxfp8, mxfp6, mxfp4, int8]
regimes: [both]
updated: 2026-11-07
---

# Matrix Cores — MFMA / SMFMAC / scaled-MFMA

> Numbers in brackets point to the Sources at the end.

## TL;DR
> The Matrix Core executes `D = A·B + C` as a **wavefront-collective** op: all **64 lanes** cooperate
> on one tile [1][5], and low-precision inputs accumulate into **FP32/INT32** [1][7]. MFMA is mandatory
> for any competitive GEMM/attention. For GEMM on MI300X, AMD finds **`mfma_16x16` typically beats
> `mfma_32x32`** even at large tiles [4] — both reach the same peak (derived below), so choose tile
> shape by register/LDS pressure, not by peak FLOPS. On MI350X, AMD says to benchmark both [4].

## Concepts

### What a Matrix Core is
- Each CU has **4 Matrix Cores**. Device total = CUs × 4: MI300X 304×4 = 1216 [3]; MI350X 256×4 = 1024
  [4].
- A single MFMA is issued by one wavefront and consumes A/B operands distributed across the 64 lanes'
  VGPRs; the result tile (C/D) is distributed back across lanes' VGPRs or AGPRs [1]. There is **no
  per-lane matmul** — the lane↔element mapping is fixed per instruction (query it with the
  `amd_matrix_instruction_calculator` [6]).
- Inputs are low precision (FP16/BF16/FP8/FP6/FP4/INT8); **accumulation is FP32 (or INT32)** [1][2][7]
  to keep long-K reductions accurate. Never down-convert the accumulator inside the K-loop.

### Naming scheme
```
v_mfma_<Dtype>_<M>x<N>x<K>_<AB-type>
        │        │  │  │     └─ input dtype of A and B (f16, bf16, fp8/bf8, f8f6f4, i8, f32, f64)
        │        └──┴──┴──────── tile dims: A is M×K, B is K×N, C/D is M×N
        └───────────────────────  output/accumulator dtype (f32, i32, f64)
```
(as in the ISA opcode tables [1][2]; the HIP builtins follow `__builtin_amdgcn_mfma_<CDfmt>_<MxNxK><ABfmt>` [5])
- Dense: `v_mfma_f32_16x16x16_bf16` → BF16 in, FP32 accumulate, 16×16 tile, K=16 [1].
- Sparse (CDNA3+): `v_smfmac_f32_16x16x32_f16` → 4:2 structured sparsity [1][2], ~2× throughput [9].
- Scaled (CDNA4): `v_mfma_scale_f32_32x32x64_f8f6f4` → per-block E8M0 scale exponents [2].

### Per-lane register footprint (wave64)
```
A entries/lane = M·K / 64       B entries/lane = K·N / 64       C entries/lane = M·N / 64
```
(derived: the 64 lanes hold each matrix evenly [1])
| Instruction | A/lane | B/lane | C/lane | Vector types |
|---|---|---|---|---|
| `f32_32x32x2_f32` | 1 | 1 | 16 | `float`, `float`, `fp32x16` |
| `f32_16x16x16_f16` | 4 | 4 | 4 | `fp16x4`, `fp16x4`, `fp32x4` |
| `f32_32x32x8_f16` | 4 | 4 | 16 | `fp16x4`, `fp16x4`, `fp32x16` |
| `f32_32x32x16_fp8_fp8` | 8 | 8 | 16 | `fp8x8`(as `long`), `fp8x8`, `fp32x16` |
| `f32_16x16x32_fp8_fp8` | 8 | 8 | 4 | `fp8x8`, `fp8x8`, `fp32x4` |
| `scale_f32_32x32x64_f8f6f4` | 32 (+1 Ax) | 32 (+1 Bx) | 16 | block-scaled (see CDNA4 file) |

### The peak-throughput formula (the only one you need)
```
peak_FLOPS = 2·M·N·K · num_matrix_cores · (clock_Hz / cycle_count)
```
Check FP16 `32x32x8` on MI300X (32 cycles [1], 1216 cores, 2.1 GHz [3]):
`2·32·32·8 · 1216 · (2.1e9/32) ≈ 1307 TFLOP/s` ✓ (data sheet: 1307.4 [3]). FP8 `16x16x32` (16 cycles
[1]): `2·16·16·32 · 1216 · (2.1e9/16) ≈ 2615 TFLOP/s` ✓. Two instructions can reach the **same** peak
(FP16 `16x16x16`@16cyc vs `32x32x8`@32cyc [1]) — pick by tile fit.

### Accumulators in AGPRs
MFMA can read/write its C/D tile from **AGPRs** (accumulation VGPRs) [1], freeing the architected VGPR
budget that limits occupancy. Data moves between them with `v_accvgpr_read_b32`/`v_accvgpr_write_b32`
[1]; AMD's guide puts the epilogue AGPR→VGPR copies at about 5% of the loop for compute-bound kernels
with large K [4]. See [wavefront_simd_vgpr_agpr.md](wavefront_simd_vgpr_agpr.md).

## The levers
1. **Prefer 16×16 over 32×32** on MI300X [4] — smaller C footprint, easier double-buffering; benchmark
   on MI350X [4].
2. **Push to the lowest viable precision**: per CU, FP16/BF16 = 8× and FP8 = 16× the FP32 matrix rate
   on CDNA3 [9]; on CDNA4, FP8 = 32× and MXFP6/MXFP4 = 64× (8192 and 16384 vs 256 FLOPs/clock/CU [10]).
3. **Feed MFMA from LDS with a swizzled layout** matching the lane mapping so `ds_read_b128` is
   conflict-free → see [memory_model_lds_bank.md](memory_model_lds_bank.md).
4. **Keep accumulators in AGPRs** for large output tiles (`amdgpu-mfma-vgpr-form=false`,
   `amdgpu-agpr-alloc=256` [4]).
5. **Match the FP8 variant to the arch**: FNUZ on CDNA3, OCP on CDNA4 [11] — see
   [dtype_numerics.md](dtype_numerics.md). A checkpoint must be re-cast, never bit-copied across.
6. **Use SMFMAC only with genuinely 4:2-sparse weights** (post structured pruning); otherwise dense.

## Cross-generation MFMA capability matrix
| Capability | CDNA1 (gfx908) | CDNA2 (gfx90a) | CDNA3 (gfx942) | CDNA4 (gfx950) |
|---|---|---|---|---|
| FP16 / BF16 MFMA | ✓ (BF16 half-rate) | ✓ | ✓ | ✓ (new larger shapes 16×16×32, 32×32×16) |
| FP32 / FP64 matrix | FP32 ✓ / FP64 ✗ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ (FP64 matrix halved) |
| INT8 MFMA | ✓ | ✓ | ✓ | ✓ |
| TF32 | ✗ | ✗ | ✓ (MFMA `xf32`) | **removed** (software emulation via BF16) |
| FP8 (E4M3/E5M2) | ✗ | ✗ | ✓ **FNUZ** | ✓ **OCP** |
| FP6 / FP4 | ✗ | ✗ | ✗ | ✓ |
| Block-scaled MXFP8/6/4 (E8M0) | ✗ | ✗ | ✗ | ✓ (`mfma_scale_*`) |
| SMFMAC (4:2 sparse) | ✗ | ✗ | ✓ | ✓ |
| Read-with-transpose LDS for MFMA | ✗ | ✗ | ✗ | ✓ |

Sources by column: CDNA1 [7][12]; CDNA2 [8][3] (the MI300X data sheet notes MI200 has no TF32, FP8 or
sparsity); CDNA3 [1][9][11]; CDNA4 [2][10][11]. Per-gen instruction tables, intrinsics, and numerics
live in each gen's `matrix_core*.md`.

## Pitfalls
- **Conflating peak with achievable.** One published study measured MI300X sustaining an average of
  **~45% of peak** FLOPs on square GEMMs across FP8/BF16/FP16 [13]. Quote achieved, never peak.
- **Wrong FP8 variant.** FNUZ vs OCP differ in bias and saturation [1][2]; mixing corrupts results
  silently.
- **Down-converting the accumulator.** Always FP32/INT32 through the K-loop.
- **Choosing 32×32 "because it's bigger."** It is not faster; it raises C-register footprint
  (16 C/lane vs 4) and hurts occupancy.

## Verify
- `amd_matrix_instruction_calculator --architecture cdna1|cdna2|cdna3 --instruction <name>
  --detail-instruction` reports opcode, register usage, **execution cycles**, FLOPs/CU/cycle, VALU
  co-execution, and ArchVGPR/AccVGPR eligibility [6]. It has no CDNA4 target [6]; for gfx950 use the
  CDNA4 ISA guide's MFMA tables and layouts [2].
- `--get-register --A-matrix --I-coordinate i --K-coordinate k` gives the register and lane of any
  element [6] (use to build conflict-free LDS swizzles).

## Sources
1. AMD, *"AMD Instinct MI300" Instruction Set Architecture* (CDNA3 ISA; §7.1 MFMA, SMFMAC and AccVGPRs, §7.2 formats) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
2. AMD, *AMD Instinct CDNA4 Instruction Set Architecture* (§7.1 MFMA incl. scaled MFMA, §7.5 SMFMAC, §11.4 transpose loads) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
3. AMD, *AMD Instinct MI300X Accelerator* data sheet — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
4. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* — https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html
5. AMD GPUOpen, *AMD matrix cores* (AMD lab notes) — https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
6. ROCm, *AMD Matrix Instruction Calculator* README (supported architectures CDNA1–CDNA3, RDNA3–RDNA4) — https://github.com/ROCm/amd_matrix_instruction_calculator
7. AMD, *AMD CDNA Architecture* white paper (MI100; Table 1) — https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna-white-paper.pdf
8. AMD, *AMD CDNA 2 Architecture* white paper (peak table) — https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf
9. AMD, *AMD CDNA 3 Architecture* white paper (Table 1; sparsity) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
10. AMD, *AMD CDNA 4 Architecture* white paper (Table 1; TF32) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
11. HIP documentation, *Low precision floating point types* (FP8/FP6/FP4 support per architecture) — https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html
12. AMD, *"AMD Instinct MI100" Instruction Set Architecture* (CDNA1 ISA) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi100-cdna1-shader-instruction-set-architecture.pdf
13. C. Ambati, T. Diep, *AMD MI300X GPU Performance Analysis*, arXiv:2510.27583 — https://arxiv.org/abs/2510.27583
