---
title: Matrix Cores — MFMA / SMFMAC / scaled-MFMA (CDNA cross-gen)
kind: hardware
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp64, fp32, tf32, bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp8_e4m3, fp8_e5m2, fp6_e2m3, fp6_e3m2, fp4_e2m1, mxfp8, mxfp6, mxfp4, int8]
regimes: [both]
updated: 2026-06-08
---

# Matrix Cores — MFMA / SMFMAC / scaled-MFMA

## TL;DR
> The Matrix Core executes `D = A·B + C` as a **wavefront-collective** op: all **64 lanes** cooperate
> on one tile, low-precision inputs accumulate into **FP32/INT32**. MFMA is mandatory for any
> competitive GEMM/attention. The single most portable fact: **`mfma_16x16` beats `mfma_32x32`** on
> MI300X/MI350X even at large tiles — choose tile shape by register/LDS pressure, not by peak FLOPS
> (both reach the same peak).

## Concepts

### What a Matrix Core is
- Each CU has **4 Matrix Cores** (one per SIMD). Device total = CUs × 4 (e.g. MI300X 304×4 = 1216;
  MI350X 256×4 = 1024).
- A single MFMA is issued by one wavefront and consumes A/B operands distributed across the 64 lanes'
  VGPRs; the result tile (C/D) is distributed back across lanes' VGPRs or AGPRs. There is **no
  per-lane matmul** — the lane↔element mapping is fixed per instruction (query it with the
  `amd_matrix_instruction_calculator`).
- Inputs are low precision (FP16/BF16/FP8/FP6/FP4/INT8); **accumulation is FP32 (or INT32)** to keep
  long-K reductions accurate. Never down-convert the accumulator inside the K-loop.

### Naming scheme
```
v_mfma_<Dtype>_<M>x<N>x<K>_<AB-type>
        │        │  │  │     └─ input dtype of A and B (f16, bf16, fp8/bf8, f8f6f4, i8, f32, f64)
        │        └──┴──┴──────── tile dims: A is M×K, B is K×N, C/D is M×N
        └───────────────────────  output/accumulator dtype (f32, i32, f64)
```
- Dense: `v_mfma_f32_16x16x16_bf16` → BF16 in, FP32 accumulate, 16×16 tile, K=16.
- Sparse (CDNA3+): `v_smfmac_f32_16x16x32_f16` → 4:2 structured sparsity, ~2× throughput.
- Scaled (CDNA4): `v_mfma_scale_f32_32x32x64_f8f6f4` → per-block E8M0 microscaling exponents.

### Per-lane register footprint (wave64)
```
A entries/lane = M·K / 64       B entries/lane = K·N / 64       C entries/lane = M·N / 64
```
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
Check FP16 `32x32x8` on MI300X (cycles=32, 1216 cores, 2.1 GHz):
`2·32·32·8 · 1216 · (2.1e9/32) ≈ 1307 TFLOP/s` ✓. FP8 `16x16x32` (cycles=16):
`2·16·16·32 · 1216 · (2.1e9/16) ≈ 2615 TFLOP/s` ✓. Two instructions can reach the **same** peak
(FP16 `16x16x16`@16cyc vs `32x32x8`@32cyc) — pick by tile fit.

### Accumulators in AGPRs
MFMA can read/write its C/D tile from **AGPRs** (accumulation VGPRs), freeing the architected VGPR
budget that limits occupancy. The compiler inserts `v_accvgpr_read_b32` in the epilogue before the
global store (~5% epilogue cost). See [wavefront_simd_vgpr_agpr.md](wavefront_simd_vgpr_agpr.md).

## The levers
1. **Prefer 16×16 over 32×32** on MI300X/MI350X — better LDS/VGPR behavior, easier double-buffering.
2. **Push to the lowest viable precision**: FP16/BF16 = 8× FP32; FP8 = 16×; FP6/FP4 (CDNA4) = 32–64×.
3. **Feed MFMA from LDS with a swizzled layout** matching the lane mapping so `ds_read_b128` is
   conflict-free → see [memory_model_lds_bank.md](memory_model_lds_bank.md).
4. **Keep accumulators in AGPRs** for large output tiles (`-mllvm -amdgpu-mfma-vgpr-form=false
   -mllvm -amdgpu-agpr-alloc=256`).
5. **Match the FP8 variant to the arch**: FNUZ on CDNA3, OCP on CDNA4 — see
   [dtype_numerics.md](dtype_numerics.md). A checkpoint must be re-cast, never bit-copied across.
6. **Use SMFMAC only with genuinely 4:2-sparse weights** (post structured pruning); otherwise dense.

## Cross-generation MFMA capability matrix
| Capability | CDNA1 (gfx908) | CDNA2 (gfx90a) | CDNA3 (gfx942) | CDNA4 (gfx950) |
|---|---|---|---|---|
| FP16 / BF16 MFMA | ✓ (BF16 half-rate) | ✓ | ✓ | ✓ (new larger shapes 16×16×32, 32×32×16) |
| FP32 / FP64 matrix | FP32 ✓ / FP64 ✗ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ (FP64 matrix halved) |
| INT8 MFMA | ✓ | ✓ | ✓ | ✓ |
| TF32 (emulated) | ✗ | ✗ | ✓ | **removed** (BF16/FP32 emulation only) |
| FP8 (E4M3/E5M2) | ✗ | ✗ | ✓ **FNUZ** | ✓ **OCP** |
| FP6 / FP4 | ✗ | ✗ | ✗ | ✓ |
| Block-scaled MXFP8/6/4 (E8M0) | ✗ | ✗ | ✗ | ✓ (`mfma_scale_*`) |
| SMFMAC (4:2 sparse) | ✗ | ✗ | ✓ | ✓ |
| Read-with-transpose LDS for MFMA | ✗ | ✗ | ✗ | ✓ |

Per-gen instruction tables, intrinsics, and numerics live in each gen's `matrix_core*.md`.

## Pitfalls
- **Conflating peak with achievable.** MI300X commonly sustains only **~45% of peak** FLOPs across
  FP8/BF16/FP16 (arXiv 2510.27583). Quote achieved, never peak.
- **Wrong FP8 variant.** FNUZ vs OCP differ in bias and saturation; mixing corrupts results silently.
- **Down-converting the accumulator.** Always FP32/INT32 through the K-loop.
- **Choosing 32×32 "because it's bigger."** It is not faster; it raises C-register footprint
  (16 C/lane vs 4) and hurts occupancy.

## Verify
- `amd_matrix_instruction_calculator --architecture cdna3|cdna4 --instruction <name>
  --detail-instruction` reports opcode, M/N/K, **execution cycles**, FLOPs/CU/cycle, VALU
  co-execution, per-matrix GPR counts/alignment, and ArchVGPR/AccVGPR eligibility. Treat it as
  authoritative over any blog table.
- `--get-register --A-matrix --I-coordinate i --K-coordinate k` gives the exact `Vx{lane}.sub` for
  any element (use to build conflict-free LDS swizzles).
