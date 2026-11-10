---
title: MI350X — Matrix Core, MFMA shapes, block-scaled MFMA
kind: hardware
topic: matrix_core
gens: [gfx950]
updated: 2026-08-28
---

# Matrix Core — MFMA, SMFMAC, scaled MFMA

`D = A·B + C` executed as a **wavefront-collective** op: all 64 lanes cooperate on one tile,
low-precision inputs accumulate into **FP32/INT32**. MFMA is mandatory for any competitive
GEMM or attention kernel.

## The one portable fact

**`mfma_16x16` beats `mfma_32x32`** even at large tiles. Both shapes reach the same peak; the
difference is the accumulator footprint:

```
A entries/lane = M·K / 64     B entries/lane = K·N / 64     C entries/lane = M·N / 64
```

16×16 → **4 C-registers/lane**. 32×32 → **16**. That 4× comes straight out of the 512-register budget
and drops occupancy. Choose tile shape by register/LDS pressure, not by peak FLOPS.

## Device totals

256 CU × 4 = **1024 Matrix Cores**, each at **2× the CDNA3 FP16/FP8 rate** (4096 FP16 FLOPs/cycle).
→ FP16/BF16 **2.5 PF**, FP8 **5 PF**, FP6/FP4 **10 PF**.

## Instruction naming

```
v_mfma_<Dtype>_<M>x<N>x<K>_<AB-type>
        │        │  │  │     └─ input dtype of A and B (f16, bf16, fp8/bf8, f8f6f4, i8, f32, f64)
        │        └──┴──┴──────── tile dims: A is M×K, B is K×N, C/D is M×N
        └───────────────────────  output/accumulator dtype (f32, i32, f64)
```

- Dense: `v_mfma_f32_16x16x32_f16`
- Sparse: `v_smfmac_f32_16x16x32_f16` (4:2 structured, ~2× throughput — only with genuinely pruned weights)
- Scaled: `v_mfma_scale_f32_32x32x64_f8f6f4` (per-block E8M0 microscaling)

## Shape / cycle table

| Type (out ← in) | Shapes | Cycles |
|---|---|---|
| FP64 ← FP64 | 16×16×4 | 64 |
| FP32 ← FP32 | 32×32×2 / 16×16×4 | 64 / 32 |
| FP32 ← FP16/BF16 | 32×32×8, 16×16×16, **+ 32×32×16, 16×16×32** | 32 / 16 |
| FP32 ← FP8 (OCP) | 16×16×32, 32×32×16 | 16 / 32 |
| FP32 ← {FP8/FP6/FP4} (f8f6f4) | **16×16×128, 32×32×64** | 16 or 32 / 32 or 64 |
| FP32 ← {MXFP8/6/4} (scaled) | **16×16×128, 32×32×64** | 16 or 32 / 32 or 64 |
| INT32 ← INT8 | 16×16×64, 32×32×32 | 16 / 32 |

> **Cycle rule for f8f6f4 and scaled:** the **lower** count applies when neither A nor B is FP8
> (FP6/FP4-only); the **higher** when **either** matrix is FP8. That is exactly why FP6/FP4 reach 10 PF
> while FP8 tops out at 5 PF.

## Per-lane register footprint

| Instruction | A/lane | B/lane | **C/lane** |
|---|---:|---:|---:|
| `f32_16x16x32_f16` / `_bf16` | 8 | 8 | **4** |
| `f32_32x32x16_f16` / `_bf16` | 8 | 8 | **16** |
| `f32_16x16x128_f8f6f4` | 32 | 32 | **4** |
| `f32_32x32x64_f8f6f4` | 32 | 32 | **16** |
| `scale_f32_32x32x64_f8f6f4` | 32 (+1 Ax) | 32 (+1 Bx) | **16** |
| `i32_16x16x64_i8` | 16 | 16 | **4** |
| `i32_32x32x32_i8` | 16 | 16 | **16** |

## Peak formula

```
peak_FLOPS = 2·M·N·K · num_matrix_cores · (clock_Hz / cycle_count)
```
Check with 1024 cores at ~2.4 GHz: FP16 `32x32x16` @32 cyc → `2·32·32·16 · 1024 · 2.4e9/32 ≈ 2.5 PF` ✓.
FP8 `32x32x64_f8f6f4` @64 cyc → `≈ 5 PF` ✓. FP6/FP4-only drops to 32 cyc → **10 PF**.

## Block-scaled MFMA (the headline gfx950 op)

```cpp
// gfx950, ROCm 7.0+. Type codes: 0=E4M3(fp8) 1=E5M2(bf8) 2=E2M3(fp6) 3=E3M2(bf6) 4=E2M1(fp4)
// scale_a/scale_b are E8M0 -> factor 2^(scale-127); 127 = no scaling.
acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
          a_reg, b_reg, acc,
          /*Atype*/Acode, /*Btype*/Bcode,
          /*OPSEL_A*/0, scale_a,
          /*OPSEL_B*/0, scale_b);
// also: __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4
```

- **A and B types are independent** — mix FP4 weights with FP6/FP8 activations as accuracy demands.
- Scales apply **after the dot product, before accumulation**.
- Classic (unscaled) FP8/FP6/FP4 use `v_mfma_f32_*_f8f6f4` without the scale operands.

**Layout (32×32×64):** A = 32×64, **Ax (scales) = 32×2**, B = 64×32, **Bx = 2×32**, C = 32×32.
Per-thread (wave64): 32 A, 1 Ax, 32 B, 1 Bx, 16 C. Each E8M0 scale covers a **32-element block** of K.
FP4 packs 2 values/byte; the scaled intrinsic wants its first two operands 256-bit wide, so 32 FP4
(128 bit) pad the upper half with zero.

## gfx950 capability list

| Capability | Status |
|---|---|
| FP16 / BF16 MFMA | ✓ — including the new 16×16×32, 32×32×16 shapes |
| FP32 matrix | ✓ (157 TF) |
| FP64 matrix | ✓ — **rate halved** vs CDNA3 |
| INT8 MFMA | ✓ (~5 POPS) |
| FP8 (E4M3 / E5M2) | ✓ — **OCP**, not FNUZ |
| FP6 (E2M3 / E3M2) | ✓ — runs at the **FP4 rate** |
| FP4 (E2M1) | ✓ |
| Block-scaled MXFP8/6/4 (E8M0) | ✓ — `v_mfma_scale_*`, ROCm 7.0+ |
| SMFMAC (4:2 sparse) | ✓ |
| Read-with-transpose LDS for MFMA | ✓ |
| **TF32** | ✗ — **removed**; emulate with BF16 or run FP32 |

## What it means for kernels

1. **16×16 over 32×32** — better LDS/VGPR behaviour, easier double-buffering, same peak.
2. **Push to the lowest viable precision** — FP16/BF16 = 16× FP32, FP8 = 32×, FP6/FP4 = 64×.
   Prefer **FP6 over FP4** when FP4 is too lossy: same 10 PF rate, more mantissa.
3. **Keep accumulators in AGPRs** for large output tiles (`mi350_execution.md`).
4. **Feed from conflict-free LDS** matching the MFMA lane map, over **64 banks** (`mi350_lds.md`);
   use **read-with-transpose** to skip an explicit B transpose.
5. **Use OCP FP8** — re-cast, never bit-copy, any FNUZ checkpoint (`mi350_dtypes.md`).
6. **SMFMAC only with genuinely 4:2-sparse weights**; otherwise dense.

## Pitfalls
- **Choosing 32×32 "because bigger"** — not faster, and 4× the C-register footprint.
- **Conflating peak with achievable** — ~45–55% of peak is the practical ceiling.
- **Feeding FNUZ bits to a gfx950 MFMA** — bias and saturation differ; silently wrong.
- **Assuming TF32 exists** — removed.
- **Down-converting the accumulator** inside the K-loop — always FP32/INT32 through K.
- **Wrong E8M0 (Ax/Bx) scale layout** — silent corruption; check the calculator first.
- **Assuming FP6 is slower than FP4** — same rate.

## Verify
- `amd_matrix_instruction_calculator --architecture cdna4 --instruction <name> --detail-instruction`
  → opcode, M/N/K, **execution cycles**, FLOPs/CU/cycle, VALU co-execution, per-matrix GPR counts and
  alignment, ArchVGPR/AccVGPR eligibility. **Authoritative over any table, including this one.**
- `--get-register --A-matrix --I-coordinate i --K-coordinate k` gives the exact `Vx{lane}.sub` for any
  element — use it to build conflict-free LDS swizzles.
- `--get-register --Ax` / `--Bx` for exact scale placement before wiring up MXFP.

## Related
`mi350_overview.md` · `mi350_dtypes.md` (FP6/FP4/MXFP numerics) · `mi350_execution.md` (the register
budget) · `mi350_lds.md` (feeding the core) · `mi350_isa.md` (opcodes, toolchain)
