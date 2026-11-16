---
title: RDNA 3 (gfx1100) — WMMA matrix instructions
kind: hardware
topic: matrix
gens: [gfx1100]
updated: 2026-11-16
---

# WMMA matrix instructions

gfx1100's matrix path. It replaces everything the MFMA cards say: there is no MFMA, no AGPR, no scaled or
sparse matrix op, and one shape.

## Instructions (ISA §7.9, Table 33)

| Instruction | A, B | C, D |
|---|---|---|
| `v_wmma_f32_16x16x16_f16` | FP16 | FP32 |
| `v_wmma_f32_16x16x16_bf16` | BF16 | FP32 |
| `v_wmma_f16_16x16x16_f16` | FP16 | FP16 |
| `v_wmma_bf16_16x16x16_bf16` | BF16 | BF16 |
| `v_wmma_i32_16x16x16_iu8` | INT8 / UINT8 | INT32 |
| `v_wmma_i32_16x16x16_iu4` | INT4 / UINT4 | INT32 |

- **16×16×16 is the only shape.** Tiles are multiples of 16 in M, N and K.
- **No FP8, FP4, TF32, FP64 or sparsity** on gfx11 (FP8 WMMA arrives with gfx12). rocWMMA's FP8 types require
  gfx1200/gfx1201.
- Rounding is round-to-nearest-even only.

## Operand layout

- **A and B must be replicated across half-waves.** In wave32, the data in lanes 0–15 is duplicated in lanes
  16–31; in wave64, also in lanes 32–47 and 48–63. A kernel that loads a fragment once per 16 lanes must copy it.
  (ISA §7.9, GPUOpen WMMA blog)
- A is column-major in its VGPRs; B, C and D are row-major. (ISA §7.9)
- Fragment sizes per lane:

| Operand | wave32 | wave64 |
|---|---|---|
| A or B, FP16/BF16 | 8 VGPRs | 8 VGPRs |
| A or B, IU8 | 4 VGPRs | 4 VGPRs |
| A or B, IU4 | 2 VGPRs | 2 VGPRs |
| C / D, 32-bit | 8 VGPRs | 4 VGPRs |

- For the FP16- and BF16-output forms, an OPSEL constant chooses whether the 16-bit results land in the low or
  high halves of the D registers.

## HIP builtins

```cpp
// wave32 (the gfx1100 default)
c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);   // a, b: half16; c: float8
c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
// FP16/BF16-output forms take the OPSEL bool last:
d = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c, /*opsel=*/false);
// integer forms take signedness flags for A and B and a clamp flag
c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(/*a_signed=*/true, a, /*b_signed=*/true, b, c, /*clamp=*/false);
```

- `_w64` variants exist for wave64 builds; a `_w32` builtin in a wave64 build does not compile.
- Builtins require the `wmma-256b-insts` feature (gfx11), which `--offload-arch=gfx1100` enables.
- rocWMMA supports gfx1100 and is the portable alternative to raw builtins.
- (clang `BuiltinsAMDGPU.td`, GPUOpen WMMA blog)

## Hazard

A WMMA whose A or B operand reads the D result of the immediately preceding WMMA needs one `v_nop` or an
independent VALU instruction in between. Back-to-back WMMAs chained only through C are fine. (ISA §7.9) The
compiler inserts the wait; hand-written assembly must.

## Throughput

| Op | Per CU per clock |
|---|---|
| FP16 / BF16 / IU8 WMMA | 512 ops |
| IU4 WMMA | 1024 ops |
| FP32 VALU (dual issue) | 256 FLOPs |

WMMA gives about **2×** the FP32 VALU rate (GPUOpen WMMA blog, measured on the same Navi 31 die). That is a far
smaller matrix-to-vector gap than on CDNA, so a kernel with substantial non-matrix work per tile (softmax,
dequantization, epilogues) is bound by the VALU sooner than on MI300X/MI355X.

## Verifying in the ISA

- `v_wmma_*` present in the hot loop, with no `v_mfma_*` (which would not assemble for gfx1100 anyway).
- Count the copies that replicate A/B across half-waves; if they dominate, restructure the load so the
  replication comes from the load itself.
- `v_dual_*` around the WMMA shows the epilogue VALU work is being paired.
