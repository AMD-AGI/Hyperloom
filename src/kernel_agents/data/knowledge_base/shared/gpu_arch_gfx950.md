# gfx950 (MI355X) Architecture Constraints

## Compute Units
- 304 CUs (MI300X: 304 CUs, MI355X: similar)
- Wavefront size: 64 lanes
- Max waves per CU: depends on register pressure

## Register File
- 512 VGPRs per SIMD (256 per wave at occupancy=2)
- AGPR (accumulation registers): separate from VGPR on gfx9
- SGPR: 108 per wave

## Occupancy Model
- **occupancy=2**: VGPR ≤ 256 AND LDS ≤ 80KB per workgroup
- **occupancy=1**: VGPR > 256 OR LDS > 80KB
- This is a STEP FUNCTION — 257 VGPR = occupancy=1, no gradual degradation
- Even 1 byte of spill kills occupancy=2

## LDS (Local Data Share)
- 160KB total per CU (shared across workgroups)
- At occupancy=2: 80KB per workgroup max
- Bank conflicts: 32 banks, 4-byte stride

## HBM Bandwidth
- MI300X: 5.3 TB/s aggregate
- MI355X: similar class

## MFMA Instructions (gfx950)
- `mfma_f32_32x32x16_bf16`: 32×32 output, 16-deep K reduction
- `mfma_f32_16x16x32_bf16`: 16×16 output, 32-deep K reduction
- Throughput: 1 MFMA per cycle per SIMD

### New MFMA shapes on gfx950 (vs gfx942)

| Mnemonic | Shape | Dtype | Note |
|---|---|---|---|
| `v_mfma_f32_16x16x128_f8f6f4` | 16×16×128 | union fp8/fp6/fp4 | K-depth 128; format picked via `cbsz`/`blgp` |
| `v_mfma_f32_32x32x64_f8f6f4`  | 32×32×64  | union fp8/fp6/fp4 | Same, wider tile |
| `v_mfma_scale_f32_16x16x128_f8f6f4` | 16×16×128 | f8f6f4 + per-block scale | **MXFP** — adds `scale_src0, scale_src1, op_sel, op_sel_hi, neg_lo, neg_hi` |
| `v_mfma_scale_f32_32x32x64_f8f6f4`  | 32×32×64  | f8f6f4 + scale | MXFP, wider tile |
| `v_mfma_{f32_16x16x32,f32_32x32x16}_bf16/f16` | — | bf16/f16 | K doubled vs gfx942 (`16x16x16`/`32x32x8`) |
| `v_mfma_{i32_16x16x64,i32_32x32x32}_i8` | — | i8→i32 | K doubled vs gfx942 |

Rule: when writing MXFP kernels, use `v_mfma_scale_*_f8f6f4` directly — do not dequantize-then-MFMA, that's the gfx942 pattern.

### Expanded SMFMAC (2:4 sparse)
Dense equivalent doubled in K. New or widened shapes:
- fp16/bf16: `16x16x{32,64}`, `32x32x{16,32}`
- fp8/bf8 (all four mixes): `16x16x{64,128}`, `32x32x{32,64}`
- i8: `16x16x{64,128}`, `32x32x{32,64}`

Sparsity index is packed in `src2` + modifiers — see `gfx950_operands.html`.

## LDS transpose-reads — new on gfx950

Single-instruction LDS load in MFMA B-operand lane order. Use instead of hand-swizzled LDS layouts.

| Mnemonic | Width | Lane stride | Pairs with |
|---|---|---|---|
| `ds_read_b64_tr_b4`  | 64b | 4-bit  | MX4 / fp4 MFMA |
| `ds_read_b64_tr_b8`  | 64b | 8-bit  | fp8 / bf8 / int8 MFMA |
| `ds_read_b64_tr_b16` | 64b | 16-bit | bf16 / fp16 MFMA |
| `ds_read_b96_tr_b6`  | 96b | 6-bit  | fp6 / bf6 MFMA (paired with f8f6f4) |

Rule: for any new GEMM/attention asm, load B through `ds_read_*_tr_*`; stop crafting permuted LDS layouts for the B operand.

## Global→LDS direct loads — widened on gfx950

```
global_load_lds_{ubyte, sbyte, ushort, sshort, dword, dwordx3, dwordx4}
```

Bypass VGPRs entirely on HBM→LDS. `dwordx4` = 128b/lane transfer. Use whenever VGPR count is the bottleneck (e.g. the `fp8_blockscale_vgpr_reduction` scenario).

## Packed VOP3P — new on gfx950
- `v_pk_add_f32`, `v_pk_mul_f32`, `v_pk_fma_f32` — **packed fp32** (gfx942 had packed fp16 only)
- `v_pk_fmac_f16_dpp` — packed fp16 FMAC with DPP lane shuffle in one op
- `v_pk_maximum3_f16` / `v_pk_minimum3_f16` — IEEE-NaN-propagating 3-way; distinct from `v_pk_max3_*`; useful for attention softmax max-reduction

## New atomics on gfx950
- `flat_atomic_pk_add_f16 / bf16` (+ `global_atomic_pk_add_*`) — packed fp16/bf16 atomic add
- `flat_atomic_max_f64 / min_f64` (+ `global_` variants) — fp64 min/max (gfx942 lacked these)
- `ds_pk_add_f16 / bf16` + `_rtn` — packed LDS atomic add

## What the LLVM asm page does NOT tell you
The `AMDGPUAsmGFX950.html` reference is operand-syntax only. For these you need the gfx950 ISA PDF or `AMDGPUUsage.rst`:
- MFMA latency / back-to-back same-dst hazards (scale with K-depth)
- Exact lane packing for `f8f6f4` (how fp8/fp6/fp4 share a 32-bit lane, `cbsz`/`blgp` selection)
- `s_wait_*` hazards on `v_mfma_scale_*`
- Whether a given fp8×fp4 or fp6×fp4 mix is legal

No WMMA on gfx950 — MFMA only. (Confirmed: zero `v_wmma_*` entries on the page.)

## Per-Lane MFMA Output Layout (mfma_f32_32x32x16_bf16)
```
C_col = lane_mod_32         (FIXED — always column index)
C_row = lane_div_32 * 4 + (reg_idx // 4) * 8 + (reg_idx % 4)
```
WARNING: operand assignment (A, B, C) changes which problem dimension maps to row/col.
Each kernel must derive its layout from first principles, not copy from another kernel.

## Key Performance Thresholds
- Compute-bound: wait/MFMA ratio < 5
- Balanced: wait/MFMA ratio 5-10
- Memory-bound: wait/MFMA ratio > 10
