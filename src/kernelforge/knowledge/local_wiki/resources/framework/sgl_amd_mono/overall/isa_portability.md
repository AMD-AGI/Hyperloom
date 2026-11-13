---
title: ISA portability of sgl-amd-mono launches — what is gfx950-only and what a gfx942 port needs
kind: technique
gens: [gfx942, gfx950]
updated: 2026-11-09
sources:
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/common/ops.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/common/mx.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/common/plan.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/minimax_m31/runner.py
  - https://llvm.org/docs/AMDGPUUsage.html
---

# ISA portability

Every launch sgl-amd-mono ships is built for gfx950, and `common/` uses gfx950 instructions directly.
The ISA is explicit everywhere a spec meets the hardware: launches live under `<model>/<isa>/`,
`MonoSpec.devices` is keyed by ISA, and `BuildContext.isa` reaches `build`. A port adds a
`devices` entry and the code paths that entry needs; it does not change what gfx950 builds.

## gfx950-only primitives in `common/`
| Primitive | Lowers to | Reached through | On gfx942 |
|---|---|---|---|
| `ops.permlane_swap` | `v_permlane16_swap`, `v_permlane32_swap` | `xred` and `_xpartner` for xor offsets 16 and 32, hence `wave_sum`, `butterfly` over those offsets, `wave_incl_scan`, `block_excl_scan` | no such instruction: cross a DPP row with `ds_bpermute` (`ops.lane_gather`, `ops.bpermute_i32`) or GFX9's DPP `row_bcast:15` / `row_bcast:31`, keeping the reduction order where results must stay bitwise |
| `ops.mfma_bf16` | `v_mfma_f32_16x16x32_bf16` | bf16 GEMV / GEMM steps | K is 16 per instruction (`v_mfma_f32_16x16x16_bf16`): twice the issues for the same tile |
| `ops.fp8_to_bf16x8`, `ops.mxfp4_to_bf16x8` | `v_cvt_scalef32_pk_bf16_fp8`, `..._fp4` | FP8 / MXFP4 weight dequant | no scaled converts: unpack and scale in VALU |
| `mx.mfma_scaled` | `v_mfma_scale_f32_16x16x128_f8f6f4` | MXFP4 / MXFP8 matrix steps | no block-scaled MFMA: dequantize, then a bf16 or FP8 MFMA |

`ops.fp8_pack4` and `ops.fp8_roundtrip` (`v_cvt_pk_fp8_f32`) compile on both, but encode different
formats: OCP E4M3FN on gfx950 (`mx.FP8_MAX = 448`), FNUZ E4M3 on gfx942 (max 240, SGLang tensors are
`float8_e4m3fnuz`). A quantizing stage takes its maximum and format from the target, not from `mx`.

## Budgets that differ
| | gfx950 (MI350X / MI355X) | gfx942 (MI300X) |
|---|---|---|
| CUs | 256 | 304 |
| LDS per CU | 160 KiB | 64 KiB |
| Unified VGPRs per SIMD lane | 512 | 512 |

- One 512-thread CTA a CU is 2 waves a SIMD, so a wave gets at most 256 registers on either ISA; past
  that it spills.
- LDS is the hard wall: a launch that needs more than 64 KiB a CTA is not resident on gfx942 at all.
  MiniMax-M3.1-Flash's FFN launch uses 66,816 B at 1 row and 163,264 B at 16 rows (cross-compiled for
  gfx950 with FlyDSL 0.4.0 and read from the code objects' metadata), so neither width fits gfx942 as
  written.
- `BLOCKS = 256` is resident on MI300X too but leaves 48 CUs idle; a gfx942 entry derives its grid
  from the device (`DeviceLimits.resident_blocks`) rather than inheriting 256.

## Memory model and peers
- `ops.CM_DEV = 16` (SC1, device scope), `CM_SYS = 17` (SC0 | SC1, system scope), `CM_NT = 2`
  (nontemporal) are the GFX940-family cache-policy fields, which gfx942 shares (AMDGPUUsage, GFX942
  memory model). Confirm on the gfx942 build by disassembling a mailbox store and reading its
  `sc0 sc1` modifiers.
- `PeerBuffer` (uncached allocation + HIP IPC over XGMI) works unchanged on MI300X.

## Porting discipline
1. Keep gfx950 byte-for-byte: add the gfx942 path beside the gfx950 one, selected by the build's ISA,
   and prove the gfx950 launches still compile to the same code objects
   (`skills/profile/compile_check.md`).
2. Re-derive every tile, slot count and LDS pool for the target's budgets; never inherit gfx950's.
3. Re-derive `NUMERICS` from check mode on the new ISA: the standard path it is compared against
   changes with the ISA too.
