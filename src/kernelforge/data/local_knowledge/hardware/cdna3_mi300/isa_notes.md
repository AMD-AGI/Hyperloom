---
title: CDNA3 / MI300X (gfx942) — ISA notes for kernel authors
kind: hardware
gens: [gfx942]
dtypes: []
regimes: [both]
updated: 2026-06-08
---

# CDNA3 / MI300X (gfx942) — ISA notes for kernel authors

> Just the gfx942 ISA facts that change how you write/read kernels. MFMA detail in
> [matrix_core.md](matrix_core.md); memory ops in [memory_hierarchy.md](memory_hierarchy.md).

## TL;DR
> Target **`--offload-arch=gfx942`**, **wave64 only** (no wave32). Key instruction families:
> `v_mfma_*`/`v_smfmac_*` (Matrix Core), `buffer_*`/`global_*`/`ds_*` (memory, incl. **direct
> global→LDS**), `v_accvgpr_read/write_b32` (VGPR↔AGPR), and **count-based `s_waitcnt`** (vmcnt /
> lgkmcnt / expcnt) for fine-grained overlap.

## Concepts

### Target & toolchain
| Item | Value |
|---|---|
| ISA target | `gfx942` (gfx940/941 = MI300A/early steppings) |
| Compile | `--offload-arch=gfx942` (or `-mcpu=gfx942`) |
| Wave size | **64** (no wave32 on CDNA3) |
| Calculator keyword | `cdna3` |
| Profilers | `rocprofv3`, `rocprof-compute` (Omniperf), `rocprof` |
| Sibling | `gfx950` (CDNA4, MI350/355) — see [../cdna4_mi350/isa_notes.md](../cdna4_mi350/isa_notes.md) |

### Matrix instructions
- Dense `v_mfma_<out>_<MxNxK>_<in>`; sparse `v_smfmac_*` (4:2). Modifiers `cbsz/abid/blgp` =
  broadcast/lane-group control (0 for ordinary GEMM). Full table: [matrix_core.md](matrix_core.md).
- **No** scaled (`v_mfma_scale_*`) or FP6/FP4 opcodes on gfx942 — CDNA4 only.

### Memory instruction families
- **`global_*`** — flat 64-bit addressing, default for HIP pointer loads (`global_load_dwordx4`).
- **`buffer_*`** — descriptor (V#) addressing with hardware **OOB bounds checking** and cache flags
  (`glc/slc/dlc`) → cheaper guards in tiled GEMM.
- **`ds_*`** — LDS (`ds_read_b128`/`ds_write_b128` preferred for width).
- **Direct global→LDS**: `buffer_load_dwordx4 ... lds` / `global_load_lds_dwordx4` bypass VGPRs
  (up to 32 b/lane on CDNA3). The single biggest GEMM occupancy win — see
  [memory_hierarchy.md](memory_hierarchy.md).

### Register-move & sync
- **`v_accvgpr_write_b32 acc, v`** / **`v_accvgpr_read_b32 v, acc`** move VGPR↔AGPR (epilogue
  read-out of MFMA accumulators; also cheap compiler spill/fill).
- **`s_waitcnt`** is **count-based**, not a fence: `vmcnt` (vector memory), `lgkmcnt` (LDS/scalar/const),
  `expcnt` (exports). Wait only for the specific outstanding ops the next instruction needs → deep
  overlap of loads with MFMA. `s_barrier` synchronizes a workgroup.

### Packed & special math
- Packed FP16/INT16 `pk` ops run **double-rate**; INT32 add is single-rate.
- Transcendentals (rsqrt/exp/…) at 4 ops/SIMD/cycle on the transcendental unit.
- `EXEC` is a 64-bit mask; cross-lane ops (`ds_swizzle`, `v_permlane*`, DPP) span 64 lanes.

### Reading the ISA dump
```bash
hipcc --offload-arch=gfx942 -S -o - kern.hip       # AMDGCN assembly
# or extract from the fat binary:
llvm-objdump -d --arch-name=amdgcn --mcpu=gfx942 kern.o
```
Look for `.vgpr_count`, `.agpr_count`, `.sgpr_count`, `.lds_size` in the kernel metadata, and confirm
`v_mfma_*` and `buffer_load ... lds` are actually emitted (not register-staged copies).

## The levers
1. **Emit `buffer_load ... lds`** for tile staging (check the dump).
2. **Use `buffer_*` + bounds check** instead of branchy guards in tiled GEMM.
3. **Fine-grained `s_waitcnt`** to overlap loads with MFMA, not a blanket wait-all.
4. **`v_accvgpr_*` epilogue** to keep accumulators in AGPRs through the K-loop.
5. **`ds_*_b128`** for wide LDS traffic.

## Pitfalls
- **Assuming wave32** — all masks/shuffles are 64-wide.
- **Blanket `s_waitcnt 0`** kills overlap; wait per-count.
- **Expecting CDNA4 opcodes** (`v_mfma_scale_*`, FP6/FP4) on gfx942.

## Verify
- Disassemble and confirm MFMA shape, AGPR usage, and direct-to-LDS loads.
- `amd_matrix_instruction_calculator --architecture cdna3 --list-instructions` for the full MFMA set.
