---
title: Raw GCN assembly — hand-scheduled compute/memory interleave & SMFMAC sparse
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, int8]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
  - https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX940.html
  - https://llvm.org/docs/AMDGPU/gfx9_waitcnt.html
  - https://arxiv.org/abs/2511.08083
  - https://github.com/ROCm/aiter
---

# Raw GCN assembly

## TL;DR
Raw `.s` / inline `asm volatile` is where AITER's fastest paths live — a hand-scheduled interleave of
`buffer_load` / `ds_read` / `v_mfma` that out-schedules LLVM for a specific hot kernel. The two ISA tools
that make it possible: **`s_waitcnt <counter>(N)`** (asynchronous-memory overlap) and **`s_setprio` +
scheduling barriers** (pin the interleave). Also covers **SMFMAC** (2:4 structured-sparse MFMA). Most
production work should *guide* the compiler with intrinsics + `sched_group_barrier` instead (see
[mfma_intrinsics.md](mfma_intrinsics.md)); reach for raw asm only when disassembly proves the compiler is
suboptimal.

## Core concepts
### `s_waitcnt` — the heart of overlap
CDNA memory ops are asynchronous; hardware tracks *outstanding* ops in counters. `s_waitcnt` blocks until
a counter drops to a given value (**not** "wait N instructions"):

| counter | tracks | use |
|---|---|---|
| **vmcnt(N)** | outstanding VMEM (`buffer_load`/`global_load`) | wait until ≤N global loads pending |
| **lgkmcnt(N)** | outstanding LDS (`ds_*`) + scalar (`s_load`) + msg | wait until ≤N LDS/scalar pending |
| **expcnt(N)** | outstanding exports | graphics; rare in compute |
| **q_waitcnt** (gfx950) | async load queue | CDNA4 direct-to-LDS overlap (HipKittens) |

Omitted fields default to **max** ("don't wait"). `s_waitcnt vmcnt(0) lgkmcnt(0)` = full fence.

### The relaxed-count software-pipelining trick
Overlap the *tail* of LDS reads with the MFMA on the *previous* fragment:
```asm
    ds_read_b128  v[8:11],  v20          ; load A frag for NEXT mfma
    ds_read_b128  v[12:15], v24          ; load B frag for NEXT mfma
    s_waitcnt     lgkmcnt(1)             ; proceed once all-but-ONE ds_read returned
    v_mfma_f32_16x16x16_bf16 a[0:3], v[0:3], v[4:7], a[0:3]  ; compute on PREVIOUS frag
    s_waitcnt     lgkmcnt(0)             ; last ds_read in; rotate buffers
```
This is exactly what the LLVM scheduler emits for gfx942 MFMA loops; CK/ck_tile v3/v4 pipelines generate
it for you.

### `s_setprio` + scheduling barriers
`s_setprio` raises wave priority during the compute burst so the MFMA issuer isn't starved.
`__builtin_amdgcn_sched_barrier(mask)` and `__builtin_amdgcn_sched_group_barrier(mask, size, sync_id)`
**pin** the `buffer_load`/`ds_read`/`v_mfma` interleave so the compiler can't reorder it. The
`SchedGroupMask::MFMA` bit identifies MFMA ops for the SW pipeliner — **hand-written MFMA in asm is NOT
recognized** by `SchedGroupMask`, defeating pipelining.

### SMFMAC — 2:4 structured-sparse MFMA (CDNA3)
`v_smfmac_*` implements **2:4 structured sparsity** (sparse A, dense B). The C/Src2 operand is replaced
by a **K matrix of compression indices**: for 8-bit inputs only 16 bits/lane are needed.
`CBSZ==0` → `ABID` selects the top/bottom 16-bit half of the index register; `CBSZ!=0` → `ABID` ignored,
low 16 bits used. (This re-purposes `cbsz/abid`, so they can't also broadcast A.) gfx950 adds
`v_smfmac_f32_16x16x128_*` fp8 variants. Query the index layout:
```bash
./matrix_calculator.py --architecture cdna3 --instruction v_smfmac_f32_16x16x32_f16 --compression --register-layout
# e.g. K[2][31] = v0{50}.[7:4]  -> compression bits in lane 50, bits 4..5 of the Src2 VGPR
```

### CDNA3 memory dataflow
`buffer_load` (global→VGPR) → `ds_write` (VGPR→LDS) → `ds_read` (LDS→VGPR) → `v_mfma`. **Direct-to-LDS
(DGL)** `buffer_load ... lds` collapses global→LDS in one op, skipping the VGPR stage + `ds_write` +
copy index math (a major occupancy/VGPR win). On gfx942 the two-step staging is still common; the fully
working DGL data path for scaled-GEMM operands is primarily a **gfx950** story (HipKittens uses
`buffer_load_dwordx4` for BF16/FP8, `dwordx3` for FP6, synced with `vmcnt`/`s_waitcnt`/`q_waitcnt`).

## The levers
- **Hand-schedule the K-loop** only when the compiler's interleave is provably bad (disassembly).
- **`s_waitcnt lgkmcnt(1)`** before the MFMA = the canonical prefetch-overlap pattern.
- **DGL `buffer_load ... lds`** to skip the VGPR stage where the toolchain supports it.
- **128-bit loads** (`buffer_load_dwordx4`, `ds_read_b128`/`ds_write_b128`) minimize instruction count
  and saturate the data fabric (up to 4 adjacent `dwordx4` form one fabric transaction).

## Pitfalls
- Hand-asm MFMA defeats SW pipelining (`SchedGroupMask` blind) — keep MFMA as the intrinsic, hand-schedule
  only the surrounding loads.
- `s_waitcnt N` means "≤N remaining", not "wait N" — off-by-one here = data race or stall.
- SMFMAC `cbsz/abid` mean *index selection*, not broadcast — don't carry over plain-MFMA habits.
- Inline-asm clobber bugs (early-clobber, `"memory"`, one block) — see [pitfalls.md](pitfalls.md).

## Verify
```bash
hipcc --offload-arch=gfx942 -O3 --save-temps kern.cpp -o kern
grep -E 'v_mfma|v_smfmac|s_waitcnt|s_setprio|accvgpr|ds_read|buffer_load|scratch_' kern-*.s
```
Confirm `s_waitcnt lgkmcnt(1)` before `v_mfma`, 128-bit load widths, no `scratch_` spills.

## Sources
- AMD CDNA3 ISA (waitcnt semantics, SMFMAC, buffer_load lds): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- LLVM gfx940/gfx942 instruction syntax (v_mfma_*, v_smfmac_*, ds_*, buffer_load): https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX940.html
- LLVM gfx9 s_waitcnt semantics ("wait for remaining"): https://llvm.org/docs/AMDGPU/gfx9_waitcnt.html
- amd_matrix_instruction_calculator (SMFMAC compression-index layout, `--compression`): https://github.com/ROCm/amd_matrix_instruction_calculator
- HipKittens (arXiv 2511.08083 — buffer_load_dwordx4/x3 DGL, vmcnt/s_waitcnt/q_waitcnt, AITER raw-asm baselines): https://arxiv.org/abs/2511.08083
- ROCm/aiter (raw-asm fastest paths): https://github.com/ROCm/aiter
