---
title: CDNA3 / MI300X (gfx942) — ISA notes for kernel authors
kind: hardware
gens: [gfx942]
dtypes: []
regimes: [both]
updated: 2026-11-07
---

# CDNA3 / MI300X (gfx942) — ISA notes for kernel authors

> Just the gfx942 ISA facts that change how you write/read kernels. MFMA detail in
> [matrix_core.md](matrix_core.md); memory ops in [memory_hierarchy.md](memory_hierarchy.md). Numbers in
> brackets point to the Sources at the end.

## TL;DR
> Target **`--offload-arch=gfx942`**, **wave64 only** [2][5]. Key instruction families [1]:
> `v_mfma_*`/`v_smfmac_*` (Matrix Core), `buffer_*`/`global_*`/`ds_*` (memory, incl. **direct
> global→LDS**, at most 32 bits per lane on gfx942), `v_accvgpr_read/write_b32` (VGPR↔AGPR), and
> **count-based `s_waitcnt`** (vmcnt / lgkmcnt / expcnt) for fine-grained overlap.

## Concepts

### Target & toolchain
| Item | Value |
|---|---|
| ISA target | `gfx942`: MI300X and MI300A [3], MI325X [2]; the matrix calculator also accepts `gfx940`/`gfx941` as CDNA3 [4] |
| Compile | `--offload-arch=gfx942` (or `-mcpu=gfx942` [3]) |
| Wave size | **64** [2]; on CDNA the wavefront size is always 64 [5] |
| Calculator keyword | `cdna3` [4] |
| Profilers | `rocprofv3`, `rocprof-compute` (Omniperf), `rocprof` |
| Sibling | `gfx950` (CDNA4, MI350/355) — see [../cdna4_mi350/isa_notes.md](../cdna4_mi350/isa_notes.md) |

### Matrix instructions
- Dense `v_mfma_<out>_<MxNxK>_<in>`; sparse `v_smfmac_*` (4:2 structured sparsity) [1]. Modifiers
  `cbsz/abid/blgp` = A-block broadcast and B lane-group pattern; their default 0 means no broadcast
  and the normal layout [4][6]. Full table: [matrix_core.md](matrix_core.md).
- **No** scaled (`v_mfma_scale_*`) or FP6/FP4 opcodes on gfx942 — CDNA4 only [7][8].

### Memory instruction families
- **`global_*`** — FLAT-format loads/stores to the global aperture (VGPR address, or SGPR base +
  VGPR offset) [1], e.g. `global_load_dwordx4`.
- **`buffer_*`** — descriptor (V#) addressing with hardware **range checking**: out-of-range reads
  return zero and out-of-range writes and atomics are dropped [1] → cheaper guards in tiled GEMM.
  Cache behavior is set per instruction by the scope bits `sc0/sc1` and the non-temporal bit `nt` [1].
- **`ds_*`** — LDS (`ds_read_b128`/`ds_write_b128` [1]; AMD's guide recommends the `_b128` forms [9]).
- **Direct global→LDS**: `buffer_load_dword ... lds` (MUBUF with LDS=1) and `global_load_lds_dword`
  write to LDS without passing through VGPRs [1]. On gfx942 only the ubyte/sbyte/ushort/sshort/dword
  sizes exist, so at most **32 bits per lane** [1]; gfx950 adds dwordx3/dwordx4 [7]. AMD's guide
  reports ~100 VGPRs per wave saved by `buffer_load_to_lds` on its gfx950 reference GEMM [9] — see
  [memory_hierarchy.md](memory_hierarchy.md).

### Register-move & sync
- **`v_accvgpr_write_b32 acc, v`** / **`v_accvgpr_read_b32 v, acc`** move VGPR↔AGPR [1] (epilogue
  read-out of MFMA accumulators).
- **`s_waitcnt`** is **count-based**, not a fence: `vmcnt` (vector memory), `lgkmcnt`
  (LDS/GWS/constant/message), `expcnt` (exports); it waits until each counter is at or below the given
  value [1]. Waiting only for the specific outstanding ops the next instruction needs lets loads
  overlap MFMA [9]. `s_barrier` synchronizes the waves of a workgroup [1].

### Packed & special math
- Packed math (VOP3P, `v_pk_*`) performs two 16-bit operations per 32-bit lane register [1].
- `EXEC` is a 64-bit mask, one bit per lane [1]; cross-lane ops on gfx942 are DPP, `ds_swizzle_b32`
  and `ds_bpermute_b32` [1] — there is no `v_permlane*` on gfx942 (CDNA4 has it [7]).

### Reading the ISA dump
```bash
hipcc --offload-arch=gfx942 -S -o - kern.hip       # AMDGCN assembly
# or extract from the fat binary:
llvm-objdump -d --arch-name=amdgcn --mcpu=gfx942 kern.o
```
Look for `.vgpr_count`, `.agpr_count`, `.sgpr_count` and `.group_segment_fixed_size` (LDS bytes) in the
kernel metadata [3], and confirm `v_mfma_*` and `buffer_load ... lds` are actually emitted (not
register-staged copies).

## The levers
1. **Emit `buffer_load ... lds`** for tile staging (check the dump) [1][9].
2. **Use `buffer_*` range checking** instead of branchy guards in tiled GEMM [1].
3. **Fine-grained `s_waitcnt`** to overlap loads with MFMA, not a blanket wait-all [9].
4. **`v_accvgpr_*` epilogue** to keep accumulators in AGPRs through the K-loop [1].
5. **`ds_*_b128`** for wide LDS traffic [9].

## Pitfalls
- **Assuming wave32** — all masks/shuffles are 64-wide [1][5].
- **Blanket `s_waitcnt 0`** kills overlap; wait per-count [9].
- **Expecting CDNA4 opcodes** (`v_mfma_scale_*`, FP6/FP4, `v_permlane*`, wide direct-to-LDS loads) on
  gfx942 [1][7].

## Verify
- Disassemble and confirm MFMA shape, AGPR usage, and direct-to-LDS loads.
- `amd_matrix_instruction_calculator --architecture cdna3 --list-instructions` for the full MFMA set [4].

## Sources
1. AMD, *"AMD Instinct MI300" Instruction Set Architecture* (CDNA3 ISA reference guide; §4.4 data dependency, §6.7 packed math, §7.1 MFMA, §9.1 buffer instructions incl. range checking, load to LDS and scope bits, §12 opcodes) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
2. ROCm documentation, *AMD GPU specifications* — https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
3. LLVM, *User Guide for AMDGPU Backend* (processor table, code object metadata) — https://llvm.org/docs/AMDGPUUsage.html
4. ROCm, *AMD Matrix Instruction Calculator* README — https://github.com/ROCm/amd_matrix_instruction_calculator
5. ROCm Compute Profiler documentation, *Pipeline metrics* ("the wavefront size is always 64") — https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.3.1/conceptual/pipeline-metrics.html
6. AMD GPUOpen, *AMD matrix cores* (AMD lab notes; `cbsz`, `abid`, `blgp`) — https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
7. AMD, *AMD Instinct CDNA4 Instruction Set Architecture* — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
8. HIP documentation, *Low precision floating point types* (FP4/FP6 need gfx950) — https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html
9. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* — https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html
