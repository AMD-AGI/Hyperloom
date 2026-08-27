---
name: profiling-asm
description: >
  Profile hand-written AMDGCN assembly: read the ISA hot loop (buffer_load width,
  s_waitcnt overlap, dense v_mfma, no scratch/accvgpr), turn rocprofv3 PMC into a
  VALU/MFMA/VMEM/LDS-bound verdict, check the 256+256 VGPR/AGPR occupancy budget,
  and use per-instruction measured cycle counts to find the real stall. Use when
  deciding what to change in an asm kernel from measured evidence. Usage: /profiling-asm
allowed-tools: Read Bash Grep Glob
---

# Profiling AMDGCN ASM kernels

Measurement-driven diagnosis at the ISA level. Per-instruction measured cycle counts + counter tracking
are in [../optimize/asm_levers/intellikit/instructions/](../optimize/asm_levers/intellikit/) (each doc has
the cycle count and which counter it bumps). Hardware peaks live in `local_knowledge/hardware/`.

## 1. Read the ISA first (asm has no higher abstraction to hide behind)
```bash
/opt/rocm/bin/amdclang++ -x hip --offload-device-only --offload-arch=gfx942 -O3 -S kern.cpp -o kern.s
```
| Look for (K-loop) | Good | Bad → fix |
|---|---|---|
| Global loads | `buffer_load_dwordx4` (≥128-bit) | `buffer_load_dword` (scalar) |
| Waitcnt | minimal, overlapped | `s_waitcnt vmcnt(0)` after every load = no overlap |
| MFMA | dense `v_mfma_*`, correct NOP spacing | sparse / missing hazard NOPs |
| Accumulator | AGPR, no moves in loop | `v_accvgpr_read/write` in loop |
| Scratch | none | `scratch_` → spilling |

## 2. PMC → bound verdict → lever
```bash
rocprofv3 --kernel-trace --stats -f csv -- <run>
```
| PMC signal | reading | asm lever |
|---|---|---|
| `MFMABusy` near peak | compute-bound | near roofline; only tiling/dtype helps |
| `MFMABusy` with gaps | matrix core starved | fix waitcnt overlap, prefetch depth, sched ordering |
| `VALUBusy` high / low MFMA | address/convert-bound | pack (`v_pk_*`), fewer address ops |
| LDS conflict counters | LDS-bound | swizzle / multi-buffer (`intellikit/guides/lds-patterns.md`) |
| vmcnt stalls before MFMA | load latency exposed | `buffer_load_lds` direct-to-LDS (~17% over load+ds_write) |
| `scratch_` present | spill | free VGPRs (`vgpr_liveness.py`), shrink tile |

## 3. Occupancy budget (256 VGPR + 256 AGPR, one wave/SIMD)
`max_waves`-limited by `max(VGPR, AGPR, LDS, wave-slot)`. Round VGPR to 16-granule.
`intellikit/tools/scripts/vgpr_liveness.py --json` finds dead register windows and suggests remappings to
lift occupancy; budgeting rules in `intellikit/guides/register-allocation.md`.

## 4. Attribute a stall to an instruction
When the hot loop stalls, use the per-instruction docs (measured cycles + counter it tracks) to find the
real cost — e.g. `v_exp_f32`/`v_rcp_f32` latency in a softmax, `ds_read_tr` vs `ds_read_b128`,
`v_mfma_*` issue cadence. `intellikit/guides/debugging-playbook.md` maps symptom→cause with a top-10 bug
list.

## 5. Gate against the higher level
Raw asm is only worth it if it beats the best intrinsics/CK/library kernel at your shape AND the win
survives e2e. Cross-check vs the CK `ckProfiler` top instance and the aiter tuned path
(`local_knowledge/framework/aiter/`) before committing a hand-maintained `.s`.

## Sources
- IntelliKit measured cycle counts / counters / hazards: `asm_levers/intellikit/` (MI355X-measured).
- rocprofv3 / rocprof-compute: https://rocm.docs.amd.com/projects/omniperf/en/amd-staging/what-is-rocprof-compute.html
- CDNA3/CDNA4 ISA (waitcnt, MFMA): AMD ISA reference PDFs (see SOURCES.md).
