---
title: gfx1151 — wave32/wave64 execution, registers, occupancy and VOPD
kind: hardware
topic: execution
gens: [gfx1151]
updated: 2026-09-02
---

# Execution model, registers and occupancy

The qualified Radeon 8060S reports native **wave32**, 80 SIMD32s, 40 CUs and at most
16 waves/SIMD. RDNA3.5 also supports wave64, but wave64 is a separate compiled execution
mode—not two independent wave32s hidden behind one source-level name.

## Wave32 and wave64

| Behavior | wave32 | wave64 |
|---|---|---|
| Active lanes | 32 | 64 |
| VALU issue | once | normally low half then high half |
| VMEM/LDS issue | once | normally low half then high half |
| SALU/SMEM/branch/message | once | once |
| VOPD | legal | illegal/skipped |
| EXEC/VCC bits used | low 32 | all 64 |

For wave64, other waves may interleave between low- and high-half issue. Both halves observe
wave state from before the instruction; do not model the second half as consuming the first
half's newly written values.

## Thread geometry

```text
threads_per_workgroup = waves_per_workgroup × compiled_wave_size
```

Examples:

| Waves | native wave32 threads | wave64 threads |
|---:|---:|---:|
| 1 | 32 | 64 |
| 2 | 64 | 128 |
| 4 | 128 | 256 |
| 8 | 256 | 512 |

This is why copying RDNA wave64 `nwarps` tables to gfx1151 can halve effective thread count.
A launch that remains functionally valid can still lose occupancy, tile coverage or parallelism.

## Live hierarchy

| Level | Qualified-node value |
|---|---:|
| GPU node / XCC | 1 / 1 |
| CUs | 40 |
| SIMD32s | 80 |
| SIMD32s per CU | 2 |
| SIMD32s per WGP | 4 |
| Native waves/SIMD ceiling | 16 |
| Native lanes/wave | 32 |

A work-group remains on one WGP. In CU mode, its waves stay on one CU half; in WGP mode,
they may occupy all four SIMD32s.

## Wave state

- `EXEC` masks VALU, VMEM, LDS, GDS and export lanes; scalar execution and branches are not
  lane-masked the same way.
- Wave32 uses `EXEC[31:0]` and `VCC[31:0]`; upper bits are ignored for execution/summary state.
- `EXEC==0` does **not** prove every instruction is skipped. WMMA and several memory/state cases
  have explicit exceptions.
- `VCC` is an SGPR pair and participates in dependency checks.
- `M0` is a per-wave scalar register used by LDS/GDS/SMEM and other instructions.
- `FLAT_SCRATCH` supplies the scratch/private base.

## Register allocation

AMD's ISA defines:

- up to 256 architected VGPRs per shader;
- VGPR allocation blocks of 16 for wave32 and 8 for wave64;
- 106 normal SGPRs plus named VCC and trap-temporary SGPRs;
- 64-bit SGPR operands require even alignment; wider scalar-memory destinations have stricter
  alignment.

Do not infer occupancy from `vgpr_count` alone. The compiler/code object must also disclose:

- private/scratch bytes and spills;
- SGPR use;
- LDS bytes/workgroup;
- workgroup size and compiled wave size.

Out-of-range register use may read VGPR0, discard writes or otherwise fail silently. No-crash is
not a correctness gate.

## Occupancy method

The hard live wave-slot ceiling is:

```text
80 SIMD × 16 waves/SIMD = 1280 native wave slots device-wide
```

That is only a ceiling. Kernel residency is the minimum of:

```text
wave slots
VGPR/SGPR allocation
LDS allocation
workgroup size and per-WGP limits
scratch/private resource limits
```

Use compiler metadata or profiler occupancy output; do not use MI350's 512-register/SIMD
formula or its 8-wave/SIMD cap.

## VOPD dual issue

VOPD encodes two independent VALU operations in one 64-bit instruction and executes them in
parallel. It is a gfx1151-relevant wave32 lever, but its constraints are correctness rules:

- wave32 only;
- X and Y operations must be independent;
- strict VGPR source-bank/port limits;
- limited SGPR/literal sourcing;
- destination VGPRs must form one even and one odd address;
- no DPP;
- an overlapping read sees the old value.

If the rules are violated, AMD states that hardware does not function correctly. Treat manual
VOPD scheduling as assembly-level work requiring disassembly and asymmetric correctness tests.

## Scheduling and dependencies

- Use `s_waitcnt` for outstanding memory/LDS dependencies.
- Use `s_delay_alu` to reduce ALU dependency stalls; it is not a memory correctness mechanism.
- Do not insert NOP padding as a substitute for wait counters.
- Dependent WMMA has its own overlap hazard described in `gfx1151_matrix_core.md`.
- Scalar-memory loads can complete out of order; wait before consuming their SGPR destinations.

## What it means for kernels

1. Treat wave32 as the baseline and wave64 as an isolated, code-object-proven choice.
2. Translate wave counts to threads before borrowing any donor geometry.
3. Cross VGPR allocation granules deliberately; small source edits within one tier may change nothing.
4. Prefer zero spills to nominally higher occupancy.
5. Consider VOPD only after the generated ISA exposes independent pair opportunities.
6. Distinguish launch/host overhead from device occupancy on tiny decode operations.

## Pitfalls

- Reporting `nwarps=4` without the compiled wave size.
- Assuming wave64's second half observes writes from the first half.
- Assuming `EXEC==0` means no WMMA/memory work issues.
- Reading MI/CDNA occupancy tables as RDNA facts.
- Calling a VOPD mnemonic hit a win without symbol correlation and correctness proof.
- Ignoring private/scratch state because the source declares no local array.

## Verify

- `rocminfo` and KFD topology for native wave size/SIMD/CU counts.
- `llvm-readobj --notes --symbols` for kernel metadata.
- `llvm-objdump -d --mcpu=gfx1151` for wave-sensitive instructions and VOPD.
- Compiler resource diagnostics for VGPR/SGPR/LDS/private/spills.
- A profiler or bounded occupancy probe for actual resident waves.

## Sources

- AMD RDNA3.5 ISA guide, Chapters 2, 3, 5 and 7.
- Live KFD topology on the qualified EVO-X2.

## Related

`gfx1151_topology.md` · `gfx1151_lds.md` · `gfx1151_matrix_core.md` ·
`gfx1151_isa.md` · `common_methodology/optimization/lever_occupancy.md`
