---
title: gfx1151 — WMMA matrix instructions, fragments and scheduling
kind: hardware
topic: matrix_core
gens: [gfx1151]
updated: 2026-09-02
---

# Matrix core — RDNA3.5 WMMA, not CDNA MFMA

RDNA3.5 exposes wave-level 16×16×16 WMMA operations through the VOP3P instruction
format. The entire wave cooperates on `D = A·B + C`. This is the matrix model to use for
native gfx1151 authoring; CDNA MFMA shape, register and cycle tables do not transfer.

## Architecturally listed forms

| Instruction | A/B elements | Accumulator/output |
|---|---|---|
| `V_WMMA_F32_16X16X16_F16` | F16 | F32 |
| `V_WMMA_F32_16X16X16_BF16` | BF16 | F32 |
| `V_WMMA_F16_16X16X16_F16` | F16 | F16 |
| `V_WMMA_BF16_16X16X16_BF16` | BF16 | BF16 |
| `V_WMMA_I32_16X16X16_IU8` | signed/unsigned 8-bit controls | I32 |
| `V_WMMA_I32_16X16X16_IU4` | signed/unsigned 4-bit controls | I32 |

The instruction list establishes architectural availability—not compiler intrinsic stability,
framework reachability, performance, or a particular packed-weight ABI.

## Fragment layout

AMD specifies:

- **A** is column-major in the VGPR fragment view;
- **B**, **C** and **D** are row-major;
- A/B fragment data from lanes 0–15 must be replicated into lanes 16–31;
- wave64 additionally replicates into lanes 32–47 and 48–63;
- source operands are VGPRs;
- output-coordinate ownership must be derived from the exact WMMA form/toolchain mapping.

Do not validate layout with all-ones inputs: uniform data is invariant under many wrong
permutations. Use basis matrices, nonuniform rows/columns, both wave halves when applicable,
extreme codes and every output coordinate.

## Floating-point behavior

- Float WMMA uses round-to-nearest-even.
- WMMA generates no ALU exceptions.
- The ISA's MODE rounding controls do not turn WMMA into arbitrary rounding modes.
- F32 accumulation is the robust default for F16/BF16 long reductions.
- F16/BF16 output forms are separate instructions, not permission to narrow an F32 accumulator
  arbitrarily inside K.

## Integer signedness

For `V_WMMA...IU...` and IU dot-product instructions, NEG fields select **signedness**; they are
not ordinary arithmetic negation controls. Reserved/unsupported NEG combinations can be undefined.

Before using IU4/IU8:

1. define each operand's signed/unsigned code domain;
2. derive zero-point and offset corrections algebraically;
3. accumulate products/corrections in I32;
4. prove packing order and K permutation;
5. convert to output scale only after the exact integer recurrence is complete.

An instruction mnemonic does not prove the surrounding quantization algebra.

## Scheduling hazard

Back-to-back dependent WMMA instructions require one `V_NOP` or an independent VALU operation
when the first D destination overlaps the next A or B source. A/B may overlap C when C is distinct
from D; the usual accumulator pattern has C and D equal.

Treat this as both a correctness and scheduling check. Inspect the final ISA—source-level intrinsic
ordering may be transformed by the compiler.

## Wave size

Both wave32 and wave64 can execute WMMA, but their fragment replication and issue behavior differ.
Native gfx1151 kernels normally use wave32. A deliberate wave64 matrix kernel must prove:

- compiled wave size in the code object;
- both-half fragment replication;
- lane/output mapping;
- resource use and correctness independently of a wave32 baseline.

VOPD and WMMA are distinct mechanisms. VOPD is wave32-only dual VALU issue; WMMA is a VOP3P
wave-matrix operation. Do not claim a WMMA kernel uses VOPD unless the symbol-correlated ISA does.

## Software-stack mapping

| Surface | Claim allowed without more evidence |
|---|---|
| ISA manual lists IU4/IU8 WMMA | architecture supports the instruction forms |
| HIP/LLVM accepts an intrinsic | toolchain can compile one source form |
| Code object contains target mnemonic | that symbol contains the instruction |
| Microkernel matches CPU reference | instruction + fragment/algebra work for tested cases |
| Framework route selects it | integration reached the kernel for the tested request |
| Model benchmark improves | end-to-end benefit for that exact model/workload/runtime |

Never skip rungs in this ladder.

## What it means for kernels

1. Start from a 16×16×16 WMMA fragment map, not an MFMA map.
2. Keep floating accumulators in F32 and integer accumulators in I32 unless a separately proven
   output-form choice is intentional.
3. Build operand staging around A-column/B-row fragment ownership.
4. Design LDS swizzles against the exact lane/register map.
5. Schedule an independent VALU or required NOP across dependent overlap hazards.
6. Use asymmetric correctness cases before timing.
7. Count packing, scaling, LDS and dispatch costs in end-to-end claims.

## Pitfalls

- Calling WMMA “unsupported” because one rocWMMA/framework path failed.
- Calling it “qualified” because an all-ones spike ran.
- Using CDNA MFMA names, shapes, AGPR advice or cycle tables.
- Applying the current lane's scale to every output fragment element.
- Treating NEG as numeric negation for IU4/IU8.
- Ignoring wave-half replication in wave64.
- Quoting register counts without the emitted specialization.

## Verify

- Compile natively for `gfx1151` and extract the exact target code object.
- Correlate the intended kernel symbol with the WMMA mnemonic.
- Inspect VGPR/SGPR/LDS/private/spill metadata for every host-selectable specialization.
- Use basis/nonuniform matrices and a CPU reference over full output-coordinate coverage.
- Verify the actual framework request reaches the intended route before model timing.

Use AMD's matrix-instruction tooling when it supports the target/form; otherwise derive from the
official ISA and confirm with a standalone known-answer microkernel.

## Sources

- AMD RDNA3.5 ISA guide, Sections 7.5, 7.9, 7.9.1 and VOP3P instruction catalogue.
- AMD/GPUOpen machine-readable ISA for exact instruction names/encodings.

## Related

`gfx1151_execution.md` · `gfx1151_dtypes.md` · `gfx1151_lds.md` ·
`gfx1151_isa.md` · `common_methodology/optimization/lever_numerics.md`
