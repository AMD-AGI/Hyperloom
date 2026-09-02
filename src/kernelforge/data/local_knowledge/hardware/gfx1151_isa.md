---
title: gfx1151 — ISA target, instruction families, waits and disassembly
kind: hardware
topic: isa
gens: [gfx1151]
updated: 2026-09-02
---

# gfx1151 ISA and toolchain

Use the final gfx1151 code object—not source intent—as the authority for wave size, instruction
selection, resource use and target identity.

## Target and current qualified toolchain

| Item | Value |
|---|---|
| Native target | **`gfx1151`** |
| HIP compile selector | `--offload-arch=gfx1151` |
| Qualified container Torch | `2.13.0+rocm10.0.0` |
| Qualified HIP runtime/toolchain | `7.15.26333` through modular ROCm SDK |
| Native wave reported by KFD | 32 |
| Official ISA family | RDNA 3.5 |

Do not set `HSA_OVERRIDE_GFX_VERSION`. Architecture spoofing may make an application launch while
selecting wrong instructions, wave assumptions or device libraries.

## Verified compile-time identity

In the qualified HIP headers/toolchain, gfx1151 compilation defines:

- `__GFX11__`;
- `__gfx1151__`;
- project-level `RDNA3` and `RDNA3_5` where those macros are derived from the architecture defines;
- not `RDNA3_0` where it is defined as RDNA3 excluding RDNA3.5.

Verify macros in device compilation, not only host preprocessing. A small kernel that prints or
materializes the selected branch is more reliable than a generic `clang -dM` invocation lacking the
HIP device pass.

## High-value instruction families

| Area | What matters on gfx1151 |
|---|---|
| Matrix | VOP3P `V_WMMA_*_16X16X16_*` for F16/BF16/IU8/IU4 |
| Packed dot | IU8/IU4 dot instructions with signedness-select NEG fields |
| Dual issue | `V_DUAL_*` / VOPD, wave32 only and heavily constrained |
| Vector ALU | VOP1/VOP2/VOP3/VOP3P, packed 16-bit forms, DPP lane movement |
| Scalar | SOP* flow/address/uniform math; scalar memory through SMEM |
| LDS | DS indexed/atomic plus CU-mode direct/parameter load forms |
| Global | GLOBAL and BUFFER for known-global addresses |
| Generic address | FLAT, with both global and LDS dependency implications |
| Private | SCRATCH using `FLAT_SCRATCH` |
| Scheduling | `S_WAITCNT*`, `S_DELAY_ALU`, clauses, barriers |

## Wait counters

`S_WAITCNT` waits until outstanding counts are at or below the encoded thresholds. Counters count
instructions, not lanes.

| Counter | Covers |
|---|---|
| `VMcnt` | vector loads/samples and atomics that return data |
| `VScnt` | vector stores and non-returning atomics |
| `LGKMcnt` | LDS indexed ops, SMEM loads, GDS/GWS, messages and FLAT's LDS side |
| `EXPcnt` | exports and LDS parameter/direct loads |

Memory groups may complete out of order. Use the narrowest correct wait, but after a FLAT access the
only generally sensible value can be zero because global and LDS sides may complete independently.

`S_DELAY_ALU` describes recent ALU dependencies to avoid stalls. It is optional for correctness and
must not replace memory waits.

## GLOBAL versus FLAT

- `GLOBAL` is for addresses known to be global and uses VM/VScnt—not LGKMcnt.
- `FLAT` can resolve per lane to global, scratch or LDS and therefore increments both dependency
  domains.
- A pure tensor-global path emitting FLAT deserves investigation: it can consume extra LDS/dependency
  machinery.
- Incorrectly using GLOBAL for an LDS/private address can raise MEMVIOL; make the address-space proof
  explicit.

## Cache controls

GLC, SLC and DLC fields affect first-level behavior, L2 temporal policy and MALL/Infinity Cache policy
where implemented. They can also affect atomic return semantics. Do not treat cache bits as generic
“faster/slower” flags; preserve correctness scope and measure the exact streaming/reuse pattern.

## Build and inspect

Example native compile:

```bash
hipcc -x hip --offload-arch=gfx1151 -O2 -c kernel.hip -o kernel.o
```

Inspect the offload bundle and target code object with the ROCm LLVM tools available in the selected
runtime. Typical flow:

```bash
roc-obj-ls kernel.o
roc-obj-extract -o - 'hipv4-amdgcn-amd-amdhsa--gfx1151' kernel.o > kernel-gfx1151.co
llvm-readobj --notes --symbols kernel-gfx1151.co
llvm-objdump -d --mcpu=gfx1151 kernel-gfx1151.co
```

Tool syntax varies by ROCm packaging. Preserve the original input hash before extraction and write to
a disposable output; do not let objcopy-like tools rewrite the authority in place.

## Disassembly checklist

| Check | Pass condition |
|---|---|
| Target | embedded code object is exactly `gfx1151` |
| Wave | metadata/disassembly matches the intended wave32 or deliberate wave64 |
| Route instruction | expected WMMA/dot/VALU/Triton-generated sequence exists in the target symbol |
| Address space | GLOBAL/BUFFER versus FLAT matches the proven pointer domain |
| Waits | consumers are dominated by correct VM/VScnt/LGKM/EXP waits |
| VOPD | every pair satisfies wave32 and bank/source/destination restrictions |
| WMMA hazards | dependent overlap has a valid independent op/NOP spacing |
| Resources | VGPR/SGPR/LDS/private/spill metadata stays within the declared budget |
| Scratch traffic | no unexpected hot-loop scratch loads/stores |
| Code change | candidate ISA differs from baseline in the mechanism claimed |

A whole-object mnemonic count is not symbol-correlated proof. A source intrinsic with no reachable
instantiation is not route proof.

## What it means for kernels

1. Compile for native gfx1151 with the exact runtime toolchain.
2. Verify the code object and intended symbol before benchmarking.
3. Read waits and address-space selection as correctness evidence.
4. Read resource metadata before interpreting occupancy.
5. Separate instruction availability from framework dispatch.
6. Preserve source, compile command, object hash and disassembly together.

## Pitfalls

- Spoofing an older GFX target.
- Trusting a source macro without a device-side compile check.
- Using NOPs instead of wait counters for memory dependencies.
- Missing FLAT's dual counter domains.
- Calling VOPD safe from mnemonic presence alone.
- Searching an entire library rather than the reachable specialization.
- Using a code-object tool in a mode that mutates the input.

## Verify

Run a known-answer kernel, then retain native target identity, exact symbol ISA, resource metadata and
correctness output. For formal encoding questions, cross-check AMD's machine-readable ISA XML rather
than inferring fields from disassembler formatting.

## Sources

- AMD RDNA3.5 ISA guide, Chapters 3–16.
- AMD/GPUOpen machine-readable ISA.
- Qualified ROCm10 modular SDK preflight on the EVO-X2.

## Related

`gfx1151_execution.md` · `gfx1151_matrix_core.md` · `gfx1151_lds.md` ·
`gfx1151_memory.md`
