---
title: gfx1151 — LDS capacity, 64 banks, CU/WGP modes and dependencies
kind: hardware
topic: lds
gens: [gfx1151]
updated: 2026-09-02
---

# LDS — capacity, banks, modes and synchronization

RDNA3.5 provides **128 KiB LDS per WGP**, split into two **64 KiB CU-affiliated halves**,
over **64 DWORD-wide banks**. One work-group may request at most **64 KiB**.

## Geometry

| Property | RDNA3.5 / qualified gfx1151 |
|---|---|
| WGP LDS | 128 KiB |
| CU-affiliated halves | 2 × 64 KiB |
| Live KFD `lds_size_in_kb` | 64 per CU |
| Banks | 64 total, split 32 + 32 by CU affiliation |
| Bank storage | 512 × 32-bit, two-port (one read + one write per clock) |
| Integer atomic units | 64 |
| Maximum one work-group may request | 64 KiB |
| Addressing modes | indexed DS, direct/parameter load in CU mode, wave collectives/atomics |

DWORDs map serially across banks. For a simple contiguous 32-bit view, reason from:

```text
bank = (byte_address / 4) mod 64
```

Then account for CU/WGP mode and operand width. A mapping proven conflict-free for a 32-bank
CDNA part is not proof for this 64-bank, two-half layout.

## CU mode versus WGP mode

### CU mode

- The work-group's waves stay on one CU's two SIMD32s.
- The allocation resides in the associated 64 KiB LDS half.
- Access may not cross or wrap into the other half.
- Both halves can operate in parallel for independent work-groups.
- `LDS_PARAM_LOAD` and `LDS_DIRECT_LOAD` are available.

### WGP mode

- Work-group waves may occupy all four SIMD32s.
- LDS is one contiguous 128 KiB WGP address space.
- An allocation may lie in either half or straddle the 64 KiB boundary.
- `LDS_PARAM_LOAD` and `LDS_DIRECT_LOAD` are **not** available.
- One work-group still cannot allocate more than 64 KiB.

## Bank conflicts

LDS bandwidth comes from parallel bank access. Indexed loads/stores and atomics serialize when
lanes request different addresses in the same bank. Same-address requests can broadcast where the
instruction semantics permit.

For a tiled array, inspect the byte stride:

```text
bank_stride = (byte_stride / 4) mod 64
```

A stride that repeatedly maps active lanes to the same bank is a conflict pattern. The exact
active-lane grouping and CU-half affiliation matter; prove a padding/XOR swizzle against the target
instruction and compiled wave size.

### Repair options

- **Padding:** move row/column strides away from bank-period multiples while retaining vector alignment.
- **XOR/swizzle:** permute addresses so the target WMMA fragment reads distribute across banks.
- **Layout change:** store the producer directly in the consumer's fragment order.
- **Bypass LDS:** when reuse is insufficient, global→VGPR can beat stage/sync/read overhead.

Do not optimize the bank formula in isolation and accidentally scalarize global accesses or break
fragment layout.

## Access methods

- Indexed DS operations use per-lane VGPR addresses/data.
- LDS direct load broadcasts one DWORD into a VGPR and implicitly reads `M0`; initialize it.
- Parameter loads and direct loads contribute to `EXPcnt` and are CU-mode only.
- LDS indexed operations contribute to `LGKMcnt`.
- Some global/buffer loads can target LDS, but their exact supported widths and lowering are
  toolchain/instruction-specific; prove the emitted form rather than importing gfx950 widths.
- Storing staged LDS data to global normally passes through VGPRs.

## Synchronization and waits

A work-group barrier waits until all participating waves reach the barrier. It does not replace
memory dependency waits.

- use `s_waitcnt lgkmcnt(...)` before consuming indexed LDS results;
- use `s_waitcnt expcnt(...)` for direct/parameter LDS results where applicable;
- do not use NOPs as a memory-ordering substitute;
- remember that FLAT may use LDS and global machinery simultaneously and therefore participates
  in both LGKM and VM/VScnt domains.

## Capacity and occupancy

For staged GEMM/attention operands:

```text
LDS/workgroup ≈ sum(staged operand bytes × pipeline stages) + scratch/exchange space
```

Then gate:

```text
LDS/workgroup <= 64 KiB
resident workgroups <= floor(available LDS side/mode / allocated bytes)
```

The true allocation granule and occupancy result come from compiler/code-object metadata. Do not
assume gfx950's 160 KiB denominator or allocation granule.

## What it means for kernels

1. State CU or WGP mode when the distinction affects layout or direct loads.
2. Keep every work-group ≤64 KiB LDS.
3. Re-derive swizzles for 64 banks and the target WMMA fragment.
4. Preserve vector alignment while padding.
5. Use fine-grained wait counters and barriers for their separate purposes.
6. Compare against a no-LDS path when reuse is low.
7. Inspect generated ISA; source-level shared arrays do not prove wide/conflict-free instructions.

## Pitfalls

- Reading “128 KiB/WGP” as “128 KiB per work-group.”
- Using direct/parameter LDS loads in WGP mode.
- Copying a 32-bank MI swizzle unchanged.
- Forgetting the 64 KiB CU-half boundary in CU mode.
- Treating a barrier as completion of all memory operations.
- Assuming gfx950 direct-to-LDS widths exist on gfx1151.
- Spending LDS to stage data that is consumed only once.

## Verify

- Code-object `.group_segment_fixed_size` / LDS metadata.
- Disassembly for `ds_*`, direct/parameter-load forms and wait counters.
- A targeted bank-conflict microbenchmark using the exact dtype, stride, wave size and mode.
- Correctness A/B with padding/swizzle enabled and disabled.
- Profiler counters where the installed ROCm stack exposes reliable LDS metrics on gfx1151.

## Sources

- AMD RDNA3.5 ISA guide, Chapters 1–3, 5 and 12.
- Live KFD `lds_size_in_kb=64` on the qualified EVO-X2.

## Related

`gfx1151_topology.md` · `gfx1151_execution.md` · `gfx1151_matrix_core.md` ·
`gfx1151_memory.md` · `common_methodology/optimization/lever_lds_banks.md`
