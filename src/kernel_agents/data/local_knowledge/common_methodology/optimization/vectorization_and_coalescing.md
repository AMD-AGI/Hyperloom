---
title: vectorization and coalescing (128-bit loads, alignment)
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, fp32]
regimes: [prefill, decode, training, both]
updated: 2026-06-05
---

# vectorization and coalescing

## TL;DR
Each global memory instruction should move the widest aligned chunk possible — **128-bit
(`global_load_dwordx4`)** — and the 64 lanes of a wave should touch **contiguous** addresses so the
memory system services them as a few wide transactions (coalescing). Misaligned or strided access
silently downgrades to scalar `dword` loads and multiplies transaction count. This is the cheapest big
win on memory-bound kernels (norm, elementwise, copy, decode GEMV) — see
`[[optimization/roofline_and_bottlenecks.md]]` to confirm a kernel is memory-bound first, and
`[[hardware/cdna3_mi300/memory_hierarchy.md]]`.

## Concepts (the hardware)
- **Wave = 64 lanes**: one memory instruction issues 64 lane addresses. The hardware coalesces lanes
  that fall in the same cache line into one transaction; scattered lanes generate many.
- **Vector width**: `dword`=32-bit, `dwordx2`=64-bit, `dwordx4`=**128-bit**. `global_load_dwordx4` is
  the widest single-instruction load — 4 fp32 / 8 bf16 / 16 fp8 per lane. Same for stores and `ds_*`.
- **Alignment**: 128-bit access requires the address to be **16-byte aligned**. Unaligned ⇒ compiler
  emits narrower/multiple instructions.

## The levers
- **Make types vectorizable**: load/store via `float4` / `__hip_bfloat16` packed-8 / `int4` so the
  compiler emits `*_dwordx4`. In triton, contiguous tensor blocks with the right `BLOCK` divisibility
  auto-vectorize; declare divisibility hints so the compiler can prove alignment.
- **Align base pointers and strides**: pad leading dims to 16-byte (or 128-bit element) multiples; an
  odd row stride breaks vectorization on every row.
- **Coalesce the lane mapping**: index so `lane i` reads element `base + i` (innermost dim along the
  wave), not `base + i*stride`. Transpose in LDS rather than via strided global reads
  (`[[optimization/lds_and_bank_conflicts.md]]`).
- **Grid-stride loops**: for elementwise/reduction over N elements, each thread strides by
  `gridDim*blockDim` so accesses stay contiguous per step and the kernel scales to any N with a fixed,
  occupancy-tuned grid (`[[optimization/wave_and_grid_sizing.md]]`). Pattern:
  `for (i = gid; i < N; i += stride) ...`.
- **Vectorize LDS too**: `ds_read_b128` / `ds_write_b128` for the staging path; padding must preserve
  16-byte alignment (`[[optimization/lds_and_bank_conflicts.md]]`).

## Coalescing vs bank conflicts (don't confuse them)
- **Coalescing** is about *global* memory transactions across lanes (cache-line granularity).
- **Bank conflicts** are about *LDS* banks within one instruction.
They are separate axes; a kernel can be perfectly coalesced in global but conflicted in LDS, or vice
versa. Fix each with its own lever.

## Pitfalls
- Strided/transposed global reads (column-major over a row-major tensor) ⇒ uncoalesced, 1 transaction
  per lane. Stage through LDS and transpose there.
- Unaligned base pointer or odd stride silently kills `dwordx4` — verify in the ISA dump.
- Over-wide loads on tiny tensors waste tail lanes (predication overhead) — match width to the data.
- Assuming wave32; CDNA is **wave64**, so coalescing windows are 64 lanes wide.

## Verify
- ISA dump: count `global_load_dwordx4` / `_b128` vs `dword` — want the wide forms in the hot loop
  (`[[languages/triton_amd/isa_verify.md]]`).
- Omniperf: memory transaction efficiency / fetch size, HBM BW vs roofline peak (`[[profiling/]]`).
- A/B: aligned vs misaligned base pointer; transaction count and BW should jump with alignment.
