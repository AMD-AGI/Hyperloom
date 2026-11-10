---
title: XCD / L2 locality — chiplet-aware tile scheduling
kind: lever
lever: xcd_locality
gens: [gfx950]
bottleneck: bandwidth-bound (re-fetch, not raw streaming)
updated: 2026-08-28
---

# XCD and L2 locality

## Route here when
- Bandwidth-bound, **and** the kernel re-reads an operand many times (tiled GEMM, attention over a
  shared KV panel) — i.e. the traffic is *re-fetch*, not a single stream.
- L2 hit rate is low at a shape where the working set should fit.
- Tile count is not a multiple of 8, or the launch produces fewer than ~1024 workgroups.

**Skip this lever if** the kernel streams each byte exactly once (elementwise, cast, copy). There is no
reuse to localize — go to `lever_coalescing.md` and `lever_fusion.md` instead.

## gfx950 topology

| Fact | Value |
|---|---|
| XCDs | **8** |
| Active CUs per XCD | **32** (→ 256 total) |
| L2 | **per-XCD, not unified** — a cross-XCD hit is not an L2 hit |
| I/O dies | **2** (4 XCDs each) — CDNA3 had 4 |
| Device-shared cache | **256 MiB Infinity Cache (MALL/L3)** on the IODs |
| HBM | 288 GB @ 8 TB/s, 4 stacks per IOD |
| Dispatch | HWS round-robins workgroups across the 8 XCDs in blocks |

The load-bearing fact: **L2 is per-XCD.** A tile whose operand was pulled into XCD 3's L2 gets no
benefit if the next tile that needs it lands on XCD 5 — that access falls through to Infinity Cache
or HBM.

## The default mapping defeats reuse

The hardware assigns workgroup ids to XCDs round-robin. With a plain linear `pid`, blocks that share a
B-panel get scattered across all 8 dies, so each die pulls its own copy of the panel. You pay 8× the
fetches for the same data.

## What to change, in order

### 1. ≥1024 workgroups
Fills 256 CUs with tail slack (≈4 workgroups/CU). Below the CU count, dies sit idle outright; below
~1024 the scheduler has no slack to hide the tail.

### 2. Tile count a multiple of 8
Round-robin over 8 XCDs then balances exactly. A non-multiple leaves one or more dies finishing early
while others carry the remainder — pure tail latency, typically a silent **10–15%**.

### 3. Swizzle the CTA order so reuse stays on one die
Remap `pid → (xcd, local_id)` so a contiguous run of data-sharing tiles lands on the same XCD:

```
# instead of xcd = pid % 8   (scatters reuse across all dies)
group = pid / tiles_per_xcd
xcd   = group
local = pid % tiles_per_xcd
```

Size `tiles_per_xcd` to that XCD's L2 working set — too large and you thrash the very cache you are
trying to exploit. Triton's `GROUP_SIZE_M` is the row-grouping form of the same idea; the XCD swizzle
is the die-grouping form. They compose.

### 4. Break 512 B leading-dimension strides
A GEMM whose leading-dimension byte stride is an exact multiple of **512 B** — notably the **TN**
layout — can collide in the L2 tag RAM, serializing accesses. Symptom: anomalously low L2 hit rate at
specific N/K while neighbouring shapes are fine. Fix by padding the leading dimension off the 512 B
multiple, or let a tuned library pick a swizzle/split-K that breaks it.

> This was characterized on the chiplet CDNA3 L2. The per-XCD organization is unchanged on gfx950, so
> treat it as a live hypothesis: **confirm on box before padding for it.**

### 5. Persistent kernels for explicit control
Launch exactly `256 × blocks_per_CU` workgroups that loop over tiles. You then own the tile→XCD
mapping outright instead of trusting the dispatcher, and you get natural Stream-K reduction. Cost: you
own load balancing (`lever_grid_sizing.md`).

### 6. Consider CPX partitioning for many-small-kernel workloads
`CPX` makes each XCD a 32-CU / 36 GB logical GPU with strictly local memory, removing cross-XCD traffic
by construction. Right for multi-tenant / many-small-job density, wrong for one large model.
See `hardware/mi350_chiplet.md`.

## Verify

| Check | How | Pass |
|---|---|---|
| L2 hit rate | `rocprof-compute`, per shape | rises after the swizzle at equal FLOPs |
| HBM read volume | same run | **falls** at equal FLOPs — this is the real signal |
| XCD balance | `rocprof-compute` XCD load balance | no straggler die |
| Grid sanity | arithmetic | workgroups ≥ 1024 **and** `tile_count % 8 == 0` |
| A/B | linear vs swizzled pid mapping | compare HBM read volume, not just wall time |

The pass condition is **lower HBM reads at the same FLOP count**. Wall time alone can improve for
unrelated reasons; the byte counter is what proves the locality worked.

## Expected magnitude
Non-8-multiple → 8-multiple tile count: **~10–15%** on prefill GEMM. Linear → XCD-swizzled order on a
reuse-heavy GEMM: **10–25%**, more if the panel was being re-fetched from HBM. Both are near-free code
changes.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Swizzle applied, L2 hits unchanged | groups sized larger than L2 → thrash | shrink `tiles_per_xcd` |
| One die finishes early | tile count not a multiple of 8 | round the grid |
| Idle CUs on prefill | <1024 workgroups | raise grid; for skinny M use split-K to manufacture blocks |
| "L2 should have it" but misses | assumed a unified L2 — it is **per-XCD** | localize the reuse, or accept the L3 hit |
| Sized the grid for 304 CUs | that is MI300X | gfx950 has **256** — query `hipGetDeviceProperties` |

## Deeper
`hardware/mi350_chiplet.md` (topology, per-XCD L2, Tagram detail, clock variance, partition modes) ·
`lever_grid_sizing.md` · `lever_coalescing.md`
