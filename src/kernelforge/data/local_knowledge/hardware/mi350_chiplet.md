---
title: MI350X — XCD chiplets, per-XCD L2, tile locality, partition modes
kind: hardware
topic: chiplet
gens: [gfx950]
updated: 2026-08-28
---

# XCD chiplets, L2 locality and partitioning

MI350X/MI355X is **8 XCD chiplets × 32 active CUs = 256 CUs**, behaving like 8 GPUs glued by Infinity
Fabric. A workgroup lives on **one CU on one XCD**; **L2 is per-XCD, not global**.

## Topology

```
┌─────────────── XCD (one chiplet, TSMC N3P) ─────────────┐
│ HWS (hardware scheduler)                                 │
│ ACEs (Asynchronous Compute Engines) — queue front-ends   │
│ 32 active CUs: each 4×SIMD64 + 4 MatrixCore              │
│   160 KiB LDS (64 banks, 256 B/clk), 32 KiB L1           │
│ shared per-XCD L2                                        │
└──────────────────────────────────────────────────────────┘
```

- 8 XCDs sit on **2 I/O dies** (CDNA3 used 4). Each IOD connects 4 HBM3E stacks (36 GB) → **288 GB @
  8 TB/s**. Memory closest to an XCD is on the same IOD, so "keep the working set local" now covers a
  4-XCD-wide neighbourhood.
- Infinity Fabric + the **256 MiB Infinity Cache** on the IODs are the device-shared coherence layer.
- A workgroup is dispatched to **one CU** and never migrates; its waves stripe across that CU's 4 SIMDs.
- The HWS round-robins workgroups across the 8 XCDs in blocks.
- **ACEs** are queue front-ends, so multiple HIP streams / concurrent kernels map naturally onto them.

> **Carried over from CDNA3, unconfirmed for gfx950:** the exact per-XCD L2 capacity, the ACE count per
> XCD, and the ~116–202 ns same-XCD vs cross-XCD global-atomic latency were measured on MI300X. The
> *model* (per-XCD L2, cross-XCD misses to Infinity Cache) is unchanged; verify the numbers on box
> before relying on them.

## Why the default mapping defeats reuse

The hardware assigns workgroup ids to XCDs round-robin. With a plain linear `pid`, blocks that share a
B-panel scatter across all 8 dies, so each die pulls its own copy. You pay 8× the fetches for the same
data. **Cross-XCD reuse is not an L2 hit** — it falls to Infinity Cache or HBM.

## The three grid rules

| Rule | Value | Why |
|---|---|---|
| Workgroups per launch | **≥ 1024** | fills 256 CUs (~4/CU) with tail slack |
| Tile count | **multiple of 8** | round-robin balances exactly across 8 XCDs |
| CTA order | **swizzled**, not linear | keeps a reuse group on one die's L2 |

Swizzle sketch — replace `xcd = pid % 8` (which scatters reuse):
```
group = pid / tiles_per_xcd
xcd   = group
local = pid % tiles_per_xcd
```
Size `tiles_per_xcd` to that XCD's L2 working set — too large and you thrash the cache you are trying
to exploit. Triton's `GROUP_SIZE_M` is the row-grouping form of the same idea; the XCD swizzle is the
die-grouping form. They compose.

## The 512 B stride cliff

A GEMM whose leading-dimension byte stride is an exact multiple of **512 B** — notably the **TN**
layout (A non-transposed, B transposed) — can collide in the L2 tag RAM, serializing accesses.
Symptom: anomalously low L2 hit rate at specific N/K while neighbouring shapes are fine.

Fix by padding the leading dimension off the 512 B multiple, or let a tuned library pick a swizzle /
split-K that breaks the stride (hipBLASLt and CK already encode this in solution selection).

> Characterized on the chiplet CDNA3 L2. The per-XCD organization is unchanged here, so treat it as a
> live hypothesis — **confirm on box before padding for it.**

## Clock variance across XCDs (3–10%)

The 8 XCDs do **not** all run at the same clock — process, thermal and power-delivery differences per
die give **~3–10%** spread. Consequences:

- A kernel with a **device-wide barrier** runs at the **slowest XCD's** pace; tightly coupled cross-XCD
  collectives pay this tax.
- **Per-XCD-independent work** (the ordinary embarrassingly-parallel GEMM/attention grid) is unaffected
  beyond load balance, which the 8-multiple rule handles.
- **Benchmark variance**: repeat-to-repeat spread partly reflects which XCDs the scheduler used. Use
  the median of ≥3 warm repeats and report the spread. This bites harder on the 1000 W MI350X, where
  sustained clock is power-capped (`mi350_clocks.md`).

## Partition modes (SPX / DPX / CPX × NPS1/2/4)

| Compute mode | Logical GPUs | XCDs each | CUs each | HBM each (NPS1) | Use |
|---|---|---|---|---|---|
| **SPX** (default) | 1 | 8 | **256** | **288 GB** | one big model/kernel |
| **DPX** | 2 | 4 | 128 | 144 GB | two balanced jobs |
| **CPX** | 8 | 1 | **32** | **36 GB** | many small jobs, inference density |

| Memory mode | NUMA domains | Effect |
|---|---|---|
| **NPS1** | 1 | unified 288 GB, interleaved across 8 stacks |
| **NPS2** | 2 | each half owns a memory quadrant |
| **NPS4** | 4 | each XCD's traffic stays local (CPX only) |

**Hard rule:** memory partitions must **not exceed** compute partitions → **SPX+NPS4 is invalid**.
Valid: SPX+NPS1, DPX+NPS1/2, CPX+NPS1/4.

```bash
amd-smi list
sudo amd-smi set --gpu all --compute-partition CPX
sudo amd-smi set --gpu all --memory-partition  NPS2
```

Switching mode terminates GPU processes and reloads amdgpu; it reverts to SPX/NPS1 on reboot.

**Kernel implications.** In CPX a kernel sees a 32-CU / 36 GB "GPU" with XCD-local memory → higher
effective BW and clocks because cross-XCD traffic is gone. AMD's CDNA4 material highlights **CPX+NPS2**
hosting up to **8 instances of a 70B model** on one MI355X. A single large model spanning all CUs and
>36 GB **must** use SPX.

## Pitfalls
- **Assuming 38 CU/XCD** — that is MI300X; here it is **32** (256 total).
- **Assuming 4 IODs** — CDNA4 has **2**, with 4 XCDs each.
- **Non-8-multiple grids** → straggler XCDs, scattered L2 reuse, a silent ~10–15%.
- **Tight cross-XCD sync** → bottlenecked by the slowest XCD plus Fabric latency.
- **Assuming a unified L2** — it is partitioned per XCD.
- **SPX+NPS4** — rejected by the driver.
- **Sizing a CPX instance like MI300X's 24 GB** — here a CPX slice is **36 GB**.
- **Sizing the grid for 304 CUs** — query `hipGetDeviceProperties → multiProcessorCount`.

## Verify
- `amd-smi static` / `rocm-smi --showcomputepartition --showmemorypartition` for the current mode.
- `rocprof-compute`: XCD load balance, **L2 hit rate**, and **HBM read volume**. The real pass
  condition for a locality change is **lower HBM reads at the same FLOP count** — wall time alone can
  move for unrelated reasons.
- Per-XCD clock via `amd-smi metric`.
- A/B linear vs swizzled pid mapping; A/B SPX vs CPX on a compute-bound GEMM.

## Related
`mi350_memory.md` (the ladder and Infinity Cache) · `mi350_clocks.md` (sustained clock, SKUs) ·
`mi350_overview.md` · `common_methodology/optimization/lever_xcd_locality.md`
