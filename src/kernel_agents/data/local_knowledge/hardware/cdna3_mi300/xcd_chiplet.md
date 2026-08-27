---
title: CDNA3 / MI300X (gfx942) — XCD chiplets & partitioning
kind: hardware
gens: [gfx942]
dtypes: []
regimes: [both]
updated: 2026-06-08
---

# CDNA3 / MI300X (gfx942) — XCD chiplets & partitioning

> Companion to [arch.md](arch.md) (topology) and [memory_hierarchy.md](memory_hierarchy.md). This file
> is the chiplet behavior a kernel author must design around: scheduling, locality, clock variance,
> and the compute/memory partition modes.

## TL;DR
> The MI300X is **8 XCD chiplets** behaving like 8 GPUs glued by Infinity Fabric. A workgroup lives on
> **one CU on one XCD**; L2 is **per-XCD** (4 MiB). Distribute work in **multiples of 8** for even XCD
> spread, keep reuse XCD-local, and expect a **3–10% clock variance across XCDs** that limits how
> tightly synchronized cross-XCD work can be.

## Concepts

### XCD internals
```
┌─────────────── XCD (one chiplet) ───────────────┐
│ HWS (hardware scheduler)                         │
│ 4 × ACE (Asynchronous Compute Engine)            │  dispatch workgroups to CUs
│ 40 CUs (38 active): each 4×SIMD64 + 4 MatrixCore │
│   64 KiB LDS, 32 KiB L1, 16 KiB scalar/const     │
│ 4 MiB shared L2 (16 × 256 KiB slices)            │
└──────────────────────────────────────────────────┘
```
- **ACEs** are queue front-ends — up to 4/XCD dispatch independent compute streams, so multiple HIP
  streams / concurrent kernels map naturally to ACEs.
- A workgroup is dispatched to **one CU** and never migrates; its waves stripe across that CU's
  4 SIMDs.
- The HWS round-robins workgroups across the 8 XCDs in blocks → **8-multiple grids** balance perfectly.

### Cross-XCD cost
- L2 is **per-XCD**; cross-XCD reuse misses to the 256 MiB Infinity Cache (L3) on the IODs.
- Measured global-atomic **core-to-core latency ~116–202 ns** depending on whether the two workgroups
  are on the **same vs different XCD**. Device-wide reductions stage through L3 with ~200 ns sync.
- Each IOD hosts 2 XCDs + 2 HBM stacks; memory closest to an XCD is on the same IOD.

### Clock variance across XCDs (3–10%)
The 8 XCDs do **not** all run at exactly the same clock — measured variance is **~3–10%** across
chiplets (process/thermal/power-delivery differences per die). Consequences:
- A kernel with a **device-wide barrier** runs at the **slowest XCD's** pace; tightly coupled cross-XCD
  collectives pay this tax.
- **Per-XCD-independent** work (the common, embarrassingly-parallel GEMM/attention grid) is unaffected
  beyond load-balance, which the 8-multiple rule handles.
- Benchmark variance: repeat-to-repeat timing spread partly reflects which XCDs the scheduler used —
  use median of ≥3 warm repeats and note spread.

### Partition modes (SPX/DPX/CPX × NPS1/2/4)
| Compute mode | Logical GPUs | XCDs each | CUs each | HBM each (NPS1) | Use |
|---|---|---|---|---|---|
| **SPX** (default) | 1 | 8 | 304 | 192 GB | one big model/kernel |
| **DPX** | 2 | 4 | 152 | 96 GB | two balanced jobs |
| **CPX** | 8 | 1 | 38 | 24 GB | many small jobs, inference density |

| Memory mode | NUMA domains | Effect |
|---|---|---|
| **NPS1** | 1 | unified 192 GB, interleaved across 8 stacks |
| **NPS2** | 2 | each half owns a memory quadrant (DPX) |
| **NPS4** | 4 | each XCD's traffic stays on its local IOD (CPX only) |

**Hard rule:** memory partitions must **not exceed** compute partitions → **SPX+NPS4 is invalid**.
Valid: SPX+NPS1, DPX+NPS1/2, CPX+NPS1/4.

**Kernel implications:**
- In **CPX/NPS4** a kernel sees a 38-CU/24 GB "GPU" with memory local to its XCD → higher effective
  BW and clocks (AMD reports **+10–15% compute-bound GEMM** vs SPX) because cross-XCD traffic is gone.
- A single large model spanning all CUs and >24 GB **must** use SPX; strong-scaling one kernel across
  8 XCDs pays Fabric/L3 costs for any cross-XCD sharing.
- Switching mode terminates GPU processes and reloads amdgpu; reverts to SPX/NPS1 on reboot.

```bash
amd-smi list
sudo amd-smi set --gpu all --compute-partition CPX
sudo amd-smi set --gpu all --memory-partition  NPS4
```

## The levers
1. **8-multiple grids/tiles** for even XCD spread.
2. **Keep an operand's reuse on one XCD** (tile schedule) for L2 hits.
3. **Avoid device-wide barriers**; if needed, stage through L3 and budget ~200 ns + slowest-XCD clock.
4. **Map concurrent streams to ACEs** (up to 4/XCD) for kernel overlap.
5. **CPX/NPS4** for many-small-kernel / multi-tenant density (+10–15% GEMM, XCD-local memory).
6. **SPX/NPS1** for one large model needing all CUs / >24 GB.

## Pitfalls
- **Non-8-multiple grids** → straggler XCDs + scattered L2 reuse.
- **Tight cross-XCD sync** → bottlenecked by the slowest XCD (3–10% clock spread) + ~200 ns Fabric.
- **SPX+NPS4** is rejected by the driver.
- **Forgetting MI250X is also chiplet** but as **two separate devices** (not one) — different model.

## Verify
- `amd-smi static` / `rocm-smi --showcomputepartition --showmemorypartition` for current mode.
- `rocprof-compute` XCD load balance; per-XCD clock via `amd-smi metric`.
- A/B SPX vs CPX/NPS4 on a compute-bound GEMM to confirm the +10–15%.
