---
title: CDNA3 / MI300X (gfx942) — XCD chiplets & partitioning
kind: hardware
gens: [gfx942]
dtypes: []
regimes: [both]
updated: 2026-11-07
---

# CDNA3 / MI300X (gfx942) — XCD chiplets & partitioning

> Companion to [arch.md](arch.md) (topology) and [memory_hierarchy.md](memory_hierarchy.md). This file
> is the chiplet behavior a kernel author must design around: scheduling, locality, clock variance,
> and the compute/memory partition modes. Numbers in brackets point to the Sources at the end.

## TL;DR
> The MI300X is **8 XCD chiplets** joined by Infinity Fabric [1]. A workgroup's waves all run on **one
> CU** [6]; L2 is **per-XCD** (4 MiB) [1]. Use **multiples of 8** for even XCD spread and L2
> efficiency, keep data-sharing workgroups on one XCD, and expect a **3–10% clock variance across
> XCDs** [2] that limits how tightly synchronized cross-XCD work can be.

## Concepts

### XCD internals
```
┌─────────────── XCD (one chiplet) ───────────────┐
│ HWS (hardware scheduler)                         │
│ 4 × ACE (Asynchronous Compute Engine)            │  dispatch workgroups to CUs
│ 40 CUs (38 active): each 4 SIMD + 4 Matrix Core  │
│   64 KiB LDS, 32 KiB L1; scalar cache per 2 CUs  │
│ 4 MiB shared L2 (16 channels × 256 KiB)          │
└──────────────────────────────────────────────────┘
```
(HWS [2]; ACEs, CUs, L1 and L2 [1]; SIMDs [2]; matrix cores 1216/304 [7]; LDS [6]; 16 KiB scalar cache
per 2 CUs [3].)
- **ACEs**: four per XCD send compute workgroups to the CUs [1].
- A workgroup's waves all run on the **same CU** [6].
- In SPX the hardware distributes workgroups **round-robin** across the XCDs [4][5] → grids in
  **multiples of 8** spread evenly [2].

### Cross-XCD cost
- L2 is **per-XCD** and coherent within its XCD; cross-XCD reuse misses to the 256 MiB Infinity Cache
  on the IODs, whose snoop filter resolves most coherent requests from other XCDs [1].
- Each IOD sits under 2 XCDs and fans out to 2 HBM stacks; memory closest to an XCD is on its IOD [1][4].

### Clock variance across XCDs (3–10%)
The 8 XCDs do **not** all run at the same clock: AMD gives a variation of **3–10%** among XCDs,
typically XCD0 fastest and XCD7 slowest on MI300X, and recommends the slowest XCD's average clock for
efficiency calculations [2]. Consequences (derived):
- A kernel with a **device-wide barrier** runs at the **slowest XCD's** pace; tightly coupled cross-XCD
  collectives pay this tax.
- **Per-XCD-independent** work (the common, embarrassingly-parallel GEMM/attention grid) is unaffected
  beyond load-balance, which the multiple-of-8 rule handles.
- Benchmark variance: repeat-to-repeat timing spread partly reflects clock differences — use the
  median of warm repeats and note spread.

### Partition modes (SPX/DPX/QPX/CPX × NPS1/2/4)
| Compute mode | Logical GPUs | XCDs each | CUs each | HBM each | Use |
|---|---|---|---|---|---|
| **SPX** (default) [4] | 1 | 8 | 304 | 192 GB | one big model/kernel |
| **DPX** [4] | 2 | 4 | 152 | 96 GB | two balanced jobs |
| **QPX** [1][2] | 4 | 2 | 76 (derived) | 48 GB | four jobs |
| **CPX** [4] | 8 | 1 | 38 | 24 GB | many small jobs, inference density |

| Memory mode | NUMA domains | Effect |
|---|---|---|
| **NPS1** | 1 | unified 192 GB, interleaved across all 8 stacks; works with SPX, DPX, CPX [4] |
| **NPS2** | 2 | 2 × 96 GB, with DPX [4] |
| **NPS4** | 4 | pairs of stacks (48 GB each) per partition; with CPX [4] (the white paper also pairs NPS4 with four partitions [1]) |

**Hard rule:** memory partitions must **not exceed** compute partitions [1][4] → **SPX+NPS4 is invalid**.

**Kernel implications:**
- In **CPX** a kernel sees a 38-CU/24 GB GPU; with **NPS4** that memory sits in the XCD's local
  quadrant [4]. On a compute-bound FP16 GEMM (16384×16384×4096), AMD measured CPX/NPS1 and CPX/NPS4
  at **10–15% higher total throughput** than SPX, CPX/NPS4 at higher clocks; NPS4 also gave 5–10%
  higher stream bandwidth [5].
- A single large model spanning all 304 CUs needs SPX [4]; strong-scaling one kernel across 8 XCDs
  pays Fabric/Infinity Cache costs for any cross-XCD sharing.
- The mode can be changed at runtime with `amd-smi`; the GPU returns to SPX on reboot or when the
  amdgpu driver is reloaded [4].

```bash
amd-smi list
sudo amd-smi set --gpu all --compute-partition CPX
sudo amd-smi set --gpu all --memory-partition  NPS4
```
(commands as in [5])

## The levers
1. **Multiple-of-8 grids/tiles** for even XCD spread [2].
2. **Keep data-sharing workgroups on one XCD** (tile schedule) for L2 hits [2].
3. **Avoid device-wide barriers**; if needed, budget for the slowest XCD's clock [2].
4. **CPX/NPS4** for many-small-kernel / multi-tenant density (AMD's 10–15% GEMM result above [5]).
5. **SPX/NPS1** for one large model needing all CUs / >24 GB [4].

## Pitfalls
- **Non-multiple-of-8 grids** → straggler XCDs + scattered L2 reuse [2].
- **Tight cross-XCD sync** → bottlenecked by the slowest XCD (3–10% clock spread) [2].
- **SPX+NPS4** is not a valid combination [1][4].
- **Forgetting MI250X is also chiplet** but as **two separate devices**: each of its two GCDs appears
  as a GPU [8] — a different model.

## Verify
- `amd-smi static` / `rocm-smi --showcomputepartition --showmemorypartition` for current mode.
- `rocprof-compute` XCD load balance; per-XCD clock via `amd-smi metric`.
- A/B SPX vs CPX/NPS4 on your compute-bound GEMM to check AMD's 10–15% for your shape.

## Sources
1. AMD, *AMD CDNA 3 Architecture* white paper (XCD, memory and partitioning sections) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
2. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* (clock differences among XCDs, WorkGroupMapping, partition table) — https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html
3. ROCm documentation, *AMD GPU specifications* — https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
4. AMD GPU driver documentation, *AMD Instinct MI300X GPU Partitioning Overview* — https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/gpu-partitioning/mi300x/overview.html
5. ROCm Blogs, *Deep dive into the MI300 compute and memory partition modes* — https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html
6. AMD, *"AMD Instinct MI300" Instruction Set Architecture* (CDNA3 ISA; §2.2.1 LDS, §4.3 work-groups) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
7. AMD, *AMD Instinct MI300X Accelerator* data sheet — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
8. AMD, *AMD CDNA 2 Architecture* white paper — https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf
