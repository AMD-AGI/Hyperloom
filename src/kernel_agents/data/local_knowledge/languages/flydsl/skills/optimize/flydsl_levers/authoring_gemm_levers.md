---
title: "FlyDSL — GEMM authoring levers (tiling / LDS / MFMA-loop / epilogue)"
kind: language
gens: [gfx942, gfx950]
updated: 2026-06-01
---

> **Reference (how-to), not a verdict.** Ingested from the FlyDSL authoring skill. GEMM-specific
> follow-on to [`authoring_optimization.md`](authoring_optimization.md). For the SOTA backend decision
> for dense GEMM (which path to pick, library vs author) see the operator card
> [`../../operators/dense_gemm/flydsl.md`](../../operators/dense_gemm/flydsl.md).

# FlyDSL GEMM optimization

## Overview

Use this document when a FlyDSL kernel is already clearly **GEMM-like** and the
next question is no longer "what category is this kernel?" but "which GEMM
optimization lever should I pull first?".

This is a specialized follow-on to the generic [`authoring_optimization.md`](authoring_optimization.md)
workflow. Start there when the bottleneck is still broad or ambiguous. Switch
here when the optimization work centers on GEMM structure, MFMA-loop
efficiency, LDS layout, or epilogue/store strategy.

---

## When to use

Use this guide when most of the optimization discussion is about:

- `tile_m` / `tile_n` / `tile_k` selection
- MFMA instruction shape and repeat layout
- LDS ping-pong or multi-stage buffering
- XOR swizzle or other bank-conflict avoidance
- global-load / LDS-load / MFMA overlap
- `sched_mfma` / `sched_vmem` / `sched_dsrd` / `sched_dswr` tuning
- direct-store vs shuffle/reorder epilogue
- arch VGPR pressure, occupancy, or spill risk
- ATT trace or ISA evidence from the GEMM hot loop

Do **not** start here for:

- first-time kernel authoring from scratch (see [`authoring_tile_programming.md`](authoring_tile_programming.md))
- general correctness bugs (see [`debugging.md`](debugging.md))
- non-GEMM kernels where the bottleneck is still unclear

---

## Workflow

### 1. Confirm the kernel is really GEMM-like

Read the full device kernel and host launcher. Verify that the hot path is a
tiled matmul or fused matmul-style loop rather than a generic reduction or
copy-heavy kernel with only incidental MFMA use.

Questions to answer first:

- What are the logical `M`, `N`, and `K` dimensions?
- How are blocks and waves partitioned across output tiles?
- Which operands are read from global memory each iteration?
- Which data is reused enough to justify LDS staging?
- Is the epilogue simple direct store, or does it reorder output fragments?

### 2. Classify the bottleneck before rewriting

Use the strongest evidence you have:

- runtime measurements: kernel time, shape sensitivity, achieved throughput
- ATT trace if available: `vmcnt`, `lgkmcnt`, `s_barrier`, `ds_*`,
  `buffer_load_*`, `v_mfma_*`
- ISA dump if available: MFMA density, barrier count, memory-instruction mix

Rule of thumb:

- High `s_waitcnt vmcnt(0)` before MFMA -> global-load latency exposed
- High `s_waitcnt lgkmcnt(0)` or `ds_*` stall -> LDS latency or bank conflicts
- High `s_barrier` stall -> synchronization overhead
- Low MFMA density / many bubbles -> schedule or loop-shape problem
- Good MFMA density but poor overall speed -> tile shape, occupancy, or store path

### 3. Prioritize high-impact structure first

Tune in this order:

1. **Tile strategy**
2. **LDS staging and overlap**
3. **MFMA-loop scheduling**
4. **Epilogue/store strategy**
5. **Final parameter tuning**

Do not start by micro-tweaking constants if the kernel still has obvious
pipeline, tiling, or memory-layout problems.

---

## Core patterns

### Tiling

Check these constraints first:

- `tile_m` is a multiple of the MFMA M dimension
- `tile_n` is large enough to keep waves busy and maps cleanly to wave/workgroup partitioning
- `tile_k * elem_bytes` aligns with the kernel's load and MFMA packing strategy
- total per-stage LDS fits the target arch budget

Use larger `tile_k` when compute can hide memory latency and LDS budget allows.
Reduce `tile_k` when LDS usage, register pressure, or occupancy becomes the
real limit.

When shapes are irregular, prefer a tile choice that remains robust across the
benchmarked range instead of overfitting to one hotspot shape.

### LDS staging

For GEMM kernels, ask whether one of these is true:

- an operand tile is reread many times by MFMA -> stage it through LDS
- global-load latency is exposed -> prefetch earlier
- one LDS buffer is idle while compute runs -> consider ping-pong staging

If the kernel already uses LDS, inspect whether the problem is:

- **capacity**: too much LDS per workgroup
- **layout**: bank conflicts from stride or access pattern
- **timing**: `ds_write` too close to dependent `ds_read` / wait

### Swizzle and bank conflicts

Prefer XOR-style swizzle when:

- the access pattern is regular
- read/write transforms can stay consistent
- LDS headroom is tight

Prefer padding when:

- swizzle would overcomplicate address math
- LDS has enough headroom
- a small stride change removes the conflict cleanly

Always keep the read path and write path consistent. A swizzled store with an
unswizzled load is a correctness bug, not a performance optimization.

### Prefetch and scheduling

Prefetch helps only when there is enough independent work to hide the latency.
Good candidates:

- next-tile global loads
- address computation
- independent MFMA groups
- epilogue preparation that does not consume the not-yet-ready data

If prefetch adds substantial carried state, re-check VGPR pressure and
occupancy. Prefetch that causes spills often loses more than it wins.

For scheduler hints, match them to the real loop body instead of copying fixed
numbers from another kernel. The MFMA count, LDS-read count, and VMEM count per
iteration should come from the current kernel's loop structure.

### Epilogue choice

Here, "epilogue" means the final stage that maps accumulator fragments to
output-memory stores.

Use direct store when it is already reasonably coalesced and simple.

Consider a shuffle/reorder epilogue when:

- output stores are poorly coalesced
- tile shape creates fragmented writes
- the extra LDS/barrier cost is smaller than the store inefficiency

---

## Quick reference

| Symptom | Likely issue | First action |
|---|---|---|
| High `s_waitcnt vmcnt(0)` before MFMA | global-load latency exposed | move next-tile loads earlier; revisit prefetch distance |
| High `s_waitcnt lgkmcnt(0)` / `ds_*` stall | LDS latency or bank conflict | inspect LDS layout, swizzle, padding, write-read distance |
| High `s_barrier` stall | too many sync points | reduce stage boundaries or merge dependent phases |
| Low MFMA ratio in hot loop | schedule overhead or loop shape | count MFMA vs memory ops and simplify loop body |
| Speed good on one shape, bad on nearby shapes | brittle tile choice | re-check tile divisibility, occupancy, and edge handling |
| Throughput drops after adding prefetch | register pressure too high | reduce carried state or use a lighter staging strategy |

---

## Correctness constraints

Performance changes must preserve these constraints:

- respect arch-specific LDS limits
- keep tile packing and vector widths aligned with operand layout
- verify accumulator/output type conversions do not introduce unexpected overflow
- apply any swizzle or padding consistently to both producer and consumer paths
- confirm edge masking still works for non-divisible shapes

---

## Common mistakes

- starting from scheduler constants before proving the bottleneck
- copying tile sizes from another kernel without checking work decomposition
- adding multi-stage LDS buffering that destroys occupancy
- treating every LDS issue as a swizzle problem instead of checking wait distance
- comparing only one benchmark shape and overfitting to it
- assuming a trace or ISA pattern from another repository matches the current kernel

---

## Verification

After each meaningful change:

1. Re-run correctness checks first
2. Re-measure the same GEMM shapes as the baseline
3. If trace or ISA evidence is available, confirm the targeted stall actually moved
4. Check that speedup is not coming from one shape while others regress
