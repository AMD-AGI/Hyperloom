---
title: mori EP dispatch/combine v2 (FlyDSL / cco-LSA) — experimental reimplementation
kind: technique
operator: ep_dispatch_combine
gens: [gfx942, gfx950]
dtypes: [bf16, f32, fp8_e4m3_fnuz, fp4_e2m1]
regimes: [prefill, decode]
updated: 2026-08-04
sources:
  - ROCm/mori@dc4bc75a:python/mori/ops/dispatch_combine_v2/README.md
  - ROCm/mori@dc4bc75a:python/mori/ops/dispatch_combine_v2/__init__.py
  - ROCm/mori@dc4bc75a:python/mori/ops/__init__.py
---

# mori EP dispatch/combine v2 — FlyDSL / cco-LSA reimplementation

## What it is
`python/mori/ops/dispatch_combine_v2/` is a **mori-parity reimplementation** of intranode (single-node,
EP8) dispatch/combine, built on **mori-cco LSA** (intra-node P2P over a flat symmetric VA, the same
CCO-SDMA transport layer other mori collectives use) and **FlyDSL** device kernels, instead of mori-v1's
hand-written HIP kernels. Reference implementation: ROCm/FlyDSL PR #522. It supports bf16/f32/fp8
(gather-only)/fp4 (gather-only, gfx950-only) token dtypes, both gather- and scatter-style combine, fp8
combine-wire quant, StdMoE conversion, and a mori-parity host op layer.

## Its own README banner is stale — verified directly
The file's own header says *"Test-only, not a mori API (yet)... There is no `mori.ops.dispatch_combine_v2`
package export."* **This is out of date at the pinned commit** (verified by reading the actual files,
not just trusting the banner):
- `dispatch_combine_v2/__init__.py` has a real `__all__` export (`EpDispatchCombineConfig`,
  `EpDispatchCombineOp`, `EpDispatchRoutingHandle`) via relative imports (`from .dispatch_combine_op
  import ...`) — not the "import each other by top-level name, no `__init__.py`" state the README
  describes.
- `mori/ops/__init__.py` lazily loads it (`_LAZY_SUBMODULES = {"dispatch_combine_v2"}`, resolved via a
  module-level `__getattr__`) — lazy **because FlyDSL is an optional dependency**
  (`pip install amd_mori[flydsl]`), not because the API is unstable.

So `import mori.ops.dispatch_combine_v2` does work at this pin; the real caveat is **adoption**, not
importability: it is absent from `docs/MORI-EP-GUIDE.md` (the guide only documents v1), not wired into
aiter's `MoriAll2AllManager` seam, and only exercised by its own
`tests/python/ops/dispatch_combine_v2/` suite. Treat it as a second, less-adopted implementation, not
(yet) the one to build production code against — but don't dismiss it as literally untestable either.

## Measured perf (its own README, MI308X gfx942, bf16, CUDA-graph)
Per-rank bandwidth at EP8, hidden=7168, top-k=8, 256 experts, dispatch 64 blocks / combine 128 blocks ×
16 warps:

| tok/rank | dispatch | combine |
|---:|---:|---:|
| 512 | 268 GB/s | 213 GB/s |
| 2048 | 306 GB/s | 294 GB/s |
| 8192 | 314 GB/s | 323 GB/s |

**Design note directly from the source** (its README's own explanation, not re-derived): combine's remote
reads are latency-bound, so it wants **~128 blocks** to hide xGMI read latency across many warps;
dispatch's posted writes saturate at **~64 blocks** (half the CUs) because it's throughput- not
latency-bound. This is a **qualitatively different grid-sizing rule** than v1's — v1's own tuning (see
`tuning.md`) found *dispatch* wanting the larger grid (scaled toward the full CU count) and *combine*
also CU-scaled but not needing more blocks than dispatch. Don't assume v1's CU-count-scaling intuition
carries over to v2's kernel design; they are different implementations with different bottlenecks per
phase.

## Kernel-authoring detail (delegated, per KB convention)
This card documents v2 as an mori *operator variant* (what it is, its measured perf, its adoption
status) — it does not re-document how to write/read FlyDSL kernels themselves. For that, see
[`languages/flydsl/INDEX.md`](../../../../languages/flydsl/INDEX.md), which already covers FlyDSL
authoring generally (aiter's own FlyDSL-backed kernels included); the FlyDSL primitives this specific op
uses (`flydsl_prims.py`: system atomics, ordered stores, fences, volatile-spin waits) are op-specific
enough that they belong in this op's own source, not duplicated into the language folder.

## Why this is out of scope for a forge-loop tuning task today
It has no stable production integration point (no aiter seam, absent from the main guide) and its own
test/bench harness needs `sys.path.insert(0, <dir>)` gymnastics that a forge-loop `driver.py` would need
to special-case. If a future task wants to explore it, that is a **new driver**, not a config-file change
to an existing one — treat it as a candidate for a dedicated future campaign, not a knob to add to the v1
EP dispatch/combine task.

## Sources
- Design notes, measured perf table, block/warp rationale: `ROCm/mori@dc4bc75a:python/mori/ops/dispatch_combine_v2/README.md`.
- Package-export reality (contradicts the README's own stale banner): `ROCm/mori@dc4bc75a:python/mori/ops/dispatch_combine_v2/__init__.py`, `python/mori/ops/__init__.py`.
