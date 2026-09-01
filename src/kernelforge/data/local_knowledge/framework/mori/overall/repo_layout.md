---
title: mori — what it is, repo scope, relation to aiter
kind: technique
gens: [gfx942, gfx950]
updated: 2026-08-04
sources:
  - ROCm/mori@dc4bc75a:README.md
  - ROCm/mori@dc4bc75a:docs/MORI-EP-GUIDE.md
  - ROCm/mori@dc4bc75a:CMakeLists.txt
  - ROCm/aiter@a177781d4:aiter/dist/device_communicators/all2all.py
---

# mori — repo layout & scope

## What mori is
`ROCm/mori` ("Modular Optimized Runtime Interconnect", per the repo's own naming) is AMD's GPU-initiated
communication library for LLM inference/training on Instinct GPUs. It provides a **symmetric-memory
heap** (`mori.shmem`) as the foundation, then several higher-level libraries built on it:

| Component | What it does | Covered by this KB folder? |
|---|---|---|
| **MORI-EP** (`mori.ops.EpDispatchCombineOp`) | Expert-parallel dispatch/combine all-to-all — the op this KB folder documents | ✅ yes, in depth |
| MORI-SHMEM (`mori.shmem`) | The symmetric-memory / P2P foundation EP (and everything else) is built on | 🟡 only as much as EP needs (init calls) |
| MORI-CCO / SDMA (`include/mori/cco/`, `.claude/skills/cco-sdma-api/`) | Low-level System-DMA transport primitives (`cco.Window`, `lsa_ptr`) used by the experimental v2 dispatch/combine | 🟡 only as far as `v2_flydsl.md` needs |
| MORI-CCL (`python/mori/ccl/`) | Hierarchical / host-proxy allgather collectives (FSDP-style), built on CCO-SDMA | ❌ not covered |
| MORI-IO (`docs/MORI-IO-GUIDE.md`) | Storage/KV-cache transfer library | ❌ not covered |
| MORI-IR / MORI-UMBP | IR-based collective codegen; unified memory/buffer pooling | ❌ not covered |

**Build options relevant to EP**: `BUILD_OPS=ON` (dispatch/combine), `BUILD_SHMEM=ON` (required by
`BUILD_OPS`), `ENABLE_STANDARD_MOE_ADAPT=OFF` by default (turn on for DeepEP-compatible 3D layouts),
`ENABLE_PROFILER=OFF` by default (turn on for MORI-VIZ perfetto traces).

## Relation to aiter
mori and aiter are **peer libraries with a one-way dependency for EP**: aiter owns the single-GPU MoE
path (local permute/sort, fused grouped-GEMM) and, for distributed expert parallelism, **delegates the
actual cross-GPU all-to-all to mori** via `MoriAll2AllManager`
(`aiter/dist/device_communicators/all2all.py` — read that file for the exact integration; this repo
keeps no aiter-side card for it). mori has no dependency on aiter and does not know it is being called by it —
`EpDispatchCombineOp` is a standalone op usable directly (as this task's own `driver.py` does, without
ever importing aiter).

**Important asymmetry**: aiter's `MoriAll2AllManager` calls mori with a small set of **fixed kwargs
chosen once** (`MoriAll2AllManager.get_handle` in `aiter/dist/device_communicators/all2all.py` has the
exact values) — not tuned per-shape via mori's own tuning-DB mechanism
(`overall/launch_config_tuning.md`). That means today, a shape where mori's real optimum differs from
aiter's fixed default (which this KB folder's `operators/ep_dispatch_combine/tuning.md` shows is common)
gets **no benefit** from mori's tuning-DB unless something upstream of aiter (or aiter itself) is changed
to consume it. This is a real, open gap, not a documentation gap — worth flagging if you're the one
deciding where a validated tuning result should land (mori's own JSON DB is the easy, low-risk landing
spot **for mori-direct callers**; getting it to also help aiter-mediated callers needs an aiter-side
change).

## Where the source actually lives (EP-relevant subset)
```
mori/
├── docs/MORI-EP-GUIDE.md                       # the EP user guide (most current EP reference — read it
│                                                # directly; more complete than any KB card on some knobs,
│                                                # e.g. combine_zero_copy, that were added after this KB
│                                                # was last synced)
├── python/mori/
│   ├── ops/
│   │   ├── dispatch_combine.py                 # EpDispatchCombineConfig / EpDispatchCombineOp (v1, production)
│   │   ├── tuning_config.py                    # TuningConfigManager — JSON DB lookup (see launch_config_tuning.md)
│   │   ├── tuning_configs/*.json               # the actual per-(arch,model,kernel,ep_size,phase) tuned rules
│   │   └── dispatch_combine_v2/                # experimental FlyDSL/cco-LSA reimplementation (see v2_flydsl.md)
│   └── shmem/api.py                            # shmem init/finalize Python API
├── include/mori/ops/dispatch_combine/          # C++ header: config, handle, kernel args
├── src/ops/dispatch_combine/                   # dispatch_combine.cpp (core), internode_v1.cpp, low_latency_async.cpp
├── tools/batch_intranode_tuning.sh             # mori's own official per-arch tuner (see launch_config_tuning.md)
└── tests/python/ops/bench_dispatch_combine.py  # reference benchmark harness (what the tuner drives)
```

## Sources
- Component list, build options: `ROCm/mori@dc4bc75a:README.md`, `CMakeLists.txt`.
- EP guide structure/currency: `ROCm/mori@dc4bc75a:docs/MORI-EP-GUIDE.md`.
- aiter's fixed single-node kwargs: `ROCm/aiter@a177781d4:aiter/dist/device_communicators/all2all.py` (see aiter.md for the exact line).
