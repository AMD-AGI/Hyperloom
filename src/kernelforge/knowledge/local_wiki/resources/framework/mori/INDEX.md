---
title: mori knowledge map — index, file roles & problem-routing
kind: index
scope: framework/mori
updated: 2026-08-29
pinned_source: ROCm/mori@dc4bc75a
---

# mori — knowledge map

This file is the entry index for everything under `framework/mori/`. `mori` (`ROCm/mori`) is AMD's
GPU-initiated communication library — the counterpart to `framework/aiter/` for **expert-parallel
all-to-all and symmetric-memory collectives**, rather than compute-op dispatch. Before this folder
existed, mori only appeared as a *backend* of aiter's `moe_dispatch_combine` operator, documented from
the aiter side. Those aiter operator cards have since been removed (operator-level knowledge went stale
too fast to maintain), so **this folder is now the only place mori is written up** — but mori still had
no first-class identity of its own before it existed: no
per-shape tuning-DB documentation, no control-plane story, nowhere to record measurements taken
directly against mori's own API. This folder is that home.

> **Scope discipline (read before trusting a claim here):** mori's actual repo surface is much wider
> than what this folder documents — it also ships MORI-IO (storage), MORI-CCL / hierarchical allgather,
> MORI-IR, MORI-UMBP, and a SDMA/CCO transport layer. **This folder currently covers EP dispatch/combine
> only** (the `mori.ops.EpDispatchCombineOp` surface) because that is the only area anyone has read the
> source for in depth. Do not assume claims here generalize to MORI-IO/CCL/IR/UMBP — those are simply
> not covered yet.

## Reading order
1. **`overall/repo_layout.md`** — what mori actually is, full repo scope vs. what this folder covers, how it relates to aiter.
2. **`overall/launch_config_tuning.md`** — the control-plane concept every mori op shares: MANUAL vs AUTO launch mode, the per-(arch, kernel_type, ep_size, shape) JSON tuning-DB, mori's own official tuner.
3. **`operators/ep_dispatch_combine/`** — the one operator this folder has real depth on.

## Start here — problem → files → order
| Task / symptom | Read in this order |
|---|---|
| "What is mori, how does it relate to aiter?" | `overall/repo_layout.md` |
| "How do I tune mori's launch params for my shape?" | `overall/launch_config_tuning.md` → `operators/ep_dispatch_combine/tuning.md` |
| "Is `use_external_inp_buf`/zero-copy worth trying?" | `operators/ep_dispatch_combine/tuning.md` §"combine buffer mode" |
| "What did KernelForge itself measure on MI300X for this op?" | `operators/ep_dispatch_combine/tuning.md` §"KernelForge-measured results (MI300X)" |
| "Is the FlyDSL/v2 dispatch-combine rewrite usable?" | `operators/ep_dispatch_combine/v2_flydsl.md` |
| "How does aiter actually call mori in production?" | `overall/repo_layout.md` § "Relation to aiter" → the source: `aiter/dist/device_communicators/all2all.py` (`MoriAll2AllManager`) |
| Math contract / numerics / fusion for dispatch-combine | Not documented in this repo (mori-agnostic, and it rots fast) — read `aiter/fused_moe.py` and `mori/ops/` |

## Folder structure & file roles
```
framework/mori/
├── INDEX.md                              ← this map
├── overall/
│   ├── repo_layout.md                    # what mori is, full scope vs. covered scope, relation to aiter
│   └── launch_config_tuning.md           # MANUAL/AUTO mode, JSON tuning-DB schema, mori's own tuner
└── operators/
    └── ep_dispatch_combine/
        ├── overview.md                   # mori's OWN API surface (kernel types, config, buffer modes);
        │                                 # math contract and numerics are not documented here — read
        │                                 # aiter/fused_moe.py and mori/ops/
        ├── tuning.md                     # THE measured-data card: mori's official per-chip tuning-DB
        │                                 # numbers + KernelForge's own MI300X forge-loop campaign results
        └── v2_flydsl.md                  # the experimental FlyDSL/cco-LSA reimplementation (dispatch_combine_v2)
```

## Why mori has its own folder
mori used to be documented only from the aiter side, as a backend of aiter's `moe_dispatch_combine`
operator: what `MoriAll2AllManager` passes, the integration points, the aiter-side pitfalls. That
answered "how does aiter use mori" and nothing else, and it has since been deleted along with the rest
of the aiter operator cards — for that question, read
`aiter/dist/device_communicators/all2all.py` directly.

This folder answers a different question: mori **as its own library, with its own tuning control
plane** — the JSON per-shape DB, MANUAL vs AUTO mode, and our own measured MI300X numbers. The two are
not the same subject, and the gap between them is real: aiter calls mori with a fixed set of kwargs and
neither exposes nor consumes mori's tuning-DB mechanism at all today. That gap is documented in
`operators/ep_dispatch_combine/tuning.md`.

## What to sync when mori is upgraded
1. Re-verify every card against the new commit (config fields, kernel type list, tuning-config JSON schema).
2. Bump `pinned_source` here and `sources:`/`updated:` on every touched card.
3. If new tuning_configs/*.json land for an arch/model this folder cites, re-check whether the numbers changed.
