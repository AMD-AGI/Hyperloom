---
title: mori — launch config & tuning-DB control plane
kind: technique
gens: [gfx942, gfx950]
updated: 2026-08-04
sources:
  - ROCm/mori@dc4bc75a:docs/MORI-EP-GUIDE.md
  - ROCm/mori@dc4bc75a:python/mori/ops/tuning_config.py
  - ROCm/mori@dc4bc75a:python/mori/ops/tuning_configs/gfx942_mi308x_IntraNode_ep8_{dispatch,combine}.json
  - ROCm/mori@dc4bc75a:tools/batch_intranode_tuning.sh
---

# mori — launch config & tuning-DB control plane

This is mori's analog of aiter's `overall/tuning_db.md` — the mechanism, not the numbers (numbers for
EP dispatch/combine live in
[`../operators/ep_dispatch_combine/tuning.md`](../operators/ep_dispatch_combine/tuning.md)).

## Two modes: MANUAL vs AUTO
Every mori op reads `MORI_EP_LAUNCH_CONFIG_MODE` (env var, default `"MANUAL"`):

- **MANUAL** (default): launch params (`block_num`, `warp_per_block`, `rdma_block_num`) come from
  `EpDispatchCombineConfig`'s constructor defaults or your per-call override on `dispatch()`/`combine()`.
  This is what a hand-written tuning loop (including a KernelForge forge-loop task) drives.
- **AUTO**: mori looks up a per-shape JSON rule (below) if one exists for the detected
  `(gpu_arch, gpu_model, kernel_type, ep_size)`; if none exists, it falls back to a **hard-coded** value
  by kernel-type family (verified directly against `dispatch_combine.py`'s three-way `if/elif/else` at
  `dc4bc75a` — the two InterNodeV1 variants are NOT the same fallback, despite reading similarly):
  `InterNodeV1` → `block_num=96, rdma_block_num=64, warp_per_block=8`; `InterNodeV1LL` →
  `block_num=256, rdma_block_num=128, warp_per_block=8`; `IntraNode`/`InterNode`/`AsyncLL` →
  `block_num=128, rdma_block_num=0, warp_per_block=16`. **Per-call `block_num`/`warp_per_block`
  overrides are silently ignored in AUTO
  mode** — this matters if you're writing a driver that expects its overrides to take effect; check the
  env var isn't set, or your tuning loop will appear to have no effect. Measured directly on MI300X
  (no JSON entry exists for that model, so this is the hard-coded fallback path): `AUTO` lands at
  ~1.79 ms on the EP8/4096-token reference shape — ~10% faster than an untuned MANUAL config, but ~9%
  slower than a properly-searched one. See
  [`../operators/ep_dispatch_combine/tuning.md`](../operators/ep_dispatch_combine/tuning.md) §"Round 3"
  for the full measurement (including proof the config file's values are ignored under `AUTO`).

## The JSON tuning-DB (what AUTO mode reads)
Files live at `python/mori/ops/tuning_configs/{arch}_{model}_{kernel}_ep{n}_{phase}.json`, e.g.
`gfx942_mi308x_IntraNode_ep8_dispatch.json`. Each file is one `(gpu_arch, gpu_model, kernel_type,
ep_size, phase)` combination, containing a `rules` list.

**Schema differs between dispatch and combine files** (`tuning_config.py`, current schema as of
`dc4bc75a`):
- **dispatch** rules are keyed by `(dtype, num_tokens, hidden_dim, topk)` — `topk` is optional/wildcard
  for old entries written before it was added.
- **combine** rules are keyed by `(dtype, num_tokens, hidden_dim, topk, zero_copy, quant_type)` — two
  extra dimensions dispatch doesn't have, because combine has two independent modes dispatch doesn't:
  the buffer mode (`zero_copy`, i.e. `use_external_inp_buf`) and the wire codec (`quant_type`, plain
  bf16 vs `fp8_blockwise`/`fp4_blockwise`). **A rule tuned for one `zero_copy` value does not apply to
  the other** — see `operators/ep_dispatch_combine/tuning.md` for concrete numbers showing why (a ~29%
  bandwidth difference and a completely different optimal `warp_per_block`, on the same shape).
- Each rule records `block_num`, `rdma_block_num`, `warp_per_block`, `bandwidth_gbps` (the keep-best
  comparison metric), and `latency_us`. New tuning runs merge in with a **keep-best** strategy: a rule is
  only overwritten if the new bandwidth exceeds the existing one — so re-running the tuner is safe to
  repeat, it never regresses a file.

## Why `topk` was added to the schema (a real, recent change)
`tuning_config.py`'s docstring explains this directly: two models can share `hidden_dim` but route a
different number of experts per token (the docstring's own example: DeepSeek-V4-Pro top-6 vs Kimi-K3
top-16, both at hidden 7168) — that changes per-rank traffic volume and thus the best block/warp
geometry, even though `hidden_dim` alone would previously have matched them to the same (wrong-for-one)
rule. If you're extending the tuning-DB, this is the lesson: **check which dimensions actually change
per-rank byte volume**, don't assume the pre-topk schema's key set is complete.

## mori's own official tuner (methodology worth copying)
`tools/batch_intranode_tuning.sh` (intranode) and `tools/batch_internode_tuning.sh` (internode) sweep
candidate `(block_num, warp_per_block[, rdma_block_num])` values and keep whichever maximizes bandwidth
on the **bottleneck rank**. The guide recommends a **two-phase approach**:
1. **Calibrate** — full-scope sweep (~75 configs) on 2 representative token counts (128 and 4096) to
   confirm the full search space's optimum region.
2. **Quick sweep** — a ~9-12 config `quick` scope across all token counts, 6× faster, validated by step 1
   to usually land on the same optimum.

This is a stronger methodology than a single forge-loop campaign's 3-8 iteration budget can replicate
exactly, but the two-phase idea (cheap calibration pass to bound the search, then a narrower sweep) is
directly reusable guidance for scoping a forge-loop `program.md`'s search space.

## What AUTO mode + the JSON DB means for aiter callers
As noted in `repo_layout.md`: aiter's `MoriAll2AllManager` calls mori with **fixed MANUAL-mode kwargs**,
never setting `MORI_EP_LAUNCH_CONFIG_MODE=AUTO` and never consuming this JSON DB. A validated tuning
result written into `gfx942_mi300x_IntraNode_ep8_*.json` would benefit a **direct mori caller** running
with `MORI_EP_LAUNCH_CONFIG_MODE=AUTO` immediately, but would need an aiter-side code change (or an
aiter-side per-shape DB of its own, analogous to `tuned_fmoe.csv`) to benefit aiter/SGLang/vLLM callers
going through `MoriAll2AllManager`. Don't assume writing the JSON file alone closes the loop for
production aiter-mediated serving.

## Sources
- MANUAL/AUTO modes, fallback values, JSON schema example: `ROCm/mori@dc4bc75a:docs/MORI-EP-GUIDE.md` §6, §10.
- Schema evolution (topk dimension), keep-best merge, combine's extra `zero_copy`/`quant_type` keys: `ROCm/mori@dc4bc75a:python/mori/ops/tuning_config.py` (module docstring + `lookup()`).
- Two-phase calibrate/quick-sweep methodology: `ROCm/mori@dc4bc75a:tools/batch_intranode_tuning.sh` (header comment) and `docs/MORI-EP-GUIDE.md` §10.
- Real dispatch vs combine schema difference, observed directly: `ROCm/mori@dc4bc75a:python/mori/ops/tuning_configs/gfx942_mi308x_IntraNode_ep8_{dispatch,combine}.json`.
