---
title: mori EP dispatch/combine — API surface overview
kind: operator_overview
operator: ep_dispatch_combine
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3, fp4_e2m1]
regimes: [prefill, decode]
updated: 2026-08-04
sources:
  - ROCm/mori@dc4bc75a:docs/MORI-EP-GUIDE.md
  - ROCm/mori@dc4bc75a:python/mori/ops/dispatch_combine.py
---

# mori EP dispatch/combine — API surface

The **math contract** (what dispatch/combine compute), **numerics** (quant/reduction-order), and
**fusion neighbors** (grouped GEMM, routed-weight multiply placement) are mori-agnostic and are **not
documented in this repo** — read `aiter/fused_moe.py` and `aiter/dist/device_communicators/all2all.py`
at the pinned commit. This card documents mori's **own direct API** (as used by a caller
that imports `mori.ops` directly, not through aiter's `MoriAll2AllManager` seam) and the **buffer-mode**
knob that aiter's fixed integration never exposes.

## The six kernel types (decision tree)
```
Is EP within a single node (xGMI only, no NIC)?
├─ Yes → IntraNode          (IntraNodeLL exists too, but see tuning.md — it loses to IntraNode at
│                             throughput shapes; only clearly wins at very small/low-latency batches,
│                             and even that needs re-confirming per shape, not assumed)
└─ No (multi-node, RDMA required)
   ├─ Throughput priority (large batches)  → InterNodeV1
   ├─ Latency priority (small batches)     → InterNodeV1LL or AsyncLL (only kernel type with a split
   │                                          dispatch_recv, for pipelined async transfers)
   └─ Baseline / debugging                 → InterNode
```

## Config surface (`EpDispatchCombineConfig`)
Required: `data_type` (deprecated for kernel launch — dtype is inferred from the runtime input tensor at
call time; kept for API back-compat), `rank`, `world_size`, `hidden_dim`, `scale_dim`, `scale_type_size`,
`max_token_type_size`, `max_num_inp_token_per_rank`, `num_experts_per_rank`, `num_experts_per_token`.

Tunable (class-level default, always overridable per-call on `dispatch()`/`combine()`):
`warp_num_per_block` (default **8**), `block_num` (default **80**), `use_external_inp_buf` (default
**True**), `kernel_type` (default `IntraNode`), `gpu_per_node` (default 8), `rdma_block_num` (default 0,
inter-node only), `num_qp_per_pe` (default 1), `quant_type` (default `"none"`).

**Note the class default (`block_num=80, warp_num_per_block=8`) differs from both AUTO mode's fallback
(128/16) and from what aiter's `MoriAll2AllManager` actually passes (80/16)** — three different "default"
numbers exist depending which layer you're reading from. Always check which one a given benchmark or
production caller is actually using before comparing numbers across sources.

## Buffer mode: `use_external_inp_buf` (the knob no aiter integration exercises)
Combine has two buffer modes, chosen per-call via `use_external_inp_buf` (int: `-1` = use config
default, `0` = zero-copy, `1` = external):

- **External** (`True`/`1`, the class default): pass an arbitrary tensor as `combine()`'s `input`; mori
  copies it into its own internally-managed peer-visible buffer before running the combine kernel.
- **Zero-copy** (`False`/`0`): call `op.get_registered_combine_input_buffer(dtype)` to get mori's own
  pre-registered buffer, write your expert output directly into it (in real usage: have the grouped
  GEMM's epilogue write there, no separate copy at all), then pass **that buffer** as `combine()`'s
  `input` with `use_external_inp_buf=0`. This skips the internal copy the external-buffer path performs.

This is a genuinely different code path with its own optimal `block_num`/`warp_per_block` — see
[`tuning.md`](tuning.md) for measured numbers showing why (mori's own official tuning-DB has it as a
**separate schema dimension** for combine, not a boolean flag on top of the same tuned geometry).

## Split send/recv (overlap primitive)
`dispatch_send()`/`dispatch_recv()` and `combine_send()`/`combine_recv()` let you interleave the
communication with compute (e.g. issue `dispatch_send`, run something else, then `dispatch_recv` once
you actually need the result) — `dispatch_recv()`/`combine_recv()` return `None`; the payload comes back
from the `_send` half. Note `dispatch_send()` just delegates to `dispatch()` internally per the guide —
the actual overlap benefit is from where **you** place the `_recv()` call in your code, not from the
kernel doing anything different.

## Standard MoE (DeepEP) compatibility
Built with `ENABLE_STANDARD_MOE_ADAPT=ON`, `dispatch_standard_moe()`/`combine_standard_moe()` fuse the
dispatch/combine with a 3D-layout conversion frameworks expecting DeepEP's tensor shape need; this is
what aiter's grouped GEMM consumes (see the aiter SOTA card). Off by default in CMake.

## Sources
- Kernel type decision guidance, config field table, buffer-mode API, split send/recv semantics: `ROCm/mori@dc4bc75a:docs/MORI-EP-GUIDE.md` §1-3, §6.
- Config dataclass defaults, `combine()`/`dispatch()` signatures: `ROCm/mori@dc4bc75a:python/mori/ops/dispatch_combine.py`.
