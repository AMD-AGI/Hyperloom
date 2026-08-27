---
title: moe_dispatch_combine — overview
kind: operator_overview
operator: moe_dispatch_combine
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3, fp4_e2m1]
regimes: [prefill, decode]
updated: 2026-07-29
sources:
  - ROCm/mori@35e2effb6:docs/MORI-EP-GUIDE.md
  - ROCm/mori@35e2effb6:python/mori/ops/dispatch_combine.py
  - ROCm/aiter@a177781d4:aiter/dist/device_communicators/all2all.py
  - https://gau-nernst.github.io/amd-a2a/
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
---

# moe_dispatch_combine  (token permute/scatter to experts + un-permute/combine)

## TL;DR
After routing, every token must be **moved to where its top-k experts live** (dispatch / scatter), then
the expert outputs **gathered back to the originating token position** (combine / un-permute). With
expert parallelism this becomes a **sparse all-to-all** over xGMI (intra-node) / RDMA (inter-node); on a
single GPU it is a local gather/scatter (the align&sort permutation, see moe_routing_topk /
fused_moe_grouped_gemm). The single most important fact on MI300X/MI350X: this is the **dominant comm
cost of MoE EP**, it is **bandwidth-bound on xGMI**, and the win comes from **sparsity** (each token goes
only to its ≤k expert-ranks, not a dense all-to-all) plus **fusing the routed-weight multiply into
combine** and **overlapping with the grouped GEMM**.

## Math contract
- **dispatch**: given `input[T, H]`, `weights[T, k]`, `scales[T, scale_dim]` and `indices[T, k]` (int32
  expert ids from moe_routing_topk), send each token to the rank(s) holding its experts → per-rank packed
  buffers (`dispatch_output`, plus corresponding weights/scales/indices) and a received-token count, with
  enough metadata (`get_dispatch_src_token_pos`) to invert the map. Optional **on-the-wire quantization**
  (`quant_type`: `fp8_direct_cast`, or the newer blockwise `fp8_blockwise` / gfx950-only `fp4_blockwise`
  combine codecs) halves-to-quarters the bytes moved.
- **combine**: given expert outputs in the packed layout + `weights[T,k]`, gather back to `[T, H]`,
  **multiplying the per-token routing weight during the gather** (prob-mult fused into combine). bf16
  combine is common even when dispatch is fp8/fp4 (accuracy).
- Intra-node (this op's focus): all peers reachable over xGMI; no NIC. Inter-node adds RDMA (kernel types
  `InterNode` / `InterNodeV1` / `InterNodeV1LL` / `AsyncLL`).

## Shape regimes
- **prefill / high-concurrency decode**: large token batches → **throughput** kernels (`InterNodeV1`
  favors staged RDMA for >256 tokens/rank).
- **low-concurrency decode**: small batches (≤256 tokens/rank) → **low-latency** kernels
  (`InterNodeV1LL`/`AsyncLL`) that overlap comm with compute. This normal-vs-low-latency split is the same
  one DeepEP and MoRI-EP both expose (kernel-type enum), and MoRI now auto-switches on it.
- DeepSeek-V3/R1 reference: hidden 7168, top-8, E=256, up to 4096 tokens/rank.

## Where it matters (Amdahl)
On dense models this op doesn't exist. On **MoE with EP** it is often the **largest single comm term** —
gating throughput under SLO. AMD's Wide-EP shows EP16 giving **1.3× output tok/s** over EP8 (and ~2×
offline vs TP8) once dispatch/combine scale; conversely a slow/unoverlapped a2a caps the whole MoE layer.
A reference hand-written MI300X a2a went from a 93,540 µs naive baseline to **292 µs** — i.e. the kernel
quality here is worth ~300× (verified against the source worklog — see tuning.md).

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| mori (MoRI-EP) | 🟢 sota (native AMD EP all-to-all) | [backends/mori.md](backends/mori.md) |
| aiter | 🟢 (`MoriAll2AllManager` / `FlyDSLAll2AllManager` seam + `moe_sorting` local permute) | [aiter.md](aiter.md) |
| hip | 🟢 sota (the kernels: P2P symmetric-memory dispatch/combine; vLLM/sglang permute) | [backends/hip.md](backends/hip.md) |
| triton | 🟡 (local permute/scatter; GPU-initiated comms via Iris experimental) | [backends/triton.md](backends/triton.md) |

## Fusion neighbors
- **prob-mult → combine** (routing weight multiplied during gather).
- **shared-expert into the same dispatch** (shared_expert_fusion): shared experts as synthetic routed
  experts → one fused dispatch for shared+routed.
- **combine → down-proj GEMM epilogue** (north-star single-kernel design; not yet shipped on AMD).
- dispatch consumes moe_routing_topk output; feeds fused_moe_grouped_gemm. See [fusion.md](fusion.md).

## Numerics
fp8/fp4 dispatch + bf16 combine is the common recipe; combine reduction order differs from a dense path →
gate with greedy/temp=0 parity, not byte match. See [numerics.md](numerics.md).

## How to bench
mori `tests/python/ops/bench_dispatch_combine.py` (intra-node) or the inter-node examples under
`examples/ops/dispatch_combine/`; report GB/s dispatch & combine at (tokens, hidden, topk, dtype, EP). e2e:
MoE model tok/s with EP enabled, trace that dispatch/combine overlap the grouped GEMM.

## Sources
- MoRI-EP guide (dispatch/combine API, kernel types, config, benchmarking, tuning): `ROCm/mori@35e2effb6:docs/MORI-EP-GUIDE.md`.
- on-box: `ROCm/mori@35e2effb6:python/mori/ops/dispatch_combine.py` (`EpDispatchCombineConfig`/`EpDispatchCombineOp`).
- aiter EP seam: `ROCm/aiter@a177781d4:aiter/dist/device_communicators/all2all.py` (`MoriAll2AllManager`, `FlyDSLAll2AllManager`).
- MI300X a2a reference study (292 µs, grid_size=304, malloc/memset fix): https://gau-nernst.github.io/amd-a2a/
- Wide-EP EP16/EP8 1.3×, shared-expert fusion: https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
