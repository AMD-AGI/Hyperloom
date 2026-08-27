---
title: moe_dispatch_combine — tuning
kind: technique
operator: moe_dispatch_combine
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3, fp4_e2m1]
regimes: [prefill, decode]
updated: 2026-07-29
sources:
  - https://gau-nernst.github.io/amd-a2a/
  - ROCm/mori@35e2effb6:docs/MORI-EP-GUIDE.md
  - ROCm/mori@35e2effb6:python/mori/ops/dispatch_combine.py
  - https://www.lmsys.org/blog/2026-05-28-mori/
  - https://moreh.io/technical-report/21k-output-tokens-per-second-deepseek-inference-on-amd-instinct-mi300x-gpus-with-expert-parallelism-251113/
---

# moe_dispatch_combine — tuning

## What you actually tune
A bandwidth-bound, latency-sensitive sparse all-to-all. Levers: kernel type (throughput vs low-latency),
`block_num`/`warp_num_per_block`/`rdma_block_num`, on-the-wire dtype (fp8/fp4 dispatch, blockwise combine),
buffer reuse (zero-copy / no per-call malloc), and overlap with the grouped GEMM.

## The MI300X reference findings (gau-nernst, GPU MODE AMD Distributed Challenge)
The hand-written dispatch+combine reimplementing pplx-kernels-style ideas progressed
**93,540 µs (naive reference) → 1,311 µs (PyTorch-only, sort + `all_to_all_single`) → 517 µs (P2P
symmetric memory + per-token lock) → 345 µs (fuse fake grouped-GEMM into combine, tuned grid) → 303 µs
(hoist malloc/memset out of the kernel) → 292 µs (varlen-aware work distribution across threadblocks)**
at E=256/topk=8/hidden 7168/world=8 (verified against the source worklog — the ladder and final numbers
match exactly). The load-bearing knobs it surfaced:
- **`grid_size = 304`** — *exactly* the MI300X CU count. 256 gave no such speedup; 304 gave **~3×
  combine-send speedup** specifically (dispatch tuning showed no comparable win). Size the grid to the
  full CU count, not a round power of two.
- **Hoist `torch.empty` / memset out of the kernel** — the caching-allocator malloc showed up in traces
  as a real cost (this only became visible once profiling moved off PyTorch Profiler, which itself showed
  spurious multi-GPU slowdowns — use manual CUDA/HIP events for multi-GPU timing).
- **P2P symmetric memory** (each rank maps peers' buffers via `hipIpcOpenMemHandle`-equivalents) → direct
  `load/store` instead of staged copies, synchronized via acquire/release signal flags
  (`__hip_atomic_store`/`__hip_atomic_load` with `__ATOMIC_RELEASE`/`__ATOMIC_ACQUIRE`).
- **Varlen work distribution**: naive round-robin threadblock-to-source-rank assignment leaves some
  threadblocks starved and others overloaded when per-rank token counts are uneven; flattening the work
  across all threadblocks (cumulative-sum walk over the 8 source ranks) fixes it.

## MoRI-EP knobs (production path)
- **Kernel type** (`EpDispatchCombineKernelType`): `IntraNode`/`IntraNodeLL` (xGMI only), `InterNode`
  (baseline/debug), `InterNodeV1` (throughput), `InterNodeV1LL` (low latency), `AsyncLL` (latency; the only
  kernel type with a split `dispatch_recv`/`combine_recv`). MoRI **auto-switches** by concurrency when
  `MORI_EP_LAUNCH_CONFIG_MODE=AUTO`. For single-node intra-node, `IntraNode` is the path.
- **Auto-mode fixed configs** (from the guide, not per-shape tuned): `InterNodeV1`/`InterNodeV1LL` →
  `block_num=96, rdma_block_num=64, warp_per_block=8`; `IntraNode`/`InterNode`/`AsyncLL` →
  `block_num=128, rdma_block_num=0, warp_per_block=16`. **Manual mode** (default) uses
  `EpDispatchCombineConfig` defaults (`block_num=80`, `warp_num_per_block=8`) or your per-call override.
- **`use_external_inp_buf=True`** (default) → an externally-managed input buffer for combine; `False` is
  the zero-copy path via `get_registered_combine_input_buffer`.
- **`max_num_inp_token_per_rank`**, `hidden_dim`, `num_experts_per_token`, `num_experts_per_rank` — model
  dims that size the internal buffers (all required, no defaults).
- **Arch pin**: `MORI_GPU_ARCHS=gfx942`/`gfx950` to stop wrong-arch JIT; `MORI_PRECOMPILE=1
  python -c "import mori"` to move first-iteration JIT cost out of the timed loop (~22 s one-time cost per
  the JIT-architecture doc).
- **Real per-arch/per-shape tuning**: mori ships a tuner (`tools/batch_intranode_tuning.sh`,
  `tools/batch_internode_tuning.sh`) that sweeps `(block_num, rdma_block_num, warp_per_block)` per
  `(kernel_type, dtype, num_tokens, hidden_dim)` and writes JSON rule files under
  `python/mori/ops/tuning_configs/{arch}_{model}_{kernel}_ep{n}_{phase}.json`, loaded automatically when
  `MORI_EP_LAUNCH_CONFIG_MODE=AUTO`. This is a real per-shape DB (analogous to aiter's `tuned_fmoe.csv`),
  not just the two fixed auto-mode configs above — those are only the hard-coded fallback when no tuned
  JSON exists for the detected arch/model.

## On-the-wire dtype
- **fp8 dispatch (`fp8_direct_cast`) + bf16 combine** is the standard recipe: halve dispatch bytes,
  keep combine in bf16 for accuracy. Measured (mori guide's reference table, EP8/EP16/EP32, 4096 tok,
  hidden 7168, top-8, fp8 dispatch + bf16 combine): **EP8 (IntraNode) 307 GB/s dispatch / 330 GB/s
  combine**; EP16-V1 208/161 GB/s (XGMI) + 63/49 GB/s (RDMA); EP32-V1-LL 103/91 GB/s (XGMI) + 57/50 GB/s
  (RDMA).
- **Quantized all-to-all (FP4 dispatch + FP8-blockwise combine)**, shipped in SGLang on MI355X: a
  **2.56× round-trip bandwidth reduction** (28,672 → 11,200 B/token) for `amd/DeepSeek-R1-0528-MXFP4-v2`.
  MoRI-EP `fp8_blockwise` combine (EP8, BF16 input, 4096 tok, hidden 7168, `scale_dim=56`, zero-copy=0,
  dispatch/combine block=128/warp=16) measured **~736 µs** vs a **~907 µs** BF16-no-quant reference — both
  numbers independently confirmed against the LMSYS/AMD MI355X writeup. The adaptive `InterNodeV1LL`
  path gives **1.52× dispatch / 1.82× combine** over `InterNodeV1` at ≤256 tok/rank (also confirmed).
- Dispatch/combine dtype is now **auto-detected** from the model's MoE weight format in SGLang
  (`SGLANG_MORI_DISPATCH_DTYPE` / `SGLANG_MORI_COMBINE_DTYPE` to override) — this replaced an earlier
  manual-env-var-only setup.
- `fp4_blockwise` combine is **gfx950-only** (hard-asserts otherwise); fp4 dispatch likewise needs CDNA4
  FP4 HW — **not** gfx942.

## Overlap with compute (the other half of the win)
- **HIP-graph-capture** the dispatch/combine so decode doesn't eat CPU launch overhead. EP input sizes are
  **dynamic** (per-routing) → you must **pad/static-ize** tensor sizes to capture.
- Split phases (`dispatch_send`/`dispatch_recv`, `combine_send`/`combine_recv`, `AsyncLL`-only) to
  **interleave** comm with the grouped GEMM rather than serialize.
- **Two-Batch Overlap (TBO)**: run two microbatches on separate comm/compute streams — microbatch A's
  dispatch send overlaps microbatch B's attention compute, etc. On SGLang/MoRI this is up to **+25%**
  throughput at large batch; with `MORI_ENABLE_SDMA=true` the transfers move to AMD's System DMA engines
  for **zero-compute-overhead** communication (no CUs consumed by the transfer itself).
- Confirm in a rocprofv3 trace that dispatch/combine kernels **overlap** the grouped GEMM and that there is
  **no `hipMalloc`** in steady state.

## Expert load balance (a tuning concern, not just correctness)
Naive contiguous expert sharding (experts 0-31 → GPU0, 32-63 → GPU1, …) showed **up to 2× workload
imbalance** across GPUs on DeepSeek-R1/MI300X (Moreh, confirmed against the source report) — one rank's
experts get more tokens, stalling the collective. An **EPLB-style frequency-balanced** grouping (measure
per-expert activation frequency, partition 256 experts into 8 sets of 32 balanced by total frequency,
reorder so each set is contiguous, and update the gate's output permutation to match) brought the
imbalance down to **within 5%** in the same report. Frequency drifts across decoder blocks and over time,
so this is a per-decoder, periodically-refreshed computation, not a one-shot static assignment.

## How to verify a tuning win
- Bandwidth harness (`bench_dispatch_combine.py`) at your (tokens, hidden, topk, dtype, EP) → compare GB/s
  vs the mori table.
- rocprofv3: kernel type matches concurrency (LL vs throughput), grid/block_num as configured, overlap with
  the GEMM, no malloc in steady state.
- Per-GPU token-count histogram to catch expert imbalance.

## Sources
- 292 µs / grid_size=304 (~3× combine-send only) / malloc+memset hoist / P2P + varlen ladder — verified
  against the full worklog, ladder matches exactly: https://gau-nernst.github.io/amd-a2a/
- MoRI-EP kernel types, config defaults, AUTO launch configs, tuner + JSON config layout, bandwidth table:
  `ROCm/mori@35e2effb6:docs/MORI-EP-GUIDE.md`; `python/mori/ops/dispatch_combine.py`.
- Quantized A2A 2.56× BW (28672→11200 B/tok, shipped in SGLang on MI355X), fp8_blockwise combine ~736 vs
  BF16 ~907 µs, InterNodeV1LL 1.52×/1.82× ≤256 tok/rank, TBO +25%, SDMA zero-compute-overhead — all
  independently confirmed against the source post: https://www.lmsys.org/blog/2026-05-28-mori/ (2026-05-28)
- up-to-2× naive expert imbalance → within 5% with EPLB-style frequency grouping, MoRI-EP DBO
  (Dual/Two-Batch Overlap) integration, HIP-graph static-ize of EP tensors — confirmed against the source
  report: https://moreh.io/technical-report/21k-output-tokens-per-second-deepseek-inference-on-amd-instinct-mi300x-gpus-with-expert-parallelism-251113/ (2025-11-13)
