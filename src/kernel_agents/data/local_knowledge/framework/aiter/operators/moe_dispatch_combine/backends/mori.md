---
title: moe_dispatch_combine on MoRI-EP — SOTA card
kind: sota_card
operator: moe_dispatch_combine
backend: mori
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3, fp4_e2m1]
regimes: [prefill, decode]
status: sota
updated: 2026-07-30
sources:
  - ROCm/mori@35e2effb6:docs/MORI-EP-GUIDE.md
  - ROCm/mori@35e2effb6:python/mori/ops/dispatch_combine.py
  - ROCm/mori@35e2effb6:python/mori/ops/dispatch_combine_v2/README.md
  - ROCm/mori@35e2effb6:python/mori/ops/dispatch_combine_v2/__init__.py
  - ROCm/mori@35e2effb6:python/mori/ops/__init__.py
  - ROCm/aiter@a177781d4:aiter/dist/device_communicators/all2all.py
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
  - https://gau-nernst.github.io/amd-a2a/
  - https://www.lmsys.org/blog/2026-05-28-mori/
---

# moe_dispatch_combine × MoRI-EP

> **mori now has its own first-class KB home**: [`framework/mori/`](../../../../mori/INDEX.md) covers
> mori's launch-config tuning control plane (MANUAL/AUTO mode, the per-shape JSON tuning-DB) and — new —
> actual **measured MI300X numbers** from KernelForge forge-loop campaigns (this chip has no official
> mori-tuner JSON yet). This card remains the right place for "how does aiter call mori"; for "how do I
> tune mori's own launch params for a shape" or "what did we measure," go to
> [`framework/mori/operators/ep_dispatch_combine/tuning.md`](../../../../mori/operators/ep_dispatch_combine/tuning.md).

## TL;DR
> **MoRI-EP is the SOTA EP dispatch/combine on Instinct** — AMD's native, HIP-graph-capturable, GPU-initiated
> all-to-all, co-designed with AITER FusedMoE and used in AMD's own Wide-EP DeepSeek deployments. Choose it for
> production MoE EP on MI300X/MI355X. It exploits the **fully-connected xGMI mesh** (every GPU pair has a direct
> P2P link) intra-node and RDMA inter-node, with six kernel types (`IntraNode`/`IntraNodeLL`, `InterNode`,
> `InterNodeV1`, `InterNodeV1LL`, `AsyncLL`) for different topology/latency regimes. Its **quantized
> all-to-all** (FP4 dispatch + FP8-blockwise combine) gives a **2.56× round-trip bandwidth reduction**
> (28,672 → 11,200 B/token) and is now **shipped in SGLang** (not just a benchmark), where
> MI355X+MoRI SGLang beats B200 SGLang by **1.25× tok/s/GPU** at iso-latency (all confirmed against the
> source writeup). DeepEP-on-ROCm / UCCL-EP are the portable alternatives.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| MoRI-EP `dispatch`/`combine` | `ROCm/mori` (`python/mori/ops/dispatch_combine.py`) | gfx942/950, fp8/fp4 dispatch + bf16/fp8-blockwise combine | **307 GB/s dispatch / 330 GB/s combine** (EP8/IntraNode, MI300X); **208/161 GB/s XGMI + 63/49 GB/s RDMA** (EP16-V1, InterNodeV1); **103/91 + 57/50 GB/s** (EP32-V1-LL) — 4096 tok / hidden 7168 / top-8, fp8 dispatch + bf16 combine (vendor, mori guide reference table) | production intra- & inter-node EP |
| MoRI-EP in Wide-EP (32× MI300X) | Wide-EP blog | gfx942 | **32.3k in / 12.4k out tok/s per node**, ROCm 6.3.1, 2025-11 (AMD-reported) | large distributed DeepSeek |
| MoRI quantized all-to-all (FP4 dispatch + FP8-blockwise combine), shipped in SGLang | LMSYS/AMD MI355X TCO blog | gfx950 | **2.56× round-trip BW reduction** (28,672 → 11,200 B/token) for `amd/DeepSeek-R1-0528-MXFP4-v2`; MoRI-EP `fp8_blockwise` combine (EP8, BF16, 4096 tok, hidden 7168, scale_dim=56) **~736 µs** vs BF16-no-quant reference **~907 µs**; adaptive `InterNodeV1LL` **1.52× dispatch / 1.82× combine** at ≤256 tok/rank (LMSYS, 2026-05-28, confirmed against source) | production quantized EP at scale |
| MoRI on MI355X (SGLang), full-stack TCO | LMSYS/AMD TCO blog | gfx950 | MI355X+MoRI SGLang **1.25× tok/s/GPU vs B200 SGLang** at iso-latency (2,436 vs 1,945 tok/s/GPU); **5% lower cost** than B200 TRT-LLM at 129 tok/s/user (confirmed) | MI355X EP serving, cost-competitive vs NVIDIA |
| hand-rolled a2a (reference point) | gau-nernst blog | gfx942 | **292 µs** dispatch+combine (from a 93,540 µs naive baseline), grid=304 (1 block/CU), GPU MODE AMD Distributed Challenge, 2025 (community, confirmed against source worklog) | shows the achievable floor / xGMI P2P model |

## Config space / knobs (on-box `EpDispatchCombineConfig`, mori @ 35e2effb6)
| field | default | note |
|---|---|---|
| `kernel_type` | `IntraNode` | `IntraNode`/`IntraNodeLL` (xGMI), `InterNode` (baseline/debug), `InterNodeV1` (throughput), `InterNodeV1LL` (low latency), `AsyncLL` (latency; only kernel with split `dispatch_recv`) |
| `block_num` | **80** | main-kernel GPU blocks |
| `warp_num_per_block` | **8** | warps/block |
| `gpu_per_node` | **8** | affects all kernel types (xGMI fan-out) |
| `rdma_block_num` | 0 | inter-node RDMA blocks |
| `num_qp_per_pe` | 1 | RDMA queue pairs per PE |
| `max_num_inp_token_per_rank` | *(required)* | per-rank input cap (memory) |
| `num_experts_per_rank` | *(required)* | local experts hosted on this rank |
| `num_experts_per_token` | *(required)* | top-k |
| `max_total_recv_tokens` | 0 (uncapped) | derives the max receivable tokens; exceeding it **asserts** |
| `use_external_inp_buf` | True | zero-copy when False |
| `quant_type` | `"none"` | `"fp8_direct_cast"`, plus newer **combine-only** blockwise codecs `"fp8_blockwise"` and (gfx950-only) `"fp4_blockwise"` |
| `data_type` | *(required)* | **deprecated for kernel launch** — dtype is now inferred from the runtime input tensor; kept only for test/example back-compat |

- **Auto kernel select**: `MORI_EP_LAUNCH_CONFIG_MODE=AUTO` switches launch params by a per-arch/per-shape
  tuned JSON config (`python/mori/ops/tuning_configs/`) when one exists, else a hard-coded fallback
  (`InterNodeV1`/`InterNodeV1LL` → 96/64/8; others → 128/0/16).
- **Layouts**: native 2D `[T,H]`; **DeepEP-compatible 3D** via `dispatch_standard_moe`/`combine_standard_moe`
  (needs build flag `ENABLE_STANDARD_MOE_ADAPT=ON`, CMake default OFF).
- **Split phases**: `dispatch_send` / `dispatch_recv` (AsyncLL only), `combine_send/recv` for overlap.
- **Arch/JIT**: `MORI_GPU_ARCHS=gfx942/gfx950`, `MORI_PRECOMPILE=1 python -c "import mori"` (avoids
  first-iteration JIT cost, ~22 s one-time per the JIT-architecture doc).
- **Emerging, undocumented (but importable)**: `python/mori/ops/dispatch_combine_v2/` reimplements this op
  on mori-cco LSA + FlyDSL kernels ("mori-parity reimplementation"). Its own README banner says *"not a
  mori API (yet)... no package export"* — that's **stale at this pin**: `dispatch_combine_v2/__init__.py`
  already has a real `__all__` export (`EpDispatchCombineConfig`/`EpDispatchCombineOp`/
  `EpDispatchRoutingHandle`) with relative imports throughout (`dispatch_combine_op.py` does
  `from .intranode_kernels import ...`), and `mori/ops/__init__.py` lazily re-exports it
  (`mori.ops.dispatch_combine_v2` resolves via a module-level `__getattr__`/`__dir__`) — lazy only because
  `flydsl` is an optional install extra, not because the API is unstable. What's still actually true: it's
  absent from `MORI-EP-GUIDE.md`, not wired into aiter's `MoriAll2AllManager` seam, and only exercised by
  its own `tests/python/ops/dispatch_combine_v2/` suite — so treat it as a second, less-adopted
  implementation, not (yet) the one to build production code against.

### API shape (on-box)
```python
op.dispatch(input, weights, scales, indices, block_num=-1, rdma_block_num=-1,
            warp_per_block=-1, call_local_expert_count=False)   # -1 ⇒ tuned/default launch params
op.combine(input, weights, indices, block_num=-1, rdma_block_num=-1, warp_per_block=-1,
           use_external_inp_buf=-1, call_reset=False)
```
`-1` launch params fall back to the config default, or the `AUTO`-mode tuned/fallback params.
`dispatch()` returns `(dispatch_output, dispatch_weights, dispatch_scales, dispatch_indices,
recv_num_token)`; `combine()` returns `(combine_output, combine_weights)`. `combine`
`use_external_inp_buf=0` ⇒ zero-copy via the registered buffer.

## Numerics / parity
`fp8_direct_cast` dispatch (quant gate) + bf16 combine is the default recipe; `fp8_blockwise`/
`fp4_blockwise` are newer **combine**-side codecs (`fp4_blockwise` requires gfx950 — it raises at
construction on other archs). Combine multiplies the **unbiased** routing weight; static-pad tokens must be
**masked** from the reduction. EP keeps each expert whole → GEMM math identical to single-GPU (best fp8
accuracy vs TP, which adds a cross-rank down-proj reduce). Greedy/temp=0 parity vs torch MoE. See
[numerics.md](../numerics.md).

## Integration (rebind seam)
- AITER: `MoriAll2AllManager` in `aiter/dist/device_communicators/all2all.py` (**not**
  `aiter/moe_op/mori_all2all.py` — that path is stale; it's still what MoRI's own guide cites, but the
  `aiter/moe_op/` directory no longer exists on the current aiter tree).
- vLLM/SGLang: register MoRI as all2all backend. Init via
  `mori.shmem.shmem_torch_process_group_init("default")`; `reset()` between iterations (or
  `call_reset=True` on `combine`).
- SGLang auto-detects dispatch/combine dtype from the model's weight format
  (`SGLANG_MORI_DISPATCH_DTYPE`/`SGLANG_MORI_COMBINE_DTYPE` to override).
- ⚠ `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` is **incompatible** with MoRI — shared-expert fusion is done
  MoRI-side in the Wide-EP path.

## Pitfalls & anti-patterns
- Forgetting `ENABLE_STANDARD_MOE_ADAPT=ON` → no 3D AITER-compatible API (`RuntimeError`).
- JIT first-iteration cost (`~/.mori/jit/`); `MORI_PRECOMPILE=1` + warm before timing.
- Dynamic EP shapes vs HIP-graph static requirement → pad/static-ize (and **mask** the pad in combine).
- Expert load imbalance up to ~2× with naive contiguous sharding → EPLB-style frequency grouping (see
  tuning.md) brings it to within ~5%.
- `max_total_recv_tokens` cap exceeded → kernel **asserts** (docstring warns); size it to worst-case.
- `fp4_blockwise` combine on non-gfx950 → raises `ValueError` at op construction, not silently wrong.
- Using `dispatch_recv`/`combine_recv` outside `AsyncLL` → only `AsyncLL` supports the split recv phase.
- Trusting `dispatch_combine_v2`'s own README banner ("no package export, test-only") at face value — it's
  stale at this pin; the package imports fine via `mori.ops.dispatch_combine_v2`. The real caveat is
  adoption, not importability: it's undocumented in the main guide and not the path aiter wires up.

## How to verify
`tests/python/ops/test_dispatch_combine_intranode.py` / other `test_dispatch_combine_*.py` (correctness),
`tests/python/ops/bench_dispatch_combine.py` (bandwidth vs the table above); rocprofv3 to confirm overlap
with the grouped GEMM and the correct kernel mode; round-trip + greedy parity.

## Worked example (EP8 DeepSeek-V3, single MI300X node)
EP=8, E=256 (32/rank), hidden 7168, top-8, 4096 tok/rank, fp8 dispatch.
1. `EpDispatchCombineConfig(rank=rank, world_size=8, hidden_dim=7168, scale_dim=0,
   scale_type_size=..., max_token_type_size=..., max_num_inp_token_per_rank=4096,
   num_experts_per_rank=32, num_experts_per_token=8, kernel_type=IntraNode, quant_type="fp8_direct_cast")`.
2. `MORI_PRECOMPILE=1` warmup (avoid JIT in the timed loop).
3. `op.dispatch(x, w, scales, indices)` (fp8) → per-rank tokens; run fused_moe_grouped_gemm;
   `op.combine(y, w, indices)` (bf16) → source ranks.
4. Verify: `bench_dispatch_combine.py` ≈ 307/330 GB/s; rocprof overlap with the GEMM; parity vs single-GPU.
Inter-node: switch to `InterNodeV1` (throughput) or `InterNodeV1LL`/`AsyncLL` (latency), set `rdma_block_num`.

## Alternatives / cross-links
moe_dispatch_combine · [hip.md](hip.md) · [../aiter.md](../aiter.md) (the EP seam) · [triton.md](triton.md) ·
fused_moe_grouped_gemm · [overview.md](../overview.md) · [numerics.md](../numerics.md) ·
[framework/mori/](../../../../mori/INDEX.md) (mori's own tuning control-plane + measured MI300X data,
mori-direct API surface, the experimental FlyDSL v2 reimplementation).

## Sources
- MoRI-EP guide + bandwidth/latency table + kernel types + config fields: `ROCm/mori@35e2effb6:docs/MORI-EP-GUIDE.md`.
- on-box: `ROCm/mori@35e2effb6:python/mori/ops/dispatch_combine.py` (`EpDispatchCombineConfig` defaults incl.
  `max_total_recv_tokens`, `dispatch`/`combine`/`dispatch_send`/`dispatch_recv`, `dispatch_standard_moe`/
  `combine_standard_moe`, `EpDispatchCombineQuantType`); `python/mori/ops/dispatch_combine_v2/README.md`
  (design/perf notes — its "no package export, test-only" banner is stale at this pin);
  `python/mori/ops/dispatch_combine_v2/__init__.py` + `dispatch_combine_op.py` + `python/mori/ops/__init__.py`
  (confirm the package export is real and lazily loaded, not missing).
- aiter integration seam (path correction): `ROCm/aiter@a177781d4:aiter/dist/device_communicators/all2all.py`.
- Wide-EP 32-GPU numbers: https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
- MI355X TCO / SGLang+MoRI (quantized A2A 2.56× BW; fp8_blockwise combine ~736 vs BF16 ~907 µs;
  InterNodeV1LL 1.52×/1.82× ≤256 tok/rank; 1.25× tok/s/GPU vs B200; auto-dtype-select): https://www.lmsys.org/blog/2026-05-28-mori/ (2026-05-28)
- a2a reference point (292 µs from 93,540 µs, grid=304, xGMI P2P) — full worklog verified: https://gau-nernst.github.io/amd-a2a/
