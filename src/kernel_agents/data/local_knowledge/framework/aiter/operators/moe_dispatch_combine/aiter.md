---
title: moe_dispatch_combine on aiter — SOTA card
kind: sota_card
operator: moe_dispatch_combine
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3, fp4_e2m1]
regimes: [prefill, decode]
status: sota
updated: 2026-07-29
sources:
  - ROCm/aiter@a177781d4:aiter/dist/device_communicators/all2all.py
  - ROCm/aiter@a177781d4:aiter/dist/device_communicators/communicator_cuda.py
  - ROCm/aiter@a177781d4:aiter/ops/moe_sorting.py
  - ROCm/aiter@a177781d4:csrc/kernels/moe_align_block_size_kernels.cu
  - ROCm/mori@35e2effb6:docs/MORI-EP-GUIDE.md
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
---

# moe_dispatch_combine × aiter

## TL;DR
> aiter owns the **single-GPU** side (local token permute/scatter via `moe_sorting` align&sort, fused into
> `fused_moe`) and the **EP seam** to MoRI-EP / FlyDSL. For *distributed* EP, aiter does **not** implement
> the all-to-all itself — it **delegates to MoRI-EP** (`MoriAll2AllManager`) or, on an intranode-only
> emerging path, to FlyDSL-generated kernels over mori's shmem heap (`FlyDSLAll2AllManager`) — and
> consumes the resulting packed layout. Use aiter for the local permute and as the FusedMoE driver; pair it
> with MoRI-EP for the cross-GPU comm. The `moe_sorting` align&sort is the same kernel SGLang rewrote for
> **7× on MI300X** via XCD-aware multi-block grids.
>
> ⚠ **Corrected vs. older write-ups (including MoRI's own docs):** the AITER↔MoRI integration point is
> **not** `aiter/moe_op/mori_all2all.py` — that path, and the `aiter/moe_op/` directory itself, do not
> exist in the current tree. `MoriAll2AllManager` now lives at
> `aiter/dist/device_communicators/all2all.py`.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `moe_sorting` (local permute + align&sort) | `aiter/ops/moe_sorting.py`, `csrc/kernels/moe_align_block_size_kernels.cu` | gfx942/950 | shares the **7× MI300X** align&sort win (SGLang multi-block, vendor) | single-GPU MoE (no EP) |
| `MoriAll2AllManager` (EP seam, mori HIP kernels) | `aiter/dist/device_communicators/all2all.py` | gfx942/950, fp8/fp4/bf16 | inherits MoRI-EP bandwidth (**307 GB/s dispatch / 330 GB/s combine**, EP8/MI300X, mori guide, vendor) | production EP MoE; wraps MoRI dispatch/combine for FusedMoE |
| `FlyDSLAll2AllManager` (EP seam, FlyDSL kernels + mori shmem) | `aiter/dist/device_communicators/all2all.py` | gfx942/950, intranode only | — (emerging) | FlyDSL-authored dispatch/combine, intranode; raises `NotImplementedError` for inter-node |
| `fused_moe` (drives permute→GEMM→combine) | `aiter/fused_moe.py` | gfx942/950 | up to **3×** vs unfused (AMD-reported) | the integrated single-GPU MoE pipeline |

### What `moe_sorting` does (the align&sort)
It takes `topk_ids [T, topk]` and produces `sorted_token_ids` / `expert_ids` / `num_tokens_post_pad`: tokens
are grouped by expert and each expert's run is **padded to a multiple of `block_size` (BLOCK_M)** so the
downstream grouped GEMM (fused_moe_grouped_gemm) is dense per block. This is a memory-bound op; the
SGLang/AMD rewrite (HF blog) made it multi-block and **XCD-aware** — for MI300X the rule is: if grid < 8 XCDs,
pin to the slowest die (XCD7) to avoid die-die sync; if grid ≥ 8, make grid a multiple of 8. That, plus
aggressive LDS (5 kB) / VGPR (52) use, took it from 39 → 66 active CUs and **7×** on MI300X.

### The EP seam (both managers, current source)
Both `MoriAll2AllManager` and `FlyDSLAll2AllManager` subclass `All2AllManagerBase` and are consumed
identically by `AiterCommunicator.dispatch()`/`.combine()` in `communicator_cuda.py`. `MoriAll2AllManager`
picks its mori kwargs by topology: single-node → `kernel_type=IntraNode, warp_num_per_block=16,
block_num=80, rdma_block_num=0`; multi-node → `kernel_type=InterNodeV1, warp_num_per_block=16,
block_num=32, rdma_block_num=16`. `FlyDSLAll2AllManager` still calls mori's
`shmem_torch_process_group_init` for the symmetric heap (FlyDSL only replaces the comm *kernels*, not the
memory model) and is intranode-only today.

### Measured perf (align&sort rewrite, vendor/community)
| op | config | metric | value @ hw / date | source |
|---|---|---|---|---|
| `moe_align_block_size` (multi-block, XCD-aware) | E=256, MAX_EXPERT=256 | speedup vs old single-block | **7× @ MI300X**, **10× @ MI100**, 3× A100/H200 (SGLang PR#3613), 2025 (vendor) | HF/AMD align&sort blog |
| active-CU utilization | same rewrite | active CUs | **39 → 66** (crosses 2 dies), LDS 5 kB / 52 VGPR | HF align&sort blog |
| `MoriAll2AllManager` (EP seam) | EP8, 4096 tok, H7168, top8 | dispatch / combine bw | 307 / 330 GB/s @ MI300X+CX7 (inherited from MoRI-EP), mori guide (vendor) | mori guide |

(Note the counter-intuitive finding: for this **memory-bound** op the multi-die MI300X can be *slower* than the
single-die MI100 because die-die exchange dominates — hence the XCD-aware grid rule below.)

## Config space / knobs
- `moe_sorting`: `block_size` (BLOCK_M) **must match** the grouped-GEMM tile; XCD-aware grid (multiple of 8 on
  MI300X, or pinned to XCD7 for small grids). On-box API: `moe_sorting_fwd(topk_ids, ..., sorted_token_ids,
  sorted_weights, sorted_expert_ids, num_valid_ids, ...)`.
- EP: `MoriAll2AllManager` forwards MoRI-EP config (kernel type, `block_num`, quant type). MoRI must be built
  with `ENABLE_STANDARD_MOE_ADAPT=ON` for the 3D layout aiter's grouped GEMM consumes
  (`dispatch_standard_moe`/`combine_standard_moe`).
- `doweight_stage1` / `MulRoutedWeight{0,1}`: where the routed weight multiply lands (stage-1 vs combine) —
  keep consistent with the GEMM epilogue or the weight is applied twice / not at all.

## Numerics / parity
fp8/fp4 dispatch quant gate (`fp8_direct_cast`/`fp8_blockwise`/`fp4_blockwise` in MoRI); bf16 combine by
default. The routed-weight multiply point is configurable — keep it bf16/fp32 and consistent with the
reference. **EP vs TP**: EP keeps each expert's weight whole on one rank → the dispatch/combine is the only
cross-GPU traffic and the GEMM math is identical to single-GPU (best for fp8 MoE accuracy). TP shards the
inter dim → an extra cross-rank reduce in the down-proj that must stay bf16/fp32. See [numerics.md](numerics.md).

## Integration (rebind seam)
- **Single-GPU**: `fused_moe` calls `moe_sorting` internally — no extra wiring.
- **EP**: aiter's `AiterCommunicator` selects `MoriAll2AllManager` (or, intranode, optionally
  `FlyDSLAll2AllManager`) and bridges dispatch/combine into FusedMoE.
- ⚠ `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` is **incompatible** with MoRI — do shared-expert fusion
  MoRI-side (as in the Wide-EP path).

## Pitfalls & anti-patterns
- Expecting aiter to do the distributed all-to-all itself — it **delegates** to MoRI-EP (or, intranode-only
  and still emerging, FlyDSL).
- `block_size` mismatch with the grouped GEMM → wasted M padding (every expert run rounded up).
- `moe_sorting` grid not XCD-aligned → idle dies; the 7× win is exactly this fix.
- Capture/live shape mismatch in the FusedMoE DB (`cu_num`, token-bucket) → tuned config misses (same failure
  mode as fused_moe_grouped_gemm).
- Forgetting `ENABLE_STANDARD_MOE_ADAPT=ON` on the MoRI build → no 3D AITER-compatible API.
- Trying `FlyDSLAll2AllManager` for inter-node EP → `NotImplementedError` (mori/HIP backend only, today).

## How to verify
`AITER_LOG_MORE=1` confirms `moe_sorting` + the FusedMoE kernels fire; rocprof confirms the `moe_sorting` grid
is a multiple of XCD=8. For EP: confirm the MoRI all2all banner, rocprofv3 overlap of dispatch/combine with
the grouped GEMM, and greedy parity on a round-trip.

## Worked example (EP8 DeepSeek-V3, MI300X node)
E=256, EP=8 (32 experts/GPU), hidden 7168, top-8, fp8 dispatch.
1. Router (`biased_grouped_topk`, moe_routing_topk) → `topk_ids/topk_weights`.
2. MoRI `dispatch` (fp8) sends tokens to expert-owning ranks → 3D `[E_local, capacity, H]` layout via
   `dispatch_standard_moe`.
3. Per-rank: `moe_sorting(block_size=BLOCK_M)` on the local tokens → `fused_moe` grouped GEMM.
4. MoRI `combine` (bf16 or `fp8_blockwise`) returns weighted outputs to source ranks.
5. Verify: `moe_sorting` grid = multiple of 8; rocprof shows dispatch/combine overlapping the GEMM; e2e parity
   vs single-GPU torch MoE. Anti-pattern: enabling `..._FUSION_SHARED_EXPERTS` with MoRI → breaks.

## Alternatives / cross-links
moe_dispatch_combine · [backends/mori.md](backends/mori.md) (the actual EP comm) · [backends/hip.md](backends/hip.md) ·
[backends/triton.md](backends/triton.md) ·
fused_moe_grouped_gemm · moe_routing_topk (shares `moe_sorting`) ·
[fmoe.md](../../skills/optimize/aiter_levers/fmoe.md) · [overview.md](overview.md) ·
[numerics.md](numerics.md).

## Sources
- on-box: `ROCm/aiter@a177781d4:aiter/ops/moe_sorting.py`, `csrc/kernels/moe_align_block_size_kernels.cu`,
  `aiter/fused_moe.py`, `aiter/dist/device_communicators/all2all.py` (`MoriAll2AllManager`,
  `FlyDSLAll2AllManager`), `aiter/dist/device_communicators/communicator_cuda.py` (dispatch/combine call
  sites — corrects the stale `aiter/moe_op/mori_all2all.py` path still cited in MoRI's own guide).
- MoRI-EP seam + 3D layout + bandwidth: `ROCm/mori@35e2effb6:docs/MORI-EP-GUIDE.md`.
- align&sort 7× MI300X / XCD grid rule: https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang ; https://www.amd.com/en/blogs/2025/revolutionizing-mixture-of-experts-performance-10.html
- Wide-EP (aiter + MoRI co-design): https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
