---
title: moe_dispatch_combine on HIP/C++ — SOTA card
kind: sota_card
operator: moe_dispatch_combine
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode]
status: sota
updated: 2026-07-29
sources:
  - https://gau-nernst.github.io/amd-a2a/
  - ROCm/mori@35e2effb6:src/ops/dispatch_combine/dispatch_combine.cpp
  - ROCm/aiter@a177781d4:aiter/ops/moe_sorting.py
---

# moe_dispatch_combine × HIP/C++

## TL;DR
> HIP/C++ is where dispatch/combine is actually written — MoRI-EP's kernels
> (`src/ops/dispatch_combine/{dispatch_combine.cpp,internode_v1.cpp,low_latency_async.cpp}`), vLLM/SGLang's
> local permute/scatter, and the reference single-kernel a2a studies are all HIP. Reach for HIP to author a
> custom intra-node dispatch (P2P symmetric memory + xGMI), to fuse the grouped GEMM into the a2a sweep,
> or to own the exact grid/buffer behavior. This is the Tier-C seam and the source of the 292 µs result.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Reference single-kernel a2a | gau-nernst (GPU MODE AMD Distributed Challenge) | gfx942, bf16 | **292 µs** (from 93,540 µs naive ref) @ MI300X, E=256/topk=8/hidden 7168/world=8, 2025 (verified against the source worklog) | study/blueprint for a custom intra-node a2a |
| MoRI-EP HIP kernels | `ROCm/mori`, `src/ops/dispatch_combine/` | gfx942/950, fp8/fp4/bf16 | 307/330 GB/s (see [mori.md](mori.md)) | production (consume, don't re-author) |
| aiter `moe_sorting` (local permute) | `aiter/ops/moe_sorting.py`, `csrc/kernels/moe_align_block_size_kernels.cu` | gfx942/950 | — | single-GPU token permute/scatter |

Recommend: consume MoRI-EP for production EP; author HIP only for a custom fusion or an unsupported topology.

## Config space / knobs
- **`grid_size = 304`** (the MI300X CU count) — 304 gave a **~3× speedup specifically on combine-send**
  (dispatch tuning showed no comparable win); 256 did not reproduce it. Size to full CU count, not a round
  power of two.
- **Pre-allocate registered buffers**; **hoist `torch.empty`/memset** out of the kernel (malloc shows in
  traces as a real cost — the biggest non-obvious win once profiling artifacts are ruled out).
- **P2P symmetric memory**: map each peer's buffer, use direct `load/store` over xGMI instead of staged
  copies, synchronized with acquire/release signal flags (`__hip_atomic_store`/`__hip_atomic_load` with
  `__ATOMIC_RELEASE`/`__ATOMIC_ACQUIRE` — the "undocumented" intrinsics also used by rocSHMEM; HIP lacks
  CUDA's masked `__syncwarp()`, use `__builtin_amdgcn_wave_barrier()` instead). Combine with the
  grouped-GEMM sweep to remove a kernel boundary.
- **Varlen work distribution**: don't assign threadblocks to source ranks round-robin when per-rank token
  counts are skewed — flatten the work across all threadblocks (walk the cumulative token-count sum) so no
  threadblock idles while another is overloaded.
- **Block/warp**: wave64; `__launch_bounds__` to cap VGPRs; one block per CU (304) for the all-to-all.
- fp8/fp4 dispatch: quantize+scale in the send path; bf16 (or blockwise-quantized) combine.
- Profiling caveat: PyTorch Profiler can show spurious multi-GPU slowdowns (e.g. a spin-lock loop appearing
  anomalously slow only under the profiler) — prefer manual HIP/CUDA-event timing for multi-GPU kernels,
  and intra-kernel timestamp buffers (`s_memrealtime`-style) for line-level bottleneck hunting.

## Numerics / parity
fp8/fp4 dispatch quant gate; bf16 (or blockwise-quantized) combine; unbiased weight; mask static-pad
tokens. Round-trip identity test + greedy parity. See [numerics.md](../numerics.md).

## Integration (rebind seam)
- aiter: kernels JIT/AOT into `aiter/jit/`; `aiter/ops/moe_sorting.py` for local permute.
- MoRI: `python/mori/ops/dispatch_combine.py` (Python API) over HIP kernels in `src/ops/dispatch_combine/`.
- vLLM/SGLang: `sgl-kernel`/`csrc` permute ops registered as torch ops; rebuild after editing `.cu`/`.cpp`.

## Pitfalls & anti-patterns
- `grid_size != 304` (e.g. a power of two) → leaves combine bandwidth on the table.
- `hipMalloc`/memset inside the steady-state kernel → caching-allocator cost in the trace.
- Round-robin threadblock-to-rank assignment under skewed per-rank token counts → uneven stalling; flatten
  the work distribution instead.
- Forgetting xGMI is a **fully-connected mesh** (no switch): all 8 GPUs needed for full bandwidth.

## How to verify
rocprofv3: grid=304, no malloc steady state, overlap with GEMM; bandwidth vs the mori table; round-trip +
greedy parity. For custom kernels: intra-kernel timestamp profiling (Chrome/Perfetto trace) to find uneven
per-threadblock stalls that kernel-level profiling can't see.

## Alternatives / cross-links
[mori.md](mori.md) · [../aiter.md](../aiter.md) · [triton.md](triton.md) · [overview.md](../overview.md).

## Sources
- 292 µs / grid_size=304 (combine-send ~3×) / malloc+memset hoist / P2P + varlen ladder, intra-kernel
  profiling technique — full worklog verified: https://gau-nernst.github.io/amd-a2a/
- MoRI-EP HIP kernels: `ROCm/mori@35e2effb6:src/ops/dispatch_combine/{dispatch_combine.cpp,internode_v1.cpp,low_latency_async.cpp}`.
- aiter local permute: `ROCm/aiter@a177781d4:aiter/ops/moe_sorting.py`, `csrc/kernels/moe_align_block_size_kernels.cu`.
