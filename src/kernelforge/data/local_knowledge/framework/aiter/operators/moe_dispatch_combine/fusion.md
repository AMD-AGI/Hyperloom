---
title: moe_dispatch_combine — fusion
kind: technique
operator: moe_dispatch_combine
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3, fp4_e2m1]
regimes: [prefill, decode]
updated: 2026-07-30
sources:
  - ROCm/mori@35e2effb6:docs/MORI-EP-GUIDE.md
  - ROCm/mori@35e2effb6:python/mori/ops/dispatch_combine_v2/__init__.py
  - ROCm/mori@35e2effb6:python/mori/ops/__init__.py
  - ROCm/aiter@a177781d4:aiter/dist/device_communicators/all2all.py
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
  - https://arxiv.org/abs/2506.04667
---

# moe_dispatch_combine — fusion

The MoE EP pipeline is `route → dispatch → grouped GEMM (gate/up, down) → combine`. Fusion here means
collapsing those boundaries and **overlapping comm with compute**.

## Fusions that exist on AMD today
| fusion | what it merges | where | payoff |
|---|---|---|---|
| **prob-mult → combine** | routing-weight multiply done during the combine gather | MoRI-EP `combine(..., weights=)` | removes a separate weighted-combine kernel |
| **shared-expert → dispatch** | shared experts injected as synthetic routed experts (top-k slots via `grouped_topk`) → one dispatch for shared+routed | Wide-EP / shared_expert_fusion | one fused dispatch, no separate shared Linear+add |
| **fp8/fp4 quantize → dispatch** | token quant + scale computed inside the dispatch send | MoRI-EP `quant_type=fp8_direct_cast` (blockwise fp8/fp4 codecs live on the **combine** side) | halves-to-quarters wire bytes, no separate quant pass |
| **split-phase comm ↔ grouped GEMM overlap** | `dispatch_send/recv`, `combine_send/recv` interleaved with the grouped GEMM mainloop, or full **Two-Batch Overlap** (two microbatches on separate comm/compute streams) | MoRI-EP split API + HIP graph; SGLang TBO on MoRI (`MORI_ENABLE_SDMA=true` for zero-compute-overhead transfers on System DMA engines) | hides comm under compute |
| **dispatch + grouped-GEMM (partial)** | the a2a reference fused grouped-GEMM into the same launch sweep | gau-nernst study (345→292 µs) | one kernel boundary removed |

## The AITER FusedMoE seam (DeepEP-compatible 3D layout)
MoRI-EP's native layout is **2D** `[num_tokens, hidden]`; AITER's grouped GEMM wants the **3D**
`packed_recv_x / packed_recv_count / packed_recv_src_info / packed_recv_layout_range`. The bridge is
`dispatch_standard_moe()` / `combine_standard_moe()` / `convert_dispatch_output()` — **requires building
mori with `ENABLE_STANDARD_MOE_ADAPT=ON`** (CMake default OFF), else those methods raise `RuntimeError`.

⚠ **Docs vs. current source (corrected here):** MoRI's own `MORI-EP-GUIDE.md` still cites the AITER
integration point as `aiter/moe_op/mori_all2all.py`, and that is what earlier knowledge bases (including
this one, before this page existed) inherited. On the **current** aiter tree that path — and the whole
`aiter/moe_op/` directory — no longer exists. `MoriAll2AllManager` now lives at
`aiter/dist/device_communicators/all2all.py` (subclassing `All2AllManagerBase`), wired into
`AiterCommunicator.dispatch()`/`.combine()` in `communicator_cuda.py`. This is how dispatch hands off to
fused_moe_grouped_gemm without re-materializing tokens.

## An emerging alternative: FlyDSL-authored dispatch/combine
Both sides of the stack are growing a second, FlyDSL-based EP path that isn't in older write-ups:
- **aiter**: `FlyDSLAll2AllManager` (same file as `MoriAll2AllManager`) still uses **mori's shmem heap**
  for P2P buffer allocation, but the dispatch/combine *kernels* are FlyDSL-generated instead of mori's HIP
  kernels; it currently supports **intranode only** (raises `NotImplementedError` for inter-node — use the
  mori backend there).
- **mori**: an in-tree `dispatch_combine_v2` (`python/mori/ops/dispatch_combine_v2/`) reimplements
  dispatch/combine on **mori-cco LSA** + FlyDSL device kernels ("mori-parity reimplementation"). Its own
  README banner still says *"not a mori API (yet)... no package export"*, but that's **stale at this
  pin**: `dispatch_combine_v2/__init__.py` already has a real `__all__` export with relative imports
  throughout, lazily re-exported from `mori.ops` (`mori.ops.dispatch_combine_v2` resolves via
  `__getattr__`/`__dir__`) — lazy only because `flydsl` is an optional install extra. What's still true:
  it's undocumented in `MORI-EP-GUIDE.md`, unused by aiter's integration seam, and only exercised by its
  own test/bench suite.

Treat both as **emerging, not SOTA** until they're documented in the main guide and wired into aiter's
integration seam; the shipping production path is still MoRI-EP's native HIP kernels via
`MoriAll2AllManager` (see [aiter.md](aiter.md) / [backends/mori.md](backends/mori.md)). Also new since the
mori-guide's last-cited numbers: `quant_type` grew from `{"none", "fp8_direct_cast"}` to include
**blockwise** `fp8_blockwise` and (gfx950-only)
`fp4_blockwise` combine codecs, and SGLang's MoRI integration now **auto-detects** dispatch/combine dtype
from the model's weight format (`SGLANG_MORI_DISPATCH_DTYPE`/`SGLANG_MORI_COMBINE_DTYPE` override it) —
this LMSYS SGLang MI355X TCO writeup shows the FP4-dispatch/FP8-combine 2.56× BW win is now
production-shipped, not just a benchmark result.

## The north-star (not yet shipped on AMD)
**Fold the combine reduction into the down-proj GEMM epilogue** so expert outputs are never
re-materialized to global memory before the gather (cf. FlashDMoE/FlashMoE single-kernel design, NVIDIA
CUTLASS+NVSHMEM, arXiv 2506.04667). MoRI-EP's **zero-copy registered buffers**
(`get_registered_combine_input_buffer`) move toward this, but a single fused combine+GEMM kernel does not
exist on AMD yet — flag it as a gap.

## Decode-path discipline
Capture dispatch→GEMM→combine into **one HIP graph** (pad token counts static), hoist all `torch.empty`
out, use the **low-latency** kernel mode (`InterNodeV1LL`/`AsyncLL`). The decode a2a is the latency tail of
the MoE layer.

## Cross-links
moe_routing_topk (feeds dispatch) · fused_moe_grouped_gemm (consumes dispatch) · shared_expert_fusion ·
[aiter.md](aiter.md) · [backends/mori.md](backends/mori.md).

## Sources
- prob-mult-in-combine, 3D adapter, `ENABLE_STANDARD_MOE_ADAPT`, zero-copy buffers, integration table
  (checked against current aiter — see correction above): `ROCm/mori@35e2effb6:docs/MORI-EP-GUIDE.md`.
- on-box: `ROCm/aiter@a177781d4:aiter/dist/device_communicators/all2all.py` (`MoriAll2AllManager`,
  `FlyDSLAll2AllManager`), `aiter/dist/device_communicators/communicator_cuda.py` (dispatch/combine call
  sites); `ROCm/mori@35e2effb6:python/mori/ops/dispatch_combine_v2/{__init__.py,README.md}` +
  `python/mori/ops/__init__.py` (README's "no package export" banner is stale at this pin — the export is
  real; see backends/mori.md).
- shared-expert fusion (Wide-EP): https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
- FP4/FP8 quantized all-to-all shipped in SGLang, auto-dtype-select, TBO+SDMA: https://www.lmsys.org/blog/2026-05-28-mori/
- single-kernel combine+GEMM north-star: https://arxiv.org/abs/2506.04667
