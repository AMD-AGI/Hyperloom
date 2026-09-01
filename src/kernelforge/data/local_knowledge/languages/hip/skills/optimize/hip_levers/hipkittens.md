---
title: HipKittens — C++ tile primitives, scheduling & register pinning for CDNA
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3, fp8_e5m2, fp6]
regimes: [both]
status: experimental
updated: 2026-07-08
sources:
  - https://arxiv.org/abs/2511.08083
  - https://arxiv.org/html/2511.08083v1
  - https://hazyresearch.stanford.edu/blog/2025-11-09-hk
  - https://github.com/HazyResearch/HipKittens
---

# HipKittens (HK) — tile-primitive HIP

## TL;DR
HipKittens is a **minimal C++ embedded tile-primitive library** for writing fast AMD Matrix-Core kernels
without dropping to raw assembly — the AMD entry in Stanford HazyResearch's "Kittens" family. Its thesis:
the **tile abstraction is portable** but the **backend (swizzling, register scheduling, wave scheduling)
must be AMD-specific**. On MI355X (CDNA4) HK reports SOTA or near-SOTA across GEMM, attention fwd/bwd, and
memory-bound kernels — beating AMD's own hand-tuned **AITER assembly** and hipBLASLt on several shapes —
while keeping kernels short (attention fwd ~500 LoC, GEMM hot loop <100 LoC). HK is now an **official
AITER backend** (ROCm/aiter PR #2039). Treat it as a **perf reference / source of ideas**: pin a commit,
re-measure per shape; APIs are unstable (research artifact, arXiv 2511.08083, Nov 2025).

## Concepts
- **Tile = the unit of data and compute.** Register or shared (LDS) tiles parametrized by `dtype`
  (FP32, BF16, FP16, FP8, FP6), `rows`, `cols` (multiples of the matrix-core shape), `layout`
  (row/col major). Bulk ops (`mma`, `exp`, `add`, load/store) are PyTorch/NumPy-flavored and wrap raw
  CDNA asm/HIP with no overhead.
- **Interface portable, implementation not.** Tile types/ops translate from NVIDIA ThunderKittens to
  AMD; what changes is memory access (swizzling) and register/wave scheduling. AMD's matrix layouts are
  **not compositional** (NVIDIA derives everything from a 16×16 core matrix), causing an "explosion of
  layouts."
- **AMD lacks NVIDIA's wave-specialization enablers.** No TMA, no `wgmma`/`tcgen05`, no `mbarrier` HW
  sync, **no register reallocation**. HK compensates with a 2× larger register file, small MFMA shapes
  (`16×16×32`) for deep pipelines, and shared-memory atomics in place of mbarriers.

## The levers (when authoring with HK)
- **Register tiles default to the smallest MFMA shape `16×16×32`** for maximal scheduling control;
  parameterize by MFMA shape for edge cases.
- **Pinned register tiles** expose the same interface as compiler-managed tiles but let the developer
  own register placement — bypassing HIPCC so **AGPRs can be fed directly to matrix instructions**
  (HIPCC otherwise inserts redundant `v_accvgpr_read` AGPR→VGPR moves before every MFMA). This is the
  key to HK's SOTA backward attention.
- **Wave scheduling pattern** (replaces NVIDIA producer/consumer wave specialization, which
  underperforms on CDNA — static register split starves output):

| pattern | layout | idea | code size | example (FP8 GEMM) |
|---|---|---|---|---|
| **8-wave ping-pong** (default) | 8 waves/block, 2/SIMD | two waves/SIMD alternate memory-cluster vs compute-cluster, swap via a conditional barrier; long identical runs over **large** tiles | compact (48 LoC) | 3222 TFLOPS |
| **4-wave interleave** | 1 wave/SIMD | finely staggered compute+memory per wave; needs **small** base tiles | large (183 LoC) | 3327 TFLOPS (+3%, ~4× code) |

  8-wave ping-pong is the default — already SOTA for GEMM/attention-fwd. Interleave buys ~22% on MHA
  backward at ~3× the code.
- **HBM-address swizzling** for conflict-free async HBM→LDS loads (AMD swizzles the *global* address,
  not the shared-memory address). LDS access phases are non-sequential and per-instruction
  (`ds_read_b128` = 4 phases/64 banks; `ds_read_b96` = 8 phases/32 banks); HK uses a **solver** to find
  conflict-free swizzles rather than hand-deriving them.
- **XCD-aware grid swizzle** (chiplet scheduling) for L2/LLC reuse on the 8-XCD MI355X: group `C`
  consecutive block IDs onto the same XCD, then traverse in vertical windows of height `W`. L2 tiles of
  `8×4` / `4×8` best on MI355X; up to ~15–19% over naïve row-major (L2 hit 55%→75%).

## Measured perf (author-reported, MI355X gfx950, arXiv 2511.08083v1, 2025-11)
All numbers are **author/vendor-reported**, single-source, MI355X-centric — treat as vendor-labeled
until re-measured on-box.
| workload | HK | best baseline | note |
|---|---|---|---|
| BF16 GEMM (8192³, 256² tile, 8-wave) | 1610 TFLOPS | hipBLASLt 1561 | matches/edges hipBLASLt |
| FP8 GEMM (8-wave ping-pong) | 3222 TFLOPS (48 LoC) | 4-wave 3327 (183 LoC) | interleave +3% at ~4× code |
| MHA non-causal bwd, seq 4096 | HK+pinned 1024 | AITER 1018 | pinned tiles reach AITER asm |
| GQA non-causal bwd | 4-wave 2.3× over baseline | AITER 272–384 / SDPA 259 | AITER GQA bwd weak |
| attention fwd (various) | beats AITER 1.0–2.1×, SDPA 1.3–4.5×, CK 1.0–1.4×, Triton 1.2–4.5× | — | ~FlashAttention-3 class |
| memory-bound (fused dropout-residual-LN, rotary) | beats AITER & torch.compile 1.1–2.2× | — | — |

## Cross-DSL takeaway (the durable finding)
The load-bearing claim is not HK's absolute TFLOPS but its measured indictment of competing AMD DSLs:
**AMD Triton** underperforms even a vanilla BF16 GEMM (HK 1.3–3.0× faster); **Mojo** MHA ~50% of peak
from bank conflicts; **TileLang** is CDNA3-only and only "competitive with PyTorch"; **AITER assembly**
is strong on fwd/GEMM but weak on GQA backward. This explains why the backend landscape ranks
aiter/hipBLASLt/CK/asm above Triton/Mojo/TileLang for production AMD kernels today.

## Pitfalls
- **Research artifact, not a maintained library.** Pin a commit; APIs unstable; no AMD support
  contract. For production prefer aiter/CK/hipBLASLt; use HK as a perf reference.
- **CDNA3/CDNA4 only** (gfx942/gfx950); use the CDNA3 branch for MI300X/MI325X. Benchmarks MI355X-centric.
- **Pinned register tiles are sharp** — you bypass the compiler's register allocator; mistakes are
  silent correctness or occupancy cliffs. Validate parity + ISA-check that pinned tiles actually feed
  AGPRs to MFMA (no spurious `v_accvgpr_read`).
- Reported wins are **per-shape**; do not assume a blanket speedup over AITER/hipBLASLt — re-measure.

## Terminology map (CUDA → HIP/HK)
warp→**wave** (32→64), SM→CU, SMEM→**LDS**, tensor core→**matrix core**, WGMMA/WMMA/TCGEN05→**MFMA**,
TMA→buffer-load-to-lds, CUDA/NVCC→HIP/HIPCC.

## Sources
- HipKittens paper (tiles/ops §3, wave scheduling/register pinning/swizzling/XCD §4, Tables 1–5): https://arxiv.org/html/2511.08083v1 ; abstract https://arxiv.org/abs/2511.08083
- Blog "AMD GPUs go brrr": https://hazyresearch.stanford.edu/blog/2025-11-09-hk
- Code: https://github.com/HazyResearch/HipKittens
- AGPR/`v_accvgpr_read` & LDS bank/phase facts also in [hip_builtins.md](hip_builtins.md) / [hip_lds_staging.md](hip_lds_staging.md).
