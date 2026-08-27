---
title: moe_routing_topk — tuning
kind: technique
operator: moe_routing_topk
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang
  - https://www.amd.com/en/blogs/2025/revolutionizing-mixture-of-experts-performance-10.html
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/moe_align_block_size_kernels.cu
---

# moe_routing_topk — tuning

## What you actually tune
The router is **memory-bound** and **latency-bound**; you are not chasing FLOPs, you are minimizing
launch count, XCD-crossing synchronization, and LDS/bank conflicts. Two kernels matter: the **fused gate
+ top-k** (softmax/sigmoid + bias + grouped select) and **align&sort** (`moe_align_block_size`).

## The MI300X multi-die (XCD) rule — the dominant lever
MI300X is **8 XCDs × 38 CU = 304 CU**; **synchronization across XCDs is expensive** and L2 is per-XCD.
The SGLang align&sort rewrite is the canonical case study:
- **Grid sizing rule**: if needed blocks < num_XCD (8), pin to the *slowest* die (XCD7) to avoid
  cross-die traffic; if > 8, make grid a **multiple of 8** so dies stay balanced. Naive grid (e.g. 39
  active CUs spanning 2 dies) left perf on the table; the tuned multi-block version used **66 active CUs**
  but paid an unavoidable Die-Die exchange in the block-wise reduction.
- **Counterintuitive**: for this memory-bound op MI100 (single die) beat MI300X in the author's test —
  inter-die comm cost outweighed MI300X's raw bandwidth. Lesson: **don't over-parallelize a small
  memory-bound router across all 8 XCDs.**
- **Avoid `hipCooperativeLaunch`** unless required: cooperative grids increase L2 pressure (texture
  addresser stall) when Die-Die data exchange grows.

Result: **7× on MI300X / 10× on MI100** for align&sort after the multi-block rewrite (PR #3613).

## Resource budget (align&sort, tuned config)
- **52 VGPRs/wave, no spills**; **48 SGPRs**; **~5 kB LDS/CU** with only **6.8% bank-conflict rate**.
- Threads-block size tuned to the best fit (initial size adjusted down to avoid spills).
- Supports **arbitrary expert count up to MAX_EXPERT_NUMBER=256** with concurrent multi-block execution.

## Gate + top-k knobs
- **Kernel family** (aiter auto-dispatch by arch): gfx9 → **DPP** path (`__builtin_amdgcn_*` cross-lane,
  no LDS round-trip for the reduction); else CK fallback. DPP is **1.42–1.94× (avg 1.66×)** faster than
  CK and supports E=256 + fp32 (CK capped at E=192, fp16/bf16 only). Prefer DPP on MI300X/MI350X.
- **`THREAD_PER_GRP = 64 / num_expert_group`** drives the cross-lane reduction. The kernel implements
  reduction only for `THREAD_PER_GRP ∈ {2,4,8}` (i.e. `num_expert_group ∈ {32,16,8}` for wave64). Other
  group counts (e.g. `num_expert_group=4` → THREAD_PER_GRP=16, or values that make it 0) get **incomplete
  reductions / wrong scores** (aiter #2153) — verify your `num_expert_group` lands on a supported value.
- **`need_renorm` / `routed_scaling_factor`**: fold into the same kernel (epilogue), don't add a pass.
- **`block_size` (BLOCK_M)** in align&sort: must match the grouped-GEMM tile (commonly 32/64/128). A
  mismatch re-pads and wastes the GEMM's M dimension.

## Decode vs prefill
- **Decode (T small)**: launch overhead dominates → keep the router to **one or two kernel launches**,
  HIP-graph-capture it, pin grid small (XCD rule above). Hoist any `torch.empty`/memset out of the hot
  loop (caching-allocator malloc shows up in traces — same finding as the a2a kernel,
  moe_dispatch_combine).
- **Prefill (T large)**: align&sort grid scales to a multiple of 8 XCDs; the gate kernel is bandwidth-bound
  on `[T,E]` reads — coalesce the logit load.

## How to verify a tuning win
- rocprof-compute the router kernels: check **active CU count**, **LDS bank-conflict %**, **VGPR spills=0**,
  and that grid is a multiple of 8 (or pinned small). Confirm DPP path fired (not CK) for E=256.
- Isolated timing of `moe_align_block_size` + the gate kernel at your (T,E,k,BLOCK_M); then e2e tok/s.
- Parity gate (expert ids+weights) after any kernel/arch swap — argmax ties flip benignly, but a wrong
  `num_expert_group` or sigmoid/softmax mismatch is a **real** regression (see [numerics.md](numerics.md)).

## Sources
- align&sort multi-die design (grid rule, 52 VGPR / 5 kB LDS / 6.8% bank conflict, 66 CUs, 7× MI300X):
  https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang ;
  https://www.amd.com/en/blogs/2025/revolutionizing-mixture-of-experts-performance-10.html
- DPP vs CK dispatch + 1.66× + E=256/fp32: https://github.com/ROCm/aiter/pull/1909
- THREAD_PER_GRP reduction limitation: https://github.com/ROCm/aiter/issues/2153
- on-box kernel: `ROCm/aiter@a6bb49937:csrc/kernels/moe_align_block_size_kernels.cu`, `topk_softmax_kernels_group.cu`.
