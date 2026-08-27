---
title: Triton on AMD Instinct (ROCm backend) — authoring overview
kind: language
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp8_e4m3, fp8_e5m2, int8]
regimes: [prefill, decode, training, both]
status: competitive
updated: 2026-07-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - https://arxiv.org/abs/2511.08083
---

# Triton on AMD — authoring overview

## TL;DR
Triton is the fastest way to **author and iterate** an MI300X/MI350X kernel, and it is the backend
PyTorch-Inductor `max-autotune` emits. The Python API is **identical** to NVIDIA; what changes is the
lowering (`TritonGPU → TritonAMDGPU → AMDGCN`) and the hardware mapping. The recurring AMD facts that
break CUDA habits: **wavefront = 64 lanes**, **LDS = 64 KB/CU** (CDNA3) / 160 KB (CDNA4),
**512 VGPR/EU** (16-granule), **FNUZ fp8** on CDNA3, and **`num_stages` semantics differ** (a single
GEMM pipelines best at 1–2, not 3–4). On a *plain* dense GEMM, AMD Triton still typically **loses to
tuned hipBLASLt/aiter** — the honest win is **fusion** (epilogue/attention) or skinny split-K decode.
Honest limit corroborated by HipKittens (arXiv 2511.08083): compiler backends including Triton
under-perform hand-tuned assembly/CK on CDNA3/CDNA4 GEMM and attention.

## Where it fits in the backend landscape
| Use Triton when | Reach for something else when |
|---|---|
| You need a **fused epilogue/attention** the library can't express | Plain dense GEMM → `hipblaslt`/`aiter`/`flydsl` |
| Rapid prototyping / shape exploration before committing to CK/asm | Last 10–20% of peak → `ck_tile`, `asm`, `flydsl`, HipKittens |
| `torch.compile` / Inductor codegen path | Block-scaled MXFP4/6 GEMM on CDNA4 → `ck`/`aiter` tuned |
| Skinny/decode GEMM with `SPLIT_K` to fill 304 CUs | Production serving hot path with a known tuned table |

## The two distributions
- **Upstream `triton-lang/triton`** — AMD backend lives in `third_party/amd/`; CDNA3/CDNA4 are
  first-class targets, built by default. Arch auto-detected from the active HIP device
  (`gfx942`=MI300X/MI300A/MI325X, `gfx950`=MI350X/MI355X).
- **`ROCm/triton`** (AMD staging fork) — carries AMD perf patches + tuning utilities (e.g. `occ.sh`)
  ahead of upstream; ROCm PyTorch wheels ship Triton built from here. Knob names/defaults drift
  between the two — always `grep third_party/amd/backend/compiler.py` for `HIPOptions` on your build.

## `third_party/amd/` layout (where the facts live)
```
third_party/amd/
├── backend/compiler.py   # HIPOptions (matrix_instr_nonkdim, kpack, waves_per_eu, num_stages,
│                         #   schedule_hint, supported_fp8_dtypes), pass pipeline
├── backend/driver.py     # HIP runtime, kernel launch
├── lib/                  # MLIR passes: TritonGPU→TritonAMDGPU→AMDGCN, MFMA dot conversion,
│                         #   stream-pipeliner, sched-group-barrier insertion, LDS layout
├── include/              # TritonAMDGPU dialect headers
└── language/hip/         # AMD device-library hooks
```

## The cheat-sheet (NVIDIA → AMD CDNA3)
| Topic | NVIDIA (contrast) | AMD MI300X / CDNA3 (`gfx942`) |
|---|---|---|
| Warp / wavefront | 32 lanes | **64 lanes** (`num_warps=N` → `N·64` threads) |
| Matrix engine | Tensor Core (`mma`/`wgmma`) | **Matrix Core / MFMA** (`v_mfma_*`) via `tl.dot` |
| MFMA tile (`matrix_instr_nonkdim`) | n/a | **16** (mfma_16x16, preferred) or 32 |
| Shared memory | 228 KB/SM (H100) | **64 KB LDS/CU** (CDNA3); 160 KB (CDNA4) |
| VGPRs | 65536/SM, 256/thread cap | **512/EU**, granularity 16 |
| FP8 matrix dtype | OCP `e4m3fn`/`e5m2` | **FNUZ** `e4m3fnuz`/`e5m2fnuz` (CDNA3); OCP on CDNA4 |
| `num_stages` (single GEMM) | 3–4 | **1–2** (stream pipeliner; 1 for fused FA) |
| Backend dir | `third_party/nvidia` | `third_party/amd` |

## The compilation pipeline
```
@triton.jit (Python AST)
  → Triton IR (TTIR)          # arch-independent
  → TritonGPU IR (TTGIR)      # blocked / MFMA layouts assigned
  → TritonAMDGPU IR           # MFMA dot conversion, LDS swizzle, stream-pipeliner, sched barriers
  → LLVM IR (AMDGPU)          # amdgpu-waves-per-eu, denormal-fp-math attrs
  → AMDGCN ISA (gfx942/950)   # v_mfma_*, ds_read/write_b128, global_load_dwordx4, buffer_load
  → HSACO                     # loaded by HIP runtime
```
Inspect any stage with env vars (`AMDGCN_ENABLE_DUMP=1`, `MLIR_ENABLE_DUMP=1`) — see
[isa_verify.md](isa_verify.md). The key AMD-only stage is **TritonAMDGPU**, where `tl.dot` becomes an
MFMA layout op and the K-loop is software-pipelined.

## The five AMD mistakes that kill Triton perf (Amdahl-relevant)
1. Assuming `warpSize==32` in grid/occupancy math. It is **64**.
2. Carrying `num_warps=8` from NVIDIA → VGPR spill to scratch (HBM) → **3–5× slowdown**. Cut warps first.
3. OCP `float8_e4m3fn` into `tl.dot` on gfx942 → `Unsupported conversion from 'f8E4M3FN'`. Use **fnuz**.
4. `num_stages=3/4` for a single GEMM — pipelines *worse* than 1–2 on the AMD stream pipeliner.
5. Forgetting LDS is 64 KB (CDNA3) — large tiles silently drop occupancy to 1 wg/CU or fail to compile.

## Deep-dive map (this levers folder)
- [deep_codegen.md](deep_codegen.md) — pipeline internals, `tl.dot`→MFMA, layouts, stream-pipeliner.
- [patterns.md](patterns.md) — GEMM/attention/reduction templates tuned for CDNA3/4.
- [knobs.md](knobs.md) — full `HIPOptions` knob set, ranges, autotune config space, baking winners.
- [pitfalls.md](pitfalls.md) — porting checklist + AMD-specific anti-patterns.
- [isa_verify.md](isa_verify.md) — `AMDGCN_ENABLE_DUMP` workflow; what good ISA looks like.

## Sources
- Optimizing Triton kernels (knobs, ISA verify): https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- MI300X workload optimization (Triton tuning, ≥1024 grid, Tagram): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Triton AMD backend `HIPOptions` / pass pipeline: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- Enabling vLLM V1 on AMD GPUs with Triton (num_warps spill, per-shape configs): https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/
- Honest AMD-Triton limits (compiler baselines vs asm/CK): HipKittens, https://arxiv.org/abs/2511.08083
