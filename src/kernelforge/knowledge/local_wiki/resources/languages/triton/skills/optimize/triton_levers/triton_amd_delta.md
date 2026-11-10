---
title: Triton on AMD — what changes vs NVIDIA, and where Triton fits
kind: language
lever: triton_amd_delta
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - https://arxiv.org/abs/2511.08083
---

# Triton on AMD — the delta

**Read this first.** The Python API is **identical to NVIDIA**; everything that matters is underneath.
This card is the porting delta and the honest positioning.

## Route here when
- Starting any Triton work on Instinct.
- Porting a kernel that works on NVIDIA.
- Deciding whether Triton is even the right tool for this kernel.

## Should you use Triton at all?

| Use Triton when | Reach for something else when |
|---|---|
| A **fused epilogue/attention** the library cannot express | Plain dense GEMM → `hipblaslt` / `aiter` / `flydsl` |
| Rapid prototyping / shape exploration before committing to CK or asm | The last 10–20% of peak → `ck_tile`, `asm`, HipKittens |
| The `torch.compile` / Inductor codegen path | Block-scaled MXFP4/6 GEMM → tuned `ck` / `aiter` |
| Skinny/decode GEMM with `SPLIT_K` to fill 256 CUs | A production hot path that already has a tuned table |

**On a *plain* dense GEMM, AMD Triton typically loses to tuned hipBLASLt/aiter.** That is not a bug to
fix; it is the tool's position. HipKittens (arXiv 2511.08083) corroborates: compiler backends including
Triton under-perform hand-tuned asm/CK on CDNA GEMM and attention. **The honest win is fusion, or
skinny split-K decode.**

## The porting cheat sheet

| Topic | NVIDIA | **AMD gfx950** |
|---|---|---|
| Warp / wavefront | 32 lanes | **64 lanes** (`num_warps=N` → N·64 threads) |
| Matrix engine | Tensor Core (`mma`/`wgmma`) | **Matrix Core / MFMA** (`v_mfma_*`) via `tl.dot` |
| MFMA tile (`matrix_instr_nonkdim`) | n/a | **16** (preferred) or 32 |
| Shared memory | 228 KB/SM (H100) | **160 KiB LDS/CU**, **64 banks** |
| Registers | 65536/SM, 256/thread cap | **512/SIMD**, 16-granule |
| FP8 matrix dtype | OCP `e4m3fn`/`e5m2` | **OCP** on gfx950 (**FNUZ** on gfx942 — the porting trap) |
| `num_stages` (single GEMM) | 3–4 | **1–2** (stream pipeliner; 1 for fused FA) |
| `tf32` | available | **removed on CDNA4** — `input_precision` is `"ieee"` |
| CU count | — | **256** (8 XCD × 32), not 304 |
| Backend dir | `third_party/nvidia` | `third_party/amd` |

## The five mistakes that kill Triton perf here

1. **Assuming `warpSize == 32`** in grid/occupancy math. It is **64**.
2. **Carrying `num_warps=8` from NVIDIA** → VGPR spill to scratch (HBM) → **3–5× slowdown**. Cut warps
   first, before anything else.
3. **`num_stages=3/4` for a single GEMM** — pipelines *worse* than 1–2 on the AMD stream pipeliner.
4. **Feeding the wrong fp8 dialect.** gfx950 is OCP; gfx942 is FNUZ. Wrong one = silent ~2× error or a
   lowering failure, depending on direction.
5. **Setting AMD knobs as Python variables.** They only take effect inside `triton.Config({...})`.

## The two distributions

- **Upstream `triton-lang/triton`** — the AMD backend lives in `third_party/amd/`; CDNA3/CDNA4 are
  first-class and built by default. Arch is auto-detected from the active HIP device.
- **`ROCm/triton`** (AMD staging fork) — carries AMD perf patches and tuning utilities (e.g. `occ.sh`)
  ahead of upstream; ROCm PyTorch wheels ship Triton built from here.

**Knob names and defaults drift between the two.** Always
`grep HIPOptions third_party/amd/backend/compiler.py` on *your* build rather than trusting any doc —
including this one.

## Where the facts live

```
third_party/amd/
├── backend/compiler.py   # HIPOptions (matrix_instr_nonkdim, kpack, waves_per_eu, num_stages,
│                         #   schedule_hint, supported_fp8_dtypes), the pass pipeline
├── backend/driver.py     # HIP runtime, kernel launch
├── lib/                  # MLIR passes: TritonGPU→TritonAMDGPU→AMDGCN, MFMA dot conversion,
│                         #   stream-pipeliner, sched-group-barrier insertion, LDS layout
├── include/              # TritonAMDGPU dialect headers
└── language/hip/         # AMD device-library hooks
```

## The compilation pipeline

```
@triton.jit (Python AST)
  → Triton IR (TTIR)          # arch-independent
  → TritonGPU IR (TTGIR)      # blocked / MFMA layouts assigned
  → TritonAMDGPU IR           # MFMA dot conversion, LDS swizzle, stream-pipeliner, sched barriers
  → LLVM IR (AMDGPU)          # amdgpu-waves-per-eu, denormal-fp-math attrs
  → AMDGCN ISA (gfx950)       # v_mfma_*, ds_*_b128, global_load_dwordx4, buffer_load
  → HSACO                     # loaded by the HIP runtime
```

**TritonAMDGPU is the AMD-only stage** — where `tl.dot` becomes an MFMA layout op and the K-loop gets
software-pipelined. That is where every AMD-specific knob acts.

## Where next

| Question | Card |
|---|---|
| Which knob, what range, how to autotune | `triton_knob_space.md` |
| Give me a starting kernel body | `triton_templates.md` |
| Why does this knob help? What does `tl.dot` become? | `triton_lowering.md` |
| Did my config actually land? | `triton_isa_check.md` |
| It compiles but is wrong / slow | `triton_traps.md` |

## Sources
- Optimizing Triton kernels on AMD (knobs, ISA verification): https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- MI300X workload optimization (Triton tuning, grid sizing, Tagram): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Triton AMD backend `HIPOptions` / pass pipeline: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- Enabling vLLM V1 on AMD GPUs with Triton (`num_warps` spill, per-shape configs): https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/
- Honest compiler-vs-asm limits: HipKittens, https://arxiv.org/abs/2511.08083
