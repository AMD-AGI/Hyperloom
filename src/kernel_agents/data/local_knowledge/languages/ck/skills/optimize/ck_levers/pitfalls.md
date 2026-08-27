---
title: CK pitfalls & anti-patterns
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, mxfp4]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/ROCm/composable_kernel
  - https://github.com/ROCm/composable_kernel/issues/1727
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
---

# CK pitfalls

## TL;DR
The recurring ways CK bites: trusting ck_tile to beat classic for dense GEMM, skipping
`IsSupportedArgument`, mis-pinning build-specific instances, building for all archs, and assuming
attention should use classic softmax-GEMM. Read this before integrating CK into a serving stack.

## The pitfalls
1. **"ck_tile is the new fast path" — not for dense square GEMM.** Issue #1727: 4096³ bf16 ck_tile
   `universal_gemm` ~359 TFLOP/s vs classic `DeviceGemmXdlUniversal` v3 ~615 TFLOP/s at the same
   256×256×64 tile (~1.7× slower). ck_tile wins on **fusion + attention/MoE**, not raw square GEMM.
   → Benchmark classic v3 first for dense paths.
2. **Skipping `IsSupportedArgument()`.** It gates M/N/K divisibility, K vs `KPerBlock×KBatch`, pointer
   alignment vs `AK1/BK1`, layout/spec. **An instance forced past a `false` returns garbage**, not an
   error. Always gate.
3. **Pinning a build-specific instance as portable.** Tile/pipeline IDs and the tuned instance DB drift
   across CK/ROCm versions. A hand-copied winning table is valid only for the build it was swept on —
   re-sweep after every bump.
4. **Repo confusion.** Standalone `ROCm/composable_kernel` is **DEPRECATED** → development is in
   `ROCm/rocm-libraries:projects/composablekernel`; `develop` is a read-only mirror. Pin the monorepo;
   don't file issues / expect merges on the old repo.
5. **Building for every gfx and every dtype.** CK's full build is huge/slow. Scope `GPU_TARGETS=gfx942`
   (or `gfx950`) and build only the needed instance group; gfx950 fp4/mxfp4 are behind `DTYPES` flags
   (won't appear unless enabled).
6. **`ckProfiler` missing in deployment images.** No CK instance sweep there → can't tune on-box. Build
   it on a dev node; fall back to aiter/Triton if CK profiling is unavailable.
7. **Classic softmax-GEMM for attention.** `DeviceBatchedGemmSoftmaxGemm*` is **legacy**. Use CK-Tile
   FMHA (`example/ck_tile/01_fmha`, paged-KV) — see [fmha_template.md](fmha_template.md).
8. **Over-large block tile → spills.** Growing the tile past VGPR/AGPR headroom triggers `v_accvgpr`
   moves / `scratch_` spills (LLVM #131954) and throughput silently drops to a smaller-tile class.
   Check disassembly, not just the config string.
9. **Sub-128-bit `AK1/BK1`.** Halves effective HBM bandwidth. Size loads to ≥128 bit (bf16 `AK1=8`,
   fp8 `AK1=16`); align pointers accordingly.
10. **Defaulting to 32×32 MFMA.** 16×16×16 usually yields higher *achievable* FLOPs on MI300X (power /
    clock). Test both.
11. **fp8 encoding mismatch.** CDNA3 is **fnuz** fp8 (different bias from OCP). Match the dequant scale to
    the encoding or get silent numeric garbage; OCP fp8/MXFP is the gfx950 story.

## Verify
- Greedy temp=0 parity vs a reference at your shapes before trusting a pinned config.
- Disassemble the hot loop (`--save-temps`); grep `buffer_load`, `accvgpr`, `scratch_`, `s_waitcnt`.
- Confirm the active repo/commit pin matches `ROCm/rocm-libraries:projects/composablekernel`.

## Sources
- Repo deprecation banner: https://github.com/ROCm/composable_kernel
- Issue #1727 (ck_tile vs classic v3 perf gap): https://github.com/ROCm/composable_kernel/issues/1727
- LLVM #131954 (large MFMA tiles → v_accvgpr/spill): https://github.com/llvm/llvm-project/issues/131954
- ROCm "Optimizing with Composable Kernel" (IsSupportedArgument, instance selection): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
- MI300X workload optimization (128-bit load, 16×16 vs 32×32, fnuz fp8): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
