---
title: CK — the eleven traps, by symptom
kind: language
lever: ck_traps
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://github.com/ROCm/composable_kernel
  - https://github.com/ROCm/composable_kernel/issues/1727
  - https://github.com/llvm/llvm-project/issues/131954
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
---

# CK traps

Indexed **by symptom**. Read this before integrating CK into a serving stack — several of these fail
silently rather than erroring.

## Symptom → trap

| What you observe | Trap | § |
|---|---|---|
| Silently wrong output, no error | forced past `IsSupportedArgument` · fp8 encoding mismatch | §1 §9 |
| ck_tile slower than expected on dense GEMM | it is not the fast path for that case | §2 |
| Pinned config regressed after an upgrade | the pin is build-specific | §3 |
| Issues/PRs going nowhere | wrong repo | §4 |
| Build takes an hour+ | building every arch and dtype | §5 |
| Cannot sweep on the deployment box | `ckProfiler` absent | §6 |
| Attention path slow / unmaintained | using classic softmax-GEMM | §7 |
| **Throughput drops as the tile grows** | spills | §8 |
| Bandwidth ~half of expected | sub-128-bit loads | §10 |
| Slower at 32×32 than 16×16 | two independent reasons | §11 |

---

### §1 Skipping `IsSupportedArgument()`
It gates M/N/K divisibility against the tile, K against `KPerBlock × KBatch`, pointer alignment against
`AK1`/`BK1`, and layout/spec. **An instance forced past a `false` returns garbage, not an error.**
**Fix:** always gate. For non-divisible shapes use `GemmSpecialization::MNKPadding` (small perf cost).

### §2 "ck_tile is the new fast path" — not for dense square GEMM
Issue #1727, MI300X: 4096³ bf16, **same** 256×256×64 tile — ck_tile `universal_gemm` ~359 TFLOP/s vs
classic `DeviceGemmXdlUniversal` v3 ~615 TFLOP/s (~1.7× slower).
**ck_tile's edge is fusion + attention/MoE, not raw square GEMM.**
**Fix:** benchmark classic v3 first for any dense path. → `ck_frontend_classic.md`

### §3 Pinning a build-specific instance as portable
Tile/pipeline IDs and the tuned instance DB drift across CK/ROCm versions. A hand-copied winning table
is valid **only for the build it was swept on**.
**Fix:** re-sweep after every bump; record the pin as `instance <idx> @ CK <commit>, ROCm <ver>, <date>`.
→ `ck_instance_codegen.md`

### §4 Repo confusion
Standalone `ROCm/composable_kernel` is **DEPRECATED** → development is in
`ROCm/rocm-libraries:projects/composablekernel`; `develop` is a read-only mirror.
**Fix:** pin the monorepo. Do not file issues or expect merges on the old repo.

### §5 Building for every gfx and every dtype
CK's full build is huge and slow.
**Fix:** scope `GPU_TARGETS=gfx950` and build only the needed instance group. Note gfx950 fp4/mxfp4 sit
behind `DTYPES` flags and will not appear unless enabled. → `ck_instance_codegen.md`

### §6 `ckProfiler` missing in deployment images
No CK instance sweep there, so you cannot tune on box.
**Fix:** build it on a dev node; or fall back to aiter/Triton for that shape.

### §7 Classic softmax-GEMM for attention
`DeviceBatchedGemmSoftmaxGemm*` is **legacy**.
**Fix:** use CK-Tile FMHA (`example/ck_tile/01_fmha`, paged-KV). → `ck_fmha_stack.md`

### §8 Over-large block tile → spills
Growing the tile past VGPR/AGPR headroom triggers `v_accvgpr` moves and `scratch_` spills
(LLVM #131954), and throughput **silently drops to a smaller-tile class**.
**Signature: TFLOP/s plateaus or regresses as the tile grows** — the one symptom that reliably
identifies this.
**Fix:** `grep -cE 'v_accvgpr|scratch_'` the disassembly. Check the ISA, not the config string.

### §9 fp8 encoding mismatch
**gfx950 FP8 is OCP** (E4M3FN bias 7, max ±448). Earlier CDNA parts used **FNUZ** (bias 8, max ±240).
A dequant scale matched to the wrong encoding gives silent numeric garbage.
**Fix:** convert the checkpoint; never bit-copy. → `hardware/mi350_dtypes.md`

### §10 Sub-128-bit `AK1` / `BK1`
Halves effective HBM bandwidth, silently.
**Fix:** size loads to ≥128 bit — bf16 `AK1=8`, fp8 `AK1=16` — and align pointers accordingly.
→ `ck_gemm_stack.md`

### §11 Defaulting to 32×32 MFMA
Two independent reasons it loses: **16 C-registers/lane vs 16×16's 4** (occupancy), and it **draws more
power so the part clocks lower** (max-achievable FLOPs).
**Fix:** default 16×16; test 32×32 only for a specific large square shape.

---

## Grid sizing note (gfx950)

Several older CK write-ups size the grid against **304 CUs** (MI300X). **gfx950 has 256.** A block count
of `≈ k·304` leaves a quantization tail here. Query `hipGetDeviceProperties → multiProcessorCount`
rather than hardcoding either number.

## The standard diagnostic pass

```bash
# build with --save-temps, then:
grep -E 'buffer_load|accvgpr|scratch_|s_waitcnt|v_mfma' kern-*.s
```

| Also | For |
|---|---|
| greedy temp=0 parity vs a reference at your shapes | §1 §9, before trusting any pinned config |
| `ckProfiler` sweep / example `-v 1` | §2 §3 |
| confirm the repo/commit pin | §4 |

## Sources
- Repo deprecation banner: https://github.com/ROCm/composable_kernel
- Issue #1727 (ck_tile vs classic v3 perf gap): https://github.com/ROCm/composable_kernel/issues/1727
- LLVM #131954 (large MFMA tiles → `v_accvgpr` / spills): https://github.com/llvm/llvm-project/issues/131954
- ROCm "Optimizing with Composable Kernel" (`IsSupportedArgument`, instance selection): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
- MI300X workload optimization (128-bit load, MFMA shape guidance): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
