---
name: debug-ck-kernel
description: >
  Diagnose Composable Kernel (CK) kernels that are wrong, won't compile/select, or
  run slow on CDNA3/CDNA4. Covers the IsSupportedArgument garbage-on-false gate,
  the ck_tile-vs-classic dense-GEMM trap (#1727), build-specific instance pinning,
  repo deprecation (rocm-libraries), over-large-tile VGPR/AGPR spills, sub-128-bit
  AK1/BK1 bandwidth loss, fnuz-vs-OCP fp8, and 16x16-vs-32x32 MFMA. Use when a CK
  kernel is incorrect or underperforms. Usage: /debug-ck-kernel
allowed-tools: Read Bash Grep Glob
---

# Debug CK kernel

Diagnostic workflow for Composable Kernel (classic `DeviceGemm*` + `ck_tile`) on MI300X gfx942 /
MI350 gfx950. Reference: [../optimize/ck_levers/ck_traps.md](../optimize/ck_levers/ck_traps.md),
[../optimize/ck_levers/ck_tuning_knobs.md](../optimize/ck_levers/ck_tuning_knobs.md),
[../optimize/ck_levers/ck_gemm_stack.md](../optimize/ck_levers/ck_gemm_stack.md).

## Step 1: classify the symptom
| Symptom | Likely cause | Go to |
|---|---|---|
| Silent garbage output (no error) | ran an instance past `IsSupportedArgument()==false` | §2 |
| Instance "not found" / won't build | wrong repo, arch/dtype not built, spec mismatch | §3 |
| ck_tile GEMM slower than expected | using ck_tile for dense square GEMM (#1727) | §4 |
| Correct but slow, MFMA-bound | 32×32 MFMA / tile spills | §5 |
| Correct but slow, memory-bound | sub-128-bit `AK1/BK1`, wrong scheduler | §5 |
| Wrong numbers, fp8 | fnuz vs OCP encoding mismatch | §6 |
| A pinned config regressed after upgrade | build-specific instance drift | §3 |

## 2. IsSupportedArgument — the #1 correctness gate
`op.IsSupportedArgument(arg)` checks M/N/K divisibility vs the tile, `K` vs `KPerBlock×KBatch`, pointer
alignment vs `AK1/BK1`, and layout/spec. **Forcing an instance past a `false` returns garbage, not an
error.** Always gate; for non-divisible shapes add `GemmSpecialization::MNKPadding` (small perf cost)
rather than bypassing the check.

## 3. Build / instance / repo traps
- **Repo deprecation**: standalone `ROCm/composable_kernel` is DEPRECATED → use
  `ROCm/rocm-libraries:projects/composablekernel`. `develop` is a read-only mirror.
- **Arch/dtype not built**: CK's full build is huge — scope `GPU_TARGETS=gfx942` and build only the
  needed instance group; gfx950 fp4/mxfp4 are behind `DTYPES` cmake flags (won't appear otherwise).
- **`ckProfiler` missing** in deployment images → no on-box sweep; build it on a dev node.
- **Build-specific pin drift**: tile/pipeline IDs and the tuned instance DB drift across CK/ROCm
  versions. Re-sweep after any bump; never ship a hand-copied instance table as portable.
Details: [../optimize/ck_levers/ck_instance_codegen.md](../optimize/ck_levers/ck_instance_codegen.md).

## 4. ck_tile vs classic — pick the right front-end
- **Dense square bf16 GEMM**: classic `DeviceGemmXdlUniversal` v3/Intrawave is often ~1.7× faster than
  ck_tile `universal_gemm` at the same 256×256×64 tile (#1727: 615 vs 359 TFLOP/s). Benchmark classic
  first for dense paths.
- **Fusion / attention / MoE**: use **ck_tile** (FMHA, fused-MoE, fp8/mxfp4). Classic
  `DeviceBatchedGemmSoftmaxGemm*` is legacy — don't use it for new attention.

## 5. Perf: tile, MFMA, load width, scheduler
- **16×16 vs 32×32 MFMA**: 16×16×16 usually yields higher *achievable* FLOPs on MI300X (32×32 draws more
  power, clocks lower). Test both; don't default to 32×32.
- **Over-large block tile → spills**: growing past VGPR/AGPR headroom triggers `v_accvgpr` moves /
  `scratch_` spills (LLVM #131954) → throughput drops to a smaller-tile class. Check disassembly, not the
  config string.
- **Sub-128-bit `AK1/BK1`** halves HBM bandwidth. Size loads to ≥128 bit (bf16 `AK1=8`, fp8 `AK1=16`);
  align pointers.
- **Scheduler**: Intrawave (compute-bound prefill) vs Interwave (memory-bound / skinny decode). Decode
  also wants split-K (`KBatch≥2`) + small M tile to fill the 304 CUs.

## 6. fp8 encoding
CDNA3 is **fnuz** fp8 (different exponent bias from OCP); match the dequant scale to the encoding or get
silent numeric garbage. OCP fp8 / MXFP block-scaled is the gfx950 story.

## 7. ISA verification
Build with `--save-temps` and confirm in the K-loop: `buffer_load_dwordx4` (≥128-bit loads),
`s_waitcnt lgkmcnt(1)` before `v_mfma`, dense `v_mfma_*`, no `v_accvgpr_*` / `scratch_` spam. Parity:
fp32 accumulate; greedy temp=0 vs a reference (≥10 prompts for attention). The MFMA/ISA facts are in
`languages/hip/skills/optimize/hip_levers/hip_builtins.md`.

## 8. When to author CK vs use a library
If aiter/hipBLASLt already dispatch a strong CK instance for your shape, tune via the aiter DB first
(`local_knowledge/framework/aiter/skills/optimize/aiter_levers/tuning_db.md`). Author/modify CK templates only for
a fusion the library can't express or a shape it doesn't cover.
