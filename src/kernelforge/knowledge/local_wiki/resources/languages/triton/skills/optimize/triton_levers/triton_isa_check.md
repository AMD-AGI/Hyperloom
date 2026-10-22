---
title: Triton on AMD — verifying a config against the ISA
kind: language
lever: triton_isa_check
gens: [gfx950]
updated: 2026-10-19
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - https://llvm.org/docs/AMDGPUUsage.html
---

# Verify against the ISA

**A tuned config is not trusted until you have read the AMDGCN.** Autotune timing tells you *what* won;
the ISA tells you *why*, and catches silent slow paths — scalar loads, scratch spills, the wrong fp8
dialect — that timing alone will not attribute.

## Route here when
- Autotune picked a config and you are about to ship it.
- A config that "should" be fast is not.
- You changed a knob and want to confirm it actually took effect (AMD knobs fail silently when set
  outside `triton.Config`).

## 1. Dump everything

```bash
# AMDGCN_ENABLE_DUMP      final AMDGCN ISA (printed to stdout)
# MLIR_ENABLE_DUMP        TTGIR / TritonAMDGPU IR after each pass (stderr)
# TRITON_PRINT_AUTOTUNING winning config + timing
# TRITON_ALWAYS_COMPILE   bypass the kernel cache so the dump is for THIS run
AMDGCN_ENABLE_DUMP=1 MLIR_ENABLE_DUMP=1 TRITON_PRINT_AUTOTUNING=1 TRITON_ALWAYS_COMPILE=1 \
  python my_kernel.py > dump.txt 2>&1
```

Keep the comments off the continuation lines: a `\` followed by a comment ends the command there, and
the variables never reach `python`.

`TRITON_ALWAYS_COMPILE=1` matters more than it looks — without it you can spend an afternoon reading a
cached dump of a previous config.

```bash
grep ".vgpr_count"                 dump.txt   # VGPRs/lane
grep ".sgpr_count"                 dump.txt
grep ".group_segment_fixed_size"   dump.txt   # static LDS only — 0 for Triton, which sizes LDS at launch
grep ".private_segment_fixed_size" dump.txt   # scratch — MUST be 0
grep "num-warps"                   dump.txt
grep "ttg.shared ="                dump.txt   # LDS bytes the kernel requests (MLIR dump; = kernel.metadata.shared)
```

`ROCm/triton` ships `occ.sh` to turn `.vgpr_count` / LDS / `num-warps` into wg/CU occupancy.

## 2. What good ISA looks like (GEMM inner loop)

| Look for | Good | Bad → retune |
|---|---|---|
| Global loads | `global_load_dwordx4` / `buffer_load_dwordx4` | `global_load_dword` (scalar) |
| Masked tail | `buffer_load_*` (HW bounds check) | `global_load_*` + `v_cmp` predication |
| LDS access | `ds_read_b128` / `ds_write_b128` | `ds_read_b32` |
| MFMA | dense `v_mfma_f32_16x16x32` | sparse, with gaps — a starved core |
| Accumulator | stays put in VGPRs or AGPRs (`.agpr_count` says which) | `v_accvgpr_read/write` **inside** the loop |
| **Scratch** | **`.private_segment_fixed_size: 0`** | nonzero → spilling to HBM, **3–5× slower** |
| Waitcnt | minimal, overlapped | `s_waitcnt vmcnt(0)` after every load = no overlap at all |

## 3. The occupancy boundary check

1. `grep .vgpr_count` → round up to a multiple of 8.
2. `max_waves = min(8, floor(512 / round_up_8(vgpr)))` (8 waves per SIMD is the hardware cap).
3. If you are **just over a boundary** (e.g. 176 → 2 waves), set `waves_per_eu = target+1` so
   LLVM shaves VGPRs (176 → ≤168 → 3 waves). Re-dump.
4. **If that introduced `.private_segment_fixed_size > 0`, you went too far.** Back off — a spill costs
   more than the wave buys.

## 4. MFMA shape and dtype sanity

| Expectation | If you see something else |
|---|---|
| fp16/bf16 with `nonkdim=16` → `v_mfma_f32_16x16x32` | `32x32x16` means `nonkdim` is 32 (or auto picked it) — compare timings |
| fp8 on gfx950 → the OCP `f8f6f4` family | `v_mfma_f32_*_f16` plus a `UserWarning` means you passed the **FNUZ** types (gfx942 dialect) and Triton 3.7.0 upcast them to fp16 |
| MXFP → `v_mfma_scale_f32_*_f8f6f4` | plain `f8f6f4` means the scales are not wired up |

## 5. LDS width check

On gfx950 you should see **`ds_read_b128` without `kpack`** — `kpack` is deprecated and forced to 1
there. If the reads are still `ds_read_b64` / `b32`, `BLOCK_K` is probably too small (< 64) or the
swizzle did not apply. Bump `BLOCK_K` and re-check.

(On gfx942, `kpack=2` was what produced `ds_read_b128`. Carrying that config to gfx950 just triggers a
backend warning.)

## 6. Cross-check against the library

Isolated-bench the tuned kernel against the library default so you know the real gap:

```bash
ROCBLAS_LAYER=2 HIPBLASLT_LOG_LEVEL=2 python compare.py   # logs the lib solution + fallbacks
```

Then **e2e-gate through the actual serving seam** (aiter), not isolated TFLOPS — a kernel that is faster
in isolation and never dispatched is worth nothing. See `triton_traps.md`, integration section.

## 7. Drill to the object if needed

```bash
roc-obj-ls kernel.hsaco
llvm-objdump -d --arch=amdgcn --mcpu=gfx950 kernel.hsaco | less
```

Counter and instruction semantics (`s_waitcnt vmcnt/lgkmcnt`, buffer descriptors, sched barriers) are in
the LLVM AMDGPU backend guide.

## The one-line rule

**A "win" whose ISA is byte-identical to the baseline is measurement noise.** If you cannot point at
what changed in the disassembly, you have not changed anything.

## Sources
- `AMDGCN_ENABLE_DUMP` / `ds_read_b128` / `global_load_dwordx4` / `OPTIMIZE_EPILOGUE`: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- `HIPOptions` / `knobs.amd.dump_amdgcn` / `use_buffer_ops`: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- AMDGPU backend (`s_waitcnt`, buffer descriptors, resource-usage attrs): https://llvm.org/docs/AMDGPUUsage.html
- Occupancy math on MI355X (512 regs/SIMD, VGPRs in groups of 8): https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html
