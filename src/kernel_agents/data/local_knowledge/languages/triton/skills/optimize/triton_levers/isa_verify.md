---
title: Triton on AMD — ISA verification workflow
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: competitive
updated: 2026-07-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - https://llvm.org/docs/AMDGPUUsage.html
---

# Triton on AMD — verify with the ISA

A tuned config is not trusted until you've **read the AMDGCN**. Autotune timing is necessary but the
ISA tells you *why* and catches silent slow paths (scalar loads, scratch spills, FNUZ mismatch).

## 1. Dump everything
```bash
AMDGCN_ENABLE_DUMP=1 \      # final AMDGCN ISA to stderr
MLIR_ENABLE_DUMP=1 \        # TTGIR / TritonAMDGPU IR after each pass
TRITON_PRINT_AUTOTUNING=1 \ # winning config + timing
TRITON_ALWAYS_COMPILE=1 \   # bypass kernel cache so the dump is for THIS run
python my_kernel.py 2> dump.txt
```
Pull the resource numbers:
```bash
grep ".vgpr_count"               dump.txt   # VGPRs/lane
grep ".sgpr_count"               dump.txt
grep ".group_segment_fixed_size" dump.txt   # LDS bytes
grep ".private_segment_fixed_size" dump.txt  # scratch — MUST be 0
grep "num-warps"                 dump.txt
grep "triton_gpu.shared"         dump.txt    # LDS bytes per shared layout (from MLIR dump)
```
ROCm/triton ships `occ.sh` to turn `.vgpr_count` / LDS / `num-warps` into wg/CU occupancy.

## 2. What good ISA looks like (GEMM inner loop)
| Look for | Good | Bad → retune |
|---|---|---|
| Global loads | `global_load_dwordx4` / `buffer_load_dwordx4` | `global_load_dword` (scalar) |
| Masked tail | `buffer_load_*` (HW bounds) | `global_load_*` + `v_cmp` predication |
| LDS access | `ds_read_b128` / `ds_write_b128` | `ds_read_b32` |
| MFMA | dense `v_mfma_f32_16x16x16` | sparse, gaps = starved core |
| Accumulator | acc stays in AGPR (`a[0:n]`) | `v_accvgpr_read/write` inside loop |
| Scratch | `.private_segment_fixed_size: 0` | nonzero → spilling to HBM (3–5× slower) |
| Waitcnt | minimal, overlapped | `s_waitcnt vmcnt(0)` after every load = no overlap |

## 3. Occupancy boundary check
1. `grep .vgpr_count` → round up to a 16-granule.
2. `max_waves = floor(512 / round_up_16(vgpr))`. If you're one granule over a boundary (e.g. 176 →
   2 waves), set `waves_per_eu = target+1` so LLVM shaves VGPRs (e.g. 176→160 → 3 waves). Re-dump.
3. If setting `waves_per_eu` introduced `.private_segment_fixed_size > 0`, you went too far — back off.

## 4. MFMA shape & dtype sanity
- fp16/bf16 with `matrix_instr_nonkdim=16` → `v_mfma_f32_16x16x16`. If you see `v_mfma_f32_32x32x8`,
  your `nonkdim` is 32 (or auto picked it) — compare timings.
- fp8 on gfx942 → `v_mfma_f32_16x16x32_fp8_fp8` (FNUZ). If the build refused to lower the dot, you
  passed OCP `e4m3fn` — convert to `tl.float8e4b8`.
- gfx950 block-scaled MXFP → `v_mfma_scale_f32_*_f8f6f4`.

## 5. LDS layout (kpack) check
`kpack=2` on gfx942 should turn the dot-operand LDS reads into `ds_read_b128`. If they're still
`ds_read_b64`/`b32`, either `BLOCK_K` is too small (<64) or the swizzle didn't apply — bump `BLOCK_K`,
re-check. On gfx950 expect `ds_read_b128` without `kpack` (deprecated there).

## 6. Cross-check vs library
Isolated bench the tuned Triton kernel against the library default to know the real gap:
```bash
ROCBLAS_LAYER=2 HIPBLASLT_LOG_LEVEL=2 python compare.py   # log lib solution + fallbacks
```
Then **e2e-gate** through the actual serving seam (aiter), not just isolated TFLOPS — see
[pitfalls.md](pitfalls.md) integration note.

## 7. Drill to the .s if needed
For the raw object:
```bash
# from a cached HSACO or AOT-compiled object
roc-obj-ls kernel.hsaco
llvm-objdump -d --arch=amdgcn kernel.hsaco | less
```
Counter/instruction semantics (`s_waitcnt vmcnt/lgkmcnt`, buffer descriptors, sched barriers) are in
the LLVM AMDGPU backend user guide.

## Sources
- AMDGCN_ENABLE_DUMP / ds_read_b128 / global_load_dwordx4 / OPTIMIZE_EPILOGUE: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- HIPOptions / knobs.amd.dump_amdgcn / use_buffer_ops: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- AMDGPU backend (s_waitcnt, buffer descriptors, resource usage attrs): https://llvm.org/docs/AMDGPUUsage.html
- Occupancy math (512 VGPR/EU, 16-granule) — MI300X workload opt: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
