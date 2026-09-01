---
name: debug-triton-kernel
description: >
  Diagnose Triton-on-AMD kernels that are wrong, fail to compile/lower, or run
  slow on CDNA3/CDNA4. Covers the FNUZ-vs-OCP fp8 2x silent error, the num_warps
  VGPR-spill cliff, num_stages/LDS-overflow occupancy collapse, AMD knobs silently
  ignored when set as Python vars, wave64 reduction width, and the AMDGCN ISA
  checklist that separates a real regression from autotune noise.
  Use when a Triton kernel produces wrong output, a lowering error, or underperforms.
  Usage: /debug-triton-kernel
allowed-tools: Read Edit Bash Grep Glob
---

# Debug Triton (AMD) Kernel

Diagnostic workflow for Triton kernels on AMD Instinct (MI300X gfx942, MI350/MI355X gfx950). Reference:
[../optimize/triton_levers/triton_traps.md](../optimize/triton_levers/triton_traps.md),
[../optimize/triton_levers/triton_knob_space.md](../optimize/triton_levers/triton_knob_space.md),
[../optimize/triton_levers/triton_isa_check.md](../optimize/triton_levers/triton_isa_check.md).

## Step 1: classify the symptom
| Symptom | Likely cause | Go to |
|---|---|---|
| `Unsupported conversion from 'f8E4M3FN'` | OCP fp8 into `tl.dot` on gfx942 | §2 |
| Wrong numbers, ~2× off, fp8 path | FNUZ vs OCP dialect mismatch | §2 |
| Correct but 3–5× slow | `num_warps=8` VGPR spill to scratch | §3 |
| Compile fail / occupancy = 1 | tile too big for 64 KB LDS, or `num_stages` too high | §4 |
| A tuned knob "does nothing" | AMD knob set as Python var, not in `triton.Config({...})` | §5 |
| Wrong reduction / wasted lanes | reduced dim < 64 (wave64) | §6 |
| "Win" that vanishes at e2e | isolated speedup, not gated through serving seam | §7 |

## 2. fp8 FNUZ vs OCP (the silent 2× / lowering error)
- **gfx942 MFMA consumes FNUZ fp8**: `tl.float8e4b8` (E4M3 fnuz) / `tl.float8e5b16` (E5M2 fnuz).
- Passing OCP `float8_e4m3fn` (`tl.float8e4nv`) into `tl.dot` on gfx942 → `Unsupported conversion
  'f8E4M3FN'`. Reading the wrong dialect (bias differs by 1) is a **2× silent error**, not a crash.
- Fix: normalize checkpoints with `normalize_e4m3fn_to_e4m3fnuz` (sglang PR #2601) before the matmul.
  On gfx950 use OCP fp8 / MXFP block-scaled.

## 3. num_warps VGPR spill (the #1 perf cliff)
`num_warps=N` → `N·64` threads (wave64). `num_warps=8` carried from NVIDIA → ~256 VGPR/wave → spill to
scratch (HBM) → 3–5× slower. **Start GEMM at `num_warps=4`**; memory-bound 2/4. Confirm via ISA:
`.private_segment_fixed_size` must be 0 (§8).

## 4. LDS / num_stages occupancy collapse
- LDS = **64 KB/CU** (CDNA3) / 160 KB (CDNA4). Big tiles × `num_stages` overflow LDS → occupancy 1 or
  compile failure. LDS bytes ≈ `(BLOCK_M·BLOCK_K + BLOCK_K·BLOCK_N)·elem·num_stages`.
- `num_stages`: single GEMM **2**, fused FA **1**, no-GEMM **1**. Higher only buffers more loads and
  crushes occupancy. `OPTIMIZE_EPILOGUE=1` frees the epilogue LDS round-trip for GEMM.

## 5. AMD knobs silently ignored
`matrix_instr_nonkdim`, `kpack`, `waves_per_eu`, `schedule_hint` are `HIPOptions` fields — they take
effect **only inside `triton.Config({...})` kwargs**. Setting them as module variables does nothing.
`kpack=2` is gfx942-only (warns/forced to 1 on gfx950). Verify names on your build:
`grep HIPOptions third_party/amd/backend/compiler.py`.

## 6. wave64 reductions
`tl.sum`/`tl.max` over a reduced dim < 64 wastes lanes. Set `BLOCK_SIZE = next_pow2(n_cols)` (≥64) so
the wave reduce is full. Same for softmax/RMSNorm/attention online-softmax.

## 7. e2e gating (don't trust isolated wins)
On sglang/vLLM the dense GEMM path is **aiter**, not raw torch. An authored Triton kernel must be wired
via the aiter seam and **e2e-gated**: keep it only if `pct_gpu_time × speedup` moves e2e past the noise
band. An isolated 0.99–1.47× can be a net e2e loss.

## 8. ISA verification (real regression vs autotune noise)
```bash
AMDGCN_ENABLE_DUMP=1 MLIR_ENABLE_DUMP=1 TRITON_PRINT_AUTOTUNING=1 TRITON_ALWAYS_COMPILE=1 \
  python my_kernel.py 2> dump.txt
grep -E ".vgpr_count|.private_segment_fixed_size|.group_segment_fixed_size" dump.txt
```
| Look for | Good | Bad → retune |
|---|---|---|
| Global loads | `global_load_dwordx4` / `buffer_load_dwordx4` | `global_load_dword` (scalar) |
| Masked tail | `buffer_load_*` (HW bounds) | `global_load_*` + `v_cmp` predication |
| LDS access | `ds_read_b128` | `ds_read_b32`/`b64` (bump `kpack`/`BLOCK_K`) |
| MFMA | dense `v_mfma_f32_16x16x16` | `v_mfma_f32_32x32x8` (compare) or sparse |
| Accumulator | AGPR, no moves in loop | `v_accvgpr_read/write` in loop |
| Scratch | `.private_segment_fixed_size: 0` | nonzero → spilling |

Full workflow: [../optimize/triton_levers/triton_isa_check.md](../optimize/triton_levers/triton_isa_check.md).

## 9. When Triton is the wrong tool
On a *plain* dense GEMM, tuned hipBLASLt/aiter usually win. Triton's honest wins are **fusion**
(epilogue/attention) and **skinny split-K decode**. If you've exhausted knobs and still trail the
library on plain GEMM, that's expected — switch strategy, don't keep tuning.
