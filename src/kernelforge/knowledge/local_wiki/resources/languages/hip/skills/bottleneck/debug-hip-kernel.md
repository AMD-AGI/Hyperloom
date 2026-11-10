---
name: debug-hip-kernel
description: >
  Diagnose HIP/C++ CDNA kernels that are wrong, crash, or run slow. Covers the two
  root-cause families (wave64 assumptions and 64 KB LDS), a symptom→cause table,
  wave64-correct reductions, LDS bank-conflict diagnosis, AGPR spill / v_accvgpr in
  the MFMA loop, FNUZ-vs-OCP fp8 mismatch on gfx942, occupancy prediction, and the
  ISA-verification checklist that separates a real regression from noise.
  Use when a HIP kernel produces incorrect output, compile errors, or underperforms.
  Usage: /debug-hip-kernel
allowed-tools: Read Edit Bash Grep Glob
---

# Debug HIP Kernel

Diagnostic workflow for hand-written HIP/C++ CDNA kernels (MI300X gfx942, MI350/MI355X gfx950).
Reference material: [../optimize/hip_levers/hip_traps.md](../optimize/hip_levers/hip_traps.md),
[../optimize/hip_levers/hip_lds_staging.md](../optimize/hip_levers/hip_lds_staging.md),
[../optimize/hip_levers/hip_builtins.md](../optimize/hip_levers/hip_builtins.md).

## Step 0: the two facts behind almost every HIP bug
1. **Wavefront = 64**, not 32. Every `__shfl`/`__ballot`/manual reduction, grid/occupancy calc, and the
   static-assert on mask width traces here. 32-lane CUDA code *runs* but uses **half** the machine.
2. **LDS = 64 KB/CU** (CDNA3; 160 KB CDNA4). H100 habits (228 KB) overflow LDS → launch failure or
   occupancy 1.

## Step 1: classify the symptom
| Symptom | Likely cause | Go to |
|---|---|---|
| Wrong reduction / off-by-half result | `warpSize`/mask assumed 32 | §2 |
| Compile static-assert on mask width | 32-bit mask on wave64 | §2 |
| Launch failure / occupancy collapses to 1 | LDS > 64 KB, or `__launch_bounds__` too tight | §3, §5 |
| Correct but slow, LDS-bound | bank conflicts (no pad/swizzle) | §3 |
| Correct but slow, MFMA-bound with gaps | `v_accvgpr_*` in loop / starved matrix core | §4 |
| Small numeric error (1–3%) on fp8 | FNUZ vs OCP, or scale mismatch | §6 |
| 3–5× slower than expected | scratch spill to HBM | §5 |

## 2. Wave64 correctness
- Masks must be **64-bit** (`unsigned long long`); use `__popcll`, not `__popc`.
- `__shfl*` width defaults to `warpSize` (64). A manual reduction must run `off = 32,16,8,4,2,1`.
- Block size must be a multiple of 64 (64/128/256). A 32-thread block is half a wave.
- Half-float `__shfl` is unsupported — shuffle as int/float and repack.
- Correct block reduction skeleton: see [../optimize/hip_levers/hip_templates.md](../optimize/hip_levers/hip_templates.md) §1.

## 3. LDS bank conflicts
- 32 banks × 4 B; a wave64 access is serviced in **two phases**. Same-bank/different-row = conflict.
- **Diagnose**: rocprofv3 LDS-conflict counters, or ISA showing `ds_read_b32` (scalar) and stalls on
  `s_waitcnt lgkmcnt(0)`.
- **Fix**: pad inner dim `+1`, or XOR-swizzle the column index (required for direct-to-LDS). Vectorize
  to `ds_read_b128`/`ds_write_b128`. Detail: [../optimize/hip_levers/hip_lds_staging.md](../optimize/hip_levers/hip_lds_staging.md) §2.

## 4. MFMA / AGPR issues
- On CDNA3 MFMA accumulators live in **AGPRs**. If the compiler inserts `v_accvgpr_read/write` inside
  the K-loop, perf drops to small-tile levels (LLVM #131954).
- **Fix**: carry the accumulator in a stable `__attribute__((vector_size))` variable across iterations,
  or use a framework that gives a tied accumulator (CK) / pinned register tiles (HipKittens).
- **Verify**: ISA inner loop shows dense `v_mfma_*` with accumulators in `a[...]`, no `v_accvgpr_*`.

## 5. Occupancy & spills — predict before you measure
```
occ_vgpr = floor(512 / round_up_16(vgpr_used))          # waves/SIMD from VGPR
occ_lds  = floor(LDS_CAP / lds_bytes_used)               # blocks/CU (65536 / 163840)
occ (wg/CU) = min(floor(occ_vgpr * 4 / num_warps), occ_lds)   # 4 SIMD/CU
```
`__launch_bounds__(maxTPB, minWavesPerEU)`: `=2` → VGPR ≤ 256, `=4` → ≤ 128. Too tight forces scratch
spills to HBM (3–5× slower). Check `.private_segment_fixed_size == 0`.

## 6. fp8 numeric mismatch
- **gfx942 fp8 is FNUZ** (e4m3 fnuz / e5m2 bf8). OCP fp8 + block-scaled MFMA are **gfx950 only**. Using
  an OCP path on gfx942 gives wrong results or no lowering — use FNUZ.
- fp8 PV/MFMA introduces inherent ~0.03 error vs bf16 reference — that is the data path, not a bug
  (expected `atol≈5e-3`). Also check `scale = softmax_scale * q_scale * k_scale` isn't applied twice.

## 7. ISA verification checklist (real regression vs noise)
Build with `--save-temps` (or `AMDGCN_ENABLE_DUMP=1`) and confirm in the inner loop:
| Look for | Good | Bad → retune |
|---|---|---|
| Global loads | `global_load_dwordx4` / `buffer_load_dwordx4` | `global_load_dword` (scalar) |
| LDS access | `ds_read_b128` / `ds_write_b128` | `ds_read_b32` |
| MFMA | dense `v_mfma_*` | sparse, gaps = starved core |
| Accumulator | `a[...]` (AGPR) | `v_accvgpr_read/write` in loop |
| Scratch | `.private_segment_fixed_size: 0` | nonzero → spilling |
| Waitcnt | minimal, overlapped | `s_waitcnt vmcnt(0)` after every load = no overlap |

`-Rpass-analysis=kernel-resource-usage` prints `.vgpr_count`, `.sgpr_count`,
`.group_segment_fixed_size` (LDS), `.private_segment_fixed_size` (scratch).

## 8. When NOT to keep hand-writing HIP
If you're reaching for raw `__builtin_amdgcn_mfma_*` + `sched_group_barrier` + double-buffering, first
check whether **rocWMMA**, **ck_tile / Composable Kernel**, **HipKittens**
([../optimize/hip_levers/hipkittens.md](../optimize/hip_levers/hipkittens.md)), or **FlyDSL** already
express the fusion — they encode the tied-accumulator + pipeline patterns correctly and avoid the
LLVM #131954 trap.

## 9. Recovery from GPU hang
```bash
rocm-smi                 # 100% usage, no progress → hang (barrier deadlock / OOB / infinite loop)
sudo amdgpu-reset        # or reboot
```
Common causes: divergent `__syncthreads()` (not all lanes reach the barrier), wrong loop bounds, OOB
global access (use buffer descriptors for HW bounds checking — [../optimize/hip_levers/hip_builtins.md](../optimize/hip_levers/hip_builtins.md) §2).
