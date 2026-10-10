---
name: profiling-triton
description: >
  Profile Triton-on-AMD kernels: read TRITON_PRINT_AUTOTUNING, dump AMDGCN/TTGIR,
  turn rocprofv3 PMC counters into a memory- vs compute-bound verdict, check the
  occupancy boundary from .vgpr_count, and tie each signal to a Triton knob
  (num_warps, waves_per_eu, kpack, num_stages, matrix_instr_nonkdim, SPLIT_K).
  Use when deciding what to tune next on a Triton kernel from measured evidence.
  Usage: /profiling-triton
allowed-tools: Read Bash Grep Glob
---

# Profiling Triton (AMD) kernels

Measurement-driven diagnosis for Triton kernels on CDNA3/CDNA4. The forge-loop builds → validates →
benches EVERY iteration, but **profiles only at the baseline and after a KEEP** (a kept improvement) —
not on every iteration, and reverted candidates are not profiled. You can also profile on demand
yourself with `local_knowledge/common_methodology/profiling/rocpc_profile.py` (see
`common_methodology/profiling/measure_rocpc_workflow.md`). This card explains how to read that
profiling output and which knob each signal points to. Hardware peaks live in `local_knowledge/hardware/`.

## 1. Autotune + wall time
```bash
TRITON_PRINT_AUTOTUNING=1 rocprofv3 --kernel-trace --stats -f csv -- python test_driver.py --profile-run
```
`TRITON_PRINT_AUTOTUNING=1` prints the winning `triton.Config` + timing; rocprofv3 gives kernel-level
wall time. Bench discipline: `warmup≥25`, `rep≥100` (median), and a real win must beat the current best
by more than run-to-run jitter (~2% noise floor). Autotune timing alone is not enough — confirm with ISA
(§4).

## 2. Classify the bottleneck from PMC counters
| Counter (rocprofv3) | Reads as | Triton knob |
|---|---|---|
| `MFMABusy` high, near peak | compute-bound | you're near roofline — bigger tile / better dtype only |
| `MFMABusy` high with gaps | matrix core starved | `num_stages`, `schedule_hint`, `use_block_pingpong` |
| `VALUBusy` high, `MFMABusy` low | VALU/address-bound | reduce index math, `use_buffer_ops`, wider loads |
| LDS bank-conflict counters high | LDS-bound | `kpack=2` (gfx942), tile shape; ISA shows `ds_read_b32` |
| `s_waitcnt vmcnt(0)` stalls before MFMA | global-load latency exposed | `num_stages=2`, `use_async_copy`, larger `BLOCK_K` |
| VGPR spill (`.private_segment_fixed_size>0`) | scratch spill to HBM | **cut `num_warps`**, `waves_per_eu`, smaller tile |
| L2 / Infinity Cache hit rate low | poor locality | `GROUP_SIZE_M` (×XCD=8) L2 swizzle |
| Only a few programs, CUs idle | grid too small (skinny) | `SPLIT_K` to reach ≥1024 programs |
| HBM BW near roofline | memory-bound at peak | reduce bytes (dtype, fusion, reuse) |

## 3. Occupancy boundary from .vgpr_count
```
max_waves = floor(512 / round_up_16(vgpr_used))     # 512 VGPR/EU, 16-granule
```
One granule over a boundary (e.g. 176 → 2 waves) → set `waves_per_eu = target+1` so LLVM shaves VGPRs
(176→160 → 3 waves). If that introduces scratch spill, back off. `occ.sh` (ROCm/triton) automates this.

## 4. Always cross-check with ISA
PMC says *what* is slow; the AMDGCN says *why*. Confirm the inner loop against the checklist in
[../bottleneck/debug-triton-kernel.md](../bottleneck/debug-triton-kernel.md) §8 and
[../optimize/triton_levers/triton_isa_check.md](../optimize/triton_levers/triton_isa_check.md) before and after a
change — a "win" that doesn't change the ISA as expected is usually autotune noise.

## 5. e2e gate through the serving seam
Isolated TFLOPS ≠ e2e win. Gate the tuned kernel through the actual seam (aiter for sglang/vLLM GEMM):
keep it only if `pct_gpu_time × speedup` moves e2e past the noise band
([../optimize/triton_levers/triton_traps.md](../optimize/triton_levers/triton_traps.md) integration note).

## Sources
- Optimizing Triton kernels (autotune, ISA dump, OPTIMIZE_EPILOGUE): https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- rocprofv3 / rocprof-compute (omniperf): https://rocm.docs.amd.com/projects/omniperf/en/amd-staging/what-is-rocprof-compute.html
- MI300X workload optimization (occupancy, ≥1024 grid, L2): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
