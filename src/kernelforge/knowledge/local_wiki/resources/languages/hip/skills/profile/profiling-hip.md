---
name: profiling-hip
description: >
  Profile HIP/C++ CDNA kernels with rocprofv3 and ISA inspection: capture wall
  time, classify memory- vs compute-bound from PMC counters (VALUBusy, MFMABusy,
  LDS conflicts, cache hit rates, HBM BW), read the roofline, and tie each counter
  back to a HIP lever (vectorization, LDS swizzle, occupancy, MFMA scheduling).
  Use when deciding what to optimize next on a HIP kernel from measured evidence.
  Usage: /profiling-hip
allowed-tools: Read Bash Grep Glob
---

# Profiling HIP kernels

Measurement-driven diagnosis for HIP/C++ CDNA kernels. The forge-loop builds → validates → benches
EVERY iteration, but **profiles only at the baseline and after a KEEP** (a kept improvement) — not on
every iteration, and reverted candidates are not profiled. You can also profile on demand yourself with
`local_knowledge/common_methodology/profiling/rocpc_profile.py` (see
`common_methodology/profiling/measure_rocpc_workflow.md`). This card explains how to read that
profiling output and what lever each signal points to. Hardware peaks live in `local_knowledge/hardware/`.

## 1. Wall time & basic trace
```bash
# kernel-level timing + stats
rocprofv3 --kernel-trace --stats -f csv -- python test_driver.py --profile-run
# isolate your kernel by name (rocprofv3 also records torch/library/runtime dispatches)
```
Bench discipline: warmup ≥ several hundred iters, report **median** of ≥3 reps; a change must beat the
current best by more than run-to-run jitter (~2% noise floor) to count as real.

## 2. Classify the bottleneck from PMC counters
| Counter (rocprofv3) | Reads as | Lever |
|---|---|---|
| `VALUBusy` high, `MFMABusy` low | VALU/address-bound | vectorize, reduce index math, buffer descriptors |
| `MFMABusy` high, gaps between MFMA | matrix core starved / `v_accvgpr_*` | tied accumulator, sched_group_barrier, double-buffer |
| `MFMABusy` near peak | compute-bound | you're near roofline — only bigger tiles / better dtype help |
| LDS bank-conflict counters high | LDS-bound | pad `+1` / XOR-swizzle, `ds_*_b128` |
| `s_waitcnt vmcnt(0)` stalls before MFMA | global-load latency exposed | prefetch/async-copy overlap, larger tile_k |
| L2 / Infinity Cache hit rate low | poor locality | XCD-aware grid swizzle, tile ordering |
| HBM BW near ~5.3 TB/s (MI300X) | memory-bound at roofline | reduce bytes moved (dtype, fusion, reuse) |

## 3. Roofline
```
Arithmetic Intensity = FLOPs / bytes_moved
AI < crossover  → memory-bound  (optimize bandwidth: vectorize, coalesce, cache reuse)
AI > crossover  → compute-bound (optimize MFMA: utilization, dtype, tile size)
```
Practical rule for GEMM-like ops: small M (≤ 512) tends memory-bound; large M compute-bound.

## 4. Tie PMC → HIP lever (decision, not prescription)
The forge-loop passes the PMC *finding* (memory / compute / spill) as a neutral fact; you choose the
technique. The mapping above is the menu:
- memory-bound → [../optimize/hip_levers/hip_lds_staging.md](../optimize/hip_levers/hip_lds_staging.md),
  [../optimize/hip_levers/hip_templates.md](../optimize/hip_levers/hip_templates.md) (vectorize / coalesce / async copy)
- compute-bound → [../optimize/hip_levers/hip_builtins.md](../optimize/hip_levers/hip_builtins.md) (MFMA
  shape, sched_group_barrier), [../optimize/hip_levers/hipkittens.md](../optimize/hip_levers/hipkittens.md) (scheduling patterns)
- register spill → [../optimize/hip_levers/hip_traps.md](../optimize/hip_levers/hip_traps.md) (occupancy,
  `__launch_bounds__`)

## 5. Always cross-check with ISA
PMC tells you *what* is slow; the ISA tells you *why*. Confirm the inner loop with `--save-temps`
against the checklist in [../bottleneck/debug-hip-kernel.md](../bottleneck/debug-hip-kernel.md) §7
before and after a change — a "win" that doesn't change the ISA in the expected way is usually noise.

## Sources
- rocprofv3 / rocprof-compute (omniperf): https://rocm.docs.amd.com/projects/omniperf/en/amd-staging/what-is-rocprof-compute.html
- MI300X workload optimization (roofline, BW, occupancy): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Matrix Core counters: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
