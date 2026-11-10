---
name: profiling-ck
description: >
  Profile Composable Kernel (CK) kernels: sweep instances with ckProfiler, read
  achieved TFLOP/s vs the ~615 reference and vs hipBLASLt/aiter, classify memory-
  vs compute-bound from rocprofv3 PMC, and tie each signal to a CK knob (block tile,
  KPerBlock, pipeline version/scheduler, MFMA tile, AK1/BK1, split-K). Use when
  deciding which CK instance/knob to change from measured evidence. Usage: /profiling-ck
allowed-tools: Read Bash Grep Glob
---

# Profiling CK kernels

Measurement-driven diagnosis for Composable Kernel. Hardware peaks live in `local_knowledge/hardware/`.

## 1. Sweep instances (classic) or validate (ck_tile)
```bash
ckProfiler gemm <layout,dtype,M,N,K,...>     # classic: prints every instance's TFLOP/s + GB/s
./bin/tile_example_universal_gemm -m=4096 -n=4096 -k=4096 -v=1   # ck_tile: built-in reference check
```
The top `ckProfiler` line is your pinned instance. Record its TFLOP/s + GB/s + the instance string, with
date — re-sweep after any CK/ROCm bump (instance IDs drift). Reference: bf16 4096³ RCR MI300X winning
instance ≈ **615 TFLOP/s** (256×256×64, 32×32, v3/Intrawave, #1727).

## 2. Classify the bottleneck from PMC
```bash
rocprofv3 --kernel-trace --stats -f csv -- <run>
```
| signal | reading | CK knob |
|---|---|---|
| `MFMABusy` near peak | compute-bound | near roofline — bigger tile/dtype only |
| `MFMABusy` with gaps | matrix core starved | pipeline `v3`→`v4`, Intrawave, `KPerBlock`↑ |
| `VALUBusy` high / low MFMA | address/overhead-bound | fewer transforms, wider loads |
| LDS bank-conflict counters | LDS-bound | XOR swizzle (`make_xor_transform`), tile shape |
| `s_waitcnt vmcnt(0)` stalls | global-load latency exposed | `AK1/BK1`≥128-bit, prefetch stages, async input |
| VGPR spill (`.private_segment_fixed_size>0`) | tile too big | shrink block tile / MFMA tile |
| few blocks, CUs idle (small M) | wave-quantization tail | split-K (`KBatch≥2`), smaller M tile, Interwave |
| HBM BW near roofline | memory-bound at peak | reduce bytes (dtype, fusion) |

## 3. Tune order (highest-leverage first)
Per [../optimize/ck_levers/ck_tuning_knobs.md](../optimize/ck_levers/ck_tuning_knobs.md): **block tile → KPerBlock → pipeline
version/scheduler → MFMA tile → wave map → load vector width**. Tune block tile + pipeline first;
everything else is second-order. Aim `ceil(M/MPerBlock)·ceil(N/NPerBlock) ≈ k·304`.

## 4. Cross-check & gate
- Compare the pinned CK instance against **hipBLASLt solidx** and the **aiter tuned config** at the same
  shape — CK is only worth pinning if it wins there.
- Confirm the ISA (K-loop `buffer_load_dwordx4`, no `scratch_`/`v_accvgpr`) — see
  [../bottleneck/debug-ck-kernel.md](../bottleneck/debug-ck-kernel.md) §7.
- Parity: fp32 accumulate; greedy temp=0 vs reference before pinning.
- If CK is consumed via aiter, e2e-gate through the aiter seam (`local_knowledge/framework/aiter/`), not just
  isolated `ckProfiler` TFLOP/s.

## Sources
- Optimizing with Composable Kernel (ckProfiler, instance selection): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
- rocprofv3 / rocprof-compute: https://rocm.docs.amd.com/projects/omniperf/en/amd-staging/what-is-rocprof-compute.html
- Reference 615 TFLOP/s instance (#1727): https://github.com/ROCm/composable_kernel/issues/1727
