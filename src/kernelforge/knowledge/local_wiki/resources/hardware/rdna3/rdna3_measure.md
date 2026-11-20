---
title: RDNA 3 (gfx1100) — profiling, clock control and library support
kind: hardware
topic: measurement
gens: [gfx1100]
updated: 2026-11-16
---

# Profiling, clock control and library support

## What does not work here

- **rocprof-compute (Omniperf) does not support gfx1100.** Its only RDNA support is RDNA 3.5 (gfx1151). The
  `rocpc_profile.py` script in `common_methodology/profiling/` and every "wait/MFMA" ratio depend on it; neither
  applies on this chip. KernelForge keeps the Analysis Agent static on gfx1100 and gives the implementer a
  rocprofv3 Self-profiling section instead.

## rocprofv3

rocprofv3 collects counters and traces on gfx1100 (rocprofiler-SDK docs show `rocprofv3-avail` output for it).

```bash
rocprofv3-avail info -d 0                         # the counters THIS device and ROCm version expose
rocprofv3 --kernel-trace --stats -d out -- python driver.py --bench
rocprofv3 --pmc SQ_WAVES GRBM_GUI_ACTIVE -d out -- <command ...>
```

`<command ...>` launches the kernel under test and little else; choosing it is covered in
`common_methodology/profiling/measure_rocpc_workflow.md` (*The command to profile*).

- **List the counters before naming one.** Counter sets differ between CDNA and RDNA and between ROCm releases;
  a CDNA counter name copied from a methodology card may not exist here. Documented for gfx1100: `SQ_WAVES`,
  `SQ_WAIT_INST_LDS`, `GRBM_GUI_ACTIVE`, `GRBM_COUNT`, and the derived `GPU_UTIL` and `ALUStalledByLDS`.
- Kernel tracing (`--kernel-trace`) is the reliable first step: it gives per-kernel durations, which is enough to
  find the hot kernel and to check a speedup independently of the driver.
- Thread trace on RDNA covers a single SIMD; prefer counters and the ISA unless a stall pattern needs it.

## Reading a bottleneck without rocprof-compute

1. Compute arithmetic intensity from the kernel's FLOPs and bytes and compare with the ridge (≈ 142 FLOP/byte
   FP16 WMMA, ≈ 71 FP32): that decides memory- vs compute-bound for most kernels.
2. Achieved bandwidth = bytes moved ÷ kernel time; near 864 GB/s (or above, from Infinity Cache) means
   bandwidth-bound.
3. Occupancy from the register counts (`rdna3_execution.md`) and LDS per work-group (`rdna3_memory.md`).
4. The ISA: WMMA density in the hot loop, `v_dual_*` pairing, `s_waitcnt` placement before uses of loaded data.
5. `ALUStalledByLDS` high → LDS-bound: bank conflicts or too little ILP around LDS reads.

## Clock control for benchmarking

- **Performance determinism is not supported on Navi 31**: `amd-smi set --perf-determinism` returns
  `AMDSMI_STATUS_NOT_SUPPORTED`, so `rocm-smi --setperfdeterminism` fails too. (AMD SMI docs, tested on RX 7900
  XTX; the W7900 is the same die.)
- The performance level can be set: `sudo rocm-smi --setperflevel high` (or `amd-smi set -g 0 --perf-level
  HIGH`), restored with `--setperflevel auto`. That holds clocks high but not fixed, so expect more run-to-run
  spread than on an Instinct part; repeat measurements rather than trusting one.
- Measured: a 32 µs Triton softmax on a W7900D at the default (auto) performance level scored 1.12×, 1.06× and
  0.95× in three independent benchmarks of one candidate. That spread (σ ≈ 0.08) raised the keep bar to 1.14×, and
  a correct candidate with a 4% mean gain was reverted. On this board, for kernels this short, a gain under ~10%
  is not measurable without help: hold the performance level high, raise `--bench-repeat`, and prefer benchmark
  cases large enough that one launch runs for hundreds of microseconds.

## Library support on gfx1100

| Library | Status | Note |
|---|---|---|
| Triton (AMD backend) | supported | `tl.dot` lowers to WMMA in current releases; confirm `v_wmma_*` in the dump |
| HIP / hipcc | supported | `--offload-arch=gfx1100` |
| rocWMMA | supported | wave32 |
| hipBLASLt | in the build targets | confirm the installed build has gfx1100 kernels before relying on it |
| Composable Kernel | partial | WMMA-based instances exist for some ops; check the op you need |
| AITER | experimental | prebuilt wheels are gfx942/gfx950 only |
| FlyDSL | source only | an `rdna3` target exists; prebuilt wheels are CDNA only |
| Gluon | `amd.rdna3` namespace | has a WMMA op; most Gluon guidance and kernels are written for CDNA |
| HipKittens | CDNA only | |
