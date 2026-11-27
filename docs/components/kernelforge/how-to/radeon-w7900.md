---
myst:
  html_meta:
    "description": "Run KernelForge forge-loop campaigns on a Radeon PRO W7900 (RDNA 3, gfx1100): supported backends, what changes from Instinct, and how to measure."
    "keywords": "KernelForge, Radeon PRO W7900, RDNA 3, gfx1100, WMMA, wave32, forge-loop, Triton, HIP, hipBLASLt"
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# KernelForge on Radeon PRO W7900 (gfx1100)

KernelForge was built on AMD Instinct™ (CDNA) GPUs. A Radeon PRO W7900 is an RDNA 3 board, ISA gfx1100, and
`forge-loop` runs on it with the differences below. Everything else — the driver contract, the SNR gate, the
benchmark and the keep-or-revert decision — is architecture-neutral and unchanged. Everything here applies to
the W7900D as well: the same gfx1100 silicon at a lower boost clock (2156 MHz against 2500 MHz), registered
as its own board (`w7900d`) with its own peaks.

## What differs from an Instinct run

The following table summarizes the differences between running forge-loop on AMD Instinct and W7900 hardware.

| | AMD Instinct (gfx942 / gfx950) | W7900 (gfx1100) |
|---|---|---|
| Kernel backends | all ten | `triton`, `hip`, `hipblaslt`, `ck` |
| Hardware knowledge the agents read | `hardware/` (MI350) | `hardware/rdna3/` |
| Matrix instructions | MFMA, AGPR accumulators | WMMA, accumulators in VGPRs |
| Default wave size | wave64 | wave32 (HIP; Triton 3.8 measured) |
| Profiling on long runs | Profiled Analysis Agent; implementer self-profiles with rocprof-compute | Static-only Analysis Agent (rocprof-compute does not support gfx1100); implementer self-profiles with rocprofv3 |
| Experience-KB records | `mi300x` / `mi355x` | `w7900` (W7900 and W7900D) |

- A fresh campaign that names `aiter`, `flydsl`, `assembly`, `gluon`, `fusion`, or `ave` on gfx1100 is refused
  before any GPU work, with the list of backends that do build for it. The backend is also inferred from the
  kernel source when `--kernel-backend` is omitted, so the same refusal applies to an inferred one.
- Every agent prompt on gfx1100 carries a short architecture note at the head of its knowledge block. It maps
  the AMD Instinct vocabulary of the backend guidance (MFMA, AGPR, wave64, FP8, HBM) onto RDNA 3 and is inlined even
  when the knowledge maps themselves are deferred to pointers.
- The per-candidate register check disassembles for gfx1100 and reports VGPR and LDS against RDNA 3's occupancy
  steps instead of CDNA's.

## Run a campaign

The target is detected through `rocminfo`; pass it explicitly when the host has more than one GPU model.

```bash
kernelforge forge-loop \
  --workspace "$W" --kernel "$W/kernel.py" --driver "$W/driver.py" \
  --kernel-backend triton --gpu-target gfx1100 \
  --max-hours 1 --git-branch forge-w7900
```

The packaged Triton softmax example detects the target the same way and is the quickest end-to-end check:

```bash
GPU_TARGET=gfx1100 src/kernelforge/examples/triton-softmax-forge-loop/run_example.sh
```

## Measure on a steady clock

A workstation board raises and lowers its clocks with load and temperature far more than an AMD Instinct part,
and the keep decision asks a candidate to beat the incumbent by as little as 0.1%. Navi 31 does not support
performance determinism (`rocm-smi --setperfdeterminism` fails), so the closest setting is to hold the
performance level high for the length of the campaign:

```bash
sudo rocm-smi --setperflevel high
```

Restore automatic clock management afterwards with `sudo rocm-smi --setperflevel auto`. That keeps clocks high
but not fixed, so expect more run-to-run spread than on Instinct; `--bench-repeat` above 1 takes the per-case
median of several in-process repeats and narrows it.

How much this matters, measured: on a W7900D at the default performance level, the packaged Triton softmax (about
32 µs per launch) scored one candidate at 1.12×, 1.06× and 0.95× in its three independent benchmarks. The keep bar
rose to 1.14× and a correct 4% gain was reverted. For short kernels, raise `--bench-repeat`, use benchmark cases
that run for hundreds of microseconds, and budget at least two hours: the 30-minute finalize reserve and the
per-round admission leave a 1.5-hour campaign room for about one round.

## Not covered on gfx1100

The following features and tools are not supported on gfx1100.

- `forge-fuse`, `kernel-rewrite-controller` and `gemm-tune` drive vLLM, SGLang or AITER, none of which this
  change validates on gfx1100.
- The profiled Analysis tier: it runs on rocprof-compute, which does not support gfx1100, so the Analysis Agent
  is static-only there. On a long run the implementer's Self-profiling section points it at `rocprofv3` instead
  (kernel traces, and the counters `rocprofv3-avail` lists for the device).
