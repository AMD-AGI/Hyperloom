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

# Radeon PRO W7900 (gfx1100)

KernelForge was built on AMD Instinct (CDNA) GPUs. A Radeon PRO W7900 is an RDNA 3 board, ISA gfx1100, and
`forge-loop` runs on it with the differences below. Everything else — the driver contract, the SNR gate, the
benchmark and the keep-or-revert decision — is architecture-neutral and unchanged.

## What differs from an Instinct run

| | Instinct (gfx942 / gfx950) | W7900 (gfx1100) |
|:--|:--|:--|
| Kernel backends | all ten | `triton`, `hip`, `hipblaslt`, `ck` |
| Hardware knowledge the agents read | `hardware/` (MI350) | `hardware/rdna3/` |
| Matrix instructions | MFMA, AGPR accumulators | WMMA, accumulators in VGPRs |
| Default wave size | wave64 | wave32 for HIP; check a Triton kernel's dump |
| Analysis Agent | profiled on long runs | static-only: rocprof-compute does not support gfx1100 |
| Experience-KB records | `mi300x` / `mi355x` | `w7900` |

- A fresh campaign that names `aiter`, `flydsl`, `assembly`, `gluon`, `fusion` or `ave` on gfx1100 is refused
  before any GPU work, with the list of backends that do build for it. The backend is also inferred from the
  kernel source when `--kernel-backend` is omitted, so the same refusal applies to an inferred one.
- Every agent prompt on gfx1100 carries a short architecture note at the head of its knowledge block. It maps
  the Instinct vocabulary of the backend guidance (MFMA, AGPR, wave64, FP8, HBM) onto RDNA 3 and is inlined even
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

A workstation board raises and lowers its clocks with load and temperature far more than an Instinct part,
and the keep decision asks a candidate to beat the incumbent by as little as 0.1%. Navi 31 does not support
performance determinism (`rocm-smi --setperfdeterminism` fails), so the closest setting is to hold the
performance level high for the length of the campaign:

```bash
sudo rocm-smi --setperflevel high
```

Restore automatic clock management afterwards with `sudo rocm-smi --setperflevel auto`. That keeps clocks high
but not fixed, so expect more run-to-run spread than on Instinct; `--bench-repeat` above 1 takes the per-case
median of several in-process repeats and narrows it.

## Not covered on gfx1100

- `forge-fuse`, `kernel-rewrite-controller` and `gemm-tune` drive vLLM, SGLang or AITER, none of which this
  change validates on gfx1100.
- Hardware-counter profiling: the profiled Analysis tier and the implementer's self-profiling script run on
  rocprof-compute. On gfx1100 the agents can still run `rocprofv3` themselves for kernel traces and the
  counters it lists for the device; the `hardware/rdna3/` cards say which ones answer which question.
