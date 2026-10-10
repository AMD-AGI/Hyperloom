<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Experimental Halo (gfx1151) custom workloads

Hyperloom recognizes the Radeon 8065S as `radeon8065s`, with ISA `gfx1151`
and 40 compute units. This is initial platform identification and custom-workload
bring-up, validated on a Ryzen AI Max+ PRO 495 with ROCm 10 and AMD PyTorch 2.13.
It does not establish support for every Halo SKU or a complete kernel optimizer.

Pass `--gpu-type radeon8065s` when HIP reports only `AMD Radeon Graphics`.
The ISA alone cannot distinguish boards or their compute-unit counts, so generic
`gfx1151` does not automatically select a board. Explicit Radeon product names
are recognized from rocm-smi or PyTorch device properties.

## Smoke test

Use an existing virtual environment containing a ROCm PyTorch build compatible
with gfx1151. From the repository root:

```bash
source .venv/bin/activate
RESULT_DIR=/tmp/hyperloom-gfx1151-smoke bash examples/gfx1151-smoke/custom_radeon8065s.sh
```

The benchmark checks GPU FP16 attention against an independent CPU FP32
reference before emitting `inferencex_result.json`. It requires no model download
or LLM credentials. Its throughput unit is attention calls per second, not LLM
tokens per second. HIP's `multi_processor_count` is preserved as
`hip_multiprocessors`; it must not be interpreted as the board's CU count.

## Custom workload integration

Use `--framework custom`, `HYPERLOOM_BENCHMARK_BACKEND=bypass`, and a dedicated
workload checkout as described in the [custom workload guide](../../docs/how-to/optimize-custom-workload.md).
Supply `custom_radeon8065s.sh` through `--benchmark-scripts-dir` and keep TP=1.
Set `INFERENCE_OPTIMIZER_RAY_EXEC=0` when Ray does not discover the APU GPU.

No Magpie serving runner or Radeon peak-performance constants are introduced.
Instinct serving recipes, profiler hotfixes, and kernel optimization paths need
separate platform validation. In particular, do not apply the ROCm 7.2 bare-metal
framework installer to an existing ROCm 10 environment.

A local Qwen3-0.6B BF16 custom workload completed baseline and configuration
benchmarks. A 15-minute source/kernel trial reached its time limit without an
accepted change; the GEAK kernel lane did not execute because its phase budget
was exhausted. These trials establish benchmark execution, not an optimization
gain or validated end-to-end kernel rewriting on Halo.
