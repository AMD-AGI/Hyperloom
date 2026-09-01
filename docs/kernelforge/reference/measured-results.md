<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Measured results

Campaign results from forge's standalone period, kept because they are
measurements: each number came off an MI355X and is what the loop's own
KEEP/REVERT gate accepted. They are a record of what forge achieved on these
kernels, not a promise about yours.

The launch note these tables were extracted from described a repository that no
longer exists, so its architecture and scale sections went with it -- see
{doc}`Architecture </kernelforge/conceptual/architecture>` for the current
shape.

## Sparse linear attention -- CK backend

Full forward + backward attention kernel, written from scratch to beat the
published Triton autotuned baseline. `B=1 H=24 S=65536 D=128`, sparsity 0.90.

| Config | Stage | Triton (published) | CK (forge) | Speedup |
|---|---|---|---|---|
| B (`BLKQ=64`) | forward | 13.29 ms | 11.06 ms | 1.20x |
| B (`BLKQ=64`) | backward | 47.56 ms | 33.79 ms | 1.41x |
| B (`BLKQ=64`) | **total** | 60.85 ms | 44.97 ms | **1.35x** |
| C (`BLKQ=128`) | forward | 11.41 ms | 8.94 ms | 1.28x |
| C (`BLKQ=128`) | backward | 46.71 ms | 33.84 ms | 1.38x |
| C (`BLKQ=128`) | **total** | 58.12 ms | 42.68 ms | **1.33x** |

The backward kernel took six optimization phases -- split pipeline, constexpr
masks, bf16 delta preprocessing, occupancy-2 for both the dkdv and dq
sub-kernels -- each one validated against hardware counters. VGPR count went
296 -> 238 with zero spills, which is what moved occupancy from 1 to 2.

## Kimi-K2 MoE mxfp4 inference

FlyDSL replacement for the CK mxfp4 MoE kernels, across every decode and
prefill shape. `TP=4 E=384 topk=8 hidden=7168`.

| Tokens | CK baseline | FlyDSL (forge) | Speedup |
|---|---|---|---|
| 64 (decode) | 287 us | 268 us | 1.08x |
| 256 | | | 1.06x |
| 512 | | | 1.07x |
| 1024 | | | 1.18x |
| 2048 | 823 us | 620 us | **1.33x** (peak) |
| 4096 | | | 1.22x |
| 8192 | 2159 us | 1745 us | 1.24x |

Absolute savings scale with sequence length: 414 us per layer at 8K tokens,
which comes straight off time-to-first-token for long prompts. Steady state
above 4096 tokens is ~1.22x; the remaining gap is the MFMA scheduler.

## Sage attention sparse forward

A fully autonomous campaign -- agents ran overnight on a SLURM cluster with no
human in the loop.

- The Triton campaign reached **1.15x** (1169 -> 1363 TFLOPS).
- Hardware-counter evidence put the ceiling at HBM bandwidth and named CK int8
  as the path past it.
- The follow-on CK campaign started from a complete int8 integration package
  (7 of 7 pieces).
