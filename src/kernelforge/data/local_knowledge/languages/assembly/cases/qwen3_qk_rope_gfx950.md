---
title: Qwen3 Q/K RMSNorm and RoPE through standalone HIP assembly
kind: case
scope: languages/assembly
updated: 2026-09-09
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Qwen3 Q/K normalization and RoPE

Use this case when profiling shows separate small Q/K normalization and rotary
position kernels in a supported Qwen3 model. This is an independently developed
Forge example, not Neha/Evolve source and not a new LLM-discovered KEEP.

## Runnable source and measured scope

The repository's `examples/qwen3-qk-assembly/` contains:

- `forge_qwen3_assembly/qk_norm_rope.s`: complete handwritten AMDHSA source.
- `forge_qwen3_assembly/kernel.py`: shape/layout checks, assembly build, and
  `kernelforge.assembly.hip.HipKernel` with an explicit ABI.
- `forge_qwen3_assembly/__init__.py`: opt-in vLLM general plugin with a custom
  torch operation and a process-local Qwen3 attention wrapper.

Tested on Qwen3-4B snapshot `1cfa9a7208912126459214e8b04321603b3df60c`, BF16,
TP=1, MI355X/gfx950, ROCm 7.2.3, vLLM 0.25.1+rocm723, PyTorch
2.11.0+gitd0c8b1f, and Triton 3.6.0. The model has 36 layers, 32 query heads,
eight KV heads, head size 128, and packed QKV width 6144. Only this plugin's
matching configuration is replaced; other shapes retain the original forward.
Installed framework/library sources and model checkpoints were not edited.

## Kernel contract

One wave handles one token/head. The grid is `(tokens, q_heads + k_heads, 1)`
with a `(64, 1, 1)` block and no LDS. The kernel reads packed BF16 QKV and
int64 positions, independently normalizes Q/K per head, applies full-width
NeoX RoPE from the BF16 cache, and writes new Q/K tensors. V remains the
original view; projections and attention are unchanged.

The 72-byte kernarg segment contains seven pointers (`qkv`, `positions`,
`cache`, `qw`, `kw`, `qout`, `kout`), FP32 epsilon, and three int32 values
(`qheads`, `kheads`, QKV byte stride). Match this metadata when changing code.
Positions must be valid cache row indices. The wrapper requires tensor spans
below 4 GiB because relative offsets are uint32. Base-pointer additions carry
into the high half, including when a valid tensor crosses an absolute 4-GiB
address boundary; that does not imply support for a 4-GiB relative span.

FP32 RMS reduction rounds to BF16 before multiplying the norm weight, then
rounds the weighted values and each RoPE product/addition. This models eager
BF16 intermediates; compiled native math may fuse operations and eliminate
some intermediate rounding. Preserve the experiment's numerical contract
when comparing candidates rather than silently treating the two as bitwise
equivalent.

The source keeps dependency spacing around DPP chains/readlane and `v_rsq_f32`.
Earlier variants without sufficient spacing produced wrong results despite
successful assembly. Any scheduling edit needs new GPU correctness evidence;
an assembler accepting an instruction sequence does not establish safety.

## Dispatch evidence

Select the general plugin with `VLLM_PLUGINS=forge_qwen3_assembly` in every
worker and isolate compilation caches per candidate. Two initial scratch
runs launched the original kernels; changing a private cache alone did not
fix worker activation. They were excluded from assembly performance evidence.
GPU traces of both final packaged-code serving runs contain 288 calls to
`forge_qk_norm_rope_h128` for an eight-token profiling request. Profiling,
compilation, loading, and warmup are outside the timed requests.

## Measured results and attribution

Final serving validation used native, assembly, fused Triton, fused Triton,
assembly, native order. Each independently started server ran five rounds of
each workload; 2,640 measured streaming HTTP requests completed. The last
native run encountered a client `ServerDisconnectedError`; its incomplete
data was retained and replaced by one complete server run with the same
configuration. No kernel fault was logged, but the transport cause remains
unresolved. Valid timing outliers were retained. Each entry below is the mean
of two server medians over five rounds, not a pooled request percentile.

| Workload: concurrency, input/output tokens | Native output tok/s | Assembly output tok/s | Fused Triton output tok/s | Assembly vs native |
| --- | ---: | ---: | ---: | ---: |
| Decode: 32, 128/128 | 6945.45 | 7295.34 | 7244.48 | +5.04% |
| Longer input: 8, 1024/32 | 1022.85 | 1038.56 | 1042.79 | +1.54% |
| Single request: 1, 128/128 | 340.80 | 346.32 | 350.19 | +1.62% |

Assembly request latency was 557.13/239.41/370.61 ms versus native
585.20/242.76/375.55 ms. TTFT was 51.27/71.14/20.18 ms versus
51.10/69.61/17.87 ms: decode throughput gains do not establish a TTFT gain.
An earlier scratch-plugin A-B-B-A experiment measured +6.36%/+3.51%/+5.73%
throughput; the smaller final single-request/long-input gains show sensitivity
to run conditions. Two servers per implementation are not a broad confidence
interval or a production SLO result.

The same-fusion Triton control uses one wave per token/head and the same BF16
round points, with FP contraction disabled. Interleaved graph microbenchmarks
(nine rounds, 100 calls per replay) measured:

| Tokens | Assembly us | Fused Triton us |
| ---: | ---: | ---: |
| 1 | 1.779 | 1.936 |
| 8 | 1.772 | 1.912 |
| 32 | 1.882 | 1.953 |
| 64 | 2.109 | 2.164 |
| 512 | 6.578 | 6.654 |
| 2048 | 20.215 | 20.654 |

The fused control explains most of the model-level gain. Assembly is +0.70%,
-0.41%, and -1.11% in throughput relative to that control across the three
serving workloads. This does not establish a stable assembly-exclusive E2E
advantage, or superiority over an exhaustively tuned Triton kernel. At large
token counts the assembly can also regress against native staged compilation;
do not extrapolate small-batch ratios or sum microbenchmark gains across layers.

## Numerical validation and remaining limits

The GPU regression compares against independent FP64 math with explicit BF16
round points at fixed `rtol=0.01, atol=0.01`, checks multiple shapes/scales,
input preservation, rebinding, nondefault streams, graph replay, and rejection
of a deliberately wrong assembly edit. Additional experiments cover valid
allocations spanning an absolute 4-GiB boundary. These are experimental kernel
bounds, not permission to change a production driver's tolerances.

Whole generated sequences are not always identical. On 260 identical prefixes,
both packaged assembly runs chose the same next token as native at 258 positions;
the fused Triton runs agreed at 259. The differing selections involved tied or
nearby leading token probabilities. Shared top-five probability differences
had a maximum absolute value around 0.117, so they must not all be described as
negligible. This investigation is not a perplexity/task-quality certification.
Keep model-quality evaluation and the original numerical acceptance criteria
as separate requirements before promoting the experimental plugin.
