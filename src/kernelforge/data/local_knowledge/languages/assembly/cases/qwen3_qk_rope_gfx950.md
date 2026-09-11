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
Forge example, not Neha/Evolve source. The initial fusion and numerical repair
were authored manually; the later Forge campaign changed only assembly
scheduling, as recorded below.

## Runnable source and measured scope

Historical source at Hyperloom commit `a7f41272b0b2ee8a312e94ef1132d9f3543281e0`,
under `examples/qwen3-qk-assembly/`, contains the following files. This model
integration is outside the minimal PORT example shipped by this PR:

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

FP64 RMS reduction and normalization convert through FP32 to BF16 before multiplying the norm weight, then
rounds the weighted values and each RoPE product/addition. This models eager
BF16 intermediates; compiled native math may fuse operations and eliminate
some intermediate rounding. Preserve the experiment's numerical contract
when comparing candidates rather than silently treating the two as bitwise
equivalent.

The source keeps dependency spacing around DPP chains/readlane and `v_rsq_f64`.
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

## Forge agent campaign after the numerical repair

Campaign `27bc4ff7` used the PR's curated Assembly knowledge with
`gpt-5.6-sol`, `--kernel-backend assembly`, and the protected eight-case driver.
The assembly file was the optimization anchor; the fixed fusion, ABI, launch
geometry, Python launcher, driver, and reference were not editable. External
experience-KB warm starts were disabled. This is a real agent campaign, separate
from the manual implementation and from deterministic replay/fault injection.

The agent's KEEP (`67abbe957e76b2b32cca1aea8cbd896e02b20f11` in the campaign
workspace) replaces the initial full VMEM wait with `vmcnt(4)`, releasing the
two activation loads first. It then uses `vmcnt(2)` before unpacking weights and
`vmcnt(0)` before unpacking cosine/sine data. The deferred unpack instructions
occupy two existing reduction dependency gaps, with residual `s_nop 2` spacing.
It retains the arithmetic, reduction order, registers, and launch shape.
These counts follow this exact six-load stream; do not copy them into a kernel
with different outstanding memory operations.

The three Forge score measurements were 1.010365x, 1.009364x, and 1.007566x;
their mean, 1.009099x, cleared that run's noise gate of 1.002391x. This is a
0.91% equal-case kernel score gain, not a 3% usefulness result or an E2E gain.
The published baseline and candidate wall means both rounded to 0.0058 ms,
so Forge retained `total_improved=false` and its aggregate-contradiction message.
Preserve that limitation rather than overriding it with the KEEP label.

The one-hour budget completed one KEEP. A second round reached planning but
was refused before implementation because it no longer fit the remaining
budget. Planning consumed 32.7 minutes of the 43.9-minute run, with two
specialist timeouts. This is not evidence that the search exhausted useful
instruction changes. Also, this fixture did not ship arena `config.yaml` or
`baseline_perf.yaml`: fixed-driver validation passed, while arena acceptance
and its golden performance anchor remained unverified.

Both the native Forge and Hyperloom export patches applied in clean clones,
reproduced the selected source bytes, and passed the unchanged driver. An
independent 24-input holdout set (four seeds, six shapes, including 31/65/129/
513/2048 tokens and zero/small/large scales) passed for the candidate, repaired
baseline, and corrected Triton control. A deliberate early-return assembly
fault went through the real loop, failed the oracle, skipped benchmarking,
and reverted to the exact selected source without changing the driver. That
negative control is not another LLM optimization iteration.

Eleven alternating graph rounds, 100 calls per replay, measured the following
full-precision medians. All samples, including large timing outliers, were
retained; small differences should not be treated as uniform improvements.

| Tokens | Repaired assembly baseline us | Agent assembly us | FP64 fused Triton us |
| ---: | ---: | ---: | ---: |
| 1 | 1.808 | 1.758 | 1.892 |
| 8 | 1.833 | 1.781 | 1.923 |
| 32 | 1.943 | 1.958 | 1.939 |
| 64 | 2.202 | 2.178 | 2.163 |
| 512 | 6.972 | 6.739 | 6.937 |
| 2048 | 21.349 | 21.361 | 21.395 |

This independent equal-case ratio is 1.0159x, with regressions/parity on some
shapes. It does not establish a model-serving benefit from the agent edit.
The Triton control uses FP64 normalization and explicit FP32-to-BF16 rounding;
the historical FP32 control below is a different experiment.

## Four-way serving and model-quality screen

The selected candidate, manual repaired baseline, matched FP64 Triton control,
and native vLLM each ran in three independently started servers, with five
rounds per workload: 5,280 timed streaming requests, all completed without a
server retry. The order was native/Triton/baseline/candidate,
candidate/baseline/Triton/native, then Triton/native/candidate/baseline.
The additional manual-baseline control was fixed before any serving results.
All variants used TP=1, max model length and batched tokens 2048, max sequences
64, prefix caching disabled, fixed greedy output lengths, and isolated caches.
Separate traces confirmed 288 calls to the expected assembly or Triton symbol
for each profiled eight-token request. Final Assembly regression: 141 passed.

The primary statistic was fixed in advance: median of five rounds per server,
then mean across three servers. Throughput changes below use that statistic.

| Concurrency, input/output tokens | Native tok/s | Candidate tok/s | vs native | vs repaired assembly | vs fused Triton |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32, 128/128 | 6941.88 | 7222.21 | +4.04% | +0.94% | +0.20% |
| 8, 1024/32 | 1022.98 | 1040.13 | +1.68% | +0.45% | -0.35% |
| 1, 128/128 | 336.95 | 347.28 | +3.06% | +0.56% | -0.19% |

The candidate/manual-baseline changes in the three comparison blocks were
+0.63%/+2.48%/-0.27% for decode, -0.08%/+1.16%/+0.28% for longer input, and
+0.45%/+0.55%/+0.66% for single requests. This suggests a small scheduling
benefit in some conditions; it does not establish a uniform or 3% E2E gain.
Native/candidate TTFT was 50.53/54.24, 68.26/71.03, and 18.86/20.88 ms.

Second-scale latency outliers occurred in several implementations and were
retained. They materially change results when every round's elapsed time is
included. This descriptive pooled statistic is total output tokens divided by
total measured round time, without selecting or dropping rounds:

| Concurrency, input/output tokens | Native tok/s | Repaired assembly tok/s | Candidate tok/s | Fused Triton tok/s |
| --- | ---: | ---: | ---: | ---: |
| 32, 128/128 | 6162.28 | 7154.51 | 6179.77 | 6359.98 |
| 8, 1024/32 | 779.39 | 1034.37 | 1039.74 | 1043.12 |
| 1, 128/128 | 322.79 | 318.84 | 306.81 | 336.67 |

For single requests, pooled candidate throughput regresses against native and
the manual baseline; candidate p99 request latency is 2.102 s versus 0.387 s
native. The long-tail cause has not been attributed to a kernel or host
component. Median gains alone therefore do not establish stable serving or
tail-latency improvement. Investigate the stalls with time-correlated host/GPU
evidence before claiming a production E2E win; do not rerun only slow rounds.

Quality was measured outside timing on the first server of each variant, using
WikiText-2 raw test (`Salesforce/wikitext`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`), 128 nonoverlapping 1024-token
chunks, and 130,944 scored teacher-forced tokens. Dataset selection and the
screening rule (mean NLL increase no greater than 0.01 nats/token versus native)
were fixed before evaluation. This is an experimental screen, not a production
quality SLA, task-accuracy benchmark, or full-dataset perplexity result.

| Implementation | Mean NLL | Perplexity | NLL increase vs native |
| --- | ---: | ---: | ---: |
| Native | 2.690566 | 14.740020 | 0 |
| Repaired assembly baseline | 2.691358 | 14.751701 | 0.000792 |
| Forge candidate | 2.691293 | 14.750743 | 0.000727 |
| Matched Triton | 2.691230 | 14.749812 | 0.000664 |

All variants pass that limited screen. Candidate/native leading-token agreement
from the returned prompt log probabilities was 129,305/130,944 (98.75%); the
maximum absolute token-NLL difference was 3.4805 nats. A small mean difference
does not imply every token distribution is close or generated text is identical.

## Historical FP32 results and attribution

The following measurements used commit `5fb681a9e`, before the FP64 numerical
repair described below. They remain evidence for those tested inputs and source
bytes; do not use them as acceptance or performance evidence for the repaired
kernel or a later agent candidate.

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

An expanded fixed driver exposed a rounding-boundary failure in the FP32
implementation: seed 910, 2048 tokens, BF16 random QKV scaled by 0.5 yielded two
Q elements outside the unchanged `rtol=0.01, atol=0.01`. At `[156, 2608]`, assembly
returned 0.47265625 against the FP64 oracle's 0.48828125; at `[202, 2160]`, it
returned -0.30859375 against -0.29296875. The second discrepancy also occurred in
the FP32 same-fusion Triton control. Prior seeds passing did not establish
coverage of this boundary.

The repaired baseline uses FP64 sum-of-squares, reduction, reciprocal square
root, and normalization multiplication, retaining the BF16 intermediate round
points. DPP shuffles move both halves of each FP64 value. This was a manual
correctness repair before the Forge campaign, not an agent-discovered speedup.
The permanent GPU regression retains the failing seed, shape, scale, and
tolerance. Treat new precision, scheduling, and reduction edits as fresh
candidates requiring both fixed-driver and independent holdout validation.

Match intermediate conversions in a high-level control too: direct Triton
FP64-to-BF16 lowering uses a round-to-odd intermediate and is not the same as
the assembly/PyTorch reference's FP64-to-FP32-to-BF16 path. The direct conversion
failed two holdout coordinates despite FP64 accumulation. Explicit FP32
conversion before BF16 made the control pass the unchanged holdouts. Preserve
both versions and their diagnostics when attributing precision or speed changes;
more precise arithmetic alone does not imply the same rounding contract.

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
