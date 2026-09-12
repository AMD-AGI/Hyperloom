---
title: Kimi-K3 A8W4 MoE stage1 - Forge ASM and model E2E
kind: case
gens: [gfx950]
status: GPU-validated Forge KEEP; small E2E reduction repeated on one workload
updated: 2026-09-12
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Screen a real MoE hotspot without changing its fused computation

Use this case for AITER's FlyDSL MXFP4-weight / FP8-activation MoE stage1
with fused SiTUv2 and FP8 output quantization. This is different from the
[signed INT4 / BF16 case](aiter_w4a16_roundtrip_gfx950.md).

## Select the execution path from the model trace

A complete five-step Kimi-K3 decode trace on eight MI355X GPUs contained
460 calls on one TP rank to
`mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_xcd4_situv2_v33`:
92 calls per step, mean 36.675 us, and 12.2% of summed kernel duration in that
trace window. Summed kernel time is not model latency attribution.

The implementation is AITER's
`aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py`, reached through
`compile_flydsl_moe_stage1` and `_flydsl_moe_stage1_impl` in
`aiter/ops/flydsl/moe_kernels.py`. It is a dynamically compiled FlyDSL kernel,
despite the installation also containing many prebuilt MoE code objects.
The inspected AITER checkout was commit
`4ad99832823dde2315b361cbd3b54b1c5c12acd5`; the common kernel file SHA-256 was
`7c43585c51bfee5506515d0efec908611c93ac8d0583addb437296d442a91a91`.

The isolated specialization used 16 tokens, 896 experts, topk 16, model
dimension 3584, local intermediate dimension 384, tile 32x128x256,
`persist_m=1`, `waves_per_eu=4`, `b_nt=2`, `xcd_swizzle=4`, interleaved gate/up,
and asynchronous copies. It has no bias or stage1 routing-weight multiply.
SiTUv2 beta 4 and linear beta 25 remain runtime arguments.
These dimensions match the model configuration and tuned dispatch row;
synthetic routing does not reproduce the model's actual routing distribution.

## Preserve the scale layout and launch contract

The host callable accepts eleven pointers, four integers, five floats and
a stream. The GPU metadata describes a 124-byte kernarg segment; the stream
is a host launch parameter. Do not treat the Python argument tuple as a raw
HIP argument ABI.

Prepare packed MXFP4 weights and their scales with
`shuffle_weight_a16w4(..., 16, True)` and
`shuffle_scale_a16w4(..., E, True)`. Generate routing with `moe_sorting`.
After FP8 group-32 activation quantization, pass the activation scale through
`mxfp4_moe_sort_fwd` with the sorted IDs and valid count before launching
stage1. A plain token-major scale tensor has the wrong layout and allocation
size: the first invalid harness produced nonrepeatable results even with the
original source kernel. That is a harness failure, not an ASM discrepancy.

For the independent FP32 reference, keep the original unshuffled packed
weight and token-major scale. `torch_moe_stage1` derives intermediate size
from both packed weight shapes, including the factor of two for MXFP4.
Dequantize the output using the actual sorted E8M0 scale. Raw FP8 codes alone
cannot be compared across quantizers that choose different scale exponents.
The screen's source-to-FP32 relative L2 error was 0.02961 against a predefined
0.05 screening limit. Candidate acceptance additionally required byte-exact
equality to the source for both FP8 output and the raw sorted scale buffer.
This screen is not a replacement for a model's existing acceptance suite.

## Verified compiler roundtrip on FlyDSL 0.3.2

The experiment used FlyDSL 0.3.2, ROCm 7.2 and SGLang 0.5.19. The existing
Forge `with_assembly` adapter worked for this self-contained artifact without
changing the installed FlyDSL compiler or the model server.

The ISA dump spelled its target `amdgcn-amd-amdhsa-unknown-gfx950`. The
experiment canonicalized only the unspecified environment spelling to
`amdgcn-amd-amdhsa--gfx950`, in the target directive and metadata. It did not
remove a GPU feature. After Forge assembly/linking, the original and rebuilt
`.text` and `.rodata` sections were byte-identical. The `.text` SHA-256 was
`e5ce8182c734eceb1d574d429dc54408caa79743e60094adf30a7cad5352f533`.
This is evidence for one specialization, not blanket compatibility with all
FlyDSL 0.3.2 artifacts or target formats.

The roundtrip passed output/scale byte comparison, original repeatability,
nondefault-stream execution and graph replay. Candidates also passed changed
activations/scales, dispersed and concentrated routes, and runtime beta pairs
1/1, 4/25 and 8/40. Preserve the raw output-scale allocation when initializing
and comparing outputs; a wrapper can return a rearranged scale copy.
An invalid-instruction probe propagated the assembler error. A separate
zero-output probe changed 77484 output bytes and failed the comparison;
the independently retained source and unedited candidate still agreed.
The probe must affect active output lanes: changing only the second half of
a 32-row tile did not affect this 16-token case.

## Five instruction candidates, no stable performance winner

All candidates retained the same frontend source, tiling, launcher and fused
computation. This was direct screening through Forge's assembler/adapter,
not an agent `forge-loop` campaign or a statistical KEEP decision.

| Candidate | Repeated graph calls | After 512 MiB eviction |
| --- | ---: | ---: |
| Original FlyDSL | 52.715 us | 75.401 us |
| Unedited assembly roundtrip | 52.792 us | 75.561 us |
| Guarded integer divide simplification | 53.023 us | 75.261 us |
| Remove `s_setprio` transitions | 52.711 us | 75.501 us |
| Raise active priority from 1 to 2 | 52.791 us | 75.401 us |
| Remove `nt` from weight loads | 53.868 us | 117.401 us |
| Divide simplification plus no priority transitions | 52.927 us | 75.241 us |

Seven randomized-order rounds were used. Warm samples timed 48 calls per
graph; eviction samples used ordinary GPU events around GEMM after a queued
512 MiB buffer fill. The eviction and input preparation were outside the
event bracket. These two timing modes are distinct experiments. Neither is
the model's measured 36.675 us, and their absolute values must not be combined
with the trace to estimate E2E savings. ROCm rejected external timing events
inside graph capture; ordinary events outside capture were used instead.

Routing holdouts used pools of 16, 64, 128 and 224 experts, with actual unique
counts 16, 64, 114 and 153. At 114 unique experts, removing `nt` reduced warm
latency from 42.036 to 40.189 us, but increased eviction latency from 51.960 to
72.241 us. A warm-only improvement of about 4.4% would hide a 39% regression
in the other condition. Retain the original cache policy for model integration
until actual routing and cache reuse demonstrate otherwise.

The assembly declares 134 VGPRs, 33792 bytes of LDS, and zero private scratch
or spills. A bounded liveness inspection found no simple unused contiguous
range for renaming the high registers into 128 VGPRs. Reducing allocation
would require a new load-scheduling/liveness analysis; merely lowering the
descriptor is invalid. Static resource counts do not establish the active
occupancy or bandwidth bottleneck.

In that initial screen, no candidate was installed into the model and no MoE
E2E gain was measured.
The next useful evidence is the model's actual routing plus memory/occupancy
counters. Preserve synchronization when considering a new load schedule:
moving VMEM instructions also changes the meaning of outstanding-count waits.

## Actual Forge campaign: reduce allocation through verified liveness changes

A later real `forge-loop --kernel-backend assembly` campaign used PR revision
`ad99f06be07fe51033b6ca8902171bc0bc1d3553`, Codex / `gpt-5.6-sol`, and no
fallback provider. Its verified PORT used Forge `assemble` and `HipKernel` for
standalone loading. After PORT, the driver, fixture, frontend source, launch
glue, grid, block size and ABI were frozen. Only `kernel.s` changed.

The retained candidate reduced 134 VGPRs to 128 by shortening live ranges,
not by lowering the descriptor alone. It placed transient weight loads in
`v2:v5`, delayed one `ds_read_b128` so `v62:v65` could temporarily hold another
weight fragment, and reconstructed four address temporaries before their
next uses. Three reconstruction instructions occupied existing dependency
latency slots. Arithmetic, cache policy, LDS size and synchronization semantics
were retained; all register references and allocation metadata were updated.

HIP's theoretical occupancy query reported three 256-thread blocks per CU for
the seed and four for the candidate: 12 versus 16 waves. This is a residency
upper bound, not measured active occupancy or proof of a bottleneck.

The PORT commit was `72678ac9d7834cb698b65aa751ac972c0aeac742`; the actual Forge
KEEP was `a945c8bbf0378e721f96d7147c6982126eb5235c`. The source-relative mean-case
score was 1.052457x, versus 1.045174x for PORT. The incremental search score
was therefore only 1.006968x. Do not attribute the entire 5.2% source-relative
score to instruction changes.

Independent validation covered 48 hidden inputs (tokens 2/3/7/9/15/16, expert
pools 16/64/224/896, two seeds and additional runtime SiTUv2 parameters),
changed-input nondefault-stream graph replay, and the exact source/seed/candidate
serving bridge. Outputs and raw scales were byte exact. Thirteen randomized
timing rounds showed that routing changes which resource regime benefits:

| Tokens / expert pool | Warm PORT -> candidate | Evicted PORT -> candidate |
| --- | ---: | ---: |
| 8 / 16 | 20.520 -> 20.694 us | 25.881 -> 25.920 us |
| 16 / 16 | 21.121 -> 21.252 us | 26.120 -> 26.200 us |
| 16 / 64 | 40.451 -> 38.861 us | 45.201 -> 43.440 us |
| 16 / 224 | 43.672 -> 41.968 us | 60.681 -> 59.681 us |
| 16 / 896 | 52.104 -> 51.989 us | 73.921 -> 73.321 us |

For token 16, pools 64 and 224 improved in all 13 paired rounds in both cache
conditions. The concentrated pool-16 warm case regressed in all rounds.
An expert pool is a synthetic routing generator parameter; it is not the
observed number of active experts in a model request. Measure actual E2E
before treating any one routing result as representative.

The tested replacement domain is only 1-16 tokens. A 64-token concentrated
preflight showed source repeatability differences whose cause was not resolved;
it is excluded rather than silently relaxed. The standalone workspace has no
arena `config.yaml` or `baseline_perf.yaml`, so its full custom driver and hidden
tests are not arena canonical certification.

Pass the current capture stream explicitly to the original FlyDSL callable
when recording reference graphs. An earlier setup passed a stale default
stream and timed an empty graph. Reject empty-graph warnings, clear outputs,
replay and verify both output and scales before accepting timing data.

Evidence: `/shared_nfs/chenyi/forge-neha-repro-20260911/moe-forge-r3b`.
The candidate ASM SHA-256 is
`1560bf7b471d08df69e5b325611348c2e8c4dbb1a3ed44de9240dc70aa889690`;
its independently assembled code object SHA-256 is
`cc59cf7b4d8263003560aa3b1024bbff7ab3e17eac8b628b381005fb1ccbff5c`.

## Full-model measurements: a small workload-specific reduction

The actual standalone candidate was installed through the fixed HIP launcher
in Kimi-K3 on SGLang TP8, eight MI355X GPUs. Every measured server recorded the
actual loaded image hash on all eight workers; short GPU traces verified target
dispatch. No microbenchmark overlapped model inference, and the AITER source
was restored after each trial. The measured Forge revision is pinned above;
later PR documentation or main rebases are not the measured campaign revision.

The first workload used sixteen 1024-token inputs differing at only one token,
512 output tokens, and five repeats per batch. Batch 1 was effectively flat.
At batch 16, PORT-before / candidate / PORT-after medians were 12.673587 /
12.704876 / 12.787473 seconds: no stable additional E2E gain. Retain that result.

A separate fixed workload used sixteen diverse topics/languages, the same
1024/512 lengths, batch 16 and eight repetitions per server. Sampling was
greedy with ignore-EOS and no prefix cache. Every timed sample was retained.

| Trial | First server median | Middle server median | Last server median |
| --- | ---: | ---: | ---: |
| PORT / candidate / PORT | 13.874957 s | 13.831462 s | 13.874669 s |
| Independent candidate / PORT / candidate | 13.820021 s | 13.879234 s | 13.825247 s |

The exploratory candidate reduced latency by 0.313% and 0.311% against its two
controls. A new reverse-order process-level repeat reproduced reductions of
0.427% and 0.389%; mean latency improved as well. This establishes a small
observed serving benefit for this fixed workload on this node, not a general
Kimi-K3 improvement. Actual model routing counts were not recorded, so do not
assert that its expert distribution equals a synthetic holdout pool.

The exploratory effect did not meet the initial 0.5% prioritization threshold.
Both controls agreed and every candidate sample was below both control ranges,
which motivated a separately recorded small-signal protocol before the reverse
repeat. Report that change and the small effect; do not claim the original
threshold passed or omit the first workload's negative result.

Complete generated token sequences differed even between unchanged PORT
repetitions on the diverse prompts. The isolated operator's outputs and raw
scales were byte exact, but these serving runs do not certify full-model
determinism or model quality. The result remains bounded by the fixed prompt
set, runtime versions, hardware and numerical checks described here.
