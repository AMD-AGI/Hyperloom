---
title: AITER W4A16 MoE stage1 - verified FlyDSL assembly roundtrip
kind: case
gens: [gfx950]
status: GPU-validated roundtrip and tile/dispatch gain; no assembly-specific gain
updated: 2026-09-08
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# AITER W4A16: validate the launcher before tuning instructions

Use this case for `aiter/ops/flydsl/kernels/moe_gemm_2stage.py` with BF16
activations and signed INT4 weights (`in_dtype="int4_bf16"`, group size 32).
The FP4 route selects `mixed_moe_gemm_2stage.py` and has different layouts.

## Source and execution contract

Validated on one MI355X, ROCm 7.2.3, PyTorch 2.11.0+gitd0c8b1f,
FlyDSL 0.2.0, and `amd-aiter` 0.1.16.post2 from a vLLM 0.25.1 image.
The installed `moe_gemm_2stage.py` SHA-256 was
`099a3ef49da4eead946da90c8856d1350986b08c45461e4b2b79a3414b560275`.
Version numbers alone do not guarantee identical kernel source.

`compile_flydsl_moe_stage1` returns a FlyDSL launcher. Compile it with
`flyc.compile` and use [the adapter](../INDEX.md#flydsl-launcher-adapter) to
replace its code object. The inspected launcher takes nine pointers, four
runtime integers, and a stream. `_s1_args_std` packs those arguments using
typed pointers. Its output is `[tokens, topk, inter_dim]`, with fused
`SiLU(gate) * up` and optional routing-weight multiplication.

FlyDSL 0.2.0 defines `CallState` in `jit_function`; 0.2.4 imports it there
from `jit_executor`. Import through `jit_function` for both layouts. Preserve
the original call specification and artifact lifetime. Changing the global
FlyDSL installation or JIT cache is not required for assembly replacement.

## Build a trustworthy oracle

- Generate signed quantized weights in `[-8, 7]` with logical shape
  `[experts, 2*inter_dim, model_dim]`. Use AITER's
  `pack_int8_to_packed_int4(shuffle_weight(q, (16, 16)))`; reshuffle before
  packing. This is not adjacent-nibble packing of the original matrix.
- Start scales in `[experts, model_dim/32, 2*inter_dim]`. For BF16 scales,
  `shuffle_scale_for_int4` packs adjacent groups into dwords using
  `[E, G/2, N, 2]`. Use `moe_sorting` to generate the padded token/slot IDs,
  expert IDs, routing weights, and valid count.
- Compute the oracle from the original unpacked weights and scales. Round
  dequantized weights to BF16 before FP32 PyTorch matrix multiplication,
  SiLU, multiplication, and BF16 output: this kernel rounds its matrix
  operands to BF16. Nonuniform power-of-two scales make dequantization exact;
  ordinary random BF16 scales additionally exercise that rounding policy.
- Test the original kernel first, then require bitwise equality between it
  and the unedited assembly. Keep the independent oracle unchanged. A
  disposable edit reversing the sign of the SiLU exponential must fail it.
- Rebind all input/output pointers and token counts, including partially
  filled tiles and inactive experts, with random, balanced, and concentrated
  routes. Retain tensor ownership because the launch arguments are pointers.
  Check input preservation, reference/old/new candidate isolation, and
  nondefault-stream graph replay.

The optional regression is
`src/kernelforge/tests/test_assembly_aiter_moe_gpu.py`. It compiles the installed
kernel and covers token counts 1, 37, 65, 129, and 257. Its synthetic-oracle
tolerances are not permission to relax a campaign's protected driver.

## An instruction reduction without a useful latency reduction

Adjacent independent scalar FP32 multiplies sharing a routing weight can
sometimes become one `v_pk_mul_f32`. Preserve operand selection: broadcast
the routing weight to both results with `op_sel_hi:[1,0]`. Register pairs
must be 64-bit aligned; an odd pair such as `v[1:2]` is rejected by `llvm-mc`.
Do not renumber registers without a liveness analysis.

Two stage1 specializations with tile `[32,256,256]` were measured using
11 interleaved timing rounds, 100 kernel calls per graph, and 10 replays per
sample. Compilation, loading, routing, and the oracle were outside timing;
inputs were reused, so this measures warm repeated-kernel execution.

| Tokens / model dim / inter dim / experts / topk | FlyDSL | Reassembled | Packed multiply |
| --- | --- | --- | --- |
| 37 / 1024 / 256 / 4 / 2 | 28.930 us | 28.924 us | 28.919 us |
| 129 / 7168 / 2048 / 8 / 2 | 144.197 us | 144.164 us | 144.194 us |

Both edits passed bitwise comparison, including new inputs, but the change
was approximately +/-0.02% and is not a useful improvement. It did not meet
the experiment's 3% target; Forge's actual KEEP threshold accounts for
measurement noise. A fresh-process rebuild with an empty cache reproduced
the larger case's assembly hashes and all correctness checks; the tiny timing
difference reversed sign, reinforcing the lack of a meaningful gain.
The smaller specialization declared 508 bytes of
private scratch and contained scratch loads/stores. Register pressure is
a next profiling hypothesis; the descriptor alone does not prove a
bottleneck. If layout or tile changes are warranted, make them in FlyDSL,
regenerate assembly, and repeat the roundtrip check.

## Attribute tile changes separately from instruction edits

A subsequent Forge agent campaign on the same installation used fixed
`model_dim=7168`, `inter_dim=2048`, `experts=8`, and `topk=2`. The agent selected
tile `[32,128,256]` for `tokens <= 129`, reassembled that compiler output, and
retained the original `[32,256,256]` FlyDSL callable for larger token counts.
This is a shape-dependent dispatch decision, not a runtime fallback on an
assembly build or correctness failure. The threshold is specific to this
experiment; it is not a recommended production dispatch rule.

The protected seven-case driver passed, including ordinary BF16 scales,
input preservation, and nondefault-stream graph replay. The two equally
weighted scored cases (129 and 257 tokens) produced a 1.424x mean per-case
speedup through Forge's statistical KEEP gate. Replaying the exact candidate
with three fresh outer benchmarks reproduced 1.4243x; deliberately reversing
the SiLU exponent sign triggered correctness failure and REVERT. Native and
Hyperloom patches restored both `kernel.py` and `kernel.s` byte-for-byte and
passed the unchanged driver in new processes with empty compiler caches.

An independent comparison replaced the assembly route with the equivalent
tuned FlyDSL callable while preserving the same dispatch condition. Seven
interleaved rounds, nine samples per round, and 100 real calls per graph gave
these warm-execution medians; compilation, loading, sorting, and oracle work
were excluded:

| Tokens, random routing / power-of-two scales | Original FlyDSL | Tuned FlyDSL dispatch | Assembly dispatch |
| --- | --- | --- | --- |
| 129 | 144.397 us | 77.987 us | 77.990 us |
| 257 | 145.413 us | 145.446 us | 145.427 us |

All three implementations passed the seven original cases and eight holdout
cases: tokens 32, 127, 128, 130, 256, 258, and 513, plus 129 tokens with
concentrated routing and ordinary random scales. Holdouts were independent
validation, not additions to the campaign's scoring set. One 256-token timing
round had large outliers; retain raw rounds and use medians rather than
claiming tiny differences as improvements.

The winning `.s` SHA-256 was
`e49eeb81ea690551b6c086653f192820ef0fc9a609d9701746e5793aefc79c5a`, identical
to the `[32,128,256]` compiler output. Therefore the useful gain is from
tile/dispatch selection. Reassembly provided no meaningful additional gain
over equivalent FlyDSL. Do not report the campaign's 1.424x score as a
hand-written assembly speedup or as an aggregate latency ratio.

Static assembly inspection found the original specialization declared 412
bytes of private scratch and contained 31 scratch loads and 31 stores. The
tuned specialization declared zero private scratch and contained none; LDS
remained 32768 bytes. These are static resource observations, not counter-based
bottleneck attribution. The original scratch instructions were outside the
repeated main K loop. Measure counters before assigning the gain to spill
traffic, occupancy, or a specific pipeline.

For future searches, compare the original launcher, regenerated assembly, and
equivalent tuned high-level launcher. Keep both launcher and assembly in each
published bundle, apply its patch in a clean checkout, and rerun the protected
driver before treating the result as reproducible.

These experiments validate stage1 candidate execution, not stage2/reduction,
a full Kimi configuration, Neha's reported gains, the arena acceptance suite,
or vLLM end-to-end performance.
