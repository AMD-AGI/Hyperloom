---
title: AITER W4A16 MoE stage1 - verified FlyDSL assembly roundtrip
kind: case
gens: [gfx950]
status: GPU-validated; no measured optimization benefit
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
- Compute the oracle from the original unpacked weights and scales with
  FP32 PyTorch matrix multiplication, SiLU, multiplication, and BF16 output.
  The regression uses nonuniform power-of-two scales so dequantization is
  exact; arbitrary scales require checking the kernel's BF16 rounding policy.
- Test the original kernel first, then require bitwise equality between it
  and the unedited assembly. Keep the independent oracle unchanged. A
  disposable edit reversing the sign of the SiLU exponential must fail it.
- Rebind all input/output pointers and token counts, including partially
  filled tiles and inactive experts. Retain tensor ownership because the
  launch arguments are pointers. Check reference/old/new candidate isolation
  and nondefault-stream graph replay.

The optional regression is
`src/kernelforge/tests/test_assembly_aiter_moe_gpu.py`. It compiles the installed
kernel and covers token counts 1, 37, 65, and 257. Its synthetic-oracle
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
was approximately +/-0.02% and is not a useful improvement. Reject it under
a 3% improvement gate. A fresh-process rebuild with an empty cache reproduced
the larger case's assembly hashes and all correctness checks; the tiny timing
difference reversed sign, reinforcing the lack of a meaningful gain.
The smaller specialization declared 508 bytes of
private scratch and contained scratch loads/stores. Register pressure is
a next profiling hypothesis; the descriptor alone does not prove a
bottleneck. If layout or tile changes are warranted, make them in FlyDSL,
regenerate assembly, and repeat the roundtrip check.

This validates stage1 candidate execution, not stage2/reduction, a full
Kimi configuration, Neha's reported gains, or vLLM end-to-end performance.
