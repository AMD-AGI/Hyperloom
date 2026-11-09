---
title: MiniMax-M3.1-Flash FFN launch — stages, budgets and measured times
kind: technique
gens: [gfx950]
updated: 2026-11-09
sources:
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/minimax_m31/ffn.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/minimax_m31/spec.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/minimax_m31/runner.py
  - https://github.com/sgl-project/sgl-amd-mono/pull/56
  - https://github.com/sgl-project/sgl-amd-mono/pull/59
---

# MiniMax-M3.1-Flash — the FFN launch (`minimax_m31/`)

The smallest complete monokernel in the repository, and the reference for a span that starts after
SGLang's attention. Spec `minimax_m31_ffn_hip`: gfx950, TP 4, widths 1–16 rows, `BLOCKS = 256`;
`TOLERANCE` numerics; it accepts speculative decoding, ragged verify, EAGLE3 hidden states and KV
offload, and requires bf16 and `SGLANG_USE_AITER`.

## What one launch does (one a sparse layer and rank)
From the attention output to the reduced MoE output, o_proj included (`ffn.py` docstring):

| Stage | Tasks | Work |
|---|---|---|
| push | 128 × 48 columns | o_proj as SGLang runs it (per-token FP8 rows × its rowwise-FP8 weight); the partial's bf16 triples to every rank |
| reduce | 192 | the ranks' pairs in rank order, f32 → bf16; + residual in f32 → residual out, the f32 sum and its sum of squares |
| norm | S | Gemma RMSNorm of the f32 sum → X (bf16) |
| router | 8 × 16 experts × 16 K parts | FP32 weight × f32(X), f32 MFMA |
| route | S | parts summed in order, sigmoid, + bias for selection, top-4, renormalized × 2.0 |
| table | every CTA | the union of picked experts in LDS; slot 0 is the shared expert over all rows |
| up | 48 × slots | 16 gate + 16 up rows of a slot's expert, A16W4 MFMA, SwiGLU-OAI → bf16 MID |
| down | 384 × 16 rows | every slot's A16W4 GEMV of MID, routed and shared experts combined → pushed to every rank |
| final | 192 | the ranks' partials in rank order → `out` |

Shapes: hidden 6144, 128 routed experts, top-4, a TP4 rank's expert intermediate 768. Routed experts
are SGLang's post-load MXFP4 tensors, read in place; the shared expert keeps the checkpoint's MXFP8
values (`load_tensor` keeps this rank's shard, `a16w8.py` lays it out). Reading it as MXFP4 instead
had about 4.6× the output error and halved DSpark's real acceptance in a test build (#56).

## Budgets (cross-compiled for gfx950, FlyDSL 0.4.0, TP 4)
| Rows | VGPRs | LDS | Spills |
|---|---|---|---|
| 1 | 184 | 66,816 B | none |
| 16 | 256 | 163,264 B of 163,840 | none |

Steps of 9–16 rows keep the normed rows (XN) in global memory: 16 rows of them do not fit in LDS with
the rest (#59); up to 8 rows they go to LDS once a CTA. At
16 rows the launch is 576 B from the LDS wall and at the 256-VGPR wave limit: a change that adds state
there has to remove some.

## Measured
- Standard path at c1 (TP4, MI350X, 150K context, DSpark with 7 drafts): about seven launches a sparse
  layer after attention, about 76 µs a layer, roughly half of an 8.66 ms step (#56).
- One launch instead: c1 decode 390.8 → 422.3 tok/s (+8.1%) with acceptance held at 3 (#56).
- Launch alone, TP=1 build: 26.3 / 45.7 / 65.8 / 108.8 µs at 1 / 4 / 8 / 16 rows (#56).
- #59 (o_proj in the launch, routing on the norm's published `rstd`, B operands from LDS, the next
  unit's weights issued during the current one): TP4, 8 rows, weights rotating through four copies,
  54.7 → 44.8 µs at 8 distinct experts (−18%), 59.6 → 51.8 at 14 (−13%), 66.0 → 58.2 at 18 (−12%);
  c1 decode +7.4% over `main`. Every output bit-identical to `main` at 1, 3, 8, 9 and 16 rows.

## Tests
`test/kernels/test_minimax_m31_ffn.py` (every stage at 1, 3, 8 and 16 rows, TP=1 build; a timed-out
wait stops; a tag cycle ends with its receive half zeroed) and `test/e2e/test_minimax_m31_server.py`
(4 × MI350X, `MINIMAX_M31_FLASH_MODEL_PATH`; GSM8K, check mode against `NUMERICS`, a stalled rank stops
the server without tokens). SGLang needs MiniMax-M3.1-Flash support (sgl-project/sglang#45925) and its
gfx950 path (#45936).

## Not covered upstream (#59)
MI355X, TP other than 4, and the attention side: the next layer's add-RMSNorm, qkv GEMM, q / k norms,
RoPE, cache writes and sparse decode still run as about 7 kernels a layer (about 40 µs).
