---
title: MiniMax-M3.1-Flash FFN launch — stages, budgets and measured times
kind: technique
gens: [gfx950]
updated: 2026-11-17
sources:
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/minimax_m31/ffn.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/minimax_m31/spec.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/minimax_m31/runner.py
  - https://github.com/sgl-project/sgl-amd-mono/pull/56
  - https://github.com/sgl-project/sgl-amd-mono/pull/59
  - https://github.com/sgl-project/sgl-amd-mono/pull/145
  - https://github.com/sgl-project/sgl-amd-mono/pull/146
  - sgl-project/sgl-amd-mono@b504de5:scripts/ci/models.py
  - sgl-project/sglang@cbd4439f:python/sglang/srt/layers/quantization/fp8_utils.py
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

## Measured on MI355X (`main` at `b504de5`, SGLang `cbd4439f`, FlyDSL 0.4.0, TP 4)
The launch serves MI355X as it is: the serving tests (`test/e2e/test_minimax_m31_server.py`) pass on
4 x MI355X, GSM8K 0.97 / 0.985. The CI does not run it there (`scripts/ci/models.py`: `gpus: mi350x`).
- Kernel time, weights rotating through four copies (eight launches a graph, slowest rank):
  44.9 / 49.7 / 60.4 / 80.4 / 109.4 / 129.5 µs at 1 / 2 / 4 / 8 / 12 / 16 rows. Against one copy
  replayed (which stays in the MALL), rotation moved wall time by 0.2-3.7% (ROCm 7.2 / FlyDSL 0.3.4).
- Per-user decode (decode steps, no DSpark, 1024 in / 512 out): 278 / 180 / 129 tok/s at concurrency
  1 / 8 / 16 with the monokernels, 184 / 137 / 119 without; at 20 (past the 16-row width) both 107.
- A stage timeline at 16 rows: MFMA busy 1.4-4.1%; the up stage (about 40 µs) already streams at about
  5.6 TB/s; the rest is the critical chain of hand-offs (push -> reduce -> norm -> router -> route ->
  table -> up -> down -> final), with the shared expert and route on CTAs that serve a prefix stage first.
- o_proj: SGLang's `fp8_utils` runs AITER's `gemm_a8w8_bpreshuffle` for o_proj's (6144, 2048) since
  sgl-project/sglang#46693 (no `smallm_fp8_gemm` config); the launch's partial is bit-identical to it at
  1, 3, 8 and 16 rows. `test_o_proj_is_sglangs_to_the_bit` compared with `smallm_fp8_gemm` only and fails
  on that SGLang; #146 holds it to the dispatched GEMM.

## Levers measured on this launch (#145, open as of 2026-11-17)
Six bit-exact changes from two forge campaigns, each committed and measured on its own; cumulative
kernel time at 16 rows -0.2, -1.2, -4.0, -5.3, -6.2, -8.8% (1 row: -4.3 ... -8.5%); per-user decode
+7.3 / +4.7 / +7.5% at concurrency 1 / 8 / 16:
1. The shared expert's up tiles on CTAs 128..175, which have no prefix task (`SHARED0 = SEAM_TASKS`);
   the down tasks past one a CTA to the CTAs that finish up first (`stage_down_placed`).
2. Route on CTAs with no prefix task (`ROUTE0`), its bias load before the logits wait, the top-4
   winner's sigmoid read from its lane instead of a butterfly.
3. Routed up units publish UPF in pairs (`pair_flag`, `UPF_TAIL`): a drain covers two units' loads.
4. At 16 rows o_proj on all eight waves, one accumulator chain a wave, summed in the original order.
5. The prefix stages' inputs (residual, Gemma weights, o_proj's weight scales, SRES) issued before
   their waits.
6. A down slot whose UPF tiles the previous slot already saw carry this launch's tag skips its poll and
   issues its first MID batch ahead of its weights.

Measured slower or within noise (bench, ROCm 7.2 / FlyDSL 0.3.4): the reduce, norm and router loads
batched before their waits as one change; other UPF-pairing tails; probing the next down slot together
with hoisting the table work before the route poll; the router computing rstd from the reduce's partial
sums with the extra down tasks split by slot; pairing down slots; folding route into the table stage.

## Not covered upstream (#59)
TP other than 4, and the attention side: the next layer's add-RMSNorm, qkv GEMM, q / k norms,
RoPE, cache writes and sparse decode still run as about 7 kernels a layer (about 40 µs). MI355X is
measured above, by hand.
