---
title: opus (split-K GEMM + MoE stage2) — aiter backend card
kind: sota_card
operator: dense_gemm
backend: opus
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp8_e4m3_fnuz, fp4_e2m1]
regimes: [prefill, decode]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/opus/gemm_op_a16w16.py
  - ROCm/aiter@b467ce342:aiter/ops/opus/gemm_op_a8w8.py
  - ROCm/aiter@b467ce342:aiter/ops/opus/moe_stage2_a8w4.py
  - ROCm/aiter@b467ce342:aiter/tuned_gemm.py
  - ROCm/aiter@b467ce342:csrc/opus_gemm/opus_gemm.cu
  - ROCm/aiter@b467ce342:csrc/gemm_a16w16/gemm_a16w16_tune.py
---

# opus × aiter (split-K GEMM + MoE stage2)

## TL;DR
**opus** is aiter's in-tree, JIT-compiled ROCm kernel family for high-performance GEMM (a16w16 bf16, a8w8
fp8) and MoE stage-2 (a8w4) on **gfx950 / gfx942 / gfx1250**. The GEMM path is built around **split-K**:
many kernel instances (`kid`s) partition K across workgroups, write **fp32 partial accumulators** into a
**persistent per-stream workspace**, then a dedicated **split-K reduce** kernel folds the partials (and
optional bias) into the bf16/fp32 output. It is one of the `libtype`s the dense-GEMM DB can select
(`solMap["opus"]`), and it is tuned by the multi-backend tuner `gemm_a16w16_tune.py --libtype opus`. The
one thing an agent must get right: opus needs its split-K workspace **registered and grown before HIP-graph
capture** — a grow inside capture aborts the graph and replay silently writes zeros.

## SOTA implementation(s)
| impl | source | gens/dtypes | notes |
|---|---|---|---|
| `gemm_a16w16_opus` (bf16 GEMM) | `aiter/ops/opus/gemm_op_a16w16.py:426`; C++ `opus_gemm` (`csrc/opus_gemm/opus_gemm.cu`); JIT `module_deepgemm_opus` | gfx942/950/1250; bf16 in, bf16/fp32 out; **B plain (not preshuffled)** | split-K + split-barrier / persistent / mono-tile / (gfx1250) cluster-TDM pipelines |
| `opus_gemm_a8w8_blockscale_bpreshuffle_tune` (fp8) | `aiter/ops/opus/gemm_op_a8w8.py:38` | **gfx942**; fp8 XQ/WQ + x_scale/w_scale, bf16 out; default `kernelId=11000` | A8W8 blockscale + bpreshuffle |
| a8w8 noscale/scale (fp8) | C++ `opus_gemm` when `XQ.dtype()==fp8` | **gfx950**; fp32 out, no bias | dispatched inside `opus_gemm()` |
| `opus_moe_stage2_a8w4_decode_fwd` (MoE stage-2) | `aiter/ops/opus/moe_stage2_a8w4.py:125`; JIT `module_moe_opus` | **gfx950**; FP8 act × FP4 weight → bf16 / route-out | shape families K3 (`inter_dim=512`), K5 (`inter_dim=768`); `block_m ∈ {16,32,64}` |

Public entry symbols (star-imported into `aiter`): `gemm_a16w16_opus`, `opus_gemm_a16w16_tune`,
`opus_gemm_workspace_init`, `opus_gemm_a8w8_blockscale_bpreshuffle_tune`. On an unsupported GPU
`aiter/ops/opus/__init__.py` swaps these for stubs that raise `RuntimeError`.

## Dispatch (how opus runs on the live path)
1. `tuned_gemm.gemm_a16w16` resolves the 10-tuple `(gfx, cu_num, padded_M, N, K, bias, dtype, otype,
   scaleAB, bpreshuffle)`; a row with `libtype="opus"` routes to `solMap["opus"] = opus_gemm`.
2. `opus_gemm(...)` (`tuned_gemm.py:584`) asserts **no scaling and no bpreshuffle**, reads
   `splitK = config["splitK"]`, **pre-warms the capture-stream workspace**, then calls the tuned launcher
   `opus_gemm_a16w16_tune(inp, weights, Y, bias=bias, kernelId=solidx, splitK=splitK)`.
3. Direct use (`gemm_a16w16_opus`, bypassing `tuned_gemm`) is three-tier: explicit `kernelId` → Python CSV
   lookup (`common.lookup_tuned`) → C++ heuristic dispatch.

## Tuning
| item | detail |
|---|---|
| tuner | `csrc/gemm_a16w16/gemm_a16w16_tune.py --libtype opus` (or `all`); opus candidates from `csrc/opus_gemm/opus_gemm_tune.py` |
| skip conditions | opus rows are skipped if opus is unavailable, `scaleAB=True`, or `indtype != bf16` |
| output | rows stamped `libtype='opus'`, `scaleAB=False`, `bpreshuffle=False` in `bf16_tuned_gemm.csv` (10-tuple key + `solidx/splitK/kernelName`) |
| subset compile | the tuner expands a JIT sidecar and rebuilds `module_deepgemm_opus.so` for the candidate kids |
| debug tuner | `csrc/opus_gemm/opus_gemm_tune.py -m M -n N -k K [--kid …]` → `/tmp/opus_debug_tuned.csv` |
| A8W8 blockscale bpreshuffle | tuned via `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py --libtype opus` (gfx942, `preshuffleB=True`) |
| MoE stage2 a8w4 | tuned via `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` with `TUNE_ONLY=opus` / `OPUS_ONLY=1`; winner in `kernelName2` |

## gfx support
- **gfx950** (CDNA4): full a16w16 (split-barrier/flatmm/persistent/mono-tile), a8w8 noscale/scale, MoE a8w4 stage2.
- **gfx942** (CDNA3): a16w16 nosplit + split-K families, a8w8 blockscale bpreshuffle (kid 11000).
- **gfx1250**: a16w16 cluster/TDM split-K + clusterlaunch (kids 20000+); **tune-id entry only** (no shape-heuristic dispatch yet).

## Pitfalls & anti-patterns
- ⚠ **HIP-graph workspace trap (the #1 opus gotcha)**: the split-K launcher grows a per-stream fp32
  workspace with `hipMalloc`, which is **stream-capture-illegal**. If the workspace isn't sized before
  capture, the grow aborts the graph → replay writes **zeros (garbage logits)**. `tuned_gemm.opus_gemm`
  auto-prewarms on `torch.cuda.graphs.graph.default_capture_stream` (the vLLM/ATOM implicit-capture case);
  for a custom capture stream call `opus_gemm_workspace_init()` **eagerly per stream** and run the largest
  expected shape before capture. TBO (two streams) needs init + warm on each.
- ⚠ **Bias is folded by the split-K reduce kernel** — do **not** add bias again above `opus_gemm` (the old
  `Y = Y + bias` double-counted → `A@Bᵀ + 2·bias`, ~54% miscompare). Bias layout is per-output-feature
  `[N]` / `[batch,N]` (F.linear), not per-row `[M]`. gfx942 split-K kids do **not** support bias.
- ⚠ **No scaling / no bpreshuffle** through the `tuned_gemm` opus path (asserts) — A8W8 blockscale
  bpreshuffle uses the separate `opus_gemm_a8w8_blockscale_bpreshuffle_tune` entry.
- ⚠ **Layout**: launchers hardcode `stride_b_batch == N*K`, `stride_b == K`; `B.unsqueeze(0).expand(...)`
  views are rejected; `Y` must be contiguous.
- ⚠ **CSV/JIT staleness**: Python CSV lookup is process-lifetime cached (restart to pick up edits); the C++
  baked lookup needs `AITER_REBUILD=1` after a CSV edit.
- ⚠ **MoE stage2 a8w4**: gfx950 only; no EP (`expert_mask`/`topk_ids`), no `bias2`, requires
  `a2_scale`/`w2_scale`.
- Fallback: if opus is unavailable at import, `tuned_gemm.opus_gemm` falls back to `torch_gemm` with a warning.

## Cross-links
operators/dense_gemm/backends/aiter (dispatch + tune flow) · operators/skinny_gemv_decode/backends/aiter
(small-M alternative) · overall/tuning_db · overall/dispatch_and_rebind
· operators/fused_moe_grouped_gemm (MoE stage2 consumer).

## Sources
- On-box `ROCm/aiter@b467ce342`: `aiter/ops/opus/{gemm_op_a16w16.py,gemm_op_a8w8.py,moe_stage2_a8w4*.py,common.py,_arch.py,__init__.py}`,
  `aiter/tuned_gemm.py` (`opus_gemm:584`, `_opus_prewarm_capture_workspace`, `_OPUS_WS_ARCHS`),
  `csrc/opus_gemm/` (`opus_gemm.cu`, `opus_gemm_tune.py`, `opus_gemm_common.py`),
  `csrc/gemm_a16w16/gemm_a16w16_tune.py` (`--libtype opus`), `aiter/jit/optCompilerConfig.json`
  (`module_deepgemm_opus`, `module_moe_opus`).
