---
title: paged_mqa_logits — overview
kind: operator_overview
operator: paged_mqa_logits
gens: [gfx942, gfx950, gfx1250]
dtypes: [fp8_e4m3, fp32]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/triton/attention/pa_mqa_logits.py
  - ROCm/aiter@b467ce342:aiter/ops/triton/_triton_kernels/attention/pa_mqa_logits.py
  - ROCm/aiter@b467ce342:aiter/ops/deepgemm.py
---

# paged_mqa_logits

## TL;DR
> The FP8 **paged MQA-logits** kernel is the DeepSeek-V4 sparse-attention **indexer**: for every KV
> token it computes a scalar relevance **logit** = `Σ_head ReLU(q·k · k_scale) · weight`. A downstream
> top-k over these logits selects the sparse KV set that sparse_attention_mla then attends to. It reads
> a **paged fp8 KV cache** with a per-token fp32 scale packed inline (D+4 layout). Separately,
> `aiter.ops.deepgemm.deepgemm` is a CK-backed **grouped GEMM** front-end (the "DeepGEMM" name) — a
> different op that shares the DSV4 lineage, not the logits kernel.

## What it is
- **Indexer / lightning-indexer** score. Per program the Triton/Gluon kernel does, over KV chunks
  (`_triton_kernels/attention/pa_mqa_logits.py:348-440`):
  `o = q @ kᵀ` → `o *= k_scale` (per-token fp8 descale) → `o = max(o, 0)` (ReLU) → `o *= weight`
  (per-head indexer weight) → causal mask by `next_n` → `logits = Σ over q-heads (reduce axis=0)`.
- **Output** `out_logits` is fp32 `[batch·next_n, max_model_len]` — one relevance logit per KV
  position per (request, MTP token). It does **not** produce an attention output; it feeds top-k
  selection.
- **`next_n`** is the multi-token-predict (MTP) / speculative token dimension of the indexer query.

## Entry points (API)
| symbol | path:line | signature (abridged) | purpose |
|---|---|---|---|
| `deepgemm_fp8_paged_mqa_logits` | `aiter/ops/triton/attention/pa_mqa_logits.py:448` | `(q_fp8, kv_cache, weights, out_logits, context_lens, kv_indices, max_model_len, Preshuffle=False, KVBlockSize=1, ChunkK=256, TotalCuCount=None, WavePerEU=2, VarCtxSchedule=None)` | paged fp8 MQA-logits (indexer) — main entry |
| `deepgemm_fp8_paged_mqa_logits_schedule` | `pa_mqa_logits.py:411` | `(batch_size, next_n, context_lens, max_model_len, ChunkK=256, TotalCuCount=None, WavePerEU=2)` | precompute the VarCtx `safe_chunks_per_cta` schedule tensor |
| `deepgemm_fp8_paged_mqa_logits_ragged_k` | `pa_mqa_logits.py:82` | `(q_fp8, kv_cache_fp8, weights, out_logits, prefix_sum_context_lens, kv_indices, max_model_len, ChunkK=64, SplitKV=5)` | ragged/CSR KV (varlen) logits variant |
| `deepgemm_fp8_paged_mqa_logits_stage1` / `_stage1_ragged_k` | `pa_mqa_logits.py:184` / `:133` | stage-1 (per-head QK, no head reduction) | two-stage split (writes `out_qk`, reduced later) |
| `deepgemm` | `aiter/ops/deepgemm.py:36` | `(XQ, WQ, Y, group_layout, x_scale=None, w_scale=None)` | CK grouped/masked GEMM front-end (`deepgemm_ck`), fp8-capable — **not** the logits op |

## Dispatch / backends
- **Backend selection** in `deepgemm_fp8_paged_mqa_logits` (`pa_mqa_logits.py:40-79, 515-623`):
  - **Gluon** when `enable_gluon_pa_mqa_logits` — JIT-compiled via `_compile_deepgemm_fp8_paged_mqa_logits`
    on Triton ≥ 3.5, or an **AOT**-loaded Gluon binary on lower Triton (env
    `AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1`; AOT gen/load recipe in the file header).
  - else the plain **Triton JIT** kernel `_deepgemm_fp8_paged_mqa_logits` (requires `KVBlockSize == 1`,
    no `Preshuffle`).
- **Gens & fp8 dialect** (`_compile_deepgemm_fp8_paged_mqa_logits`): asserts gfx ∈
  `{gfx942, gfx950, gfx1250}`; pointer dtype is `*fp8e4b8` (fnuz) on **gfx942** vs `*fp8e4nv` (OCP
  e4m3) on **gfx950/gfx1250** (`deepgemm.py`→`pa_mqa_logits.py:260, 279-281`). gfx1250 uses warp_size 32.
- **Preshuffle / TDM block-load**: gfx1250 preshuffle needs `KVBlockSize > 1` and
  `ChunkK % KVBlockSize == 0`; the gfx1250 base kernel needs `KVBlockSize == 1`
  (`pa_mqa_logits.py:262-274`).
- `deepgemm` (grouped GEMM) dispatches to the CK op `deepgemm_ck` (`@compile_ops("module_deepgemm",
  fc_name="deepgemm")`, `deepgemm.py:25-44`).

## Config / knobs
- **`ChunkK`** (default 256; ragged_k uses 64), **`KVBlockSize`** (1 base; ≥16, `%16==0` for Preshuffle),
  **`ChunkQ`** (= `heads`), **`SplitKV`** — auto-derived from `TotalCuCount` / `next_n` / `WavePerEU`
  (`pa_mqa_logits.py:475-482`), **`WavePerEU`** (gfx1250 overrides to 1 or 4).
- **VarCtx scheduling**: pass `VarCtxSchedule` from `deepgemm_fp8_paged_mqa_logits_schedule` to switch to
  the variable-context grid (`pa_mqa_logits.py:500-513`); **not implemented on gfx1250** (warns + falls
  back, `:500-507`).
- **KV cache layout**: `[num_blocks, block_size, 1, D+4]` fp8 — the trailing 4 bytes per row are one
  fp32 per-token scale (`kv_cache[..., :D]` = fp8, `kv_cache[..., D:].view(float32)` = scale,
  `pa_mqa_logits.py:491-498`).
- **`weights`** fp32 `[batch·next_n, heads]`; **`out_logits`** fp32 `[batch·next_n, max_model_len]`.

## Numerics / parity
- Scores accumulate in fp32; **ReLU** clamps negatives to 0 before the per-head weighted sum; positions
  beyond `context_length - next_n + pid_next_n` are set to `-inf` (causal for the MTP token)
  (`_triton_kernels/attention/pa_mqa_logits.py:423-433`).
- **fp8 dialect trap**: fnuz (gfx942) vs OCP e4m3 (gfx950/gfx1250) — a mismatched dialect corrupts the
  descale (`pa_mqa_logits.py:279-281`).
- Reference parity check: `op_tests/op_benchmarks/triton/bench_deepgemm_attention.py`
  (`ref_fp8_paged_mqa_logits*`, `calc_diff`).

## Pitfalls
- **`deepgemm` ≠ the logits op.** `aiter/ops/deepgemm.py:deepgemm` is a CK **grouped/masked GEMM**
  (takes `group_layout`), used elsewhere in the DSV4 stack; the indexer is
  `deepgemm_fp8_paged_mqa_logits`. Don't conflate the two.
- `opus_gemm_a16w16_tune` re-exported from `deepgemm.py` is a **deprecation shim** — moved to
  `aiter.ops.opus.gemm_op_a16w16` (`deepgemm.py:47-61`).
- AOT Gluon on Triton < 3.4 needs a trailing dummy pointer arg (ABI break in triton#7258); the file
  guards this (`pa_mqa_logits.py:310-312, 559-594`).
- Non-Gluon Triton path only supports `KVBlockSize == 1` and no Preshuffle (`pa_mqa_logits.py:596-597`).

## Cross-links
- `../sparse_attention_mla/overview.md` — consumes the top-k derived from these logits.
- `../mla_attention/aiter.md` — the MLA attention family (DSV4 decode/prefill) this indexer serves.

## Sources
- `ROCm/aiter@b467ce342:aiter/ops/triton/attention/pa_mqa_logits.py` (`deepgemm_fp8_paged_mqa_logits`:448,
  `_schedule`:411, `_ragged_k`:82, `_stage1`:184/`_stage1_ragged_k`:133, backend/AOT selection:40-79,
  `_compile_*` gfx+fp8 dialect:248-281, Preshuffle/TDM:262-274, KV D+4 layout:491-498, SplitKV auto:475-482,
  VarCtx gfx1250 fallback:500-507).
- `ROCm/aiter@b467ce342:aiter/ops/triton/_triton_kernels/attention/pa_mqa_logits.py`
  (`_deepgemm_fp8_paged_mqa_logits`:348, indexer math dot/descale/ReLU/weight/mask/sum:423-440,
  `_sum_combine`:9).
- `ROCm/aiter@b467ce342:aiter/ops/deepgemm.py` (`deepgemm`:36, `deepgemm_ck`:26, CK front-end docstring:3-13,
  deprecation shim:47-61).
