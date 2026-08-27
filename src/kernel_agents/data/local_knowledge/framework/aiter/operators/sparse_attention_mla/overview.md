---
title: sparse_attention_mla — overview
kind: operator_overview
operator: sparse_attention_mla
gens: [gfx950]
dtypes: [bf16, fp16, fp8_e4m3]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/triton/attention/sparse_attention_dsv4.py
  - ROCm/aiter@b467ce342:aiter/ops/triton/attention/unified_attention_sparse_mla.py
  - ROCm/aiter@b467ce342:aiter/ops/pa_sparse_prefill_opus.py
  - ROCm/aiter@b467ce342:aiter/ops/triton/gluon/mla_gluon.py
---

# sparse_attention_mla

## TL;DR
> DeepSeek-V4 **sparse MLA** attention: instead of attending to the whole KV history, each query
> attends only to a per-token **top-k** set of KV rows (selected upstream by the indexer — see
> paged_mqa_logits). aiter carries several entries: a Gluon CDNA4 kernel (`mla_gluon`, gfx950), an
> OPUS HIP prefill (`pa_sparse_prefill_opus`, gfx950), and Triton wrappers (`sparse_attn_prefill`,
> `unified_attention_sparse_mla`). The Triton `sparse_attn_prefill` auto-prefers Gluon when
> **Triton ≥ 3.6 + arch == gfx950**, else falls back to a plain Triton kernel.

## What it is
- Sparse-MLA attention over the **absorbed latent** KV: `head_dim = nope_head_dim + rope_head_dim`.
  For the DSV4 sparse prefill path the split is fixed to **448 NoPE + 64 RoPE = 512** and RoPE is
  folded into one contiguous query/KV row (`sparse_attention_dsv4.py:55-74`).
- Input is a **ragged/CSR** set of KV indices per query (top-k, optionally interleaved with a
  sliding-window region), so the attention only touches the selected rows in the unified KV pool.
- An optional per-head **attention sink** (fp32) can be folded into the softmax denominator.

## Entry points (API)
| symbol | path:line | signature (abridged) | purpose |
|---|---|---|---|
| `sparse_attn_prefill` | `aiter/ops/triton/attention/sparse_attention_dsv4.py:355` | `(q, kv, indices, topk_length, scale, head_dim, nope_head_dim, rope_head_dim, attn_sink, output, ragged_indices=None, ragged_indptr=None) -> None` | DSV4 sparse-MLA prefill, bf16 KV, dense-or-ragged top-k indices; writes `output` in place |
| `unified_attention_sparse_mla` | `aiter/ops/triton/attention/unified_attention_sparse_mla.py:6` | `(q, kv, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, topk_indices, block_table, kv_lora_rank)` | paged sparse MLA over `block_table` + per-token `topk_indices`; dev/reference path |
| `pa_sparse_prefill_opus` | `aiter/ops/pa_sparse_prefill_opus.py:67` | `(q, unified_kv, kv_indices_prefix, kv_indptr_prefix, kv, kv_indices_extend, kv_indptr_extend, attn_sink, softmax_scale, out=None) -> Tensor` | gfx950 HIP two-region sparse prefill (paged prefix + flat extend), bf16/fp16 |
| `pa_sparse_prefill_fp8_opus` | `aiter/ops/pa_sparse_prefill_opus.py:185` | `(q_nope, q_rope, unified_kv_nope, unified_kv_rope, …, attn_sink, softmax_scale, out=None) -> Tensor` | fp8 NoPE (512) + bf16 RoPE (64) variant of the OPUS prefill |
| `mla_gluon` | `aiter/ops/triton/gluon/mla_gluon.py:760` | `(q_nope, q_pe, kv_c, o, page_table, seq_info, sm_scale, …, has_pe=True, attn_sink=None)` | unified Gluon (CDNA4) MLA: decode + DSV4 sparse prefill via `has_pe=False` |

## Dispatch / backends
- **Gluon preference (Triton path):** `sparse_attn_prefill` imports `mla_gluon` only when
  `Version(triton) >= 3.6.0` **and** `arch == "gfx950"`; if that import succeeds the wrapper routes to
  `mla_gluon(..., has_pe=False)` (single 448+64 row, 1-D VarLen, single-split fast path). Otherwise it
  launches the plain Triton `_sparse_attn_prefill_kernel` (`sparse_attention_dsv4.py:22-40, 310-347`).
- **`mla_gluon` regime selection** by `(nhead, kv dtype)` (`mla_gluon.py:822-836`): `nhead ∈ {64,128}`
  → `bh64` (bf16 KV, BLOCK_H=64/BLOCK_N=64); `1 ≤ nhead ≤ 16` + bf16 KV → `bh16bn64`; `nhead ≤ 16` +
  `float8_e4m3fn` KV → `bh16bn128`. Requires gfx950, `head_dim_ckv=512`, `head_dim_kpe=64`
  (`mla_gluon.py:808-814`).
- **OPUS** (`pa_sparse_prefill_opus` / `_fp8_opus`) are `@compile_ops` JIT HIP kernels with a hard
  gfx950 guard — they `raise RuntimeError` on any other arch (`pa_sparse_prefill_opus.py:105-107,
  227-229`). Single compiled config: `D == 512`.
- Adapted from the vLLM DSV4 sparse-MLA backend `rocm_aiter_mla_sparse_dsv4.py`
  (`sparse_attention_dsv4.py:4-5`).

## Config / knobs
- **Fixed DSV4 sparse dims:** NoPE=448, RoPE=64, head_dim=512 (asserted, `sparse_attention_dsv4.py:55-74`).
- **Ragged index builders:** `build_ragged_indices_from_dense` (:112), `compute_global_topk_ragged_indices_and_indptr`
  (:159, local top-k → global slot ids via `block_table`), `combine_topk_swa_indices_ragged` (:209,
  interleave top-k + sliding-window into disjoint ranges of one KV buffer).
- **`attn_sink`:** optional `[H]` fp32 per-head softmax-denom bias; gated by `HAS_ATTN_SINK` constexpr
  (`sparse_attention_dsv4.py:96-104`); OPUS takes it as a required arg; Gluon folds it in
  (`mla_gluon.py:643-654`).
- **`mla_gluon` splitting:** `NUM_KV_SPLITS` auto-picked to fill ~256 workgroups (bh64 ∈ {1,2,4}); when
  `NUM_KV_SPLITS == 1` stage-1 writes O directly (no stage-2 reduce). `min_kv_seq_len` lower bound per
  regime; bh64 needs `batch_size % 64 == 0`, bh16bn128 needs `batch_size == 1`
  (`mla_gluon.py:840-886`).
- **KV > 2 GB:** kernel switches `buffer_load` (32-bit offsets) → `global_load` (64-bit) via
  `WITHIN_2GB` (`mla_gluon.py:894-897`).

## Numerics / parity
- fp32 online softmax; the two OPUS KV regions share one accumulator, making order region-invariant
  (`pa_sparse_prefill_opus.py:6-9`). Gluon uses `exp2`/LOG2E and a fp32 accumulator, output bf16
  (`mla_gluon.py:528-531, 657-658`).
- **fp8 KV temperature fold:** Gluon sets `qk_scale = sm_scale * kv_scale` because softmax is shift-
  but not scale-invariant — `kv_scale` must scale QK, not just the accumulator (`mla_gluon.py:361-365`).
- **fp8 dialect:** the Gluon bh16bn128 path uses `float8_e4m3fn` (gfx950 OCP e4m3, *not* e4m3fnuz)
  (`mla_gluon.py:827`). OPUS fp8 path splits fp8 NoPE + bf16 RoPE; bf16 output (`pa_sparse_prefill_opus.py:200-226`).

## Pitfalls
- **Gluon is gfx950 + Triton ≥ 3.6 only.** On any other arch/Triton, `sparse_attn_prefill` silently
  uses the un-optimized Triton kernel — confirm which path ran (`sparse_attention_dsv4.py:29-35`).
- `unified_attention_sparse_mla` is explicitly **"not optimized and simplified for initial
  development"** — treat it as a reference, not SOTA (`unified_attention_sparse_mla.py:38`).
- `topk_indices` index the **KV cache**, not the `block_table` (`unified_attention_sparse_mla.py:22`).
- OPUS is one hard-coded config: wrong gfx/dtype/`D`/out-shape raises (`pa_sparse_prefill_opus.py:105-128`).
- `mla_gluon` asserts `PAGE_SIZE == 1` and the regime batch/seq bounds above; violating them raises at
  launch (`mla_gluon.py:370, 848-892`).

## Cross-links
- `../mla_attention/aiter.md` — dense MLA decode/prefill and the `mla_decode_fwd_v4_nm` sparse asm
  decode; this card is the Triton/Gluon/OPUS sparse-prefill sibling.
- `../paged_mqa_logits/overview.md` — the FP8 indexer that produces the per-token top-k this operator
  consumes.

## Sources
- `ROCm/aiter@b467ce342:aiter/ops/triton/attention/sparse_attention_dsv4.py` (`sparse_attn_prefill`:355,
  `_sparse_attn_prefill_ragged`:279, Gluon gating:22-40/310-347, DSV4 dims 448/64:55-74, ragged builders
  :112/:159/:209, vLLM origin:4-5).
- `ROCm/aiter@b467ce342:aiter/ops/triton/attention/unified_attention_sparse_mla.py`
  (`unified_attention_sparse_mla`:6, "not optimized":38, topk indexes KV:22).
- `ROCm/aiter@b467ce342:aiter/ops/pa_sparse_prefill_opus.py` (`pa_sparse_prefill_opus`:67,
  `pa_sparse_prefill_opus_fwd`:37, gfx950 guard:105-107, single-config note:15-22,
  `pa_sparse_prefill_fp8_opus`:185).
- `ROCm/aiter@b467ce342:aiter/ops/triton/gluon/mla_gluon.py` (`mla_gluon`:760, regimes:6-27, dispatch
  :822-836, gfx950/512/64 asserts:808-814, `has_pe=False` prefill:800-806, NUM_KV_SPLITS auto:840-886,
  attn_sink fold:643-654, fp8 temperature fold:361-365).
