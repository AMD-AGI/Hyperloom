---
title: fmha_sink_mxfp8 — overview
kind: operator_overview
operator: fmha_sink_mxfp8
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8_e4m3, mxfp8]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/mha.py
---

# fmha_sink_mxfp8

## TL;DR
> FMHA forward variants that add two features on top of the standard `flash_attn_func`: an **attention
> sink** (a per-head extra logit added to the softmax denominator, GPT-OSS style) and **MXFP8** inputs
> (fp8 e4m3 Q/K/V with e8m0 microscaling). aiter exposes dedicated gfx1250 ASM entries
> (`fmha_fwd_with_sink_asm`, `fmha_fwd_mxfp8_asm`) plus the general `flash_attn_func`, which carries the
> sink through its `window_size`/`sink_ptr` params and dispatches to asm / CK-tile / Triton.

## What it is
- **Attention sink**: a per-Q-head fp32 bias whose contribution is added **only to the softmax
  denominator** — `denom += exp((sink - max) · scale)` — with no V contribution. It lets a head "dump"
  probability mass, stabilizing long-context/streaming attention (`mha.py:1755-1757`).
- **MXFP8 FMHA**: Q/K/V are fp8 (e4m3) with **e8m0** (`float8_e8m0fnu`) block-scale descale buffers,
  block_size 32 along head_dim; output bf16. This is distinct from the per-tensor fp8 path (which uses
  fp32 descales) (`mha.py:541-545, 1769-1773`).
- **Two separate "sink" knobs** in `flash_attn_func`: `window_size[2]` = `sink_size` (an integer count
  of sink tokens, only effective under a mask) vs `sink_ptr` (the per-head fp32 sink-bias tensor).

## Entry points (API)
| symbol | path:line | signature (abridged) | purpose |
|---|---|---|---|
| `flash_attn_func` | `aiter/ops/mha.py:2533` | `(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1,-1,0), bias=None, alibi_slopes=None, deterministic=True, return_lse=False, return_attn_probs=False, how_v3_bf16_cvt=1, cu_seqlens_q=None, cu_seqlens_kv=None, sink_ptr=None, num_splits=0)` | general FA fwd; `window_size=(left,right,sink_size)`, `sink_ptr`=`[nheads]` fp32 sink bias |
| `fmha_fwd_with_sink_asm` | `aiter/ops/mha.py:393` | `(q, k, v, softmax_scale, is_causal, return_lse, sink=None, out=None) -> (out, lse)` | gfx1250 bf16 ASM FMHA with per-head fp32 attention sink (bshd) |
| `fmha_fwd_with_sink_varlen_asm` | `aiter/ops/mha.py:465` | `(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, softmax_scale, is_causal, return_lse, sink=None, out=None)` | packed THD varlen bf16 ASM sink forward (gfx1250) |
| `fmha_fwd_mxfp8_asm` | `aiter/ops/mha.py:567` | `(q, k, v, q_scale, k_scale, v_scale, softmax_scale=None, is_causal=False, return_lse=False, out=None) -> (out, lse)` | gfx1250 MXFP8 ASM FMHA (fp8 e4m3 + e8m0 microscaling), bf16 out |

## Dispatch / backends
`flash_attn_func` → `FlashAttnFunc.apply` → `_flash_attn_forward` (`mha.py:1660`), which tries paths in
order (`mha.py:1875-1998`):
1. **native split-K** (`can_impl_fmha_native` + `num_splits`) → `mha_fwd_native_splitkv`.
2. **gfx1250 `fmha_fwd_with_sink_asm`** (`can_impl_fmha_fwd_with_sink_asm`:1734) — bf16, hdim ∈ {64,128},
   `sink_size == 0`. Per-hdim rule: **D128** kernels compile `ENABLE_SINK=0` (require `sink_ptr is None`,
   else fall back to CK); **D64** kernels always read sink (require `sink_ptr is not None`). **Passes
   `sink_ptr` verbatim** (`:1898-1918`).
3. **gfx1250 `fmha_fwd_mxfp8_asm`** (`can_impl_fmha_fwd_mxfp8_asm`:1769) — fp8 (e4m3) Q/K/V with e8m0
   microscaling descales (`:1921-1938`).
4. **gfx950 opus D128 bf16** (`can_impl_fmha_fwd_hd128_bf16_opus`, `:1939-1952`).
5. **gfx9 `fmha_v3_fwd`** (`can_impl_fmha_v3_fwd`:1714 **and `seqlen_q > 128`**) — hand-written gfx942/950
   ASM, bf16 or per-tensor fp8, hdim 128/192, no SWA. **This asm path does NOT take `sink_ptr`** (the
   call forwards `None`, `:1953-1972`), so sinks on gfx9 go through the CK fallback below.
6. else **CK-tile `mha_fwd`** — the general fallback; consumes both `sink_size` (`:1984`) **and**
   `sink_ptr` (`:1995`). CK picks `_sink`/`_nsink` batch_prefill blobs via `has_effective_sink`
   (`:1406-1419`).
- Above all of this, when `ENABLE_CK` is unset `flash_attn_func` returns the **Triton** `flash_attn_func`
  early, forwarding `sink=sink_ptr` + `window_size` (`mha.py:2607-2624`).

## Config / knobs
- **`window_size = (left, right, sink)`** — `-1` = infinite context, `0` = no sink (`mha.py:2540`).
  Unpacked into `window_size_left/right` and `sink_size = window_size[2]` (`mha.py:2409-2411`).
- **`sink_size`** is only effective with a mask: `has_effective_sink = sink_size > 0 and (causal or
  window)` — full (no-mask) attention ignores it (`mha.py:1406-1410`).
- **`sink_ptr`** must be `[nhead_q]`; cast to fp32 if needed (`mha.py:1686-1690`).
- **`fmha_fwd_with_sink_asm`**: `sink` passed **verbatim** (no host-side scaling); 1-D fp32
  `[q_head_num]`; q/k/v are bshd-shaped, only `stride(-1)==1` required; LSE always allocated
  (`mha.py:363-373, 406-414`).
- **`fmha_fwd_mxfp8_asm`**: q/k/v bshd shape in **bhsd memory order** (`stride_head > stride_seq`);
  `q/k/v_scale` are `float8_e8m0fnu` (block_size 32 along head_dim); `softmax_scale` defaults to
  `hdim_q**-0.5` (`mha.py:541-545, 579-599`).
- `num_splits` (native split-K, 0=auto), `how_v3_bf16_cvt` — general `flash_attn_func` knobs
  (`mha.py:2594-2597`).

## Numerics / parity
- Sink affects the softmax **denominator only**; algebraically it lowers all probabilities uniformly for
  that head. bf16/per-tensor-fp8 accumulate in fp32.
- **fp8 dialect**: per-tensor fp8 v3 path is FNUZ on gfx942 / OCP e4m3 on gfx950; MXFP8 (e4m3 + e8m0) is
  a gfx1250-only path — mismatched dialect corrupts descale (`mha.py:1769-1773, 1698-1713`).
- The ASM entries **always touch the LSE buffer**; when `return_lse=False` its contents are undefined —
  ignore them (`mha.py:412-414, 592-593`).

## Pitfalls
- **`sink_size` (int) ≠ `sink_ptr` (tensor).** `window_size[2]` is a masked sink-token count routed to
  the CK `_sink` variant; `sink_ptr` is the per-head fp32 sink-bias. They are different mechanisms.
- gfx1250 `fmha_fwd_with_sink_asm` has the D128-vs-D64 sink asymmetry above — a D128 call with a
  non-None `sink_ptr` silently routes to CK; a D64 call requires an explicit `sink_ptr`
  (`mha.py:1749-1766`).
- **On gfx9 the `fmha_v3_fwd` ASM path does not take `sink_ptr`** (forwards `None`, `mha.py:1972`); a
  sink on gfx942/950 is applied only through the CK fallback (`:1995`). Verify the engaged path when a
  sink matters.
- MXFP8 is a **dedicated gfx1250 path** with its own C++ TU / kernarg ABI; wrong shape/memory-order or
  descale dtype won't hit it (`mha.py:531-545`).
- The gfx1250 sink ASM path rejects varlen / dropout / SWA / quant / alibi / bias — those fall back to
  CK (`mha.py:1734-1748`).
- `sink_size` under full attention is a no-op (`mha.py:1406-1410`).

## Cross-links
- `../attention_prefill_fmha/aiter.md` — the base FMHA prefill SOTA card (`flash_attn_func` /
  `mha_batch_prefill`); this card covers its sink + MXFP8 extensions.
- `../mla_attention/aiter.md`, `../sparse_attention_mla/overview.md` — sinks also appear in MLA/sparse-MLA.

## Sources
- `ROCm/aiter@b467ce342:aiter/ops/mha.py` (`fmha_fwd_with_sink_asm`:393 + `@compile_ops`:375 + gfx1250/
  verbatim-sink header:361-373; `fmha_fwd_with_sink_varlen_asm`:465; `fmha_fwd_mxfp8_asm`:567 + e8m0/
  bhsd header:531-545; `flash_attn_func`:2533, `window_size` default:2540, `sink_ptr`:2549, ENABLE_CK
  Triton fallback:2607-2624; window_size unpack:2409-2411; `has_effective_sink`:1406-1410; sink_ptr
  asserts:1686-1690; `_flash_attn_forward`:1660; dispatch predicates `can_impl_fmha_v3_fwd`:1714-1732 /
  `can_impl_fmha_fwd_with_sink_asm`:1734-1767 / `can_impl_fmha_fwd_mxfp8_asm`:1769-1774; dispatch
  body:1875-1998 (sink_ptr→sink asm:1916, mxfp8:1925, v3 no-sink:1972, CK sink_size+sink_ptr:1984/1995);
  sink→denom math:1755-1757).
