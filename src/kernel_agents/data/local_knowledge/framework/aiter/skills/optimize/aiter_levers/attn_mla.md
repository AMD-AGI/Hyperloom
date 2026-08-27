---
title: aiter attention & MLA — flash_attn, paged decode, mla_decode_fwd
kind: backend
backend: aiter
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/mla.py
  - ROCm/aiter@b467ce342:csrc/py_itfs_cu/asm_mla_v4.cu
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
  - https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.html
---

# aiter attention & MLA

## TL;DR
aiter owns the attention kernels on AMD serving: **`flash_attn_func`** (MHA prefill), **paged/decode
attention**, and **`mla_decode_fwd`** (DeepSeek Multi-head Latent Attention decode). The MLA decode kernel
is the headline: AMD reports up to **17× vs naive decode** on MI300X, achieved via matrix-absorption (run
MQA instead of MHA) + a hand-tuned asm kernel that the Triton MLA path is several × slower than. These are
the *default* attention kernels; `--attention-backend` only overrides which one runs, not whether aiter is
on.

## Concepts

### MLA decode (`aiter/mla.py:197 mla_decode_fwd`)
```python
mla_decode_fwd(q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
               max_seqlen_q, sm_scale=None, logit_cap=0.0, num_kv_splits=None, ...,
               return_lse=False, g_kv_indptr=None, cp_world_size=1, cp_rank=0)   # + context-parallel
```
- `q`: `[B*q_seqlen, num_heads, kv_lora_rank + qk_rope_head_dim]` (e.g. 512 + 64).
- `kv_buffer`: `[num_pages, page_size, num_heads_kv(=1), qk_head_dim]`; in decode `num_heads_kv==1` and
  `page_size==1` use the original (unpaged) representation.
- `o`: `[B*q_seqlen, num_heads, kv_lora_rank]`.
- `sm_scale` defaults to `1/sqrt(qk_head_dim)`.
- **Context-parallel**: pass the GLOBAL `g_kv_indptr` + `cp_world_size`/`cp_rank` → gfx950 `cprr` asm kernels.
- **DeepSeek-V4 sparse decode** is a separate entry `mla_decode_fwd_v4_nm(q, qrope, kv_buffer, kvrope, out,
  …, *, sink)` (`aiter/mla.py:1215`) — FP8 Q/KV + bf16 rope, **required `sink`**, gqa≤128; gfx950 + gfx1250.

**Matrix absorption**: the `kv_proj_up` weight is split — `Wuk` absorbed into `q_nope`, `Wuv` into the
attention output — so the layer runs as MQA. This collapses bandwidth and lets the asm kernel saturate
the MFMA pipe.

**`num_kv_splits` (split-KV)**: `get_meta_param` auto-picks the KV-split count and builds
`num_kv_splits_indptr` from batch/total-KV/heads heuristics; a Triton stage-2 (`_fwd_kernel_stage2_asm`)
combines the per-split partials. This is the decode analog of split-K — it fills CUs when batch is small.
Leave `num_kv_splits=None` (auto) unless you are an expert.

### MHA prefill / paged decode
`aiter.flash_attn_func(q, k, v, causal=..., softmax_scale=...)` (in `aiter/ops/mha.py`) is the flash MHA
prefill entry (asm/CK/Triton under the hood). Paged/decode attention is exposed via `aiter.paged_attn` /
`attention.py`. FP8 KV-cache and fp8 fmha are supported on gfx942 (FNUZ). Sparse-MLA / DSV4 paths are
**gen-specific, not blanket gfx942→Triton**: v4 asm (`mla_decode_fwd_v4_nm`) ships on gfx950 + gfx1250;
gfx942 has expanded fp8 asm behind `AITER_ENABLE_EXPERIMENTAL`; the Gluon/Triton sparse path
(`sparse_attention_dsv4` → `mla_gluon`) is gfx950.

## The levers
- Pick the right entry: `mla_decode_fwd` for DeepSeek decode, `flash_attn_func` for MHA prefill.
- vLLM (framework-side flags, not in the aiter tree): `VLLM_MLA_DISABLE=0`, `VLLM_USE_AITER_MLA`,
  `VLLM_USE_TRITON_FLASH_ATTN=0` to keep the asm/CK MLA path; `--attention-backend` selects the kernel but
  `VLLM_ROCM_USE_AITER=1` is still required. Verify these against the current vLLM/sglang, not this card.
- KV-cache dtype (bf16 vs fp8 FNUZ) trades accuracy for bandwidth.

## Numerics / parity
MLA matrix absorption is algebraically equivalent to standard MLA (parity-safe in bf16). fp8 KV-cache /
fp8 fmha introduce quant error — validate model accuracy. The Triton MLA reference exists for
correctness cross-checks but is much slower.

## Pitfalls
- Sparse/V4 MLA is **gen-specific** (v4 asm on gfx950/gfx1250; gfx942 experimental; Gluon on gfx950) — not a
  blanket gfx942→Triton fallback. Confirm the actual kernel with `AITER_LOG_MORE=1`; several experimental
  fp8/HK paths need `AITER_ENABLE_EXPERIMENTAL=1` (else excluded at JIT build).
- `num_heads_kv`/`page_size` must match the decode contract (`==1`) to hit the fast unpaged path.
- Don't hand-set `num_kv_splits` unless you have measured it; the auto heuristic is shape-aware.

## How to verify
`AITER_LOG_MORE=1` to confirm the asm MLA kernel (not Triton) fires; benchmark decode tok/s. The AMD
MLA blog + AI-Developer-Hub notebook give a runnable `mla_decode_fwd` example to confirm the path on-box.

## Alternatives / cross-links
operators: `mla_attention`, `attention_prefill_fmha`, `attention_decode_paged` ·
`backends/flash_attention_rocm/` · [integration.md](../../../overall/dispatch_and_rebind.md).

## Sources
- On-box: `ROCm/aiter@b467ce342`: `aiter/mla.py` (`mla_decode_fwd:197`, `get_meta_param:125`,
  `mla_decode_fwd_v4_nm:1215`, `_fwd_kernel_stage2_asm`, CP args), `csrc/py_itfs_cu/asm_mla_v4.cu`,
  `aiter/ops/mha.py` (`flash_attn_func`).
- 17× MLA decode + matrix absorption (AMD-reported, MI300X, tested 2025-03):
  https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
- `mla_decode_fwd` signature/example: https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.html
