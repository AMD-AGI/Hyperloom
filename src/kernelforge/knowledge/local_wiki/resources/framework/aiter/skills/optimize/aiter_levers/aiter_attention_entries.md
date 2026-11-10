---
title: aiter attention entries — which call, which generation, which kernel actually ran
kind: lever
backend: aiter
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: sota
updated: 2026-08-28
sources:
  - ROCm/aiter@b467ce342:aiter/mla.py
  - ROCm/aiter@b467ce342:aiter/ops/mha.py
  - ROCm/aiter@b467ce342:csrc/py_itfs_cu/asm_mla_v4.cu
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
  - https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.html
---

# aiter attention entries

## Route here when
- You need to pick between `flash_attn_func`, paged decode, and `mla_decode_fwd`.
- A DeepSeek model's decode is slow and you want to know whether the asm MLA kernel is even running.
- Sparse / DeepSeek-V4 attention behaves differently across gfx942 / gfx950 / gfx1250.
- An attention env var "should" have changed something and didn't.

**Skip this if** the question is "what does MLA compute" — that is model architecture, not covered
here. This card is about which aiter entry maps to which kernel, and how to confirm it.

## The decision in one table
| Workload | Entry | Notes |
|---|---|---|
| MHA prefill | `aiter.flash_attn_func(q, k, v, causal=…, softmax_scale=…)` | `aiter/ops/mha.py`; asm / CK / Triton underneath |
| Paged decode | `aiter.paged_attn` / `attention.py` | fp8 KV-cache supported on gfx942 (FNUZ) |
| DeepSeek MLA decode | `aiter.mla.mla_decode_fwd` | the headline kernel — see below |
| DeepSeek-V4 sparse decode | `aiter.mla.mla_decode_fwd_v4_nm` | **gfx950 + gfx1250 only**, separate entry, extra required arg |

These are the *defaults* on an aiter-enabled stack. `--attention-backend` chooses among them; it does
not decide whether aiter is on at all — that is `VLLM_ROCM_USE_AITER=1` / `SGLANG_USE_AITER=1`.

> **Framework-side flags are not documented here on purpose.** vLLM and SGLang each carry their own
> MLA / flash-attention switches (the `VLLM_*MLA*` and `VLLM_USE_TRITON_FLASH_ATTN` family), and they
> are renamed, defaulted differently, or removed between releases. Read them out of the framework
> version you are actually running, then confirm the outcome with `AITER_LOG_MORE=1` — the log is the
> only claim that stays true.

## Why MLA decode is fast: matrix absorption
The `kv_proj_up` weight is split and folded into its neighbours — `Wuk` absorbed into `q_nope`, `Wuv`
into the attention output. The layer then runs as **MQA instead of MHA**. Two consequences:

1. Bandwidth collapses (one KV head instead of many), which is what makes decode fast.
2. It is **algebraically exact**, so bf16 parity holds. The speedup costs no accuracy.

On top of that sits a hand-tuned asm kernel. AMD reports up to **17× versus naive decode** on MI300X.
The Triton MLA path exists as a correctness reference and is several times slower.

## `mla_decode_fwd` — the contract
```python
mla_decode_fwd(q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
               max_seqlen_q, sm_scale=None, logit_cap=0.0, num_kv_splits=None, ...,
               return_lse=False, g_kv_indptr=None, cp_world_size=1, cp_rank=0)
```

| Argument | Shape / value | Gotcha |
|---|---|---|
| `q` | `[B*q_seqlen, num_heads, kv_lora_rank + qk_rope_head_dim]` | e.g. 512 + 64 |
| `kv_buffer` | `[num_pages, page_size, num_heads_kv, qk_head_dim]` | the **absorbed latent**, not raw KV |
| `o` | `[B*q_seqlen, num_heads, kv_lora_rank]` | note: rank, not full head dim |
| `num_heads_kv` | must be `1` | the latent is MQA |
| `page_size` | `1` for the fast unpaged path | >1 takes a different, slower route |
| `sm_scale` | defaults to `1/sqrt(qk_head_dim)` | |
| `num_kv_splits` | leave `None` | see below |
| `g_kv_indptr`, `cp_world_size`, `cp_rank` | context-parallel | pass the **global** indptr → gfx950 `cprr` asm kernels |

### `num_kv_splits` — leave it alone
`get_meta_param` auto-picks the KV-split count and builds `num_kv_splits_indptr` from batch, total-KV,
and head-count heuristics; a Triton stage-2 (`_fwd_kernel_stage2_asm`) merges the per-split partials.

This is the decode analogue of split-K: it exists to fill CUs when the batch is small. The heuristic
is shape- and CU-aware; hand-setting it is an expert move and the source calls it out as such. If you
do override it, measure — do not reason about it.

## Generation differences are real, and not a simple fallback ladder
The tempting mental model — "gfx950 gets asm, gfx942 falls back to Triton" — is wrong. The actual
layout for sparse / DeepSeek-V4:

| Path | Where it ships |
|---|---|
| v4 asm (`mla_decode_fwd_v4_nm`) | gfx950 **and** gfx1250 |
| expanded fp8 asm | gfx942, behind `AITER_ENABLE_EXPERIMENTAL` |
| Gluon / Triton sparse (`sparse_attention_dsv4` → `mla_gluon`) | gfx950 |

So gfx942 does not get v4 asm at all, gfx950 has *two* sparse paths, and several fp8 / HipKittens
paths are excluded at JIT build time unless `AITER_ENABLE_EXPERIMENTAL=1` is set. Never infer the
kernel from the arch — read it from the log.

`mla_decode_fwd_v4_nm(q, qrope, kv_buffer, kvrope, out, …, *, sink)` additionally takes fp8 Q/KV with
bf16 rope, caps `gqa` at 128, and **requires `sink`** — it is keyword-only and has no default.

## Verify
| Check | Command / signal | Pass condition |
|---|---|---|
| asm MLA actually ran | `AITER_LOG_MORE=1` | an asm MLA kernel, **not** a Triton `_fwd_kernel_*` name |
| Which sparse path fired | `AITER_LOG_MORE=1` | matches what you expect for this arch, per the table above |
| Experimental paths available | `AITER_ENABLE_EXPERIMENTAL=1` set at **build** time | otherwise they were never compiled in |
| The win is real | decode tok/s, then end-to-end TPOT | isolated decode gains do not always survive |
| fp8 KV is safe | a task metric (e.g. gsm8k), not `allclose` | quantized KV needs an accuracy gate |

The AMD MLA blog and the AI-Developer-Hub notebook both give a runnable `mla_decode_fwd` example if
you need to confirm the path on a fresh box.

## Failure modes
| Symptom | Cause | Fix |
|---|---|---|
| Triton kernel name in the trace, expected asm | shape violates the fast-path contract | check `num_heads_kv == 1` and `page_size == 1` |
| A v4 / experimental path "isn't there" | `AITER_ENABLE_EXPERIMENTAL` unset when the JIT built | rebuild with it set; a runtime flip is too late |
| Sparse MLA behaves differently after a box change | gen-specific paths, not a fallback ladder | re-read the generation table; confirm with the log |
| Setting `num_kv_splits` made it slower | overrode a CU-aware heuristic | set it back to `None` |
| Accuracy drift after enabling fp8 KV | quantization error, not a bug | gate on a task metric; consider bf16 KV |

## Numerics
Matrix absorption is exact — bf16 MLA is parity-safe against standard MLA. What is *not* parity-safe:
fp8 KV-cache and fp8 fmha, both of which introduce quantization error that a `allclose` check will
happily pass while the model degrades. Use the Triton MLA reference for correctness cross-checks and
a task metric for the accuracy gate.

## Deeper
[operator_catalog.md](../../../overall/operator_catalog.md) — entry points and signatures for
`mla_attention`, `attention_prefill_fmha`, `attention_decode_paged` ·
[dispatch_and_rebind.md](../../../overall/dispatch_and_rebind.md) (how a backend is chosen and how to
prove engagement).

## Sources
- On-box `ROCm/aiter@b467ce342`: `aiter/mla.py` (`mla_decode_fwd:197`, `get_meta_param:125`,
  `mla_decode_fwd_v4_nm:1215`, `_fwd_kernel_stage2_asm`, context-parallel args),
  `csrc/py_itfs_cu/asm_mla_v4.cu`, `aiter/ops/mha.py` (`flash_attn_func`).
- 17× MLA decode and matrix absorption (AMD-reported, MI300X, tested 2025-03):
  https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
- `mla_decode_fwd` signature and a runnable example:
  https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.html
