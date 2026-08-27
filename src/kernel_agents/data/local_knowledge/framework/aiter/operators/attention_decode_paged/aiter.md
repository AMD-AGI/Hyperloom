---
title: attention_decode_paged on aiter — SOTA card
kind: sota_card
operator: attention_decode_paged
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/paged_attn.py
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
  - https://github.com/ROCm/aiter
---

# attention_decode_paged × aiter

## TL;DR (one-line decision)
> aiter owns the **default paged decode attention** on AMD serving (`paged_attention_v1`/`v2` →
> `paged_attention_rocm` asm, plus the AITER FA decode path). It dispatches to a hand-tuned asm/CK kernel
> with **split-KV (partition 256)** and a Triton fallback, supports **fp8 KV-cache (FNUZ)**, GQA
> `gqa_ratio` 1–32, head sizes 64–256, and `seqlen ≤ 65536`. Backend ranking (vLLM ROCm, Feb 2026):
> **ROCM_AITER_FA > ROCM_AITER_UNIFIED_ATTN > TRITON_ATTN > ROCM_ATTN**. ROCM_AITER_FA vs ROCM_ATTN TPS
> (64/128 req): **MI300X 3.82×/2.65×, MI325X 4.36×/3.12×, MI355X 3.61×/2.88×**; TPOT **2.8–4.6×** faster.
> Use it as the default; reach for vLLM's custom HIP paged-attn only for an editable kernel or when aiter
> lacks a path for your shape.

## SOTA implementation(s)
| impl | source (`repo@commit:path`) | gens / dtypes / shapes | measured perf (`value @ hw, lib, date`) | when it's best |
|---|---|---|---|---|
| aiter `paged_attention_v2` (split-KV) | `ROCm/aiter@b467ce342:aiter/paged_attn.py:90` | gfx942/950; bf16/fp16/fp8 KV; head sizes 64/80/96/112/120/128/192/256; gqa 1–32; seqlen ≤65536; **partition 256** | **ROCM_AITER_FA vs ROCM_ATTN TPS (64/128 req): MI300X 3.82×/2.65×, MI325X 4.36×/3.12×, MI355X 3.61×/2.88×; TPOT 2.8–4.6×** @ vLLM ROCm blog, Feb 2026 (vendor) | default decode, long ctx, large batch |
| aiter `paged_attention_v1` (single-pass) | `aiter/paged_attn.py:46` | short context (fits 1 partition) | fewer launches at short ctx | low-latency short-context decode |
| `PagedAttention.forward_decode` (custom rocm) | `aiter/paged_attn.py:225` | `gqa 1–32 & seqlen ≤65536 & not Navi` gate | the production fast path; fp8 out + MTP | sglang/vLLM decode |

**Real gate + partition + supported sizes** (`aiter/paged_attn.py`):
```python
_PARTITION_SIZE_ROCM = 256
def _use_rocm_custom_paged_attention(qtype, head_size, block_size, gqa_ratio, max_seq_len):
    return (not _ON_NAVI and (gqa_ratio >= 1 and gqa_ratio <= 32) and max_seq_len <= 65536)
class PagedAttention:
    @staticmethod
    def get_supported_head_sizes(): return [64, 80, 96, 112, 120, 128, 192, 256]
# forward_decode: max_num_partitions = (max_seq_len + 256 - 1) // 256
torch.ops.aiter.paged_attention_rocm(output, exp_sums, max_logits, tmp_output, query,
    key_cache, value_cache, num_kv_heads, scale, block_tables, seq_lens, block_size,
    max_seq_len, alibi_slopes, kv_cache_dtype, k_scale, v_scale, fp8_out_scale,
    _PARTITION_SIZE_ROCM, q_scale=q_scale, mtp=mtp)
```
KV-cache is **split-layout** (`split_kv_cache`): key reshaped `(blocks, kv_heads, d//x, block, x)` with
`x = 16 // elem_size` (the "x-factor" 128-bit-vector layout), value `(blocks, kv_heads, d, block)`.

## Config space / knobs
| param | range / values | effect | default |
|---|---|---|---|
| `_PARTITION_SIZE_ROCM` | 256 (fixed) | split-KV chunk; `max_num_partitions = ceil(seqlen/256)` | 256 |
| `block_size` (`--page-size`) | 16 / 32 / 64 | KV page granularity | 16 |
| `kv_cache_dtype` | `auto` / `fp8` (e4m3 **FNUZ**) | KV memory + bandwidth | auto (bf16) |
| `gqa_ratio` | 1–32 | KV broadcast in-register | model |
| `head_size` | 64/80/96/112/120/128/192/256 | else → Triton fallback | model |
| `k_scale`/`v_scale`/`q_scale` | fp32 tensors | fp8 (de)scale | required for fp8 |
| `mtp` | ≥1 | multi-token-predict decode width | 1 |
| `fp8_out_scale` | tensor / None | fp8 output | None |

Framework gates: `VLLM_ROCM_USE_AITER=1` + `VLLM_ROCM_USE_AITER_MHA=1`; sglang `SGLANG_USE_AITER=1`,
`--decode-attention-backend aiter`; `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` (AITER FA, concurrency ≥32).

## Numerics / parity
fp32 online-softmax accumulate; split-KV reduce uses per-split `m_i`/`exp_sums`/`max_logits` (the three
scratch tensors above). fp8 KV is a task-accuracy gate — **FNUZ on gfx942, OCP on gfx950**; wrong
dialect off ~2×. Cross-backend bf16 tie-flips benign. See [numerics.md](numerics.md).

## Integration (rebind seam)
- **sglang:** `--attention-backend aiter` / `--decode-attention-backend aiter`.
- **vLLM:** `--attention-backend ROCM_AITER_FA` (+ master `VLLM_ROCM_USE_AITER=1`).
- `PagedAttention.forward_decode(...)` / `paged_attention_v2(...)` is the Python seam (monkeypatchable).
- **Verify it engaged:** `AITER_LOG_MORE=1`; rocprofv3 → `paged_attention_rocm`/asm (not Triton).

## Pitfalls & anti-patterns
- Head size **not** in 64/80/96/112/120/128/192/256 → Triton decode fallback (`ROCM_ATTN` cliff,
  2.7–4.4× slower).
- `max_seq_len > 65536` leaves the custom fast path.
- **Navi (gfx1*) not supported** — `_use_rocm_custom_paged_attention` returns False on Navi.
- `gqa_ratio > 32` falls back.
- fp8 requires `k_scale`/`v_scale` tensors (not floats in the new path) — and the right FNUZ/OCP dialect.
- Master switch `VLLM_ROCM_USE_AITER=1` required even when forcing the backend.

## Worked example
Llama-3-70B decode, GQA 64/8 (ratio 8), d=128, ctx grows to 32K, fp8 KV on MI300X:
1. head=128 ∈ supported, gqa=8 ∈ [1,32], seqlen 32K ≤ 65536, not Navi → custom path engages.
2. `max_num_partitions = ceil(32768/256) = 128` split-KV chunks → `paged_attention_v2`.
3. fp8 KV: set `kv_cache_dtype=fp8`, pass `k_scale`/`v_scale`, confirm **FNUZ** on gfx942.
4. `AITER_LOG_MORE=1` → asm; bench TPOT vs HIP/Triton; gsm8k accuracy gate for fp8.

## How to verify (bench + oracle)
`AITER_LOG_MORE=1`; rocprofv3 Top-N → `paged_attention_rocm`/asm-CK decode (not Triton). Isolated decode
bench vs vLLM-HIP / Triton at `(batch, heads, kv_heads, head_dim, context, block_size, dtype)`. TPOT +
greedy temp=0 parity. Gate: win AND parity AND engaged.

## Alternatives / cross-links
./ck.md · ./triton.md (fallback cliff) · ../../attention_prefill_fmha/backends/aiter.md ·
../../mla_attention/backends/aiter.md · ../../gqa_mqa_attention/backends/aiter.md ·
`backends/aiter/attn_mla.md` · ../overview.md.

## Sources
- On-box `ROCm/aiter@b467ce342:aiter/paged_attn.py` (v1:46, v2:90, `_PARTITION_SIZE_ROCM=256`, `get_supported_head_sizes`, gqa 1–32, seqlen ≤65536, split_kv_cache x-layout, `forward_decode:225`/`paged_attention_rocm`, fp8 k/v/q_scale, mtp).
- ROCM_AITER_FA vs ROCM_ATTN TPS (MI300X 3.82×/2.65×, MI325X 4.36×/3.12×, MI355X 3.61×/2.88× @ 64/128 req; TPOT 2.8–4.6×), backend ranking ROCM_AITER_FA > ROCM_AITER_UNIFIED_ATTN > TRITON_ATTN > ROCM_ATTN (vendor, Feb 2026): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- aiter overview: https://github.com/ROCm/aiter
