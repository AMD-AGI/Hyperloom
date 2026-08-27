---
title: gqa_mqa_attention on aiter — SOTA card
kind: sota_card
operator: gqa_mqa_attention
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/paged_attn.py
  - ROCm/aiter@b467ce342:aiter/ops/triton/attention/unified_attention.py
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
  - https://github.com/ROCm/aiter
---

# gqa_mqa_attention × aiter

## TL;DR (one-line decision)
> aiter's FMHA and paged-attn kernels handle GQA/MQA **natively via in-register KV broadcast** —
> `gqa_ratio` 1..32 in the custom paged path; `BLOCK_M` packs the query group in the Triton path. Since
> modern LLMs (Llama-3, Qwen, Mistral) are all GQA, this is just the **default attention** with the right
> head pairing, and the AITER FA **1.2–4.4× TPS** numbers are measured on these GQA models. Use aiter; the
> only thing to get right is sizing the KV-cache to `num_kv_heads` (no replication).

## SOTA implementation(s)
| impl | source (`repo@commit:path`) | gens / dtypes / shapes | measured perf (`value @ hw, date`) | when it's best |
|---|---|---|---|---|
| aiter custom paged-attn (GQA) | `ROCm/aiter@b467ce342:aiter/paged_attn.py` | gfx942/950; bf16/fp16/fp8 KV; **gqa_ratio 1..32**; head sizes 64..256 | **1.2–4.4× TPS vs generic FA** on GQA models @ MI300X/325X/355X, 2026-01-29 (vendor) | default GQA/MQA decode |
| aiter FA (GQA/MQA via fewer KV heads) | `aiter/ops/mha.py:2533` | head_dim ≤256; q heads divisible by kv heads | the prefill GQA path (vendor envelope above) | default GQA/MQA prefill |
| aiter Triton unified (GQA `BLOCK_M` pack; RDNA overrides `BLOCK_M=64`) | `aiter/ops/triton/attention/unified_attention.py` | any head dim | editable GQA path | mixed-batch / fallback |

**Real GQA mechanics** — custom path gates `gqa_ratio ∈ [1,32]` and sizes the cache to `num_kv_heads`
(`aiter/paged_attn.py`):
```python
def _use_rocm_custom_paged_attention(qtype, head_size, block_size, gqa_ratio, max_seq_len):
    return (not _ON_NAVI and (gqa_ratio >= 1 and gqa_ratio <= 32) and max_seq_len <= 65536)
@staticmethod
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
    return (2, num_blocks, block_size * num_kv_heads * head_size)   # sized to KV heads, no replication
```
Triton path packs the query group into the tile rather than replicating KV
(`unified_attention.py`):
```python
num_queries_per_kv = num_query_heads // num_kv_heads
BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
BLOCK_Q = BLOCK_M // num_queries_per_kv     # one KV load serves the whole group
```
`flash_attn_func` docstring states it directly: "Supports MQA/GQA by passing in KV with fewer heads than
Q… the number of heads in Q must be divisible by the number of heads in KV."

## Config space / knobs
| param | range / values | effect | default |
|---|---|---|---|
| `(num_q_heads, num_kv_heads)` | model (R = q/kv) | the GQA ratio | model |
| `gqa_ratio` | 1..32 (custom path) | KV broadcast in-register | model |
| head size | 64/80/96/112/120/128/192/256 | else Triton fallback | model |
| KV dtype | bf16 / fp8_e4m3 **FNUZ** | memory + bandwidth | bf16 |
| `block_size` (`--page-size`) | 16/32/64 | KV page granularity | 16 |

Framework: `VLLM_ROCM_USE_AITER=1` + `VLLM_ROCM_USE_AITER_MHA=1`; sglang `--attention-backend aiter`.
Decode levers inherited from ../../attention_decode_paged/backends/aiter.md.

## Numerics / parity
GQA is **bit-identical** to MHA-with-shared-KV (broadcast has zero numerical cost vs explicit `repeat_kv`).
fp8 KV with GQA is slightly more accuracy-sensitive (one KV head's quant error is shared across the whole
query group) — task-accuracy gate. fp8 dialect FNUZ gfx942 / OCP gfx950. See [numerics.md](numerics.md).

## Integration (rebind seam)
- **sglang:** `--attention-backend aiter`. **vLLM:** `ROCM_AITER_FA` (+ master `VLLM_ROCM_USE_AITER=1`).
- The KV-cache is allocated at `num_kv_heads` via `get_kv_cache_shape` — the framework already does this;
  don't override it to `num_q_heads`.
- **Verify it engaged:** `AITER_LOG_MORE=1` for asm/CK path; confirm KV-cache footprint = `num_kv_heads`.

## Pitfalls & anti-patterns
- `gqa_ratio` outside 1..32 → falls back (Triton decode cliff, 2.7–4.4× slower).
- **Don't `repeat_kv` before calling aiter** — that materializes KV at `num_q_heads` in HBM and throws
  away the entire GQA bandwidth win (the #1 GQA anti-pattern).
- Head size not in the supported list → Triton fallback.
- MQA (ratio = num_q_heads, kv=1) still works as long as ratio ≤32; ratio >32 (rare, huge models) falls
  back.
- Master switch `VLLM_ROCM_USE_AITER=1` required even when forcing the backend.

## Worked example
Llama-3-70B: 64 q heads, 8 kv heads (ratio 8), d=128, bf16:
1. ratio 8 ∈ [1,32], head 128 ∈ supported → custom paged path engages for decode; FA for prefill.
2. KV-cache shaped `(2, blocks, block_size·8·128)` — confirm footprint uses **8**, not 64.
3. `AITER_LOG_MORE=1` → asm; bench vs the same model forced through MHA-shaped KV to see the ~8× KV
   bandwidth/footprint win.
4. fp8 KV: set scales, FNUZ on gfx942, gsm8k gate (group-shared quant error).

## How to verify (bench + oracle)
Confirm KV-cache footprint is `num_kv_heads` (not `num_q_heads`); `AITER_LOG_MORE=1` for the asm/CK path;
isolated bench at the model's head ratio vs an MHA-shaped KV to see the bandwidth win; greedy temp=0
parity. Gate: footprint = kv_heads AND engaged AND parity.

## Alternatives / cross-links
./triton.md · ./ck.md · ../../attention_prefill_fmha/backends/aiter.md ·
../../attention_decode_paged/backends/aiter.md · ../../mla_attention/backends/aiter.md ·
../overview.md.

## Sources
- aiter `gqa_ratio` 1..32, `get_kv_cache_shape` uses num_kv_heads, GQA-native `flash_attn_func`, Triton `BLOCK_M`/`BLOCK_Q` group packing: on-box `ROCm/aiter@b467ce342:aiter/paged_attn.py`, `aiter/ops/mha.py:2533`, `aiter/ops/triton/attention/unified_attention.py`.
- 1.2–4.4× TPS on GQA models (vendor, MI300X/325X/355X, 2026-01-29): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- aiter overview: https://github.com/ROCm/aiter
