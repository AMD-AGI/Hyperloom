---
title: gqa_mqa_attention — overview
kind: operator_overview
operator: gqa_mqa_attention
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/paged_attn.py
  - https://github.com/Dao-AILab/flash-attention
  - https://arxiv.org/abs/2305.13245
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---

# gqa_mqa_attention  (grouped-query / multi-query attention)

## TL;DR
GQA/MQA share a small number of KV heads across many query heads (`num_kv_heads ≪ num_q_heads`): MQA =
1 KV head for all, GQA = `G` KV heads each shared by `num_q_heads/num_kv_heads` query heads. It is **not a
separate kernel** — it is a **KV-broadcast trait** of the FMHA/paged-attn kernels: the KV head is read
once and **broadcast in-register** across its query-head group, so the KV-cache is never replicated in
HBM. The whole point is to **cut KV-cache bandwidth** (decode) and KV memory; the kernel-level lever is
the **`gqa_ratio` = q-heads per kv-head** and making sure the broadcast happens in registers, not via a
physical KV expand.

## Math contract
Same as MHA `O = softmax(QKᵀ·scale + mask)·V`, but each KV head `j` is shared by query heads
`[j·R, (j+1)·R)` where `R = num_q_heads / num_kv_heads` (`gqa_ratio`). MQA = `num_kv_heads=1`. The kernel
maps the (head, kv_head) pairing so a loaded K/V tile serves all R query heads. dtype: bf16/fp16 (+ fp8
KV), fp32 online-softmax accumulate.

## The anti-pattern: KV replication
A naive implementation **materializes** the KV-cache to full `num_q_heads` by repeating each KV head R
times (`repeat_kv`) — this throws away the entire GQA bandwidth/memory benefit. The correct AMD kernels
(aiter paged-attn, FA-ROCm, CK FMHA) broadcast the KV head in-register. `gqa_ratio` in aiter's
`_use_rocm_custom_paged_attention` must be **1..32** for the custom path.

## Shape regimes
- **Decode**: this is where GQA matters most — KV bandwidth is the bottleneck and sharing KV heads cuts
  it by R. `gqa_ratio` is the lever; see ../attention_decode_paged/overview.md.
- **Prefill**: GQA is a trait of the FMHA forward; less bandwidth-critical (GEMM-bound) but still avoids
  KV replication. See ../attention_prefill_fmha/overview.md.

## Where it matters (Amdahl)
Modern LLMs (Llama-3, Qwen2/3, Mistral) are all GQA — so "GQA attention" *is* the attention hot path for
those models, not a niche variant. The bandwidth saving directly improves decode TPOT; it is folded into
the AITER FA 1.2–4.4× TPS story (the FA kernels broadcast KV correctly).

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (KV broadcast in FA/paged-attn) | aiter |
| triton | 🟡 (FA/paged Triton; KV broadcast trait) | triton |
| ck | 🟡 (CK FMHA MQA/GQA trait) | [ck.md](ck.md) |
| fa_rocm | 🟡 (MQA/GQA feature, both backends) | fa_rocm |

## Fusion neighbors
KV-write + RoPE + quant (pre); the broadcast is intra-kernel (not a fusion). Same neighbors as MHA. See
[fusion.md](fusion.md).

## Numerics
Identical to MHA up to the head-pairing; in-register broadcast vs physical replication are bit-identical
(same K/V values). fp8 KV is the quant gate. See [numerics.md](numerics.md).

## How to bench
Isolated FMHA/paged-attn bench with `num_q_heads / num_kv_heads` set to the model's ratio (e.g. 8 for
Llama-3 8B: 32 q / 4 kv → R=8); compare against the same shape with `num_kv_heads=num_q_heads` (MHA) to
confirm the bandwidth win. e2e via `--attention-backend` + TPOT.

## Sources
- aiter paged-attn `gqa_ratio` 1..32 (in-register broadcast): on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/paged_attn.py`.
- MQA/GQA as an FA feature (both FA-ROCm backends): https://github.com/Dao-AILab/flash-attention
- GQA definition: https://arxiv.org/abs/2305.13245
- AITER FA TPS (GQA models): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
