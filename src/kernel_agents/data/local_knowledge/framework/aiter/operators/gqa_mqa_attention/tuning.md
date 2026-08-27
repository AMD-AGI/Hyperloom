---
title: gqa_mqa_attention — tuning
kind: operator_overview
operator: gqa_mqa_attention
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/paged_attn.py
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
---

# gqa_mqa_attention — tuning

GQA/MQA is a **trait** of the FMHA and paged-attn kernels, so tuning = the FMHA/decode tuning (see
../attention_prefill_fmha/tuning.md and ../attention_decode_paged/tuning.md) **plus** the
GQA-specific mapping that keeps the KV broadcast in registers and exploits the shared KV head.

## Lever 1 — keep the broadcast in-register (the correctness-of-perf lever)
The #1 GQA mistake is `repeat_kv` (physically expanding KV to `num_q_heads`) — it discards the bandwidth
win. The right kernels map the head grid so a loaded K/V tile serves all R = `num_q_heads/num_kv_heads`
query heads. In aiter `gqa_ratio` must be **1..32** for the custom paged path; outside that range it falls
back. Verify in a profile that the KV-cache size is `num_kv_heads`, not `num_q_heads`.

## Lever 2 — pack the query-head group into the MFMA tile
With R query heads sharing one KV head, you can put the R query heads of a group into the **M dimension**
of the `q·Kᵀ` MFMA tile — so one K tile load feeds an R-row Q tile. This raises arithmetic intensity per
KV byte (the GQA win, expressed at the tile level). Choose `BLOCK_M` to be a multiple of (or aligned to) R
where the kernel supports it. This is what makes GQA decode faster than MQA-with-replication.

## Lever 3 — decode is the bandwidth regime
GQA's payoff is decode KV bandwidth. All the decode levers apply: splitKV (`num_kv_splits` / partition
256), KV-cache layout (NHD/HND, coalesced 128-bit reads), block_size, fp8 KV. The KV broadcast multiplies
the effect of fp8 KV (less bandwidth × shared across R heads).

## Lever 4 — MFMA / occupancy
`matrix_instr_nonkdim=16`, `num_stages=1` (fused FA / decode), `num_warps=4` (wave64),
`waves_per_eu∈{2,3,4}` decode. Supported head sizes (aiter paged) 64/80/96/112/120/128/192/256 — head dim
192/256 (CDNA3/4) combine with GQA in long-context models.

## CDNA3 vs CDNA4
- LDS 64 KB (gfx942) / 160 KB (gfx950): packing R query heads into the Q tile is LDS-heavier; gfx950's
  larger LDS allows bigger R-packed tiles.
- fp8 KV: FNUZ on gfx942, OCP on gfx950 (wrong dialect off by 2×).

## Model ratios (set these to the model)
| model | num_q_heads | num_kv_heads | R (gqa_ratio) |
|---|---|---|---|
| MQA (e.g. some GPT variants) | H | 1 | H |
| Llama-3 8B | 32 | 8 | 4 |
| Llama-3 70B | 64 | 8 | 8 |
| Qwen2/3 (varies) | e.g. 28 | 4 | 7 |
Set the kernel's `(num_q_heads, num_kv_heads)` to the model — do not bench MHA shapes and assume GQA perf.

## How to verify a tune helped
Isolated bench at the model's `(num_q_heads, num_kv_heads, head_dim, context, dtype)`; confirm KV-cache
footprint is `num_kv_heads` (no replication); compare vs MHA (`num_kv_heads=num_q_heads`) to see the
bandwidth win; rocprofv3 to confirm the GQA-aware kernel ran; e2e TPOT + parity.

## Sources
- aiter `gqa_ratio` 1..32 in-register broadcast: on-box `ROCm/aiter@b467ce342:aiter/paged_attn.py`.
- MFMA / occupancy / ≥1024 grid: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Triton knobs (num_stages=1, wave64): https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
