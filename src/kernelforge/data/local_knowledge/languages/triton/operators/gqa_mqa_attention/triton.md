---
title: gqa_mqa_attention on Triton — SOTA card
kind: sota_card
operator: gqa_mqa_attention
backend: triton
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/mha.py
  - https://github.com/Dao-AILab/flash-attention
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
---

# gqa_mqa_attention × Triton

## TL;DR
Triton FA / paged-attn support GQA/MQA as a KV-broadcast trait — `MQA-GQA` is an explicit feature of the
FA-ROCm Triton backend (aiter kernels). It is the **editable** and **universal-fallback** GQA path;
slower than aiter asm on most shapes, but it runs any ratio/head dim and is the reference. Use it for
portability, a custom GQA variant, or as the correctness oracle.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter Triton FA/paged (GQA trait) | `ROCm/aiter@a6bb49937:aiter/ops/triton/attention/mha.py`, `pa_decode.py` | gfx90a/942/950; bf16/fp16/fp8; any ratio/head dim | fallback; below aiter asm | portability / editable / reference |

## Config space / knobs
`BLOCK_M` aligned to R = `num_q_heads/num_kv_heads` where supported (pack the query-head group into the M
tile), `num_kv_splits` (decode), `num_warps=4` (wave64), `num_stages=1`, `matrix_instr_nonkdim=16`,
`waves_per_eu`. FA-ROCm: enable Triton backend (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`). See
[tuning.md](tuning.md).

## Numerics / parity
GQA bit-identical to MHA-with-shared-KV; fp32 accumulate. fp8 KV + GQA accuracy gate. See
[numerics.md](numerics.md).

## Integration (rebind seam)
sglang `--attention-backend triton`; vLLM `TRITON_ATTN`. The `@triton.jit` kernel is the Tier-C seam.

## Pitfalls & anti-patterns
- Don't `repeat_kv` before the kernel (replicates KV, kills the GQA win).
- `num_stages>1` hurts the fused kernel; keep at 1.
- Slower than aiter asm — the win is portability/editability.

## How to verify
`AMDGCN_ENABLE_DUMP=1` ISA check; in-register-broadcast vs `repeat_kv` reference = bit-identical (head
pairing sanity); isolated bench vs aiter at the model ratio; greedy temp=0 parity.

## Alternatives / cross-links
aiter · ck · [flash_attention_rocm.md](fa_rocm.md) ·
`triton_levers/` · ../overview.md.

## Sources
- aiter Triton FA/paged GQA (on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/mha.py`, `pa_decode.py`).
- MQA/GQA Triton FA feature: https://github.com/Dao-AILab/flash-attention
- Triton AMD knobs: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
