---
title: attention_decode_paged on HIP (vLLM custom paged-attn) — SOTA card
kind: sota_card
operator: attention_decode_paged
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/tree/main/csrc/rocm
  - https://github.com/vllm-project/vllm/blob/main/csrc/rocm/torch_bindings.cpp
  - https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
---

# attention_decode_paged × HIP (vLLM custom paged-attn)

## TL;DR
vLLM ships its **own hand-written HIP paged-attention** decode kernel (`csrc/rocm/attention.cu`, the
`ROCM_ATTN` backend) — the **editable HIP source** for a Tier-C decode rewrite, and strong on decode
without AITER. Use it when you want to own/modify the decode kernel, or as the `ROCM_ATTN` fallback when
AITER lacks a path. It is splitKV/MFMA-based and templated on `BLOCK_SIZE` / KV dtype / fp8 KV.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| vLLM custom paged-attn (`ROCM_ATTN`) | `vllm-project/vllm:csrc/rocm/attention.cu` | gfx942/950; bf16/fp16/fp8 KV | strong decode without AITER; **2.7–4.4× slower** when KV head size unsupported (falls to Triton) | editable decode; AITER-free path |

Kernels to grep in a profile:
- `paged_attention_ll4mi_QKV_mfma16_kernel` — MFMA-16 main path
- `paged_attention_ll4mi_QKV_mfma4_kernel` — MFMA-4 small-head path
- `paged_attention_ll4mi_reduce_kernel` — cross-split softmax reduce

## Config space / knobs
`VLLM_ROCM_CUSTOM_PAGED_ATTN=1` (engage). Template params: `BLOCK_SIZE` (page size), KV dtype, fp8 KV.
When rewriting: `matrix_instr_nonkdim=16`, `waves_per_eu` (decode is memory-bound → 3-4), splitKV
partition size, KV-cache layout (the reshaped `[2, num_blocks, block_size*kv_heads*head]` with `x` inner
split for 128-bit reads). Exposed via `rocm_ops.def("paged_attention", ...)` in `torch_bindings.cpp`.

## Numerics / parity
fp32 online-softmax accumulate; splitKV reduce. fp8 KV uses scaled reads (fnuz on gfx942 — wrong dialect
off by 2×). Custom-HIP vs AITER vs Triton reduction order differs → re-check greedy temp=0 parity. See
[numerics.md](numerics.md).

## Integration (rebind seam)
`--attention-backend ROCM_ATTN`. The HIP source (`attention.cu`) is the **Tier-C edit seam** — edit and
**rebuild vLLM** (not a Python-only change). `torch_bindings.cpp` is the registration surface. Dispatch
order on gfx942 (`vllm/platforms/rocm.py`): ROCM_ATTN → ROCM_AITER_UNIFIED_ATTN → TRITON_ATTN, with
AITER MLA/MHA inserted when enabled.

## Pitfalls & anti-patterns
- **V0-era vars silently ignored on V1** (`VLLM_USE_TRITON_FLASH_ATTN`, `VLLM_USE_ROCM_FP8_FLASH_ATTN`) —
  selection is the `--attention-backend` enum + AITER hierarchy.
- KV head size unsupported by the HIP path → Triton decode fallback (2.7–4.4× slower).
- Editing `csrc/rocm/*.cu` requires a vLLM rebuild.
- fp8 KV is an accuracy gate; fnuz re-check on MI300X.

## How to verify
rocprofv3 kernel-trace → confirm `paged_attention_ll4mi_*` actually ran (not a Triton fallback). Isolated
decode bench vs `ROCM_AITER_FA` / `TRITON_ATTN` at the served batch. Greedy temp=0 parity.

## Alternatives / cross-links
aiter · vllm_kernels · triton ·
`backends/vllm_kernels/rocm_kernels.md` · `hip_levers/` · ../overview.md.

## Sources
- vLLM ROCm custom HIP kernels (`attention.cu`, `ll4mi_*` names): https://github.com/vllm-project/vllm/tree/main/csrc/rocm
- ROCm op registration: https://github.com/vllm-project/vllm/blob/main/csrc/rocm/torch_bindings.cpp
- Dispatch order / ROCM_ATTN fallback cliff: https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py ; https://vllm.ai/blog/2026-02-27-rocm-attention-backend
