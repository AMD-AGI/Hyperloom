---
title: softmax on hip — SOTA card
kind: sota_card
operator: softmax
backend: hip
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# softmax × hip

## TL;DR
Hand-written HIP softmax is the reference: a wave64 `__shfl` max-reduce then sum-reduce over a 128-bit
vectorized row, fp32 exp. Reach for HIP when fusing softmax into a custom kernel (e.g. a fused router or
attention); standalone, aiter Triton already hits the bandwidth floor. vLLM's custom paged-attention HIP
kernel contains the canonical fused online softmax (`paged_attention_ll4mi_*`).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| HIP wave64 max/sum softmax | AMD lab-notes / hand-written | gfx942/950, bf16/fp16/fp32 | bandwidth-bound | fused into a custom kernel |
| vLLM paged-attn online softmax (HIP) | `vllm/csrc/rocm/attention.cu` (`paged_attention_ll4mi_*`, `_reduce_kernel`) | gfx942/950 | fp32 online accumulate; cross-block reduce kernel | decode attention softmax |
| aiter asm/Triton (recommended standalone) | aiter | gfx942/950 | floor | standalone |

## Config space / knobs
- Block ×64; two wave64 reductions (`__shfl_down` over 64 lanes, 6 steps): max, then sum.
- Vector load x (`float4`/`__half2`), `__restrict__`, fp32 exp.
- For wide N: persistent grid-stride over the row + cross-block reduce (vLLM's `_reduce_kernel` pattern).
- `__launch_bounds__(block, 4)`; `hipcc --offload-arch=gfx942 -O3`.

## Numerics / parity
Max-subtract; fp32 exp/accumulate; online correction in attention (`exp(m_old−m_new)` rescale). Reduction
order differs from Triton/CK → greedy re-gate. See [numerics.md](numerics.md).

## Integration (rebind seam)
- vLLM custom paged-attn: `--attention-backend ROCM_ATTN`; edit `csrc/rocm/attention.cu`, rebuild.
- Standalone: `torch.utils.cpp_extension`.

## Pitfalls & anti-patterns
- ⚠ No max-subtract → NaN.
- ⚠ `dim3 grid(num_tokens)` with 0 tokens → crash (guard).
- 32-bit shuffle mask on wave64 → static-assert; use `unsigned long long`.

## How to verify
`--save-temps` ISA; fp64 oracle no-NaN; rocprofv3 confirms `paged_attention_ll4mi_*` ran (attention case);
greedy parity.

## Alternatives / cross-links
aiter · triton · backends/vllm_kernels/rocm_kernels ·
hip_levers/patterns §1 · attention_decode_paged.

## Sources
- wave64 reduce: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html, https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/.
- vLLM paged-attn online softmax: https://github.com/vllm-project/vllm/tree/main/csrc/rocm.
