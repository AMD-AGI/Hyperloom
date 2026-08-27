---
title: softmax on triton — SOTA card
kind: sota_card
operator: softmax
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/triton/softmax.py
  - /sgl-workspace/aiter/aiter/ops/triton/_triton_kernels/softmax.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# softmax × triton

## TL;DR
Triton is the authorable SOTA for standalone softmax (routing/logits) on MI300X — aiter ships its own
online-softmax Triton kernel. For attention softmax, the softmax is *inside* the FMHA Triton kernel, not
this one. Memory-bound → Triton matches hand-written.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter `_softmax_kernel_online` | `aiter/ops/triton/softmax.py`, `_triton_kernels/softmax.py` | gfx942/950, bf16/fp16/fp32 | online single-pass, `num_warps=8`, fp32 exp/sum | standalone row softmax (routing/logits) |
| FMHA online softmax (inside attn) | attention_prefill_fmha triton card | gfx942/950 | `num_stages=1`, `schedule_hint="attention"` | attention (the real case) |
| Triton softmax tutorial | https://triton-lang.org tutorials | generic | row-per-program | learning |

## Config space / knobs
- `num_warps=8` for wide rows (vocab); 2–4 for narrow (n_experts).
- `BLOCK_SIZE=next_pow2(N)`; round reduced dim to pow2 for full wave64 reduce.
- Grid: row-per-program; persistent `min(M, num_sms)` for few rows.
- `num_stages=1–2`; fp32 compute.

## Numerics / parity
Max-subtraction (online running max); fp32 exp+accumulate; online = exact up to rounding; reduction order
differs from naive → greedy re-gate. See [numerics.md](numerics.md).

## Integration (rebind seam)
- Direct: `from aiter.ops.triton.softmax import softmax`.
- torch.compile: Inductor emits Triton softmax for `F.softmax` / attention decompositions under
  max-autotune.

## Pitfalls & anti-patterns
- `num_warps=8` is right for *wide* softmax but spills if the row is narrow and VGPR-heavy — drop for
  routing.
- Skipping max-subtraction → NaN on large logits.
- Materializing the full attention score matrix instead of online → LDS blowup; use the FMHA path.

## How to verify
`TRITON_PRINT_AUTOTUNING=1`; isolated bench at routing/vocab N; fp64 oracle; greedy parity.

## Alternatives / cross-links
aiter · hip · attention_prefill_fmha · moe_routing_topk ·
triton_levers/patterns §5.

## Sources
- aiter online softmax: `/sgl-workspace/aiter/aiter/ops/triton/softmax.py`, `_triton_kernels/softmax.py`.
- wave64 reduce / pow2 dim: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html.
