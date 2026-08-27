---
title: sampling_topk_topp on triton — SOTA card
kind: sota_card
operator: sampling_topk_topp
backend: triton
gens: [gfx942, gfx950]
dtypes: [fp32, bf16]
regimes: [decode, both]
status: competitive
updated: 2026-06-08
sources:
  - https://vllm.ai/blog/mrv2
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - https://flashinfer.ai/2025/03/10/sampling.html
  - https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/sampler.py
---

# sampling_topk_topp × triton

## TL;DR
Triton is **competitive and increasingly relevant** for sampling: vLLM's **Model Runner V2 (MRV2)**
reworks sampling around **custom Triton kernels** — a **Gumbel-Max** kernel that samples without
materializing softmax (stateless in-kernel RNG) and a **sort-free top-k logprobs** kernel (find top-k
logits first, compute logprobs only for the selected candidates). On AMD these Triton kernels run
unmodified (wave64/knob caveats apply). For top-k/top-p *filtering*, a Triton implementation of the
dual-pivot rejection loop is feasible and portable, but the production AMD path is aiter's HIP port — Triton
shines for the **Gumbel sample + logprobs** pieces and as the editable Tier-C prototype.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| MRV2 Gumbel-Max Triton sample (no explicit softmax, stateless RNG) | `vllm.ai/blog/mrv2` (vLLM MRV2) | gfx942/950, fp32 | softmax-free, sync-free draw; better memory/numerics control | the no-sync random draw in MRV2 |
| MRV2 sort-free top-k logprobs Triton kernel | same | gfx942/950, fp32 | logprobs only for top-k candidates (not full `[M,V]`) | logprobs alongside sampling |
| SGLang batched top-k/top-p/min-p (vectorized torch, sort-based) | `sgl-project/sglang@HEAD:.../sampler.py` | gfx942/950, fp32 | per-row masking via arange/cumsum; the sort fallback when no kernel | pytorch backend / fallback |
| hand-Triton dual-pivot rejection filter | author (triton_amd patterns.md) | gfx942/950, fp32 | portable sorting-free filter; prototyping | editable Tier-C; aiter HIP is the prod path |

## Config space / knobs
One-row-per-program (or per-block) reduction over `V` (see triton_amd knobs.md, memory-bound):
- `BLOCK` = vocab chunk per program; `num_warps=2..4` (memory-bound; **not** 8 — wave64 spill);
  `num_stages=1` (no GEMM). `knobs.amd.use_buffer_ops=1` for masked tail loads of the `V`-row.
- A block-wide reduction (max for min-p floor, sum for top-p CDF, argmax for Gumbel) over `V` — use Triton
  `tl.max`/`tl.cumsum`/`tl.argmax`; stateless RNG via `tl.rand` seeded by `(seed, row, offset)`.
- fp32 throughout; deterministic prefix-sum if reproducibility is required ([numerics.md](numerics.md)).

## Numerics / parity
fp32; Gumbel-Max draw is statistically equivalent (KL gate). Non-associative `tl.cumsum` for the CDF →
deterministic-scan concern; tie-break lowest index. MRV2 keeps Gumbel math careful for numerics; SGLang
uses fp64 for the seeded Gumbel. See [numerics.md](numerics.md).

## Integration (rebind seam)
MRV2 Triton sampling kernels are wired inside vLLM's MRV2 model runner (no user flag for the kernel — it's
the runner). A hand-Triton sampler registers as a custom op / `torch.compile` path. Verify: rocprofv3 shows
a Triton-named sampling kernel and **no D2H sync**.

## Pitfalls & anti-patterns
- `num_warps=8` (NVIDIA habit) → wave64 VGPR spill (triton_amd pitfalls).
- A sort-based Triton top-k (`tl.sort`) over full `V` defeats the purpose — do the find-top-k-then-logprobs
  pattern (MRV2) or the rejection loop.
- bf16 logits → biased thresholds; fp32.
- Forgetting the deterministic-scan option when reproducibility is needed.

## How to verify
`TRITON_PRINT_AUTOTUNING=1`; statistical KL gate vs sorted multinomial; greedy exact parity; rocprofv3
confirms Triton kernel + no D2H sync; latency vs aiter at `(M, V=128k–256k)`.

## Alternatives / cross-links
aiter (prod HIP path) · hip · vllm_kernels ·
[overview.md](overview.md) · triton_amd · argmax_topk · cumsum_scan · softmax.

## Sources
- MRV2 Triton sampling (Gumbel-Max no-softmax, stateless RNG, sort-free top-k logprobs): https://vllm.ai/blog/mrv2
- AMD Triton backend knobs / codegen: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- Sorting-free algorithm (portable): https://flashinfer.ai/2025/03/10/sampling.html
- SGLang batched per-row top-k/top-p/min-p + fp64 Gumbel: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/sampler.py
