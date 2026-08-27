---
title: sampling_topk_topp — overview
kind: operator_overview
operator: sampling_topk_topp
gens: [gfx942, gfx950]
dtypes: [fp32, bf16]
regimes: [decode, both]
updated: 2026-07-14
sources:
  - https://flashinfer.ai/2025/03/10/sampling.html
  - ROCm/aiter@b467ce342:csrc/cpp_itfs/sampling/sampling.cuh
  - https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py
  - https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/sampler.py
---

# sampling_topk_topp  (logits → temperature → softmax → top-k/top-p/min-p → token)

## TL;DR
Token sampling is **memory-bound + tail-latency critical**: per step it reads the `[M,V]` logits
(V=128k–256k) once or twice and emits one token id per request. The decisive optimization is the
**sorting-free, fused, single-kernel rejection sampler** (FlashInfer's **Dual-Pivot Rejection Sampling**,
ported to HIP in **aiter** `csrc/cpp_itfs/sampling/sampling.cuh`): it does top-k/top-p/min-p **without
sorting the vocab** and **without a host sync**, in one threadblock per request (BLOCK_THREADS=1024,
vectorized loads, hipCUB BlockScan/BlockReduce). The fallback is a **PyTorch full-vocab sort** (~20%
sampling overhead, plus a GPU→CPU sync on the top-k-only path). On AMD the live paths are **aiter sampling
ops** (vLLM `forward_hip`, SGLang) — **FlashInfer itself is NVIDIA-only** (CUB/CCCL), so AMD uses aiter's
port. Greedy (temp=0) is a separate **argmax** fast path that skips all of this.

## Math contract
Per request row `p ∈ ℝ^V` (probabilities, or logits → softmax):
- **temperature**: `logits /= T` before softmax (T=0 ⇒ greedy argmax, no softmax). vLLM/SGLang require
  `temperature` and `top_p` in **fp32** (a dtype mismatch crashes the non-GPU sampler, vLLM #21936).
- **top-k**: keep the k highest-prob tokens (renormalize over them).
- **top-p (nucleus)**: keep the smallest set whose cumulative prob ≥ p.
- **min-p**: keep tokens with `p_i ≥ min_p · max_j p_j` (dynamic floor relative to the mode).
- **filter order**: vLLM applies **top-k then top-p**; FlashInfer historically applied them **jointly**
  (`filter_apply_order`) — a semantic difference that was later reconciled (first-top-k-then-top-p). aiter's
  joint kernel takes both `top_k` and `top_p` per-row arrays. Match the framework's order or distributions
  diverge.
- **sample**: draw from the renormalized distribution. The no-sync trick: instead of `torch.multinomial`
  (which forces a CPU-GPU sync), use **exponential/Gumbel noise + argmax** (`random_sample`,
  `random_sample_outer_exponential`) — statistically equivalent, fully on-device.
- output: one token id per request (`int`), optionally logprobs (computed from **raw** logits, see
  [numerics.md](numerics.md)).

## Shape regimes
- **decode** (the only regime that matters): `M`=batch (1..256), `V`=128k–256k. One threadblock per
  request row; each block streams the full `V`-row (vectorized `VEC_SIZE` loads). Latency-bound at small
  batch, bandwidth-bound at large batch.
- Per-request heterogeneity: every request can have a different `(temperature, top_k, top_p, min_p, seed)`
  → kernels take **per-row arrays** (`maybe_top_k_arr`, `maybe_top_p_arr`), not scalars. Batches mix greedy
  + random; the sampler splits them (greedy argmax for greedy rows, rejection sampler for random rows).

## Where it matters (Amdahl)
Sampling is small in **FLOPs** but a **tail-latency** hotspot: a full-vocab **sort** costs ~20% of
sampling time and a **host sync** stalls the async scheduler — both directly inflate inter-token latency,
especially at large batch where the sort's `O(V log V)` dominates. The sorting-free fused kernel
"reduces overall sampling time by >50%" (FlashInfer, 1×H100; the algorithm/port is the same on MI300X).
The win is **latency + scheduler overlap**, not throughput-FLOPs.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (the live AMD path: HIP Dual-Pivot rejection sampler + greedy/Gumbel `module_sample`) | [aiter.md](aiter.md) |
| vllm_kernels | 🟢 sota (`forward_hip` → aiter; native sort fallback; Gumbel no-sync `random_sample`) | vllm_kernels |
| triton | 🟡 competitive (MRV2-style fused top-k/Gumbel Triton kernels; sort-free logprobs) | triton |
| hip | 🟡 competitive (the rejection sampler *is* HIP — aiter's `sampling.cuh` is the editable source) | hip |
| flashinfer | ⚪ na | NVIDIA-only (CUB/CCCL); AMD uses aiter's port — record as `na` |

## Fusion neighbors
- **softmax fused into the sampler**: the rejection kernel can sample directly from logits/probs without
  materializing a separate softmax pass (Gumbel-max avoids explicit softmax) → see [fusion.md](fusion.md),
  softmax.
- **temperature + top-k + top-p + min-p + sample** all in one kernel (no intermediate `[M,V]` writes).
- Upstream consumer of lm_head_logits (logits → sampler); the head's greedy-argmax fusion handles
  temp=0 (argmax_topk).
- penalties (repetition/frequency/presence) and logit-bias are applied **before** sampling as elementwise
  logits processors (not fused into the rejection kernel).

## Numerics
fp32 probs/logits; non-associative parallel prefix-sum needs a **deterministic-scan** option; argmax/sample
**tie-break = lowest index** (`atomicMin`); **stateless philox RNG** (no host sync). See
[numerics.md](numerics.md).

## How to bench
Isolated: time `top_k_top_p_sampling_from_probs(probs, top_k_arr, top_p_arr)` at `(M∈{1,16,64,256},
V=128k–256k)` vs the PyTorch sort path; oracle = statistical (sample N draws, compare empirical
distribution / KL to a reference renormalized multinomial), **not** exact token match (rejection sampling
is only statistically equivalent). e2e: measure **inter-token latency** + confirm **no host sync**
(rocprofv3: no `hipMemcpy D2H` mid-decode).

## Sources
- Dual-Pivot Rejection Sampling, sorting-free, single fused kernel, O(log(1/ε)) rounds, >50% faster:
  https://flashinfer.ai/2025/03/10/sampling.html
- aiter HIP port (pivot_0/pivot_1, BlockScan/BlockReduce, philox, deterministic scan, atomicMin tie-break):
  `ROCm/aiter@b467ce342:csrc/cpp_itfs/sampling/sampling.cuh` (on-box).
- vLLM `topk_topp_sampler` (forward_hip→aiter, sort fallback, Gumbel `random_sample`, fp32 req):
  https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py
- SGLang sampler (aiter greedy on HIP, flashinfer CUDA-only, sort fallback, `multinomial_with_seed`):
  https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/sampler.py
