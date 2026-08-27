---
title: sampling_topk_topp — tuning
kind: technique
operator: sampling_topk_topp
gens: [gfx942, gfx950]
dtypes: [fp32, bf16]
regimes: [decode, both]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:csrc/cpp_itfs/sampling/sampling.cuh
  - https://flashinfer.ai/2025/03/10/sampling.html
  - https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# sampling_topk_topp — tuning

## TL;DR
The decisive choices are **algorithmic**, not knob-twiddling: (1) **use the sorting-free rejection
sampler** (aiter's HIP port), never the full-vocab sort, at large V/batch; (2) keep the whole pipeline in
**one kernel with no host sync** (Gumbel/exponential sample, not `torch.multinomial`); (3) take the
**greedy argmax fast path** when the whole batch is temp=0. Beyond that, the HIP kernel's tunables are
`VEC_SIZE` (vectorized load width), `BLOCK_THREADS=1024` (one block per row), and the
**deterministic vs fast block-scan** trade-off.

## Lever 1 — algorithm: rejection sampler vs sort (the big one)
| path | cost | when |
|---|---|---|
| **rejection sampler** (aiter `top_k_top_p_sampling_from_probs`) | O(log(1/ε)) rounds, **no sort**, one fused kernel, no host sync | **default** — large V, any batch |
| PyTorch sort (`apply_top_k_top_p_pytorch`) | O(V log V) sort over `[M,V]`; ~20% sampling overhead | fallback only (per-request generators, logprobs needs processed logits, unsupported shape) |
| top-k-only with CPU sync (`apply_top_k_only`) | avoids full sort but **GPU→CPU sync** (kills async scheduling) | avoid on the hot path |

The sort path's cost **grows with batch** (it sorts the whole `[M,V]`); the rejection sampler's per-row
work is roughly independent of V except for the streaming read. At large vocab + large batch the gap is the
">50% sampling-time" win. Prefer the rejection sampler unless a feature forces the fallback.

## Lever 2 — no host sync (Gumbel/exponential sample)
Use `random_sample` / `random_sample_outer_exponential` (aiter `module_sample`) — the exponential/Gumbel
trick: add `-log(-log(u))` noise to logprobs and argmax, **on device**, instead of `torch.multinomial`
(which syncs). The rejection kernel itself uses **stateless philox** (`hiprand_init(seed, row, offset)`),
so no RNG state is copied across the bus. **Verify with rocprofv3 that there is no `hipMemcpy D2H` between
the head GEMM and the next step** — a stray sync (e.g. the top-k-only CPU path, or a `.item()` on a count)
silently serializes decode.

## Lever 3 — greedy fast path
When `is_all_greedy` (every request temp<1e-5), skip temperature, softmax, top-k/top-p, and the
argmax-invariant logit processors (min-p) entirely — just argmax the logits (argmax_topk). vLLM/SGLang
do this batch-wide; it removes the entire sampler from greedy steps. The argmax-invariant processors can
only be skipped when the **whole batch** is greedy (mixed batches still run them).

## Lever 4 — the HIP kernel knobs (aiter sampling.cuh)
- **`VEC_SIZE`** — vectorized load width per thread (`vec_t<float, VEC_SIZE>` / `cast_load`); wider =
  fewer load instructions over the `V`-row. Tune for HBM efficiency (the kernel is a streaming read of
  `V` floats per row).
- **`BLOCK_THREADS=1024`** — one threadblock per request row; the `V`-row is processed in
  `ceil_div(V, BLOCK_THREADS·VEC_SIZE)` chunks. One block/row means the grid = `M` (batch); at tiny batch
  the 304 CUs are under-occupied (latency-bound) — acceptable, sampling is small.
- **deterministic vs fast scan** — `DETERMINISTIC` template flag picks the slower
  `DeterministicInclusiveSum` (reproducible prefix sum) vs hipCUB `BlockScan` (faster, FP-order-dependent).
  Use deterministic only when bitwise reproducibility is required (see [numerics.md](numerics.md)); the
  python wrappers expose `deterministic: bool = False`.
- **early-stopping** — the dual-pivot loop exits as soon as a pivot is accepted; well-peaked distributions
  (low temperature, small top-k) converge in 1–2 rounds.

## Pitfalls
- Falling back to the **sort** path unknowingly (per-request `Generator`, or `logprobs_mode` needing
  processed logits) → 20%+ overhead. Check why the framework didn't take the rejection path.
- A hidden **host sync** (top-k-only CPU path, `.item()`, eager `torch.multinomial`) → async scheduler
  stall.
- bf16 logits into the sampler → biased thresholds; upcast to **fp32** first ([numerics.md](numerics.md)).
- Per-request top_k/top_p as scalars when the batch is heterogeneous → wrong filter for most rows; pass
  the **per-row arrays**.

## Cross-links
[overview.md](overview.md) · [numerics.md](numerics.md) · [fusion.md](fusion.md) · softmax ·
argmax_topk · lm_head_logits · cumsum_scan (prefix-sum primitive) · aiter.

## Sources
- aiter sampling.cuh (VEC_SIZE, BLOCK_THREADS=1024, deterministic scan, dual-pivot early stop, philox):
  `ROCm/aiter@b467ce342:csrc/cpp_itfs/sampling/sampling.cuh`.
- Sorting-free >50% win, O(log(1/ε)): https://flashinfer.ai/2025/03/10/sampling.html
- vLLM sort fallback, Gumbel `random_sample`, greedy fast path: https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py
- Memory-bound op tuning (coalesced/vectorized reads): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
