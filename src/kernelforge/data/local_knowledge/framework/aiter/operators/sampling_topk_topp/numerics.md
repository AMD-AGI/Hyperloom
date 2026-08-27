---
title: sampling_topk_topp — numerics
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
  - https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/sampler.py
---

# sampling_topk_topp — numerics

## TL;DR
Sampling correctness is **statistical, not bit-exact** — the rejection sampler is only *statistically
equivalent* to a sorted multinomial, so the oracle is a distribution test (KL/empirical frequency), not a
token match. The five real hazards: **(1)** fp32 everywhere (bf16 logits bias the thresholds), **(2)**
non-associative parallel **prefix-sum** can be non-monotonic → needs the deterministic-scan option for
reproducibility, **(3)** **tie-break = lowest index** (`atomicMin`) for both argmax and equal-prob samples,
**(4)** **stateless philox RNG** (seed, row, offset) so results are reproducible without host state, and
**(5)** all-NaN/-inf rows can make the kernel return an **out-of-range token id** (the SGLang aiter-greedy
guard).

## fp32 discipline
- temperature, top_p, and probs must be **fp32** — vLLM crashed when logits dtype changed under it
  (#21936); the contract is `temperature`/`top_p` as float32.
- bf16 logits → softmax in bf16 biases the top-p cumulative threshold and the min-p floor (`min_p·max p`)
  → wrong nucleus membership. Upcast logits to fp32 before softmax/threshold (lm_head_logits emits
  fp32 for exactly this reason).

## Non-associative prefix-sum (the subtle one)
The rejection sampler uses a parallel **inclusive prefix-sum** (CDF) to locate the sampled token via
inverse-transform. FlashInfer's own note: a parallel prefix-sum "cannot guarantee monotonic outputs" due
to non-associative FP add, which can break the CDF comparison. aiter's port carries a **`DETERMINISTIC`
template path** (`DeterministicInclusiveSum`, slower but reproducible) alongside the fast hipCUB
`BlockScan`. Use deterministic when you need run-to-run reproducibility or to debug a sampling divergence;
the python wrappers expose `deterministic: bool = False`.

## Tie-break (lowest index)
On exact ties the kernel selects the **lowest token index** — aiter does this with
`atomicMin(&sampled_id, idx)` over the candidates. This must match the reference and the **greedy argmax**
tie-break (lm_head_logits numerics, argmax_topk) or greedy decoding diverges at tie tokens. A
vocab-parallel argmax must break cross-shard ties the same way.

## RNG: stateless philox, no host sync
The kernel seeds per-row with `hiprand_init(philox_seed, row_idx, philox_offset, &state)` — a **stateless**
counter-based RNG. Same `(seed, offset)` → same draw, with **no RNG state copied across the bus** (a key
part of the no-host-sync design). For the no-multinomial sample path, the Gumbel/exponential trick
(`random_sample_outer_exponential`) is likewise on-device. SGLang's `multinomial_with_seed` keeps the
Gumbel math in **fp64** for numerical stability of the per-row seed→uniform→Gumbel chain.

## All-NaN / -inf rows (the safety guard)
If a logits row is all-NaN or all-`-inf` (e.g. a fully-masked request), the aiter greedy/sample kernel can
return an **out-of-range token id**. SGLang guards this with `SGLANG_DISABLE_AITER_GREEDY_SAMPLE`, which
falls back to `torch.argmax` ("always returns a valid index"). Treat an OOR token id as the tell of a
NaN/-inf row, not a kernel bug — fix the masking upstream or enable the guard.

## logprobs from raw logits
When logprobs are requested, compute them from the **raw (unprocessed) logits** for consistency regardless
of sampling params (vLLM `--logprobs-mode raw_logprobs` default). The FlashInfer/aiter rejection path
**cannot return processed logits/logprobs** → the framework falls back to the native (sort) path when
processed logprobs are required. Don't expect the fast sampler to also give you post-filter logprobs.

## The gate (statistical)
After a sampler backend swap: draw N samples per fixed `(logits, top_k, top_p, min_p, seed)` and compare
the **empirical token distribution** (KL or χ²) to the native sorted-multinomial reference within a
tolerance — **not** exact tokens (rejection sampling is only statistically equivalent). For greedy
(temp=0), use **exact token parity** (argmax is deterministic). Always run a fixed-seed reproducibility
check with `deterministic=True`.

## Cross-links
[overview.md](overview.md) · [tuning.md](tuning.md) · [fusion.md](fusion.md) · argmax_topk ·
lm_head_logits (fp32 logits, tie-break) · softmax · cumsum_scan (prefix-sum).

## Sources
- aiter sampling.cuh (deterministic scan, atomicMin tie-break, hiprand philox, dual-pivot):
  `ROCm/aiter@b467ce342:csrc/cpp_itfs/sampling/sampling.cuh`.
- Non-monotonic prefix-sum / numerical care: https://flashinfer.ai/2025/03/10/sampling.html
- fp32 requirement, raw-logprobs, statistical equivalence: https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py
- aiter-greedy OOR guard, fp64 Gumbel `multinomial_with_seed`: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/sampler.py
