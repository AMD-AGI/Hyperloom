---
title: sampling_topk_topp on aiter — SOTA card
kind: sota_card
operator: sampling_topk_topp
backend: aiter
gens: [gfx942, gfx950]
dtypes: [fp32, bf16]
regimes: [decode, both]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:csrc/cpp_itfs/sampling/sampling.cuh
  - ROCm/aiter@b467ce342:aiter/ops/sampling.py
  - ROCm/aiter@b467ce342:aiter/ops/sample.py
  - https://flashinfer.ai/2025/03/10/sampling.html
---

# sampling_topk_topp × aiter

## TL;DR
**aiter is the live AMD sampling path.** It ships a **HIP port of FlashInfer's Dual-Pivot Rejection
Sampling** (`csrc/cpp_itfs/sampling/sampling.cuh`) — sorting-free, single fused kernel, one threadblock per
request, stateless philox RNG, `atomicMin` lowest-index tie-break, optional deterministic prefix-sum.
Exposed as three probability-space ops (`top_k_renorm_probs`, `top_p_sampling_from_probs`,
`top_k_top_p_sampling_from_probs`) plus a separate `module_sample` (`greedy_sample`, `random_sample`,
`random_sample_outer_exponential` — the no-sync Gumbel/exponential draw). This is what vLLM `forward_hip`
and SGLang call on MI300X (FlashInfer itself is NVIDIA-only).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `top_k_top_p_sampling_from_probs` (joint k+p, dual-pivot) | `aiter/ops/sampling.py` → `csrc/cpp_itfs/sampling/sampling.cuh` | gfx942/950, fp32 probs | sorting-free, no host sync; algorithm is the >50%-faster-than-sort path (FlashInfer 1×H100; same algo on MI300X — not separately measured here) | joint top-k + top-p decode |
| `top_p_sampling_from_probs` (top-p only) | same | gfx942/950, fp32 | one-kernel nucleus sample | top-p only |
| `top_k_renorm_probs` (renorm for min-p combo) | same | gfx942/950, fp32 | renormalize to top-k, then min-p/multinomial | min-p path (renorm + sample) |
| `greedy_sample` / `random_sample` / `random_sample_outer_exponential` (`module_sample`) | `aiter/ops/sample.py` | gfx942/950 | greedy argmax + Gumbel/exponential no-sync draw | greedy fast path; no-multinomial sampling |

aiter exposes **no dedicated min-p kernel** — min-p is built from `top_k_renorm_probs` + sample (the
framework composes it).

## Config space / knobs
- **`deterministic: bool = False`** (python wrapper) → toggles the `DETERMINISTIC` template
  (`DeterministicInclusiveSum`, reproducible but slower) vs the fast hipCUB `BlockScan`. Set True for
  reproducibility / debugging ([numerics.md](numerics.md)).
- **per-row arrays**: `maybe_top_k_arr`/`maybe_top_p_arr` (per-request k/p) + scalar `top_k_val`/`top_p_val`
  fallback — pass arrays for heterogeneous batches.
- **`indices`** arg: the kernel takes a probs tensor + indices (supports an already-gathered/reordered
  candidate set).
- Kernel internals (Tier-C edit, not a runtime knob): `BLOCK_THREADS=1024` (one block/row), `VEC_SIZE`
  vectorized load width, dual-pivot early-stop. JIT-compiled on first use (`compile_ops`/`module_sample`).

## Numerics / parity
fp32 probs. Non-associative prefix-sum → use `deterministic=True` for reproducibility. Tie-break = lowest
index (`atomicMin`). philox RNG: same `(seed, offset)` → same draw, no host state. **Statistical**
equivalence to sorted multinomial (KL gate, not token match); greedy is exact. All-NaN/-inf row → possible
OOR token id (guard upstream). See [numerics.md](numerics.md).

## Integration (rebind seam)
Registered as torch custom ops via `direct_register_custom_op` (`top_k_top_p_sampling_from_probs`, …) and
`module_sample` JIT. Live callers: vLLM `TopKTopPSampler.forward_hip`/`aiter_sample` (lazy
`import aiter.ops.sampling`, gated on `rocm_aiter_ops.is_enabled()`), SGLang sampler (`greedy_sample` on
HIP). Verify: rocprofv3 shows the aiter sampling kernel ran (not the PyTorch sort), and **no `hipMemcpy
D2H`** mid-decode.

## Pitfalls & anti-patterns
- **No min-p kernel** — composed from renorm + sample; a naive min-p that re-sorts defeats the purpose.
- **No processed-logprobs** from the rejection path → framework falls back to native (sort) when processed
  logprobs are requested. Don't request both fast sampling and processed logprobs.
- **No per-request `Generator`** support → vLLM/SGLang fall back to native when per-request seeds are set
  (the rejection kernel uses a single philox seed/offset).
- All-NaN/-inf rows → OOR token id (SGLang `SGLANG_DISABLE_AITER_GREEDY_SAMPLE` guard).
- Passing bf16 probs/logits → biased thresholds; use fp32.

## How to verify
Statistical: N draws per fixed `(probs, k, p, seed)` vs sorted-multinomial reference (KL within band);
greedy exact-token parity; fixed-seed reproducibility with `deterministic=True`. rocprofv3: aiter kernel
present, no D2H sync. Isolated latency vs the PyTorch sort at `(M, V=128k–256k)`.

## Alternatives / cross-links
vllm_kernels (forward_hip→here + sort fallback) · hip (this *is* HIP —
the editable source) · triton · [overview.md](overview.md) · aiter ·
argmax_topk · lm_head_logits.

## Sources
- HIP Dual-Pivot rejection sampler (pivot_0/pivot_1, BlockScan/BlockReduce, hiprand philox, atomicMin,
  DETERMINISTIC, BLOCK_THREADS=1024, VEC_SIZE): `ROCm/aiter@b467ce342:csrc/cpp_itfs/sampling/sampling.cuh`.
- Python ops (3 sampling entrypoints + deterministic flag, custom-op registration): `…:aiter/ops/sampling.py`.
- `module_sample` greedy/random/exponential (no-sync Gumbel): `…:aiter/ops/sample.py`.
- Algorithm + >50% win: https://flashinfer.ai/2025/03/10/sampling.html
