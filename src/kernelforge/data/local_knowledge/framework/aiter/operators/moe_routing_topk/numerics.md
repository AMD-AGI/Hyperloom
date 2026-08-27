---
title: moe_routing_topk — numerics
kind: technique
operator: moe_routing_topk
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - https://github.com/ROCm/aiter/issues/2153
  - https://github.com/vllm-project/vllm/pull/17955
  - ROCm/aiter@b467ce342:aiter/ops/topk.py
  - ROCm/aiter@b467ce342:csrc/kernels/topk_gating_kernels.cu
---

# moe_routing_topk — numerics & parity

## What "correct" means here
The router's output (`topk_ids`, `topk_weights`) **picks which experts run and how their outputs are
weighted** — a wrong id sends a token to the wrong expert (a real quality regression), while a tiny
weight perturbation is usually benign. So the parity gate is **expert-id match + weight closeness**, not
byte parity.

## Benign vs real divergence
- **Benign (numerical-equivalence-class)**: when two experts have near-equal gate scores, fp32 reduction
  order differences across backends (HIP DPP vs CK vs Triton vs torch) flip the **tie-break** → a different
  but equally-valid expert id. On a long greedy decode this shows up as occasional token differences. Gate
  with a parity probe over ≥10 prompts; don't fail on a single argmax flip near a tie.
- **Real regression** (must fail the gate):
  1. **Wrong scoring function**. The legacy aiter biased-grouped HIP kernel (`biased_grouped_topk_hip`)
     **hardcoded `isSoftmax=false`** — it always applied **sigmoid** regardless of the requested
     `scoring_func` (aiter #2153), so using it for a **softmax-scored** model (Mixtral/Qwen-MoE) yields wrong
     weights and `test_grouped_topk` fails. The current unified **`topk_gating`** entry (`aiter/ops/topk.py`)
     fixes this: it validates `score_func ∈ {sqrtsoftplus, sigmoid, softmax}` and, for softmax, adds the
     correction bias **after** normalization (and forces `need_renorm=False`) — prefer it. On the legacy path,
     assert `scoring_func=="sigmoid"` before the biased branch and route softmax models to `topk_softmax`
     (`aiter/ops/moe_op.py`) / Triton. (vLLM #17955 review flagged exactly this — `scoring_func` is "assumed".)
  2. **Incomplete group reduction**. `THREAD_PER_GRP = 64/num_expert_group`; cross-lane reduction is only
     implemented for `THREAD_PER_GRP ∈ {2,4,8}`. For unsupported `num_expert_group` (e.g. 4 → 16, or any
     value driving `lane_steps→0`), each lane sees only a **subset** of its group's experts → wrong group
     scores → wrong groups selected. Verify your model's `num_expert_group` is supported before enabling.

## fp8 / quant interaction
The router itself runs in fp32 even when the GEMM is fp8 — the gate logits come from a bf16/fp16 linear.
The **numeric risk moves downstream**: where the routed weight gets multiplied (`doweight_stage1`,
`MulRoutedWeight1`) and the fp8 grouped GEMM quant error (fused_moe_grouped_gemm, scaled_quant_gemm).
Keep the routed-weight multiply in fp32/bf16 before any fp8 cast.

## Renormalization & scaling
`need_renorm` (divide the k weights by their sum) and `routed_scaling_factor` must be applied in the same
fp32 domain and in the **same order** as the reference, or the combined output drifts. DeepSeek applies
bias for *selection* but the **original (unbiased) score** for the *weight* — a common porting bug is using
the biased score as the weight.

## Verification recipe
1. Isolated: `aiter/op_tests/test_moeTopkSoftmax.py` and `test_moe_topk_gating.py` — compare ids+weights
   vs `biased_grouped_topk_torch` / `grouped_topk_torch` references at E∈{64,128,256}, k∈{2,6,8}.
2. e2e: greedy/temp=0 on a MoE model, `VLLM_ROCM_USE_AITER=1` vs `=0`; expect occasional benign argmax
   flips only. A systematic divergence (every prompt) ⇒ scoring/group bug above.
3. Run `tests/kernels/moe/test_routing.py::test_grouped_topk` with `VLLM_ROCM_USE_AITER=1` — it is the
   canonical detector for both bugs.

## Sources
- Hardcoded sigmoid + THREAD_PER_GRP reduction bug: https://github.com/ROCm/aiter/issues/2153
- `scoring_func` assumed, assert softmax: https://github.com/vllm-project/vllm/pull/17955
- on-box kernel source: `ROCm/aiter@b467ce342:csrc/kernels/topk_gating_kernels.cu` (unified `topk_gating`,
  `score_func` resolution, gfx1250 wave32), `csrc/kernels/topk_softmax_kernels_group.cu`, `aiter/ops/topk.py`
  (`topk_gating`, `biased_grouped_topk_torch`, `grouped_topk_torch` references).
