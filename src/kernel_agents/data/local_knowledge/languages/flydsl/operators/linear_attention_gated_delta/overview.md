---
title: linear_attention_gated_delta — overview
kind: operator_overview
operator: linear_attention_gated_delta
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill, decode]
updated: 2026-06-08
---

# linear_attention_gated_delta  (Gated DeltaNet / chunked linear-attention scan)

## TL;DR
Gated DeltaNet (Qwen3-Next / Qwen3.5, Kimi-Linear; "Gated Delta Networks: Improving Mamba2 with Delta
Rule", arXiv 2412.06464) is a **linear-attention** layer carrying a fixed-size recurrent **state matrix**
`S ∈ R^{d_k×d_v}` per head, updated with a **gated delta rule** (Mamba2-style exponential decay + DeltaNet
error-correcting update). It is NOT softmax attention — there is no seq² score matrix. The kernel comes in
two regimes: a **chunked scan** for prefill (process chunks of C tokens with intra-chunk parallelism,
O(T/C) sequential steps) and a **fused recurrent** kernel for decode (one state update per token). On AMD
this is **Triton-portable**: the reference is `fla-org/flash-linear-attention` (FLA, used by HF
transformers), and aiter ships an on-box Triton port (`aiter/ops/triton/gated_delta_net/`,
prefill `chunk*` + decode `fused_recurrent*`). The single most important fact: it is **memory-bandwidth
bound on the state**, so the win is **keeping S in registers/LDS** across the update — a decomposed path
that round-trips S to HBM is estimated **10–50× slower**.

## Math contract
Per head, per token `t` (DeltaNet + gating):
```
S_t = (diag(α_t) ⊙ S_{t-1}) + β_t · k_t (v_t - S_{t-1}^T k_t)^T     # gated delta update
o_t = S_t^T q_t                                                       # readout
```
- `α_t` = exponential decay gate (Mamba2-style, per-channel), `β_t` = delta write strength.
- Pre-steps: **causal conv1d** on Q/K/V (local mixing) and **L2-norm on Q/K** (fused in the kernel).
- **Chunked form** (prefill): within a chunk, compute intra-chunk contributions in parallel via a
  WY-representation / `solve_tril` (lower-triangular solve) + cumulative decay (`cumsum`), carry the state
  across chunks. This is what makes prefill O(T/C) not O(T).
- Shapes (aiter `fused_recurrent_gated_delta_rule`): `q,k [B,T,H,K]`, `v [B,T,HV,V]` (GVA if `HV>H`),
  `g [B,T,HV]` decays, `beta`, optional `initial_state`/`output_final_state`, `cu_seqlens` for varlen.
- bf16/fp16 in, fp32 state accumulate.

## Shape regimes
- **Prefill**: chunked scan; chunk size C (commonly 64) trades parallelism vs state-carry cost.
- **Decode**: fused recurrent — one `S` update per token; pure memory-bound on `S` (d_k×d_v per head).
- Hybrid models (Qwen3-Next/3.5, Kimi-Linear): **3:1** ratio — 3 Gated-DeltaNet layers per 1 full-attn
  layer. When 75% of layers are GDN, this kernel **dominates** inference time.

## Where it matters (Amdahl)
On Qwen3-Next-class hybrids GDN is the **majority of layers** → the dominant attention-side cost. SGLang
auto-detects the hybrid layers and uses the optimized GDN kernels; the launch command on AMD uses
`--attention-backend triton`. A decomposed (non-fused) GDN path is estimated 10–50× slower, so the fused
kernel is not optional for serving.

## Backend landscape
Competitive positioning for this operator; this folder's card is [flydsl.md](flydsl.md). Other backends
live in their own folders (`framework/aiter/`, `languages/{triton,ck,asm,hip}/`).

| backend | status |
|---|---|
| triton | 🟢 sota (FLA reference + aiter on-box port; sglang default GDN path) |
| hip | 🟡 (causal-conv1d + state-update glue; no hand-asm GDN scan) |
| tilelang | 🧪 (expressible; no published AMD-tuned GDN kernel) |
| flydsl | 🧪 experimental (`flydsl_gdr_decode`; import commented out, gfx950-tuned) |

## Fusion neighbors
Causal conv1d ([[causal_conv1d]]), L2-norm + RoPE-free QK, sigmoid gating, the chunked cumsum/solve_tril,
fp8 state quant (rare). See [fusion.md](fusion.md). The full-attn layers it interleaves with are
[[attention_prefill_fmha]] / [[gqa_mqa_attention]].

## Numerics
fp32 state accumulate; the recurrent scan accumulates error over T — chunk boundaries and the L2-norm/gate
order matter. See [numerics.md](numerics.md).

## How to bench
Reference (FLA/aiter): B,T,H,K,V with chunk C=64; bench chunked prefill + fused-recurrent decode
separately; oracle = the FLA reference scan in fp32. See [tuning.md](tuning.md).
