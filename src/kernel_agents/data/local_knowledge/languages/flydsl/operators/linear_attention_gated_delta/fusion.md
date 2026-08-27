---
title: linear_attention_gated_delta — fusion
kind: technique
operator: linear_attention_gated_delta
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill, decode]
updated: 2026-06-08
---

# linear_attention_gated_delta — fusion

## TL;DR
Fusion IS this operator — keep the state S resident and fold every pre/post step (QKVZBA split, causal conv1d, L2-norm Q/K, GDN gating, cumsum+KKᵀ, triangular WY solve, chunk output+state carry) into the scan so S never round-trips to HBM. Decomposing the scan into separate matmuls that spill S is the canonical mistake (~10–50× slower).

## Fusion IS the operator
GDN's whole performance story is fusion: keep the state S resident and fold every pre/post step into the
scan so S never round-trips to HBM. The aiter on-box kernels show exactly which pieces fuse:

| fused piece | aiter kernel (on-box) | why |
|---|---|---|
| **QKV(z,b,a) split** | `fused_qkvzba_split.py` | one projection output → split into q/k/v/gates without an HBM bounce |
| **causal conv1d + split** | `prefill/causal_conv1d_fwd_split_qkv.py`, `decode/causal_conv1d_split_qkv.py` | local mixing fused with the split ([[causal_conv1d]]) |
| **L2-norm Q/K** | in-kernel (`use_qk_l2norm_in_kernel`) | avoids a separate norm pass |
| **GDN gating (sigmoid)** | `fused_gdn_gating_prefill.py`, `decode/fused_sigmoid_gating_recurrent.py` | gate computed where it's used |
| **cumsum of decays + KKᵀ** | `fused_cumsum_kkt.py` | the chunk's serial decay/score prep |
| **triangular solve (WY)** | `fused_solve_tril_recompute.py`, `utils/solve_tril.py`, `wy_representation.py` | intra-chunk delta-rule, kept in LDS |
| **chunk output + state carry** | `chunk_o.py`, `chunk_delta_h.py` | readout + state update fused |

## The hybrid-model macro-fusion
GDN layers interleave 3:1 with full-attention layers (Qwen3-Next/3.5, Kimi-Linear). The system-level
"fusion" is the **hybrid KV-cache manager**: GDN layers store a fixed state (no growing KV), full-attn
layers store paged KV. vLLM/sglang manage both in one cache to avoid fragmentation — getting both on a
uniform backend (sglang uses `--attention-backend triton`, auto-detecting GDN layers) matters more than
any single micro-fusion.

## Anti-pattern
**Decomposing** the scan (separate matmuls reading/writing S to HBM) is the canonical mistake — estimated
10–50× slower because the bottleneck is state bandwidth, not compute. Never split the state update.

## Cross-links
- Pre-step: [[causal_conv1d]] · norm: [[rmsnorm]] (L2-norm variant) · scan: [[cumsum_scan]].
- Interleaved full-attn: [[attention_prefill_fmha]] · [[gqa_mqa_attention]].
- Languages: [[triton_amd]] (the path) · backend: [[aiter]] · [[sglang_kernels]].
