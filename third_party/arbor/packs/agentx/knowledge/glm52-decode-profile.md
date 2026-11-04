# GLM-5.2 Decode Profile — Baseline

SIKL decode profile: TP=4, DP=4, ISL=100K, CONC=64, fp8_e4m3 KV.

## Hotspots (one decode step, all 78 layers), largest first

| Op | Notes |
|---|---|
| `mlp.experts` | MXFP4 2-stage FlyDSL: `flydsl_mxmoe_g1_a4w4_16x256x256` |
| `attn.<rest>` | qk_norm + rope + absorb_bmm + kv_write + attn kernel |
| `attn.o_proj` | |
| `attn.indexer` | DSA indexer (21 of 78 layers) |
| `layer.<rest>` | collectives + residual_add |
| `attn.q_b_proj` | |
| `mlp.topk` | |

## Layer breakdown

- 3 dense+indexer layers
- 57 MoE layers ← biggest chunk of the step
- 18 MoE+indexer layers (the costliest per call)

## What to optimize

**#1 Priority: `mlp.experts`** — MoE GEMM using `flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt` + `flydsl_moe2_layout_afp4_wfp4_bf16`. Shape per rank: M=16 (batch/TP), N=6144, K=16384 for the expert projection. At CONC=64 with DP=4, each rank sees 16 requests. This is memory-bandwidth bound at small M — improving tile config, persistent kernel, or overlap with topk dispatch are the levers.

**#2 Priority: `attn.<rest>`** — fused attention kernel. The DSA tilelang backend is used. At ISL=100K and CONC=64, this is a long-context decode — KV cache reads dominate.

**#3 Priority: collectives (in `layer.<rest>`)** — allreduce between TP ranks. Currently using NCCL (custom AR disabled). Re-enabling aiter custom allreduce when stable could recover here.

## Kernel details
- MoE experts: `flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt` (stage 1) + `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_nt_sbm16` (stage 2)
- Attention: tilelang DSA backend (`dsa-decode-backend tilelang`)
- TopK: aiter `module_top_k_per_row`
- No tuned GEMM config found for M=16, N=6144, K=16384 — using torch fallback
