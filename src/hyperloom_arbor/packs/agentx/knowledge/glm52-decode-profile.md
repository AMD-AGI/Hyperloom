# GLM-5.2 Decode Profile — Baseline (2026-09-02)

SIKL decode profile: TP=4, DP=4, ISL=100K, CONC=64, fp8_e4m3 KV.

**Step time: 30.5ms — 2099 tok/s**

## Hotspots (one decode step, all 78 layers)

| Op | ms/step | Share | Notes |
|---|---|---|---|
| `mlp.experts` | 13.05 | **44.2%** | MXFP4 2-stage FlyDSL: `flydsl_mxmoe_g1_a4w4_16x256x256` |
| `attn.<rest>` | 5.47 | 18.5% | qk_norm + rope + absorb_bmm + kv_write + attn kernel |
| `attn.o_proj` | 2.93 | 9.9% | |
| `attn.indexer` | 2.38 | 8.1% | DSA indexer (21 of 78 layers) |
| `layer.<rest>` | 2.14 | 7.2% | collectives + residual_add |
| `attn.q_b_proj` | 1.13 | 3.8% | |
| `mlp.topk` | 0.64 | 2.2% | |

## Layer breakdown

- 3 dense+indexer layers: 0.336 ms/call → 1.0 ms/step
- 57 MoE layers: 0.339 ms/call → **19.3 ms/step** ← biggest chunk
- 18 MoE+indexer layers: 0.498 ms/call → 9.0 ms/step

## What to optimize

**#1 Priority: `mlp.experts` (44.2%)** — MoE GEMM using `flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt` + `flydsl_moe2_layout_afp4_wfp4_bf16`. Shape per rank: M=16 (batch/TP), N=6144, K=16384 for the expert projection. At CONC=64 with DP=4, each rank sees 16 requests. This is memory-bandwidth bound at small M — improving tile config, persistent kernel, or overlap with topk dispatch are the levers.

**#2 Priority: `attn.<rest>` (18.5%)** — fused attention kernel. The DSA tilelang backend is used. At ISL=100K and CONC=64, this is a long-context decode — KV cache reads dominate.

**#3 Priority: collectives (7.2% in `layer.<rest>`)** — allreduce between TP ranks. Currently using NCCL (custom AR disabled). Re-enabling aiter custom allreduce when stable could recover here.

## Kernel details
- MoE experts: `flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt` (stage 1) + `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_nt_sbm16` (stage 2)
- Attention: tilelang DSA backend (`dsa-decode-backend tilelang`)
- TopK: aiter `module_top_k_per_row`
- No tuned GEMM config found for M=16, N=6144, K=16384 — using torch fallback
