---
title: act_and_mul_silu_gelu — fusion neighbors
kind: technique
operator: act_and_mul_silu_gelu
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, mxfp4]
regimes: [both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/triton/activation.py
  - /sgl-workspace/aiter/aiter/ops/flydsl/kernels/silu_and_mul_fq.py
  - /sgl-workspace/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage.py
---

# act_and_mul_silu_gelu — fusion

The gated activation is a **fusion-first** op — standalone it forces an HBM round-trip between two GEMMs.
The fusion ladder:

## 1. up/gate GEMM epilogue + act_and_mul  (the basic fusion)
The up/gate Linear produces `[M, 2d]`; instead of writing it, apply `act(gate)·up` in the GEMM **epilogue**
and write only `[M, d]`. Removes the standalone read of `[M, 2d]` and write of `[M, d]`. See
gemm_epilogue_fused, dense_gemm (fusion.md notes folding act_and_mul into up/gate).

## 2. act_and_mul + output quant  (the MoE/MLP win)
While `y` is in fp32 registers, quantize to fp8/fp4 so the **down-proj GEMM reads quantized input** (½ or
¼ bytes):
- `scaled_silu_and_mul` (aiter C++): static-scale fp8.
- `act_mul_and_fp8_group_quant` (Triton): per-group dynamic fp8 + scale.
- `act_mul_and_mxfp4_quant` (Triton): mxfp4 e8m0 block-scaled (gfx950).
- `silu_and_mul_fq` (FlyDSL): fused silu·mul + fp4/fp8 quant + sorted-scale write — used in MoE.
Cross-link fused_norm_quant, quant_dequant_fp8, quant_fp4_mxfp.

## 3. inside fused-MoE stage-1  (the dominant production form)
In a fused-MoE kernel, stage-1 = grouped up/gate GEMM → act_and_mul → (quant) → stage-2 down GEMM, all
token-sorted by expert. The activation never materializes to HBM. aiter `fused_moe`; flydsl
`moe_gemm_2stage` / `mixed_moe_gemm_2stage`. This is the Kimi-K2.5 **+162% throughput** path (vendor). See
fused_moe_grouped_gemm, shared_expert_fusion.

## Fusion table
| form | impl | saves |
|---|---|---|
| GEMM epilogue + act | gemm_epilogue_fused | `[M,2d]` read + `[M,d]` write |
| act + fp8 quant | `act_mul_and_fp8_group_quant` | ½ down-proj input bytes |
| act + mxfp4 quant | `act_mul_and_mxfp4_quant` (gfx950) | ¼ down-proj input bytes |
| in fused-MoE stage-1 | aiter `fused_moe` / flydsl 2stage | no HBM materialization |

## torch.compile
vLLM registers the act_and_mul ops as custom ops (the `_ACTIVATION_AND_MUL_REGISTRY`); Inductor fuses
around them. ROCm fusion passes can stitch act+quant. See backends/vllm_kernels/aiter_integration.

## Sources
- act+quant Triton: `/sgl-workspace/aiter/aiter/ops/triton/activation.py`.
- flydsl silu_and_mul_fq + MoE 2stage: `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/silu_and_mul_fq.py`, `moe_gemm_2stage.py`.
- Kimi-K2.5 fused-MoE win: perf_knowledge languages/flydsl/kernel_families.
