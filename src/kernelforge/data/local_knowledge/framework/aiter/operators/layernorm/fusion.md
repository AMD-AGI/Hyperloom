---
title: layernorm — fusion neighbors
kind: technique
operator: layernorm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/norm.py
  - ROCm/aiter@b467ce342:aiter/ops/triton/normalization/norm.py
---

# layernorm — fusion

Identical fusion economics to rmsnorm (a bandwidth-bound anchor) — fold neighbors to share the single
read+write. aiter ships the full matrix:

## 1. residual-add + layernorm → (LayerNorm form of fused_add_rmsnorm)
`layernorm2d_fwd_with_add` / `_fused_add_layernorm` (Triton `layernorm2d_fwd_with_add`): `residual_out =
x + residual_in`, then normalize, one pass. The encoder/ViT analog of the decoder's fused-add-rmsnorm.

## 2. layernorm + fp8/int8 dynamic/smooth quant → fused_norm_quant
- `layernorm2d_fwd_with_dynamicquant` — per-token fp8 dynamic quant in the norm kernel.
- `layernorm2d_fwd_with_smoothquant` — SmoothQuant (per-channel scale) for int8.
- `layernorm2d_fwd_with_add_dynamicquant` / `_with_add_smoothquant` — residual+norm+quant triple-fusion.
Output is fp8/int8 + scale, halving downstream-GEMM input traffic. Cross-link quant_dequant_fp8,
quant_int8, fused_norm_quant.

## 3. groupnorm sibling
aiter also ships `groupnorm.py` (vision/diffusion); same reduction machinery over channel groups — out of
scope here but shares the wave64 reduce primitive.

## Fusion table
| form | aiter entrypoint | saves |
|---|---|---|
| add + LN | `layernorm2d_fwd_with_add` | residual round-trip |
| LN + dyn-quant | `layernorm2d_fwd_with_dynamicquant` | ½ output bytes (fp8) |
| LN + smoothquant | `layernorm2d_fwd_with_smoothquant` | int8 GEMM input |
| add + LN + quant | `layernorm2d_fwd_with_add_dynamicquant` | both |

## torch.compile
Same as rmsnorm: register fused ops as custom ops so Inductor fuses around them; ROCm fusion pass stitches
norm+quant. See backends/vllm_kernels/aiter_integration.

## Sources
- aiter fused layernorm variants: `aiter/ops/norm.py` (`layernorm2d_fwd_with_add`, `_with_smoothquant`, `_with_add_smoothquant`).
- Triton fused layernorm bodies: `aiter/ops/triton/normalization/norm.py` (`layernorm2d_fwd_with_add`, `_with_dynamicquant`, `_with_add_dynamicquant`).
