---
title: gqa_mqa_attention on CK — SOTA card
kind: sota_card
operator: gqa_mqa_attention
backend: ck
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha
  - https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
  - https://github.com/Dao-AILab/flash-attention
---

# gqa_mqa_attention × CK

## TL;DR
CK-Tile FMHA handles GQA/MQA as a head-pairing trait of the forward kernel (the same kernels behind
flash-attention ROCm). It is the **default CK FA** behavior for GQA models within head_dim ≤256, fp16/bf16.
On serving, GQA attention is usually routed to aiter (faster), so CK GQA is the CK-FA-path / from-source
option. Use CK for stable half-precision GQA (and the mature CK backward in training); use aiter for
serving.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| CK-Tile FMHA (GQA/MQA trait) | `ROCm/composable_kernel:example/ck_tile/01_fmha` | gfx90a/942/950; fp16/bf16; head_dim ≤256 | the CK FA path for GQA; ~CK FMHA perf | stable half-precision GQA, training bwd |

## Config space / knobs
CK template params with the GQA head-pairing (KV head shared by R query heads), `kM0` aligned to R,
WarpGemm 16×16×16 / 32×32×8, page size (decode), causal/SWA trait. Codegen via FMHA `generate.py`.
See `languages/composable_kernel/fmha_template.md`.

## Numerics / parity
GQA bit-identical to MHA-with-shared-KV; fp32 accumulate. CK FA: fp16/bf16 only (no fp8 in core CK FA;
fp8 KV is Triton-side). See [numerics.md](numerics.md).

## Integration (rebind seam)
Via flash-attention ROCm: `VLLM_USE_TRITON_FLASH_ATTN=0` → CK FA (GQA trait built in). From source: build
the FMHA example with the GQA head config.

## Pitfalls & anti-patterns
- head_dim ≤256 hard limit; no fp8 in core CK FA.
- aiter GQA beats CK on serving — measure before choosing CK.
- composable_kernel moved into `ROCm/rocm-libraries` (standalone deprecated) — pin accordingly.

## How to verify
Build FMHA example with GQA head config, `-v 1` reference compare; isolated bench vs aiter at the model
ratio; greedy temp=0 parity; confirm CK fired (`*ck_*`/`fmha_*`).

## Alternatives / cross-links
aiter (serving SOTA) · triton · [flash_attention_rocm.md](fa_rocm.md) ·
`operators/attention_prefill_fmha/backends/ck.md` · `languages/composable_kernel/fmha_template.md` ·
../overview.md.

## Sources
- CK-Tile FMHA GQA/MQA trait: https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha ; https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
- MQA/GQA as CK FA feature (head_dim ≤256, fp16/bf16): https://github.com/Dao-AILab/flash-attention
