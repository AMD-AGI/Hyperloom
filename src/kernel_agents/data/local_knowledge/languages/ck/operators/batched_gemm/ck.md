---
title: batched_gemm on ck — SOTA card
kind: sota_card
operator: batched_gemm
backend: ck
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
  - https://github.com/ROCm/composable_kernel
---

# batched_gemm × ck

## TL;DR
> CK provides `DeviceBatchedGemm*_Xdl_CShuffle` templates — choose it to author a **fused** batched
> GEMM (per-batch epilogue) or an unsupported precision, or as the substrate for grouped/MoE batched
> matmul. For plain uniform batched GEMM on the live path, prefer tuned hipBLASLt via
> aiter. CK is build-time, no env-overlay seam.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `DeviceBatchedGemmMultipleD_Xdl_CShuffle` | `ROCm/composable_kernel` (→ rocm-libraries) | gfx942/950; bf16/fp16/fp8/int8 | no published batched figure beats tuned hipBLASLt → **use hipblaslt for plain batched** | fused / quantized / custom batched GEMM |
| CK grouped GEMM (variable per-batch) | CK grouped templates | gfx942/950 | preferred over padded batched for ragged MoE | per-expert/ragged batches |

## Config space / knobs
- Instance tile: `MPerBlock/NPerBlock/KPerBlock`, `MPerXDL/NPerXDL` (16×16 preferred), XdlPerWave.
- BlockGemmPipeline version (deeper for K-deep), block-transfer thread clusters.
- Per-batch epilogue via `CDEElementwise` (bias/act/scale) → [fusion.md](fusion.md).
- For ragged batches use the **grouped** device op, not batched (avoids padding waste).

## Numerics / parity
fp32 accumulate per batch; int8/fp8 need task-accuracy gating. See [numerics.md](numerics.md).

## Integration (rebind seam)
Build-time extension; call directly or register as an aiter candidate. No env-overlay into serving.

## Pitfalls & anti-patterns
- Mismatched instance tile → far below hipBLASLt; enumerate via ckProfiler.
- Using batched (padded) for ragged MoE shapes — use grouped GEMM.
- Expecting CK to auto-replace serving GEMM — no seam.

## How to verify
`ckProfiler batched_gemm` vs `hipblaslt-bench --batch_count` on the same shape; adopt only with a seam.

## Alternatives / cross-links
hipblaslt (prefer plain) · aiter · asm ·
triton · [overview.md](overview.md) ·
grouped: operators/grouped_gemm_moe/overview.md · language ref languages/composable_kernel/....

## Sources
- CK device-op / CShuffle: ROCm "Optimizing with Composable Kernel".
- CK repo: https://github.com/ROCm/composable_kernel.
