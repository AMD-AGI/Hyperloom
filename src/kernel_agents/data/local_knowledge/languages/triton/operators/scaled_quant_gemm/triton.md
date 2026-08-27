---
title: scaled_quant_gemm on triton — SOTA card
kind: sota_card
operator: scaled_quant_gemm
backend: triton
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp4_e2m1, fp6_e2m3, fp6_e3m2]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# scaled_quant_gemm × triton

## TL;DR
> Triton ships a **generic block-scaled matmul** (mxfp4/mxfp8) that lowers to CDNA4 scaled-MFMA — the most
> accessible authorable path for low-bit GEMM on gfx950, and the natural **parity reference** for hand-tuned
> kernels. Use it for custom scaled GEMMs; for absolute peak the Gluon/asm route squeezes more (Gluon ceilings:
> BF8 99.72%, MXFP4 92.41%). On gfx942 there is **no native 32-elem block-scaled MFMA** — block scaling is
> emulated/coarse there; use fp8 tensor-scaled instead.

## SOTA implementation
The Triton tutorial `10-block-scaled-matmul` packs A/B as low-bit + an E8M0 scale tensor and emits the
scaled-MFMA. The dot is computed in fp32 accumulate, then scaled per 32-element block. The kernel autotunes
over the usual GEMM space plus the scaled-MFMA variant (32x32x64 vs 16x16x128) and split-K.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Triton block-scaled matmul (mxfp4/mxfp8, 32-elem blocks) | https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html | gfx950 native scaled MFMA (gfx942 fp8 non-block) | no first-party number reproduced; Gluon ceilings on same HW: **BF8 99.72%, MXFP4 92.41%** efficiency @ MI350/355 | authorable mxfp GEMM / parity ref |

## Config space / knobs
| param | range / typical | effect | default |
|---|---|---|---|
| `BLOCK_M/N/K` | K aligned to 32-elem blocks | tile sizes; **K must be a multiple of the 32-elem scale block** | 128×128×128 |
| scaled-MFMA variant | 32x32x64 / 16x16x128 | matrix-core shape for f8f6f4 | autotuned |
| `num_stages` | 2–4 | software pipeline (overlaps scale load with dot here, unlike plain GEMM) | 2 |
| `num_warps` | 4 / 8 | warps per block | 8 |
| `waves_per_eu` | 0–4 | occupancy hint | 0 |
| `SPLIT_K` | 1–16 | K split for large K | 1 |
| scale tensors | E8M0 per-32 | passed alongside A/B; index must track K blocks | — |

## Numerics / parity
- **E8M0** block scales, **scale-after-dot**, fp32 accumulate; accuracy gate vs bf16 reference
  ([numerics.md](numerics.md)). fp8 on gfx942 is **FNUZ**; gfx950 is **OCP** (quantization/fnuz_vs_ocp).

## Integration (rebind seam)
Overlay the Triton scaled-GEMM module the framework calls (aiter also ships Triton scaled paths and can route
to one via a tuned-CSV `triton` row). Verify the scaled kernel name + autotune key in a rocprof trace; e2e-gate
through the aiter engagement flow (operators/scaled_quant_gemm/backends/aiter).

## Pitfalls & anti-patterns
- On **gfx942 there is no native 32-elem block-scaled MFMA** → block scaling is emulated/coarse; don't assume
  gfx950 behavior. Use fp8 tensor-scaled there.
- **K not aligned to 32** → scale misindex (silently wrong, not an error).
- Like dense Triton, verify the ISA actually lowers the scaled-MFMA (`AMDGCN_ENABLE_DUMP=1`) — a fallback to
  emulated dequant-then-bf16-dot tanks throughput.
- Treat Triton as the parity reference, not the peak — expect Gluon/asm to beat it by the last 10–20%.

## How to verify (worked example)
```bash
AMDGCN_ENABLE_DUMP=1 python block_scaled_matmul.py 2>&1 | grep -i mfma_scale
# achieved TFLOPS vs dtype peak + bf16 accuracy gate
python block_scaled_matmul.py --check-accuracy --ref bf16   # err vs bf16 within tol
```

## Alternatives / cross-links
operators/scaled_quant_gemm/backends/aiter (live path) · operators/scaled_quant_gemm/backends/asm
(Gluon peak) · operators/dense_gemm/backends/triton (plain GEMM knobs) ·
quantization/block_scaling_mxfp · quantization/fnuz_vs_ocp ·
operators/scaled_quant_gemm/overview

## Sources
- Triton block-scaled matmul tutorial: https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
- Gluon efficiency ceilings (BF8 99.72%, MXFP4 92.41%): https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- CDNA scaled-MFMA / E8M0 layout: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
