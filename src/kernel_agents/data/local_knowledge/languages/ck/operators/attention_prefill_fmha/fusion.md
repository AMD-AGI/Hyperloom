---
title: attention_prefill_fmha — fusion
kind: operator_overview
operator: attention_prefill_fmha
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
  - https://github.com/ROCm/aiter
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# attention_prefill_fmha — fusion

FlashAttention is *itself* a fusion (two GEMMs + softmax in one kernel, no materialized scores). The
fusion question here is **what fuses into the pre- and post-attention boundary** so the prefill block is
one or two kernels instead of five.

## The attention block's fusion neighbors
```
hidden ─► [RMSNorm] ─► [QKV proj GEMM] ─► [RoPE + QK-norm] ─► [FMHA] ─► [O proj GEMM] ─► residual add
            ▲ pre-attn fusions ────────────────────────┘        └──── post-attn ────┘
```

### Pre-attention (into the QKV → FMHA seam)
- **QK-norm + RoPE + KV-write + quant**, fused. aiter ships a fused
  `qk-norm + RoPE + KV-cache-write + (fp8) quant` kernel — instead of four passes over Q/K you do one,
  writing the rotated+normed+quantized K/V straight into the paged cache. This is the single biggest
  pre-attention fusion win on AMD serving (see rope/ and `backends/aiter/`).
- **fp8 QKV**: quantize Q/K/V to fp8 in the projection epilogue so FMHA consumes fp8 directly
  (FNUZ on gfx942). Folds the quant into the prior GEMM's CShuffle epilogue.
- **Bias / rotary as FMHA traits**: CK-Tile FMHA fuses bias (`bias.hpp`) and rotary (`rotary.hpp`) as
  codegen traits *inside* the kernel rather than as separate passes.

### Post-attention (out of FMHA)
- The attention output goes into the **O-projection GEMM**; the FMHA epilogue can write O in the MFMA
  layout (`OPTIMIZE_EPILOGUE=1`) so the next GEMM consumes it without a reblock pass.
- **Residual add** typically folds into the next RMSNorm (`fused_add_rmsnorm`), not into FMHA.

## What does NOT fuse (and why)
- **QKV projection GEMM does not fuse into FMHA.** They are both matrix-core-bound and the QKV output is
  consumed by RoPE/norm before attention — keep them separate (a fused "QKV-GEMM+FMHA" mega-kernel
  would spill registers and lose the GEMM's tuned tiling). The cheap win is fusing the *small* ops
  (norm/rope/quant) into the boundary, not the two big GEMMs.
- **Softmax is already fused** — never run attention as separate `matmul → softmax → matmul`; that
  materializes S=[sq,sk] (O(sq·sk) HBM traffic) and defeats the whole point.

## Backend support for the fusions
| fusion | CK-Tile | Triton/aiter FA | TileLang | asm (aiter) |
|---|---|---|---|---|
| causal/SWA mask trait | yes (codegen) | yes | yes | yes |
| ALiBi bias | no (Triton-side) | **yes** | manual | yes (aiter) |
| rotary inside FMHA | trait (`rotary.hpp`) | yes | manual | yes |
| fp8 QKV / fp8 KV-cache | yes (fp8 WarpGemm) | yes (FA-v3 iface) | WIP | yes |
| fused qk-norm+rope+kv-write+quant (pre) | — (separate aiter kernel) | aiter kernel | — | **aiter** |

## Where fusion moves e2e
On dense LLMs the FMHA forward itself is a small slice when only some layers are full-attn; the pre/post
fusions (norm+rope+quant, fp8 KV write) remove **separate kernel launches and HBM round-trips** in the
attention block, which is where the realistic e2e win is (launch-bound at small batch). The two big
GEMMs (QKV, O-proj) dominate the block's FLOPs and are tuned independently — see
dense_gemm/fusion.md.

## Sources
- CK-Tile FMHA bias/rotary traits, fp8 WarpGemm, MFMA-layout epilogue: https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
- aiter fused qk-norm+RoPE+KV-write+quant, fp8 KV-cache: https://github.com/ROCm/aiter (`aiter/rotary_embedding.py`, attention ops) ; `backends/aiter/attn_mla.md`
- 2-GEMM fusion `OPTIMIZE_EPILOGUE=1` (store O in MFMA layout): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
