# Phase 0+ — Triton complex kernels on gfx1100 (beyond vector-add)

**Status: PASSED (2026-06-29)** on RX 7900 XT (gfx1100), native Windows, `triton-windows 3.6.0`.

| Kernel | What it proves | Result |
|---|---|---|
| `matmul_kernel` (`tl.dot`) | Triton's **WMMA codegen** on RDNA3 (tl.dot lowers to WMMA) — the core of any attention/GEMM | `TRITON_MATMUL_OK` — rel err 0.0 vs torch (512×512×512, fp16→fp32) |
| `softmax_kernel` | Row reduction / max / exp / sum — the other half of attention | `TRITON_SOFTMAX_OK` — max_err ~3.7e-9 vs torch |

**Why this matters:** the research was pessimistic about Triton on native-Windows ROCm for gfx1100 (citing flash-attention failures #4514 and MXFP4 compile errors). Those are *specific* kernels/dtypes. The **building blocks of attention — `tl.dot` (WMMA) matmul + softmax reduction — compile and run correctly here.** Combined with the rocWMMA HIP results in `../phase0_kernels`, both attention routes (Triton and hand-written HIP-WMMA) are de-risked for fp16 and INT8.

Run: `python triton_kernels.py` (no MSVC env needed; triton-windows JITs with its own clang/HIP toolchain).

## Still to validate (the harder Triton cases)
- A full **flash-attention** Triton kernel (the pattern that triggered #4514 stack-frame overflow on gfx1100) at head_size=128 with paged KV / GQA.
- vLLM's actual `TRITON_ATTN` backend selected and run end-to-end.
