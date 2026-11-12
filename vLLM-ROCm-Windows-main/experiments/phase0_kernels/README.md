# Phase 0+ — Real HIP kernels on gfx1100 (RMSNorm + rocWMMA GEMM)

**Status: PASSED (2026-06-29)** on RX 7900 XT (gfx1100), native Windows.

Escalates the Gate-A `.pyd` proof to *real* kernels:

| Kernel | What it proves | Result |
|---|---|---|
| `rmsnorm` (`ops.cu`) | A genuine LLM op: block reduction + normalize + weight, fp32 | `RMSNORM_OK` — max_err ~9.5e-7 vs torch |
| `wmma_gemm` (`ops.cu`) | The **RDNA3 matrix path**: fp16→fp32 16×16×16 wave32 GEMM via **rocWMMA** | `WMMA_GEMM_OK` — rel err 0.0 vs torch (256×256×256) |
| `wmma_igemm` (`ops.cu`) | **INT8 matrix path** (iu8→i32) — building block of W4A16/W8A8 quant | `WMMA_IGEMM_OK` — max_abs_diff 0 vs torch (128×128×128) |

**Why this matters:** rocWMMA compiles and runs correctly under the native-Windows HIP SDK 7.2 toolchain on gfx1100, so we can **hand-write WMMA-based GEMM / attention kernels** for RDNA3 — the fallback route for a paged-attention kernel if the Triton path underperforms (see `research/04`).

## Build recipe (same as `../phase0_hip_ext`, plus one fix)
Run with MSVC env active + `ROCM_HOME=ROCM_PATH=HIP_PATH=C:\HIP-SDK`:
```
cmd /c "\"E:\BuildTools\VC\Auxiliary\Build\vcvars64.bat\" && set ROCM_HOME=C:\HIP-SDK && set HIP_PATH=C:\HIP-SDK && set ROCM_PATH=C:\HIP-SDK && python build_and_test.py"
```
`extra_cuda_cflags` carry:
- `--rocm-device-lib-path=C:\HIP-SDK\lib\llvm\amdgcn\bitcode` (device bitcode location)
- `-U__HIP_NO_HALF_CONVERSIONS__ -U__HIP_NO_HALF_OPERATORS__` — **rocWMMA needs the `__half(float)` ctor** that torch's HIP flags otherwise strip (last flag on the clang line wins).

Plus the cpp_extension hipify-None monkeypatch from `../phase0_hip_ext`.

## Notes / next
- `block(32)` = one wave32 per 16×16 output tile on gfx1100; rocWMMA `fragment<...,16,16,16,...>` handles lane layout.
- Next escalations: a tiled/shared-memory WMMA GEMM for real throughput; a WMMA flash-attention micro-kernel (paged-KV semantics); and confirming vLLM's W4A16 HIP kernel (PR #41394) builds on this toolchain.
