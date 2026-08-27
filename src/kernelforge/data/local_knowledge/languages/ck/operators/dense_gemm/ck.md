---
title: dense_gemm on ck — SOTA card
kind: sota_card
operator: dense_gemm
backend: ck
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, mxfp4]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
  - https://github.com/ROCm/composable_kernel
  - https://rocm.blogs.amd.com/software-tools-optimization/ck-int8-gemm-sq/README.html
  - ROCm/aiter@a6bb4993:csrc/ck_gemm_a8w8/gemm_a8w8.cu
---

# dense_gemm × ck

## TL;DR
> Composable Kernel is AMD's **templated C++ kernel generator** for GEMM — the right choice when you need a
> *custom epilogue* (bias/act/residual/quant) or a precision hipBLASLt doesn't expose (int8 SmoothQuant,
> mxfp4), and you can build a kernel into your stack. For plain bf16/fp8 dense GEMM on the live serving path,
> **prefer the tuned hipBLASLt kernels that aiter dispatches** — CK rarely beats a well-tuned Tensile
> solution there. CK is also aiter's **FlyDSL fallback** for low-bit MoE. It shines as the *authoring
> substrate* for fused/quantized GEMM, not as a live-path lever (no env overlay like aiter's CSV).

## SOTA implementation
aiter ships CK GEMM as compiled `.cu` instances (`csrc/ck_gemm_a8w8/`, `ck_gemm_a4w4_blockscale/`, etc.),
each selected from its own lookup table keyed on `(gfx, cu_num, padded_m, N, K)` — the same `getPaddedM`
bucketing as the bf16 dispatcher. From `/sgl-workspace/aiter/csrc/ck_gemm_a8w8_blockscale/...cu`
(`ROCm/aiter@a6bb4993`):

```cpp
int padded_m = M;
padded_m = getPaddedM(M, N, K, 0);            // gl=0 fine-grained
it = lookup.find({gfx, cu_num, padded_m, N, K});
if (it == lookup.end()) {
    padded_m = getPaddedM(M, N, K, 1);        // gl=1 coarse (nextPow2)
    it = lookup.find({gfx, cu_num, padded_m, N, K});
}
```

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `DeviceGemmMultipleD_Xdl_CShuffle` | `ROCm/composable_kernel@develop` (now `ROCm/rocm-libraries`) | gfx942/950; bf16/fp16/fp8/int8/mxfp4 | no published isolated bf16 number beats tuned hipBLASLt → **use operators/dense_gemm/backends/hipblaslt for plain GEMM**; CK competitive | custom epilogue / unsupported dtype |
| CK int8 GEMM (SmoothQuant) `03_gemm_bias_relu` pattern | ROCm CK int8 SmoothQuant blog | gfx942; int8 W8A8 | qualitative win for SmoothQuant serving (reduced memory + latency; AMD blog notes kernel **untuned/suboptimal**, no TFLOP figure) | int8 quantized models |
| CK mxfp4 scaled GEMM | CK + CDNA4 ISA (`V_MFMA_SCALE_F32_16X16X128_F8F6F4`) | gfx950; mxfp4 | see operators/dense_gemm/backends/asm / Gluon: MXFP4 5255 TFLOPS (92.4%) | fp4 weight-quant on MI350 |

## Config space / knobs
| param | range / values | effect | default |
|---|---|---|---|
| build `DTYPES` | `fp16;bf16;fp8;int8` | limit generated instances (compile-time cost) | all |
| `CK_USE_FP8_ON_UNSUPPORTED_ARCH` | ON / OFF | OFF on MI300X (native fp8) | OFF (MI300X) |
| `MPerBlock/NPerBlock/KPerBlock` | 16–256 tiles | per-workgroup output + K tile | per-instance |
| `MPerXDL/NPerXDL` | 16×16 / 32×32 | MFMA shape — **prefer 16×16** | per-instance |
| `MXdlPerWave/NXdlPerWave` | 1–8 | MFMA tiles per wave | per-instance |
| `CShuffleMXdlPerWavePerShuffle` | 1–4 | C-shuffle granularity for the store | per-instance |
| `BlockGemmPipeline` | v1..v5 | deeper pipeline for K-deep prefill | v3/v4 |
| `ABlockTransfer`/`BBlockTransfer` | thread-cluster + scalar-per-vector | global→LDS load shape/vectorization | per-instance |
| `CDEElementwise` | op | carries bias/act/residual/scale (fusion) | passthrough |

## Numerics / parity
fp32 MFMA accumulate; same parity story as [numerics.md](numerics.md). int8/mxfp4 paths need
task-accuracy gating, not byte parity (the SmoothQuant blog evaluates LAMBADA last-token accuracy on the
first 1000 samples).

## Integration (rebind seam)
CK is **build-time**: you compile a `DeviceGemm*` instance into a `.so`/extension and call it. There is no
env-overlay into sglang's live GEMM. To engage it you either (1) register it as an aiter candidate (the
`csrc/ck_gemm_*` instances already are, with their own lookup CSVs), or (2) call it directly from your op.
This is why CK is "authorable" rather than the live bf16 lever.

## Pitfalls & anti-patterns
- Choosing a CK instance whose tile doesn't match your shape → far below hipBLASLt; you must enumerate/tune
  instances (`ckProfiler gemm`).
- Expecting CK to silently replace serving bf16 GEMM — it won't; no rebind seam like aiter's CSV (the CK
  *quant* paths are wired in aiter, the generic bf16 dense path is hipBLASLt).
- Building all `DTYPES` → huge compile time; restrict the set.
- The int8 SmoothQuant blog kernel is **untuned** by design — don't quote its latency as a tuned ceiling.

## How to verify (worked example)
```bash
# isolated CK vs hipBLASLt on the same (M,N,K,dtype)
ckProfiler gemm 1 1 1 0 0 1 0   4096 4864 32896 -1 -1 -1   # verify=1, time=1
hipblaslt-bench -m 4096 -n 4864 -k 32896 --a_type bf16_r --b_type bf16_r --compute_type f32_r
# adopt only if CK wins AND you have a call seam (aiter candidate or direct op)
```

## Alternatives / cross-links
operators/dense_gemm/backends/hipblaslt (prefer for plain GEMM) · operators/dense_gemm/backends/aiter
(live dispatch) · operators/dense_gemm/backends/asm (peak/fp4) ·
operators/dense_gemm/backends/flydsl (CK is its fallback) · operators/dense_gemm/backends/triton ·
operators/dense_gemm/overview · language ref `languages/composable_kernel/` (P1).

## Sources
- CK GEMM design / CShuffle / build flags: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
- CK int8 SmoothQuant GEMM (memory/latency win, untuned kernel, LAMBADA eval): https://rocm.blogs.amd.com/software-tools-optimization/ck-int8-gemm-sq/README.html
- CK repo (deprecated mirror; moved to rocm-libraries): https://github.com/ROCm/composable_kernel
- aiter CK instances + `getPaddedM` lookup: `/sgl-workspace/aiter/csrc/ck_gemm_a8w8_blockscale/` (`ROCm/aiter@a6bb4993`).
