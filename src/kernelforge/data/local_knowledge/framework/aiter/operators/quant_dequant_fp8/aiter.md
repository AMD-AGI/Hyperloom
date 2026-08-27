---
title: quant_dequant_fp8 on aiter — SOTA card
kind: sota_card
operator: quant_dequant_fp8
backend: aiter
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3, fp8_e5m2]
regimes: [prefill, decode, both]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/quant.py
  - ROCm/aiter@b467ce342:csrc/kernels/quant_kernels.cu
  - ROCm/aiter@b467ce342:csrc/include/smoothquant.h
  - ROCm/aiter@b467ce342:aiter/ops/triton/quant/fused_fp8_quant.py
  - ROCm/aiter@b467ce342:aiter/test_common.py
  - https://github.com/ROCm/aiter
---

# quant_dequant_fp8 × aiter

## TL;DR
aiter is the **live FP8 quant path** on sglang/vLLM. It ships the per-tensor (static/dynamic), per-token
(dynamic), and per-group/per-block FP8 quant kernels, plus — the real reason to use it — the **fused**
norm/act/KV variants that erase the standalone HBM pass. Use aiter's quant when the FP8 output feeds an
aiter `gemm_a8w8` or KV store: they share the `scale = amax/dtypeMax` convention and the dispatcher
(`get_torch_quant` / `get_hip_quant` / `get_triton_quant`) picks the HIP / Triton path per shape and
quant type. Standalone, the canonical entrypoints are `per_token_quant_hip` →
`dynamic_per_token_scaled_quant` (dynamic per-token) and `per_tensor_quant_hip` →
`dynamic_per_tensor_quant` / `static_per_tensor_quant`. Do **not** reach for it when no fused variant
exists and the op is a one-off — there a single Triton cast is just as good (the op is memory-bound, no
MFMA).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `per_token_quant_hip` → `dynamic_per_token_scaled_quant` | `aiter/ops/quant.py:407` → `csrc/kernels/quant_kernels.cu:721` | gfx942/950, e4m3/i8/fp4 | bandwidth-bound row-reduce, vectorized 16B loads (`vec_size_i=16/sizeof`) | activation quant before a linear |
| `per_tensor_quant_hip` (`static_*` / `dynamic_*`) | `aiter/ops/quant.py` → `quant_kernels.cu:601`/`:675` | gfx942/950 | single scalar scale; dynamic uses `atomicMaxFloat` amax | weights / calibrated activations |
| `per_group_quant_hip(group_size∈{32,64,128})` | `aiter/ops/quant.py:440` | gfx942/950 | block-FP8 (DeepSeek-style 1×128) | block-scaled FP8 GEMM |
| `pertoken_quant` (py ref + smooth) | `aiter/ops/quant.py:64` | all | reference + SmoothQuant `x_scale` | per-token + optional smooth scale |
| **fused** `fused_rms_fp8_group_quant`, `fused_silu_mul_fp8_per_tensor_static_quant`, `fused_reduce_act_mul_fp8_group_quant` | `aiter/ops/triton/quant/fused_fp8_quant.py` + fusion.md | gfx942/950 | removes a full HBM read+write pass | the production path |

### SOTA excerpt — dynamic per-token scale (`aiter/ops/quant.py:64`)
```python
def pertoken_quant(x, scale=None, x_scale=None,  # x_scale = smooth_scale
                   scale_dtype=dtypes.fp32, quant_dtype=dtypes.i8, dtypeMax=None):
    x = x.to(dtypes.fp32)
    hidden_states = x if x_scale is None else x * x_scale   # SmoothQuant path
    if dtypeMax is None:
        dtypeMax = get_dtype_max(quant_dtype)               # torch.finfo(...).max → fnuz vs ocp
    per_token_scale = scale
    if scale is None:
        per_token_amax, _ = torch.max(torch.abs(hidden_states), dim=-1, keepdim=True)
        per_token_scale = per_token_amax / dtypeMax
        per_token_scale[per_token_scale == 0] = 1            # zero-scale → 1 (avoid div-by-0)
    y = (hidden_states / per_token_scale).to(dtype=quant_dtype)
    return y, per_token_scale.to(scale_dtype)
```
The fast on-device path is the HIP kernel; `pertoken_quant` is the parity reference. The HIP kernel does a
16-byte-vectorized double-buffered row read and a wavefront amax reduce (`quant_kernels.cu`), so it is
HBM-bound: ~1 read + 1 write of the tensor.

### SOTA excerpt — dynamic per-tensor amax via atomic (`csrc/kernels/quant_kernels.cu:244`)
```cpp
__device__ __forceinline__ float atomicMaxFloat(float *addr, float value) {
  float old;
  old = (value >= 0)
          ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
          : __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
  return old;                          // sign-aware max trick: positive→atomicMax, negative→atomicMin
}
// scaled_quant_impl: inverted_scale = rcpf(*scale) for fp8 → per-elem op is a multiply
```

## Config space / knobs
| knob | values | effect |
|---|---|---|
| granularity (`QuantType`) | `per_Tensor` / `per_Token` / `per_1x128` / `per_1x32` | static weights vs dynamic activations vs block-FP8 vs MXFP4 |
| `group_size` | 32 / 64 / 128 | `per_group_quant_hip` only supports these three (asserted) |
| `quant_dtype` | `dtypes.fp8` (e4m3), `dtypes.fp8_e5m2`, `dtypes.i8`, `dtypes.fp4x2` | output element type; `scale_dtype` is fp32 |
| `x_scale` (smooth) | per-channel `[1,n]` | pass into `pertoken_quant`/`smoothquant_fwd` → SmoothQuant FP8 |
| `num_rows` / `num_rows_factor` | tensor / int | ragged / MoE row count (variable tokens) |
| dispatcher | `get_hip_quant` vs `get_triton_quant` | HIP kernel vs Triton fallback |
| HIP `thread_data_size` / block size | 16 / 32 | vectorization width — see tuning.md |

## Measured performance
| config | metric | value @ hw / ver / date | source |
|---|---|---|---|
| dynamic per-token e4m3, hidden 8192 | nature of bound | HBM-bound: ~1 read + 1 write of activation (vectorized 16B) | code analysis, `quant_kernels.cu:249` |
| fused norm+quant vs standalone | saved traffic | removes one full activation read+write from HBM | fusion.md, `fused_fp8_quant.py` |
| MoE per-group FP8 | correctness gate | `tol_err_ratio = 0.05` (`checkAllclose`) | `aiter/test_common.py:482` |

> No vendor throughput number is asserted here — the standalone quant is memory-bound and shape-dependent.
> Bench at your hidden size with `op_tests/test_quant.py`; report median of ≥3 warm repeats per
> conventions.

## Numerics / parity
- **Dialect:** e4m3 **FNUZ** on gfx942 (CDNA3), **OCP** e4m3 on gfx950 (CDNA4). `dtypeMax` comes from
  `torch.finfo(quant_dtype).max`, so the *torch dtype you pass* decides the dialect — pass the dtype that
  matches the checkpoint and the arch (../../../quantization/fnuz_vs_ocp).
- **Scale:** `scale = amax / dtypeMax`; zero-scale clamped to `1`. Per-element store uses `rcpf(scale)`
  (a multiply), so the dequant is `q * scale`.
- **e4m3 vs e5m2:** e4m3 (4-exp, 3-mant) for activations/weights; e5m2 (5-exp, 2-mant) only where range
  beats precision. FNUZ ("Finite, NaN, Unsigned-Zero") gives one extra exponent of range vs OCP and has
  no Inf / no −0.
- **Gate:** task accuracy / err-ratio, **never** byte parity. The aiter gate is `tol_err_ratio=0.05` (≤5%
  of elements may miss `rtol=atol=1e-2`) — see `checkAllclose`. → numerics.md.

## Integration (rebind seam)
- Call sites: `aiter.ops.quant.{per_token_quant_hip, per_tensor_quant_hip, per_group_quant_hip}` and the
  fused norm/KV ops; the type-routed factories are `get_hip_quant(qType)` / `get_triton_quant(qType)`.
- In vLLM, gated by `VLLM_ROCM_USE_AITER=1` plus `VLLM_ROCM_USE_AITER_LINEAR=1` /
  `VLLM_ROCM_USE_AITER_RMSNORM=1` (the latter pulls in the fused RMS+quant). → ../../../reference/env_vars.
- The FP8 tensor + fp32 scale produced here are consumed by `gemm_a8w8`
  (operators/scaled_quant_gemm); keep the scale granularity matched on both sides.

## Pitfalls & anti-patterns
- **FNUZ↔OCP dialect mismatch** with the checkpoint → silently ~2× error (the wrong-dialect trap). The cast
  is dictated by the torch dtype, so a copy-pasted `float8_e4m3fn` on gfx942 is the classic bug
  (../../../quantization/fnuz_vs_ocp).
- Running **standalone** quant when a fused norm/act variant exists → a wasted HBM pass (fusion.md).
- **Per-tensor on outlier activations** → underflow of small channels; use per-token (or SmoothQuant
  `x_scale`).
- `per_group_quant_hip` with `group_size ∉ {32,64,128}` → assertion failure.
- Passing a pre-computed `scale` to the per-token HIP path raises "unsupported: static per token quant".

## How to verify
- Round-trip max-rel error + err-ratio vs fp32 reference (`pertoken_quant`); gate at `tol_err_ratio=0.05`.
- `AITER_LOG_MORE=1` to confirm the HIP kernel dispatched (vs the Triton fallback).
- e2e gsm8k parity after enabling the FP8 linear; rocprofv3 to confirm the aiter quant symbol (not a vLLM
  `scaled_fp8_quant_kernel`) ran.

## Alternatives / cross-links
vllm_kernels · hip · triton · asm ·
[overview.md](overview.md) · [fusion.md](fusion.md) · [numerics.md](numerics.md) ·
operators/scaled_quant_gemm · operators/fused_norm_quant · ../../../quantization/scaling_strategies ·
../../../quantization/fnuz_vs_ocp.

## Worked example
Quantize a `[4096, 8192]` bf16 activation to per-token e4m3 on gfx942, feed `gemm_a8w8`:
```python
import torch, aiter
from aiter import dtypes
from aiter.ops.quant import get_hip_quant
from aiter.ops.enum import QuantType                     # canonical (bound from C++ module_aiter_core)
x = torch.randn(4096, 8192, dtype=torch.bfloat16, device="cuda")
quant = get_hip_quant(QuantType.per_Token)               # → per_token_quant_hip
y, scale = quant(x, quant_dtype=dtypes.fp8)              # y: e4m3_fnuz [4096,8192], scale fp32 [4096,1]
# y, scale now feed gemm_a8w8 with the matching per-token scale convention
```
Reference check: `pertoken_quant(x, quant_dtype=dtypes.fp8)` then
`aiter.test_common.checkAllclose(y_ref, y, tol_err_ratio=0.05)`.

## Sources
- On-box = `ROCm/aiter@b467ce342`:
  `aiter/ops/quant.py` (`pertoken_quant:64`, `per_token_quant_hip:407`, `per_group_quant_hip:440`,
  `get_hip_quant:367`, `get_triton_quant:390`), `csrc/kernels/quant_kernels.cu`
  (`atomicMaxFloat:244`, `static_per_tensor_quant:601`, `dynamic_per_tensor_quant:675`,
  `dynamic_per_token_scaled_quant:721`), `csrc/include/smoothquant.h`,
  `aiter/ops/triton/quant/fused_fp8_quant.py`, `aiter/test_common.py:482` (`checkAllclose`,
  `tol_err_ratio=0.05`). `QuantType` is defined in `aiter/ops/enum.py`.
- aiter as central engine: https://github.com/ROCm/aiter
- FP8 e4m3/e5m2 + FNUZ semantics: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html
