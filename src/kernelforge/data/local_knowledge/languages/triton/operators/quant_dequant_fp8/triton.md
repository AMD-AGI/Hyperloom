---
title: quant_dequant_fp8 on triton — SOTA card
kind: sota_card
operator: quant_dequant_fp8
backend: triton
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode, both]
status: competitive
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/quant/quant.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/quant/fused_fp8_quant.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/quant.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html
---

# quant_dequant_fp8 × triton

## TL;DR
Triton is the **fusion-friendly, portable** FP8 quant authoring path, and the one where AMD shines for
quant: the standalone quant is memory-bound (no MFMA), so Triton's lack of a tuned-asm edge doesn't cost
anything. The real value is that the **fused** norm+quant / act+quant kernels are written in Triton
(`fused_fp8_quant.py`), so the quant rides for free inside a kernel you would author anyway. On gfx942 you
must emit the **fnuz** fp8 type — feeding OCP `float8_e4m3fn` into `tl.dot`/a cast on gfx942 errors with
`Unsupported conversion from 'f8E4M3FN'`. Prefer Triton when (a) you are co-authoring norm/act+quant, or
(b) you need a portable fallback when the AITER HIP quant path is off. Reach for the HIP/aiter kernel only
when it is already the live dispatch.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `dynamic_per_token_quant_fp8_i8` | `aiter/ops/triton/quant/quant.py:94` | gfx942/950, e4m3/i8 | grid=`(rows,)`, `NUM_COL_POW2` block reduce per row | per-token activation quant |
| `static_per_tensor_quant_fp8_i8` | `:27` | gfx942/950 | scalar scale, `(rows,)` grid | weights / calibrated |
| `dynamic_per_tensor_quant_fp8_i8` | `:54` | gfx942/950 | row amax → tensor scale, `DTYPE_MAX` from torch finfo | dynamic per-tensor |
| **fused** `fused_rms_fp8_group_quant`, `fused_silu_mul_fp8_per_tensor_static_quant`, `fused_reduce_act_mul_fp8_group_quant` | `fused_fp8_quant.py:24/161/382/748` | gfx942/950 | **fuses the pass away** | production (norm/act + quant) |

### SOTA excerpt — DTYPE_MAX picks the dialect from the torch dtype (`quant.py:94`)
```python
def dynamic_per_token_quant_fp8_i8(qx, x_in, scale_out):
    rows, cols = x_in.shape
    NUM_COL_POW2 = triton.next_power_of_2(cols)         # pad hidden to a power of 2
    grid = lambda meta: (rows,)                         # one program per token-row
    _dynamic_per_token_quant_fp8_i8_kernel[grid](
        qx, scale_out, x_in, cols, x_in.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
        DTYPE_MAX=(torch.finfo(qx.dtype).max            # ← fnuz(224/240) vs ocp(448) auto from dtype
                   if torch.is_floating_point(qx)
                   else torch.iinfo(qx.dtype).max))
```
Because `DTYPE_MAX` is read from `torch.finfo(qx.dtype)`, the **caller's output dtype** silently selects
the FP8 dialect — allocate `qx` as the fnuz type on gfx942.

## Config space / knobs
| knob | range | note |
|---|---|---|
| `BLOCK_SIZE` / `NUM_COL_POW2` | `triton.next_power_of_2(cols)` | one row per program; whole hidden in registers/LDS |
| `num_warps` | 4–8 wide hidden / 1–2 narrow | NVIDIA-carried `8` → VGPR spill on AMD |
| `num_stages` | 1–2 | AMD stream pipeliner; keep low for fused norm+quant |
| `DTYPE_MAX` | `finfo(qx.dtype).max` | handles fnuz vs ocp transparently |
| `MXFP4_QUANT_BLOCK_SIZE`/group | n/a here | fp8 group path is per-1×128 (`per_block_quant_wrapper`) |
| decode (few rows) | launch-bound | prefer a fused variant |

## Measured performance
| config | metric | value @ hw / ver / date | source |
|---|---|---|---|
| per-token e4m3, one row/program | bound | HBM-bound row read + write (no MFMA) | code analysis, `quant.py:94` |
| fused RMS+FP8 vs RMS then quant | saved traffic | one fewer activation read+write | `fused_fp8_quant.py` |
| decode, M small | bound | launch/occupancy-bound → fuse | tuning.md |

> Standalone Triton quant matches HIP within HBM bandwidth; the differentiator is fusion, not raw quant
> speed. Bench fused vs unfused at your shapes; no vendor number asserted.

## Numerics / parity
- **Dialect:** fnuz on gfx942 / ocp on gfx950. Pick the fp8 torch dtype matching the arch or you get a
  Triton compile error (`Unsupported conversion from 'f8E4M3FN'`). Triton spells the fnuz e4m3 type
  `tl.float8e4b8` (bias-8 fnuz) vs OCP `tl.float8e4nv` / `float8_e4m3fn`.
- `scale = amax / DTYPE_MAX`, `DTYPE_MAX = torch.finfo(qx.dtype).max`. e4m3 vs e5m2 chosen by the alloc
  dtype.
- **Gate:** task accuracy / err-ratio (`tol_err_ratio=0.05`), not byte parity → numerics.md.

## Integration (rebind seam)
- `aiter.ops.triton.quant.*`; routed by `get_triton_quant(qType)` (`aiter/ops/quant.py:276`). In vLLM the
  Triton quant path is the fallback when AITER HIP quant is off.
- Pure Python kernel → overlay a tuned config via the kernel's `@triton.autotune`; **no site-packages
  edit** needed. → ../../../kernel_workflow/integrating_a_new_kernel.

## Pitfalls & anti-patterns
- **OCP `float8_e4m3fn` on gfx942** → `Unsupported conversion from 'f8E4M3FN'`; use the fnuz type
  (`tl.float8e4b8`).
- `num_warps=8` carried over from an NVIDIA kernel → VGPR spill on CDNA; start at 4.
- Standalone quant in decode → launch-bound; fuse into the norm/act kernel.
- Forgetting that `DTYPE_MAX` follows the **output** dtype — allocating `qx` as the wrong dialect is the
  2× error trap with no compile error if both dialects exist in torch.

## How to verify
- `AMDGCN_ENABLE_DUMP=1` to confirm the fp8 cast lowered (look for the `cvt` to the fnuz type).
- Round-trip max-rel error vs fp32; gate `tol_err_ratio=0.05`.
- e2e gsm8k parity; confirm via the kernel's `_LOGGER.info` marker (e.g.
  `DYNAMIC_PER_TOKEN_QUANT_FP8_I8: x=...`).

## Alternatives / cross-links
aiter (live HIP path) · hip · vllm_kernels ·
[fusion.md](fusion.md) · [numerics.md](numerics.md) · triton_levers ·
[overview.md](overview.md) · ../../../quantization/fnuz_vs_ocp.

## Worked example
Per-token FP8 quant of a `[M, 8192]` activation on gfx942 (fnuz), then a `tl.dot`-based FP8 GEMM:
```python
import torch, triton
from aiter import dtypes
from aiter.ops.triton.quant.quant import dynamic_per_token_quant_fp8_i8
x = torch.randn(2048, 8192, dtype=torch.bfloat16, device="cuda")
qx = torch.empty_like(x, dtype=dtypes.fp8)               # dtypes.fp8 == fnuz e4m3 on gfx942
scale = torch.empty(2048, 1, dtype=torch.float32, device="cuda")
dynamic_per_token_quant_fp8_i8(qx, x, scale.view(-1))    # DTYPE_MAX = finfo(dtypes.fp8).max
```
Allocating `qx` as `torch.float8_e4m3fn` instead would compile-error on gfx942 — the fnuz dtype is
mandatory.

## Sources
- Triton fp8 quant + fused variants: `ROCm/aiter@a6bb49937:aiter/ops/triton/quant/quant.py`
  (`static_per_tensor_quant_fp8_i8:27`, `dynamic_per_tensor_quant_fp8_i8:54`,
  `dynamic_per_token_quant_fp8_i8:94`), `aiter/ops/triton/quant/fused_fp8_quant.py`
  (`fused_rms_fp8_group_quant:161`, `fused_reduce_act_mul_fp8_group_quant:382`,
  `fused_silu_mul_fp8_per_tensor_static_quant:748`), dispatcher `aiter/ops/quant.py:276`.
- fnuz vs ocp Triton dtype, num_stages/num_warps: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- FP8 e4m3/e5m2 + FNUZ low-fp types: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html
