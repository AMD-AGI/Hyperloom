---
title: quant_fp4_mxfp on triton — SOTA card
kind: sota_card
operator: quant_fp4_mxfp
backend: triton
gens: [gfx950]
dtypes: [mxfp4, mxfp6]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/quant/quant.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/utility/fp4_utils.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/quant/fused_mxfp4_quant.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
---

# quant_fp4_mxfp × triton

## TL;DR
Triton is a SOTA MXFP4 quant path because (a) the quant is memory-bound so no asm edge is lost, and (b)
Triton's **`tl.dot_scaled`** consumes MXFP4 + E8M0 scales natively (lowering to the CDNA4 scaled-MFMA),
so the Triton quant + Triton block-scaled GEMM compose cleanly without a separate scale shuffle.
`dynamic_mxfp4_quant` casts activations (group 32, **shape-adaptive** tiling); the `fused_mxfp4_quant.py`
family fuses norm / act / MoE. Use the **unshuffled** scale layout for `tl.dot_scaled`; switch to aiter's
HIP `e8m0_shuffle` only when feeding the raw HW MFMA directly. gfx950-only for a HW win.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `dynamic_mxfp4_quant(x, scaling_mode="even")` | `aiter/ops/triton/quant/quant.py:134` | gfx950, mxfp4 | shape-adaptive tiling (M/N branches) | activation MXFP4 |
| `fused_rms_mxfp4_quant`, `fused_reduce_act_mul_and_mxfp4_quant` | `fused_mxfp4_quant.py:22/194` | gfx950 | fuses the pass | production norm/act |
| `fused_dynamic_mxfp4_quant_moe_sort` | `fused_mxfp4_quant.py:561` | gfx950 | sort + MXFP4 in one kernel | MoE |

### SOTA excerpt — fixed block 32 + shape-adaptive tiling (`quant.py:134`)
```python
def dynamic_mxfp4_quant(x, scaling_mode="even"):       # x: fp16/bf16, returns (x_fp4_uint8, blockscale_e8m0)
    M, N = x.shape
    assert (N // 2) % 2 == 0
    MXFP4_QUANT_BLOCK_SIZE = 32                         # "This is fixed by spec for MXFP4. Do not tune this."
    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)        # 2 fp4 per byte
    blockscale_e8m0 = torch.empty(((N + 31)//32, M), dtype=torch.uint8).T       # [(N+31)//32, M].T
    if M <= 32:                                         # decode / few rows
        NUM_ITER, BLOCK_SIZE_M, BLOCK_SIZE_N, NUM_WARPS, NUM_STAGES = 1, npow2(M), 32, 1, 1
    else:                                               # prefill / many rows
        NUM_ITER, BLOCK_SIZE_M, BLOCK_SIZE_N, NUM_WARPS, NUM_STAGES = 4, 64, 64, 4, 2
        if N <= 16384: BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 128
    if N <= 1024:                                       # narrow hidden
        NUM_ITER, NUM_STAGES, NUM_WARPS = 1, 1, 4
        BLOCK_SIZE_N = max(32, min(256, npow2(N)))      # must be a multiple of 32
        BLOCK_SIZE_M = min(8, npow2(M))
```
The tiling auto-selects by (M, N) regime — do not carry NVIDIA `num_warps` here; this picks 1–4 for AMD.

## Config space / knobs
| knob | range | note |
|---|---|---|
| `MXFP4_QUANT_BLOCK_SIZE` | **32** (fixed) | "Do not tune this" — OCP MX spec |
| `BLOCK_SIZE_M/N` | auto by M/N | 8/32/64/128 branches; `BLOCK_SIZE_N` multiple of 32 |
| `NUM_WARPS` | 1–4 (auto) | 1 for decode, 4 for prefill |
| `NUM_ITER` / `NUM_STAGES` | 1–4 / 1–2 | auto; keep low for fused |
| `scaling_mode` | "even" | Quark `even_round` |
| scale output | `[(N+31)//32, M].T` uint8 / `fp8_e8m0` | unshuffled (transposed) for `tl.dot_scaled` |

## Measured performance
| config | metric | value @ hw / ver / date | source |
|---|---|---|---|
| `dynamic_mxfp4_quant` | bound | HBM-bound per-block amax (no MFMA) | `quant.py:134` |
| Triton `tl.dot_scaled` MXFP4 GEMM | rate | lowers to CDNA4 scaled-MFMA (16x16x128 / 32x32x64) | block-scaled-matmul tutorial |
| fused RMS+MXFP4 vs unfused | saved traffic | one fewer activation read+write | `fused_mxfp4_quant.py` |
| gfx942 | speedup | none — simulation only (no FP4 MFMA) | HW matrix |

> No vendor throughput asserted; the quant cast is memory-bound. Bench the fused vs unfused chain at your
> shapes per conventions.

## Numerics / parity
- **E8M0 group-32 scale**, power-of-2. `tl.dot_scaled` repeat-interleaves the e8m0 scale across 32
  elements and applies it after the dot.
- **Layout:** Triton 3.6+ expects `rhs_scale` in transposed form `(N, K//32)`; the unshuffled scale here
  is the `tl.dot_scaled` form (not the HW `e8m0_shuffle` layout).
- MXFP4 = e2m1 (max 6.0, scaling denom 4.0); MXFP6 = e2m3/e3m2 accuracy fallback.
- **Gate:** task accuracy, never byte parity → numerics.md.

## Integration (rebind seam)
- `aiter.ops.triton.quant.dynamic_mxfp4_quant`; pairs with a `tl.dot_scaled` GEMM. Routed by
  `get_triton_quant(QuantType.per_1x32)`.
- Pure Python kernel → autotune overlay, **no site-packages edit**. → ../../../kernel_workflow/integrating_a_new_kernel.

## Pitfalls & anti-patterns
- **gfx942: no FP4 HW** — simulation only (../../../quantization/hardware_support_matrix).
- **Triton 3.6+ `rhs_scale` transpose** requirement — `(N, K//32)`; mismatched → wrong result.
- `num_warps` carried from NVIDIA → VGPR spill; the kernel already auto-picks 1–4 for AMD.
- Group ≠ 32 / tuning `MXFP4_QUANT_BLOCK_SIZE` → breaks the MX format.
- Shuffled (HW) scale fed to `tl.dot_scaled` → corruption; `tl.dot_scaled` wants **unshuffled**.

## How to verify
- `AMDGCN_ENABLE_DUMP=1` to confirm the fp4 cast / scaled-MFMA lowered.
- Round-trip per-block error vs bf16; gate `tol_err_ratio=0.05`.
- `tl.dot_scaled` result vs a bf16 reference matmul; e2e gsm8k on gfx950.

## Alternatives / cross-links
aiter · hip · ck · [overview.md](overview.md) ·
[numerics.md](numerics.md) · triton_levers · ../../../quantization/block_scaling_mxfp ·
operators/scaled_quant_gemm.

## Worked example
Quantize a `[M, 8192]` activation to MXFP4 and run a `tl.dot_scaled` GEMM on gfx950:
```python
import torch
from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant
x = torch.randn(4096, 8192, dtype=torch.bfloat16, device="cuda")
x_fp4, x_scale = dynamic_mxfp4_quant(x, scaling_mode="even")   # x_fp4 [4096,4096] uint8, scale [(8192//32),4096].T
# x_fp4 + x_scale (unshuffled) feed tl.dot_scaled; for Triton>=3.6 transpose rhs_scale to (N, K//32).
```
Passing the HW-shuffled scale (aiter `e8m0_shuffle`) into `tl.dot_scaled` would corrupt the result — use
the unshuffled layout this function emits.

## Sources
- Triton `dynamic_mxfp4_quant` + fused: `ROCm/aiter@a6bb49937:aiter/ops/triton/quant/quant.py:134`,
  `aiter/ops/triton/quant/fused_mxfp4_quant.py` (`fused_rms_mxfp4_quant:22`,
  `fused_reduce_act_mul_and_mxfp4_quant:194`, `fused_dynamic_mxfp4_quant_moe_sort:561`),
  `aiter/utility/fp4_utils.py` (`dynamic_mxfp4_quant:401`).
- Triton AMD knobs (num_warps/num_stages): https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- `tl.dot_scaled` block-scaled matmul, E8M0 repeat_interleave(32), 16x16x128 / 32x32x64:
  https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
