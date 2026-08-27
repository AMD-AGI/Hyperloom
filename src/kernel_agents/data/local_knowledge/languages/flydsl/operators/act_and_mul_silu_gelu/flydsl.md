---
title: act_and_mul_silu_gelu on FlyDSL (silu_and_mul_fq) — SOTA card
kind: sota_card
operator: act_and_mul_silu_gelu
backend: flydsl
gens: [gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp4_e2m1]
regimes: [prefill, decode]
status: sota
updated: 2026-06-09
---

# act_and_mul_silu_gelu × FlyDSL (silu_and_mul_fq)

## TL;DR
`build_silu_and_mul_fq_module` is a FlyDSL kernel that fuses **SiLU(gate)·up + optional MXFP4/MXFP8
quantization + sorted-scale write** into a single pass. It's the post-processing step for split-K MoE
stage1: the GEMM emits bf16 gate/up partials, and this kernel applies the gate activation, quantizes to
fp4/fp8 (with per-32 e8m0 scales in tiled sorted layout), or just writes bf16 when `quant_mode="none"`.
Built/cached via `_get_compiled_silu_fused` in `moe_kernels.py`. SiLU only (no GELU variant in this kernel).

## SOTA implementation
The kernel loads bf16 gate/up (interleaved or separated layout), computes
`SiLU(gate)*up = (gate · sigmoid(gate)) · up` using AMDGCN intrinsics, then quantizes per the compile-time
`quant_mode`. From `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/silu_and_mul_fq.py`:

```python
neg_log2e = arith.constant(-1.4426950408889634, type=f32)
for vi in range_constexpr(VEC):
    g = vector.extract(gate_f32, static_position=[vi], dynamic_position=[])
    u = vector.extract(up_f32,   static_position=[vi], dynamic_position=[])
    t = g * neg_log2e
    emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
    den = c1_f32 + emu
    sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den], [], [])
    act_vals.append(g * sig * u)        # SiLU(g) * u
```

For the quant path it reduces the per-32-lane abs-max via warp `shuffle_xor`, derives an e8m0 scale with a
fixed `_fp_headroom` (2 for fp4, 8 for fp8), quantizes (fp4 via hand-coded `_f32_to_e2m1`; fp8 via
`rocdl.cvt_pk_fp8_f32`), and writes the e8m0 byte into a 6-index tiled sorted scale buffer. `gui_layout`
selects block-interleaved `[gate_0:16, up_0:16, ...]` vs gate-up-separated `[gate_0:N | up_0:N]`.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `build_silu_and_mul_fq_module` (SiLU·mul + fused quant) | `kernels/silu_and_mul_fq.py` | gfx950; in bf16, out fp4 (MXFP4) / fp8 (MXFP8 e4m3fn) / bf16 | no isolated number on-box | split-K MoE stage1 epilogue; act+requant in one pass |

## Config space / knobs
From `build_silu_and_mul_fq_module(inter_dim, topk, quant_mode, gui_layout)`.

| param | range / typical | effect | default |
|---|---|---|---|
| `inter_dim` | divisible by 32 (asserted) | activation output cols (input has `inter_dim*2`) | — |
| `topk` | per-token expert slots | row addressing (`in_row = token_id*topk + slot`) | — |
| `quant_mode` | `fp4` / `fp8` / `none` | MXFP4 + e8m0 / MXFP8 e4m3fn + e8m0 / bf16 no-quant | `fp4` |
| `gui_layout` | False / True | gate-up separated vs block-interleaved (block 16) | False |
| `VEC` (derived) | `max(ceil(inter_dim/256),2)`, even, ≤16 if interleaved | elems/thread; must divide 32 | derived |
| `BLOCK_THREADS` | 256 | threads per block (grid = num_sorted_rows) | 256 (fixed) |
| `_fp_headroom` (derived) | 2 (fp4) / 8 (fp8) | e8m0 scale headroom bits | by quant_mode |

## Numerics / parity
SiLU computed in fp32 via `exp2`/`rcp` intrinsics (the `-log2(e)` trick to reuse exp2 for sigmoid).
Quantization is **MX block-scaled per 32 elements**: abs-max → e8m0 exponent (`c254 - max(exp-headroom,0)`),
then scale and convert. fp4 output is E2M1 packed 2-per-byte; fp8 is e4m3fn via `cvt_pk_fp8_f32`. Invalid
sorted rows write a zero scale. fp4/fp8 are lossy — parity is at the dequant/task level, not bit-exact;
`quant_mode="none"` is bf16-faithful (truncf from fp32). See [../numerics.md](../numerics.md).

## Integration (rebind seam)
No standalone public API — reached through MoE stage1. In `moe_kernels.py`:
```python
@functools.cache
def _get_compiled_silu_fused(inter_dim, topk, quant_mode="fp4", gui_layout=False):
    from aiter.ops.flydsl.kernels.silu_and_mul_fq import build_silu_and_mul_fq_module
    return build_silu_and_mul_fq_module(inter_dim, topk, quant_mode, gui_layout)
```
`flydsl_moe_stage1` calls it for the split-K / gate-up-interleave fused-quant cases. To use directly, import
`build_silu_and_mul_fq_module` and launch with `(x, out_buf, out_scale_sorted, sorted_ids, num_valid_ids,
token_num, num_sorted_rows, stream)`. Only meaningful when `is_flydsl_available()`.

## Pitfalls & anti-patterns
- **SiLU only** — no GELU/gelu_tanh variant in this kernel; for GELU+mul use the aiter/triton/hip act paths.
- `inter_dim` must be divisible by 32, and the derived `VEC` must divide 32 (and ≤16 for interleaved) — both
  asserted at build time.
- Input is the **stage1 gate/up partials** in the expected (separated or interleaved) layout; `gui_layout`
  must match how stage1 emitted them or activation reads the wrong columns.
- Scale buffer is written in a specific 6-index tiled sorted layout consumed by stage2 — don't reinterpret it
  as a plain row-major scale tensor.
- fp4/fp8 lossy; verify at task accuracy. Optional dependency (flydsl) as elsewhere.

## How to verify
```python
from aiter.ops.flydsl.utils import is_flydsl_available
assert is_flydsl_available()
from aiter.ops.flydsl.kernels.silu_and_mul_fq import build_silu_and_mul_fq_module
k = build_silu_and_mul_fq_module(inter_dim, topk, "none")  # bf16, compare vs aiter silu_and_mul
```
Exercised indirectly by `test_flydsl_moe_a4w4.py` (split-K stage1 fp4 path).

## Alternatives / cross-links
[[operators/act_and_mul_silu_gelu/backends/aiter]] (CK/asm `silu_and_mul`, the non-fused default) ·
[[operators/act_and_mul_silu_gelu/backends/triton]] (act+quant Triton) ·
[[operators/fused_moe_grouped_gemm/backends/flydsl]] (the consumer: fp4 MoE stage1) ·
[[operators/grouped_gemm_moe/backends/flydsl]] · [[operators/quant_fp4_mxfp]] (MXFP4 quant).
