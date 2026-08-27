---
instruction: v_pk_mul_f32
category: vector
architecture: gfx950
tags: [v_pk_mul_f32, packed, VOP3P, s_nop, attention]
---

# v_pk_mul_f32

Packed 2-wide FP32 multiply: computes two independent FP32 multiplications in a single instruction using a VGPR pair.

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | VOP3P |
| Throughput | ~1 shader cycle (standard VALU issue rate, estimated) |
| Category | VALU packed arithmetic |

## Hazards

Standard VALU forwarding latency (5 cycles). No special hazards.

### s_nop may be needed after v_pk_mul_f32

The forward attention disassembly has `s_nop 0` between `v_pk_mul_f32` and the following `v_fma_f32` in the softmax scale section. This suggests a potential pipeline hazard when the packed multiply result feeds into a subsequent VOP3 instruction. Conservative approach: insert `s_nop 0` after v_pk_mul_f32 when the result is immediately consumed.

## Known Bugs / Gotchas

### O accumulator rescaling in attention

Optimized kernels use `v_pk_mul_f32` to rescale the accumulated O output by the softmax alpha factor (ratio of old max to new max). This replaces 32 scalar `v_mul_f32` with 16 packed operations:

```asm
v_pk_mul_f32 v[2:3], v[2:3], v[122:123]     ; rescale O[0:1] by alpha pair
v_pk_mul_f32 v[4:5], v[4:5], v[122:123]     ; rescale O[2:3]
; ... 16 total v_pk_mul_f32 for 32 O accumulators
```

The alpha factor is broadcast using `op_sel_hi:[1,0]` or by placing the same value in both halves of the source pair.

### Max scaling for softmax

The softmax path simultaneously computes `old_max * log2e_scale` and `new_max * log2e_scale` in a single instruction:

```asm
v_pk_mul_f32 v[122:123], s[12:13], v[122:123]
; v122 = s12 * v122 (old_max * log2e_scale)
; v123 = s13 * v123 (new_max * log2e_scale)
```

### FP8 GEMM epilogue scale application

In GGEMM kernels, `v_pk_mul_f32` applies the combined FP8 scale factor (`a_scale * b_scale`) to accumulator pairs before BF16 conversion:

```asm
v_pk_mul_f32 v[0:1], v[138:139], v[0:1]     ; scale accum[0:1]
v_cvt_pk_bf16_f32 v0, v0, v1                 ; convert to BF16 pair
```

## Common Usage Patterns

### Attention O rescaling (2x throughput vs v_mul_f32)
```asm
; After new softmax max computed, rescale all O values:
v_pk_mul_f32 v[O0:O1], v[O0:O1], v[alpha:alpha+1]
; ... 16 ops for 32 accumulators
```

### FP8 GEMM scale + convert
```asm
v_pk_mul_f32 v[acc:acc+1], v[scale:scale+1], v[acc:acc+1]
v_cvt_pk_bf16_f32 v_packed, v_acc, v_acc1
buffer_store_dwordx2 v[packed:packed+1], ...
```

## Sources

- O rescale analysis (22 v_pk_mul_f32 per iteration)
- Empirically validated on MI355X: FP8 GEMM epilogue scale application
- Observed in production GEMM kernels: scale + convert + store pattern
