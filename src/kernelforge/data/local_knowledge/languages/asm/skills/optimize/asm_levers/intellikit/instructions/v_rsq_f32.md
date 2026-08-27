---
instruction: v_rsq_f32
category: vector
architecture: gfx950
tags: [v_rsq_f32, transcendental, NOP, hazard, s_nop, RMSNorm]
---

# v_rsq_f32

Reciprocal square root: computes `1/sqrt(x)`.

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | VOP1 |
| Throughput CPI | ~7.8 shader cycles |
| Measured throughput | 0.475 ref-clk CPI (median 486 ticks / 1024 iters) |
| Functional unit | Transcendental unit (shared) |

## Hazards

### NO hardware interlock on result read

Same hazard as v_exp_f32 and v_rcp_f32.

**Required:** `s_nop 3` between v_rsq_f32 and any consumer.

```asm
v_rsq_f32 v6, v_sum_sq
s_nop 3              ; MANDATORY
v_mul_f32 v_out, v_in, v6   ; now reads correct rsqrt
```

## Common Usage Patterns

### RMSNorm (primary use case)
```asm
; norm = x * rsqrt(mean(x^2) + eps)
v_rsq_f32 v_scale, v_variance_plus_eps
s_nop 3
v_mul_f32 v_out, v_in, v_scale
```

### Attention scale factor
```asm
; scale = 1/sqrt(d_head)
; Often precomputed on host, but if done in-kernel:
v_rsq_f32 v_scale, v_d_head_f32
s_nop 3
```

## Sources

- rmsnorm kernels, attention scale computation
- isa-bench: trans_rsq_f32 throughput kernel
