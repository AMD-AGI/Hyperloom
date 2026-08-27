---
instruction: v_rcp_f32
category: vector
architecture: gfx950
tags: [v_rcp_f32, transcendental, NOP, hazard, s_nop]
---

# v_rcp_f32

Reciprocal: computes `1/x`.

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | VOP1 |
| Latency CPI | ~5.9 shader cycles (chained, includes mandatory s_nop 3) |
| Measured latency | 1.15 ref-clk CPI (median 294 ticks / 256 iters) |
| Functional unit | Transcendental unit (shared) |

## Hazards

### NO hardware interlock on result read

Same hazard as v_exp_f32. Reading the destination immediately returns stale data.

**Required:** `s_nop 3` between v_rcp_f32 and any consumer.

```asm
v_rcp_f32 v6, v6
s_nop 3              ; MANDATORY
v_mul_f32 v7, v6, v7 ; now reads correct reciprocal
```

## Common Usage Patterns

### Softmax normalization
```asm
; output = P / sum(P)
; After computing l_acc = sum of exp values:
v_rcp_f32 v_inv_l, v_l_acc
s_nop 3
v_mul_f32 v_P0, v_P0, v_inv_l   ; normalize
v_mul_f32 v_P1, v_P1, v_inv_l
```

### RMSNorm
```asm
; rsqrt can also be used, but rcp(sqrt(x)) is sometimes used:
v_sqrt_f32 v6, v_variance
s_nop 3
v_rcp_f32 v6, v6
s_nop 3
v_mul_f32 v_out, v_in, v6
```

## Sources

- Attention kernel softmax, numerical stability pipeline
- isa-bench: trans_rcp_f32 latency kernel
