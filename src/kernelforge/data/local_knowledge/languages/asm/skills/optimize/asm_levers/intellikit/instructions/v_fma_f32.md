---
instruction: v_fma_f32
category: vector
architecture: gfx950
tags: [v_fma_f32, FMA, softmax, attention]
---

# v_fma_f32

Fused multiply-add: computes `src0 * src1 + src2` with a single rounding (IEEE 754 fusedMultiplyAdd).

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | VOP3 |
| Throughput CPI | 0.242 ref-clk (median 248 ticks / 1024 iters) |
| Normalized throughput | ~1.4 shader cycles |
| Category | VALU arithmetic |

## Hazards

Standard VALU forwarding latency (5 cycles). No special hazards.

## Known Bugs / Gotchas

### Fuses scale + exp into 2 instructions

The attention kernel uses `v_fma_f32` to combine the softmax scale multiplication and max subtraction into a single instruction, followed by `v_exp_f32`:

```asm
v_fma_f32 v50, s13, v50, -v123    ; S_scaled = log2e * scale * S - log2e * scale * max
v_exp_f32 v65, v50                  ; P = 2^(S_scaled) = exp(scale * (S - max))
```

This replaces the naive 3-instruction sequence (`v_sub_f32 + v_mul_f32 + v_exp_f32`), saving 33 instructions per softmax iteration (one per S value). The constant `s13` holds the precomputed `log2e * softmax_scale`.

### Newton-Raphson refinement in division

The full IEEE-754 compliant division in the attention epilogue uses `v_fma_f32` for Newton-Raphson refinement steps:

```asm
v_rcp_f32 v36, v34                    ; initial reciprocal estimate
v_fma_f32 v38, -v34, v36, 1.0         ; error = 1.0 - x * approx
v_fmac_f32 v36, v38, v36              ; refined = approx + error * approx
```

## Common Usage Patterns

### Softmax scale-and-shift (fused)
```asm
; Precompute s13 = log2e * softmax_scale on host
v_fma_f32 v_scaled, s13, v_S, -v_max_scaled
v_exp_f32 v_P, v_scaled
```

### O accumulator rescaling
```asm
; Rescale previous O by alpha = exp(old_max - new_max):
v_fmac_f32 v_l_acc, v_prev_l_acc, v_alpha
```

### Newton-Raphson division refinement
```asm
v_fma_f32 v_err, -v_denom, v_rcp, 1.0
v_fmac_f32 v_rcp, v_err, v_rcp
```

## Sources

- Fused scale+exp pattern (33 v_fma_f32 + 33 v_exp_f32 per iteration)
- Agents adopted fused v_fma+v_exp pattern for softmax
- isa-bench: valu_fma_f32 throughput kernel
