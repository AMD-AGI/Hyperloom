---
instruction: v_exp_f32
category: vector
architecture: gfx950
tags: [v_exp_f32, transcendental, NOP, hazard, s_nop]
---

# v_exp_f32

Transcendental base-2 exponential. Computes `2^x`, NOT `e^x`.

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | VOP1 (op 0x20) |
| Throughput CPI | ~7.8 shader cycles (shared trans unit, 8-cycle pipeline) |
| Latency CPI | ~5.9 shader cycles (chained, includes mandatory s_nop 3) |
| Measured throughput | 0.475 ref-clk CPI (median 486 ticks / 1024 iters) |
| Measured latency | 1.17 ref-clk CPI (median 300 ticks / 256 iters) |
| Functional unit | Transcendental unit (shared with v_rcp, v_rsq, v_log, v_sqrt, v_sin) |

## Hazards

### NO hardware interlock on result read

The hardware does NOT stall dependent instructions. Reading the destination register immediately after `v_exp_f32` returns the **pre-instruction stale value**.

**Required:** `s_nop 3` (or 4+ independent instructions) between `v_exp_f32` and any consumer of its result.

```asm
v_exp_f32 v6, v6
s_nop 3              ; MANDATORY -- no HW interlock
v_add_f32 v6, 1.0, v6   ; now reads correct exp result
```

**How discovered:** `silu_mul.s` computed `v_exp_f32 v6, v6` then immediately `v_add_f32 v6, 1.0, v6`. The add read the pre-exp value. silu(1.0) returned -0.44 instead of 0.73. Five debug kernels needed to isolate.

**Measured in isa-bench:** hazard_trans_nop0 and hazard_trans_nop3 kernels store float results. With 0 NOPs, the consumer reads the stale value. With 3 NOPs, the result is correct.

## Known Bugs / Gotchas

### v_exp_f32 computes 2^x, not e^x

This is base-2, not natural exponential. To compute `exp(x)`:

```asm
; exp(x) = 2^(x * log2(e))
s_mov_b32 s11, 0x3fb8aa3b    ; log2(e) = 1.44269504
v_mul_f32 v6, s11, v6         ; x * log2(e)
v_exp_f32 v6, v6              ; 2^(x * log2(e)) = e^x
s_nop 3
```

For `exp(-x)`, use `0xBFB8AA3B` (negative log2(e)).

### Hex literal as inline VALU operand is unreliable

`v_mul_f32 v6, 0x3fb8aa3b, v6` was observed to produce wrong results in some contexts. Load the constant into an SGPR first.

## Common Usage Patterns

### Softmax (exp2 variant)
```asm
; P = exp2(S - row_max)
v_sub_f32 v10, v10, v_max
v_exp_f32 v10, v10
s_nop 3
; v10 now holds exp2(S - max)
```

### SiLU activation
```asm
; silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
s_mov_b32 s11, 0xbfb8aa3b       ; -log2(e)
v_mul_f32 v6, s11, v6            ; -x * log2(e)
v_exp_f32 v6, v6                 ; 2^(-x*log2e) = exp(-x)
s_nop 3
v_add_f32 v6, 1.0, v6            ; 1 + exp(-x)
v_rcp_f32 v6, v6                 ; 1 / (1 + exp(-x))
s_nop 3
v_mul_f32 v_out, v_x, v6         ; x * sigmoid(x)
```

## Sources

- silu_mul.s debugging, transcendental latency discovery
- Softmax recomputation in BWD attention (48 transcendentals per inner loop)
- isa-bench: trans_exp_f32 throughput/latency kernels, hazard_trans_nop0/nop3 correctness kernels
- gfx950-reference.md: consolidated hazard cheat sheet
