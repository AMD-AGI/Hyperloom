---
instruction: s_setprio
category: scalar
architecture: gfx950
tags: [s_setprio, scheduling, occupancy, MFMA]
---

# s_setprio

Set wave scheduling priority. Higher priority waves get preferential access to shared execution resources (MFMA, LDS, memory).

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | SOPP |
| Overhead CPI | 0.168 ref-clk (median 172 ticks / 1024 iters) |
| Normalized overhead | ~1 shader cycle |
| Category | Scalar flow control |

## Hazards

None. This is a scheduling hint, not a data operation.

## Known Bugs / Gotchas

### Only useful at occupancy > 1

`s_setprio` adjusts relative scheduling priority between waves on the same SIMD. At single-wave occupancy (1 WG per CU), there is no competing wave to deprioritize. Effect: exactly 0%.

At 2 waves/SIMD, the effect is marginal: 0% to +0.8% depending on workload.

rocBLAS uses `s_setprio 3` around MFMA blocks for a reported ~0.8% gain at max occupancy. This was tested extensively across grouped GEMM kernels with the following results:

| Kernel | Occupancy | Effect |
|--------|-----------|--------|
| GGEMM wgrad (legacy) | 2 waves/SIMD | 0% |
| GGEMM persistent FWD | 1 wave/SIMD | 0% |
| GGEMM persistent FWD | 2 waves/SIMD | 0% to -0.3% |
| FWD attention | 3 waves/SIMD | Not used (achieves 0 NOPs without it) |

### Must pair s_setprio 3 with s_setprio 0

Using `s_setprio 3` without a corresponding `s_setprio 0` is equivalent to not using it at all -- both waves run at the same elevated priority, which is the same as both at normal priority. The optimization requires the priority differential.

```asm
s_barrier
s_setprio 3        ; boost during MFMA-heavy phase
; ... MFMAs ...
s_setprio 0        ; restore before memory-heavy phase
; ... buffer_loads ...
```

### Can serve as 1-cycle pipeline spacer

`s_setprio` executes in 1 cycle and can replace `s_nop 0` as a pipeline spacer while serving double duty as a scheduling hint. Some grouped GEMM kernels used this to replace s_nop 0 between buffer_loads without increasing code size.

## Common Usage Patterns

### Memory-bound kernel with max occupancy (rocBLAS pattern)
```asm
s_barrier
s_setprio 3           ; prioritize MFMA execution
; ... ds_reads + MFMAs ...
s_setprio 0           ; yield for memory operations
; ... buffer_loads ...
s_cbranch .loop
```

### Pipeline spacer (replacing s_nop 0)
```asm
; Instead of:
s_nop 0               ; 1-cycle delay

; Use:
s_setprio 0           ; 1-cycle delay + scheduling hint
```

## Sources

- s_setprio pattern from rocBLAS (~0.8% gain)
- Empirically validated on MI355X: zero effect at single occupancy
- Zero effect at 2-wave occupancy
- Sometimes negative effect (-0.3%) at higher occupancy
- isa-bench: scalar_setprio latency kernel
