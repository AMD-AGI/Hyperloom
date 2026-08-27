---
instruction: s_memrealtime
category: scalar
architecture: gfx950
tags: [s_memrealtime, SMEM, timer, FCLK, profiling]
---

# s_memrealtime

Load the GPU real-time clock (RTC) timestamp into an SGPR pair. Ticks at FCLK frequency (~1600 MHz on MI355X).

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | SMEM |
| Latency CPI | 1.68 ref-clk (median 860 ticks / 512 iters) |
| Normalized latency | ~9.8 shader cycles |
| Result | 64-bit timestamp in s[dst:dst+1] |
| Category | SMEM special |

## Hazards

None specific beyond standard SMEM latency. Requires `s_waitcnt lgkmcnt(0)` before reading the result.

## Known Bugs / Gotchas

### The ONLY working cycle counter on gfx950

`s_getreg_b32 s_dst, hwreg(29)` (HW_REG_SHADER_CYCLES, offset 29) returns **0** on gfx950. This register is documented in the ISA manual but non-functional. `s_memrealtime` is the only reliable high-resolution timer available.

```asm
; BROKEN on gfx950:
s_getreg_b32 s10, hwreg(29)     ; returns 0!

; WORKING:
s_memrealtime s[10:11]           ; returns 64-bit RTC timestamp
s_waitcnt lgkmcnt(0)             ; must wait before reading s[10:11]
```

### Ticks at FCLK, not shader clock

`s_memrealtime` ticks at the FCLK (fabric clock) frequency, not the shader clock. On MI355X:
- FCLK: ~1600 MHz
- Shader clock: ~2100 MHz (inferred)
- Ratio: ~1.31x (FCLK/shader) or ~5.814x when calibrated against s_nop overhead

To normalize raw ticks to shader cycles, divide by the `s_nop 0` calibration factor measured on the same hardware.

### isa-bench measurement methodology

The isa-bench microbenchmark suite uses `s_memrealtime` for all cycle measurements:

```asm
s_memrealtime s[0:1]          ; start timestamp
s_waitcnt lgkmcnt(0)
; ... measured code ...
s_memrealtime s[2:3]          ; end timestamp
s_waitcnt lgkmcnt(0)
; delta = s[2:3] - s[0:1]
```

The calibration kernel measures `s_nop 0` overhead (172 ticks / 1024 iters = 0.168 CPI) to establish the reference cycle cost. All other instruction measurements are normalized against this baseline.

## Common Usage Patterns

### Instruction throughput measurement
```asm
s_memrealtime s[0:1]
s_waitcnt lgkmcnt(0)
.rept 1024
  v_add_f32 v0, v0, v1    ; instruction under test
.endr
s_memrealtime s[2:3]
s_waitcnt lgkmcnt(0)
; Store s[2:3] - s[0:1] for host readout
```

### Kernel profiling (inline timestamps)
```asm
s_memrealtime s[T0:T0+1]
s_waitcnt lgkmcnt(0)
; ... section A ...
s_memrealtime s[T1:T1+1]
s_waitcnt lgkmcnt(0)
; delta = T1 - T0 = section A cost in FCLK ticks
```

## Sources

- hwreg(29) returns 0 on gfx950, s_memrealtime is the only reliable clock source
- FCLK normalization methodology for cycle count measurements
- isa-bench: special_getreg latency kernel
