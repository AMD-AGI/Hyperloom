---
instruction: v_cvt_scalef32_pk_f16_fp8
category: vector
architecture: gfx950
tags: [v_cvt_scalef32_pk_f16_fp8, FP8, FP16, E8M0, dequantization]
---

# v_cvt_scalef32_pk_f16_fp8

Convert FP8 (E4M3 or E5M2) to a pair of FP16 values with an E8M0 scale factor application.

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | VOP3 |
| Throughput CPI | 0.238 ref-clk (median 244 ticks / 1024 iters) |
| Normalized throughput | ~1.4 shader cycles |
| Category | VALU conversion |

## Hazards

Standard VALU forwarding latency (5 cycles). No special hazards.

## Known Bugs / Gotchas

### E8M0 scale encoding

The E8M0 scale factor is a single byte interpreted as a biased exponent: `scale_float = 2^(byte - 127)`. To apply it, the byte is loaded and left-shifted by 23 to form an IEEE 754 FP32 bit pattern:

```
E8M0 byte value -> FP32 scale: (byte << 23)
Example: byte=127 -> scale=1.0, byte=128 -> scale=2.0, byte=126 -> scale=0.5
```

This instruction combines the dequantization (FP8 to FP16) and scale application into a single operation, avoiding the two-step `v_cvt_f32_fp8 + v_mul_f32` sequence.

### Used in FP8 inference kernels

In DeepSeek-V4 and similar FP8 inference pipelines, this instruction dequantizes FP8 weight tiles with per-block E8M0 scales:

```asm
v_cvt_scalef32_pk_f16_fp8 v_out, v_fp8_data, v_scale
; Produces 2 FP16 values from 2 FP8 values with scale applied
```

### fnuz vs OCP format consideration

This instruction operates on native hardware FP8 format. When data was created as fnuz (bias=8) but hardware interprets as OCP (bias=7), each value reads as 2x its intended magnitude. A correction factor may be needed depending on the data source.

## Common Usage Patterns

### FP8 weight dequantization
```asm
; Load FP8 weights and E8M0 scale:
buffer_load_dword v_fp8, ...        ; 4 FP8 values packed in 32 bits
buffer_load_byte v_scale, ...       ; E8M0 scale byte
; Dequant with scale:
v_cvt_scalef32_pk_f16_fp8 v_out, v_fp8, v_scale
; v_out contains 2 FP16 values with scale applied
```

### MFMA source preparation
```asm
; Convert FP8 tiles to FP16 for BF16 MFMA consumption:
v_cvt_scalef32_pk_f16_fp8 v_tile0, v_fp8_0, v_scale
v_cvt_scalef32_pk_f16_fp8 v_tile1, v_fp8_1, v_scale
; Feed to v_mfma_f32_*_bf16 or similar
```

## Sources

- FP8 dequant instruction identification
- gfx950-reference.md: FP8/FP4 MFMA and conversion instruction table
- isa-bench: valu_cvt_scalef32_f16_fp8 throughput kernel
