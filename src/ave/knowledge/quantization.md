<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Hardware quantization

## Applicability

Use native conversion instructions when scalar quantization dominates part of the pipeline. Establish the target instruction's precision and scale convention first.

## Mechanism

Replace scalar conversion sequences with hardware instructions only after matching shared-exponent derivation, rounding, saturation and packed-byte ordering. For the suite's gfx950 FP4 conversion, `cvt_scalef32_pk_fp4_f32(old, a, b, scale, word_select)` divides by `scale`; a stored E8M0 exponent therefore supplies the same power-of-two floating-point scale, not its reciprocal.

## Invariants and tradeoffs

Preserve bias/activation semantics, quantization conventions and output coverage. Compare all fixed cases, including padding and multiple shapes.
