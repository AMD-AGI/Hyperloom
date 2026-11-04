<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Scale packing

## Applicability

Use native packing when scalar scale extraction dominates part of the pipeline. Establish the target instruction's operand layout, precision and scale convention first; tile sizes and selectors depend on that contract.

## Mechanism

Align value and scale storage with the matrix instruction's consumption order. Pack scale selectors into aligned words and use instruction selectors instead of repeated byte extraction where the binding supports it. Change producers and consumers together while preserving the public tensor contract.

## Invariants and tradeoffs

Update every launch site when specializing away runtime dimensions. Preserve quantization conventions and output coverage. Compare all fixed cases, including padding and multiple shapes. Measure format changes before assuming reduced instruction counts improve latency.
