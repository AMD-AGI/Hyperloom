<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Register tiling

## Applicability

Use register tiling when a matrix kernel reloads the same operands across output fragments. Establish the target instruction's operand layout first; tile sizes and lane ownership depend on that contract.

## Mechanism

Put the reduction-tile loop outside the output-fragment loops. Load each input/weight fragment once per reduction tile, reuse it across a persistent accumulator tile and preserve its lane-to-output mapping. Check register pressure when increasing the tile.

## Invariants and tradeoffs

Preserve bias/activation semantics, quantization conventions and output coverage. Compare all fixed cases, including padding and multiple shapes. A larger register tile may lose occupancy. Explicit experiment requirements determine which mechanisms belong to the current step.
