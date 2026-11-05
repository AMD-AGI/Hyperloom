<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Kernel fusion

## Applicability

Consider kernel fusion when consecutive operations consume compatible data and launch/intermediate-memory overhead matters. First establish the supported shapes of any fused implementation.

## Mechanism

Fuse adjacent preprocessing operations only where the fused API preserves both data and metadata layouts. Keep the existing path for unsupported shapes, and preserve explicit user selection when changing dispatch defaults.

## Invariants and tradeoffs

Preserve accumulation semantics, launch ordering and resource ownership. Test both sides of a fusion dispatch threshold and measure small as well as large workloads. Historical thresholds belong to the workload recipe.
