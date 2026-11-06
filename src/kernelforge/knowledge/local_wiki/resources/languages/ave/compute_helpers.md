<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Reusable compute helpers

## Applicability

Separate compute from workload routing when a step requires a reusable matrix pipeline or explicit input/output callbacks.

## Mechanism

Separate reusable compute from workload routing: compute helpers own matrix arithmetic and their documented scratch lifetime; wrappers own dispatch, routing and output placement. A refactor is useful for a stated dependency or reuse goal, not evidence of a speedup by itself.

## Invariants and tradeoffs

Update every launch site when specializing away runtime dimensions. Preserve bias/activation semantics, quantization conventions and output coverage. Explicit experiment requirements determine which mechanisms belong to the current step.
