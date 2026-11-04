<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Instruction scheduling

## Applicability

Overlap independent memory, arithmetic and matrix instructions where their dependencies permit it. Establish instruction completion rules and buffer lifetimes first.

## Mechanism

Issue independent loads before the operation that consumes them; delay waits only when the corresponding dependency is understood. Prefetch epilogue operands late enough to avoid keeping them live throughout the whole reduction loop.

Scheduling directives may overlap memory, arithmetic and matrix instructions, but they do not establish memory correctness. Use a target-specific schedule only where the experiment requests it and preserve the associated waits and barriers.

## Invariants and tradeoffs

Check waits, thread participation and arena lifetimes separately when changing cache policy or removing a barrier. Longer live ranges can erase the intended gain. Follow the experiment's exclusions so adjacent pipeline changes remain attributable.
