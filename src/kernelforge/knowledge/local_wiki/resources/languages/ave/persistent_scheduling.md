<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Persistent work scheduling

## Applicability

Use persistent work assignment when the natural task grid leaves uneven work or repeatedly rebuilds reusable state. First establish task boundaries and output ownership.

## Mechanism

Let a bounded worker set process multiple work items. Divide the full extent, including remainders, so each valid item is handled exactly once. Derive worker count from the target and workload or use the experiment's explicit choice; do not assume a fixed count is optimal everywhere.

Initialize route metadata and resource descriptors once per work item. Join the participating threads before reusing its scratch arena, including paths that skip invalid work.

## Invariants and tradeoffs

Check empty/invalid work, remainder distribution and repeated GPU graph replay. Preserve accumulation semantics, launch ordering and resource ownership. Worker counts belong to the workload recipe, not to general scheduling knowledge.
