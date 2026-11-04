<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Metadata caching

## Applicability

Use metadata caching when loop-invariant routes or descriptors are repeatedly rebuilt. Establish the work item over which each value remains valid.

## Mechanism

Hoist stable descriptors, routes and row offsets to their natural lifetime, such as one output tile or persistent work item. Preserve validity when adding per-iteration offsets, and do not retain metadata across work items that change its meaning.

## Invariants and tradeoffs

Preserve resource ownership. Extra LDS/register usage and longer live ranges can erase the intended gain. Follow the experiment's exclusions so adjacent pipeline changes remain attributable.
