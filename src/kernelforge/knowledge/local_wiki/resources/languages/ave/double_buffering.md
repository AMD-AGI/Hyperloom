<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Double buffering

## Applicability

Use double buffering when independent loads can overlap compute. Establish the producer/consumer mapping, scratch capacity, alignment and instruction completion rules first.

## Mechanism

For double buffering, prime the first tile, prefetch the next tile into a disjoint slot and compute on the current one. Separate prologue, steady state and drain. Keep the final unused prefetch in the lifetime analysis when an experiment requires it; skipping a prefetch and removing its wait are separate decisions.

## Invariants and tradeoffs

Draw the arena lifetime from producer through consumer to reuse, including DMA zero writes and epilogue reuse. Extra LDS/register usage, longer live ranges or unnecessary terminal memory traffic can erase the intended gain. Follow the experiment's exclusions so adjacent pipeline changes remain attributable.
