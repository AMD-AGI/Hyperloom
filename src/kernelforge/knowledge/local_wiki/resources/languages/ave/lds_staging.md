<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# LDS staging

## Applicability

Use LDS staging when multiple consumers reuse input data. Establish the producer/consumer mapping, scratch capacity, alignment and instruction completion rules first. gfx950 raw-buffer operations additionally require the invariants in the gfx950 knowledge document.

## Mechanism

Stage each shared input tile once, complete its writes, then release consumers with the required barrier. When introducing staging, start with a single buffer whose reuse point is explicit; the workload step determines the buffering for the current transformation.

## Invariants and tradeoffs

Draw the arena lifetime from producer through consumer to reuse, including DMA zero writes and epilogue reuse. Test invalid/padded rows, all configured tile shapes and repeated invocations. Extra LDS/register usage can erase the intended gain.
