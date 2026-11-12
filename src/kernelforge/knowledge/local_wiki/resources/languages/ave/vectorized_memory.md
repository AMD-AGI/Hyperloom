<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Vectorized memory access

## Applicability

Use coalesced traffic and aligned vectors when scale loads or output stores repeat scalar element traffic. Establish the producer/consumer mapping and alignment first.

## Mechanism

Coalesce scale traffic, pack output stores and reuse aligned vectors instead of scalar element traffic. Verify byte order, alignment and exact output coverage.

## Invariants and tradeoffs

Preserve quantization conventions and output coverage. Test invalid/padded rows and all configured tile shapes. Benchmark each workload case.
