<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# LDS layouts

## Applicability

Use an LDS layout transformation when producer and consumer access patterns cause bank conflicts. Establish the producer/consumer mapping, scratch capacity and alignment first.

## Mechanism

XOR-swizzle an LDS layout when producer and consumer access patterns cause bank conflicts. Define one mapping and apply it on both sides; changing just the read or write side changes the data. Preserve slot separation and allocation bounds when increasing tile sizes.

## Invariants and tradeoffs

Test invalid/padded rows, all configured tile shapes and repeated invocations. Benchmark each workload case. Follow the experiment's exclusions so adjacent pipeline changes remain attributable.
