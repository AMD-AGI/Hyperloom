<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Cache policy

## Applicability

Choose a cache policy according to operand reuse and workload shape. Hardware-specific coherence and completion rules remain applicable.

## Mechanism

Select cache policy from reuse distance and workload shape. Data loads, scale loads and separate compute stages can have different reuse. Treat non-temporal hints separately from coherence scope; a useful threshold for one experiment is not a universal constant.

## Invariants and tradeoffs

Preserve explicit user selection when changing dispatch defaults. Historical thresholds belong to the workload recipe. Measure small as well as large workloads.
