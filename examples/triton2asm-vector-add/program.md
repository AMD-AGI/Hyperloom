<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Optimize the captured vector-add instructions

Optimize only the prepared `kernel.s`. Preserve the original masked FP32 addition,
signature, launch dimensions, shared-memory contract and exact numerical results.
Leave the source kernel, host launcher, manifest, driver and numerical contract
unchanged. Compilation and loading must precede graph capture and timing. The
initial Triton implementation remains selected unless an ASM candidate passes
the correctness and performance gates.
