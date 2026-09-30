<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

Optimize only the compiler-emitted `kernel.s` after host preparation. Preserve
the FP32 vector addition, masked tail, ABI and original launch configuration.
Use `driver.py --mode test` and `driver.py --mode bench`. The source, manifest,
driver and input domain are fixed. Profile first; keep the original when no
repeatable instruction-level benefit is found. Do not change the FlyDSL source
or replace it with a handwritten implementation.
