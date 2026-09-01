# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""forge-rewrite: rewrite a kernel from another language into FlyDSL, then reuse
forge-loop to optimize it.

This is a thin front-end layer on top of forge-loop. It ports a source kernel in
any language ``protocol.capabilities()`` advertises into an equivalent FlyDSL
kernel (correctness-only PORT phase), then
delegates optimization to forge-loop unchanged. Measurement is operator-agnostic:
a conforming task driver is reused, while a missing or invalid one can be authored
by the rewrite-specific preparation stage. The driver uses the ORIGINAL kernel as
a live oracle + baseline, so the layer is not limited to one operator family.
"""

from kernelforge.rewrite_by_flydsl.runner import run_rewrite

__all__ = ["run_rewrite"]
