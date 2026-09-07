# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""forge-rewrite: rewrite a kernel from another language into FlyDSL, then reuse"""

from kernelforge.rewrite_by_flydsl.runner import run_rewrite

__all__ = ["run_rewrite"]
