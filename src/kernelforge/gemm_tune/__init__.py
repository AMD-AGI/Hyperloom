# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic GEMM tuning for AMD GPUs -- the ``kernelforge gemm-tune`` tree."""

#: Not the distribution version any more: this subpackage stopped shipping as
#: its own wheel when it was folded into kernelforge. It survives as the stamp
#: ``artifact_manifest`` writes into every produced manifest, so consumers can
#: tell which tuner-artifact layout they are reading. Bump it when that layout
#: changes, not when the distribution is released.
__version__ = "0.1.0"
