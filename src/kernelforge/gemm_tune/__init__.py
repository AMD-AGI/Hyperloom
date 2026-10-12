# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic GEMM tuning for AMD GPUs -- the ``kernelforge gemm-tune`` tree."""

#: The tuner-artifact layout version ``artifact_manifest`` stamps into every
#: produced manifest, so consumers can tell which layout they are reading. It
#: is not the distribution version: bump it when that layout changes, not when
#: kernelforge is released.
__version__ = "0.1.0"
