# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic GEMM tuning for AMD GPUs -- the ``kernelforge gemm-tune`` tree."""

from kernelforge.gemm_tune.evidence import (
    MOE_SHAPE_FIELDS,
    moe_dispatch_keys,
    moe_tuple,
    moe_untuned_csv_text,
    parse_log_file,
)
from kernelforge.gemm_tune.model_analyzer import analyze_model
from kernelforge.gemm_tune.router import RoutingEvidence, RoutingTarget, select_tuners

#: The tuner-artifact layout version ``artifact_manifest`` stamps into every
#: produced manifest, so consumers can tell which layout they are reading. It
#: is not the distribution version: bump it when that layout changes, not when
#: kernelforge is released.
__version__ = "0.1.0"

__all__ = [
    "MOE_SHAPE_FIELDS",
    "RoutingEvidence",
    "RoutingTarget",
    "analyze_model",
    "moe_dispatch_keys",
    "moe_tuple",
    "moe_untuned_csv_text",
    "parse_log_file",
    "select_tuners",
]
