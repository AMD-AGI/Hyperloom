# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Inference Optimizer — single-mode 4-agent runtime."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    # Single source of truth: the installed package metadata (pyproject version).
    __version__ = _pkg_version("hyperloom-inference_optimizer")
except PackageNotFoundError:  # not installed (e.g. raw source tree)
    __version__ = "0.0.0.dev0"
