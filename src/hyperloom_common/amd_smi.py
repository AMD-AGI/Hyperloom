# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only ``amd-smi`` primitives shared by GPU telemetry and the power-settings checks."""

from __future__ import annotations

__all__ = ["AmdSmiError"]


class AmdSmiError(RuntimeError):
    """``amd-smi`` could not be run or its output could not be read."""
