# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Why a forge-fuse run was refused, in words meant for the operator who started it."""

from __future__ import annotations


class FuseError(Exception):
    """The run cannot start."""


class FuseUsageError(FuseError):
    """An option, or a combination of them, the run cannot honour."""


__all__ = ["FuseError", "FuseUsageError"]
