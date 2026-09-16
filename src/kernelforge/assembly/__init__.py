# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compiler-generated AMDGPU assembly support for Forge kernel candidates."""

from __future__ import annotations

from .compiler import AssemblyError, assemble

__all__ = ["AssemblyError", "assemble"]
