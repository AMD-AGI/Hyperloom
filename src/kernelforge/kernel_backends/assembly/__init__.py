# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AMDGPU assembly kernel backend: its optimization expertise, and the compiler-generated assembly
support Forge kernel candidates run on."""

from __future__ import annotations

from kernelforge.kernel_backends.assembly.compiler import AssemblyError, assemble

__all__ = ["AssemblyError", "assemble"]
