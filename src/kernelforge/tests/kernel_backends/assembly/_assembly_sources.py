# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Assembly sources the assembly-backend tests share."""

from __future__ import annotations


ASM = """.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
add:
    s_endpgm
.amdhsa_kernel add
.end_amdhsa_kernel
.amdgpu_metadata
.end_amdgpu_metadata
"""
