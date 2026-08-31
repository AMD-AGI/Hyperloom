# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Autonomous iteration loop — autoresearch-inspired kernel optimization.

Inspired by Karpathy's autoresearch and AutoKernel patterns:
- Single-file modification per iteration
- Git keep/revert for experiment isolation
- Driver-owned full-suite validation
- Fixed time budget per experiment
- Overnight autonomous iteration
"""
