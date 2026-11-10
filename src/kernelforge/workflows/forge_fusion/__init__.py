# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel fusion - autonomous source-level fusion discovery + validation."""

from kernelforge.workflows.forge_fusion.launch_folding import foldable_launches, render_foldable_launches
from kernelforge.workflows.forge_fusion.shadow_repo import reclaim_killed_shadow

__version__ = "0.1.0"

__all__ = ["foldable_launches", "reclaim_killed_shadow", "render_foldable_launches"]
