# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The patch-text half of the single-file boundary.

The declaration itself is read in :mod:`hyperloom.common.optimization_scope`, which the
launch CLI reads too and so has to sit below this package. Only the check that reads a
patch belongs here, where the patch parser lives.
"""

from __future__ import annotations

from pathlib import Path

from hyperloom.common.optimization_scope import check_scope_root

from ..specialists.patch_safety import parse_patch_targets


def check_patch_scope(root: Path, texts: list[str]) -> None:
    """Reject patches outside the sole model artifact before applying them."""
    allowed = check_scope_root(root)
    if allowed is None:
        return
    for text in texts:
        targets = parse_patch_targets(text).all
        if not targets or set(targets) != {allowed.name}:
            raise ValueError(f"Single-file workload permits changes only to {allowed.name}")


__all__ = ["check_patch_scope"]
