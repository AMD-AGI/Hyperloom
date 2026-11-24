# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a single-file workload may change: the patch-text and non-diff-artifact halves.

The declaration itself is read in :mod:`hyperloom.common.optimization_scope`, which the
launch CLI reads too and so has to sit below this package. Only the checks that read a
deliverable belong here, where the patch parser lives.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hyperloom.common.optimization_scope import check_scope_root, optimization_scope

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


def artifacts_refused_by_scope(targets: Sequence[str]) -> list[str]:
    """Name the non-diff artifacts a single-file workload cannot deliver.

    Every one of them, whatever it targets -- so this filters by the declaration, not by
    the target. The export ships the declared module alone, so an installed artifact
    joins the measured configuration without reaching the operator, and that is as true
    of one landing beside the module as of one in another tree. A declaration that cannot
    be read counts as declared: the apply path refuses every patch under a broken one, so
    admitting artifacts would leave them the only way to change the tree.
    """
    if not targets:
        return []
    try:
        declared = optimization_scope() is not None
    except ValueError:
        declared = True
    return list(targets) if declared else []


__all__ = ["artifacts_refused_by_scope", "check_patch_scope"]
