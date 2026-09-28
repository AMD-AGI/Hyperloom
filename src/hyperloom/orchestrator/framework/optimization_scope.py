# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Optional operator-owned single-file scope for custom script workloads."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..specialists.patch_safety import parse_patch_targets


def optimization_file(directory: str | None = None) -> Path | None:
    """Read the write boundary from the frozen benchmark directory, if declared."""
    if directory is None:
        directory = os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "")
    if not directory:
        return None
    declaration = Path(directory) / "optimization_scope.json"
    if not declaration.exists():
        return None
    scope = json.loads(declaration.read_text())
    path = Path(scope["file"])
    if scope.get("schema_version") != 1 or not path.is_absolute() or path.suffix != ".py":
        raise ValueError("Invalid single-file optimization scope")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Optimization file must be an existing regular Python file")
    return path.resolve()


def check_patch_scope(root: Path, texts: list[str]) -> None:
    """Reject patches outside the sole model artifact before applying them."""
    allowed = optimization_file()
    if allowed is None:
        return
    if root.resolve() != allowed.parent:
        raise ValueError(f"Single-file workload permits changes only to {allowed}")
    for text in texts:
        targets = parse_patch_targets(text).all
        if not targets or set(targets) != {allowed.name}:
            raise ValueError(f"Single-file workload permits changes only to {allowed.name}")


def scope_instructions() -> str:
    """Describe the operator's artifact boundary to optimization agents."""
    allowed = optimization_file()
    if allowed is None:
        return ""
    return (
        f"Single-file optimization contract: only {allowed} may change. "
        "It must define hyperloom_optimize(model) returning the optimized callable. "
        "The user script owns inputs, weights, validation and latency measurement. "
        "Do not edit that script or installed packages. All reusable optimization logic, "
        "including custom kernels and runtime setup, must live in this one Python module. "
        "External environment changes, overlays and auxiliary source files are not deliverables."
    )
