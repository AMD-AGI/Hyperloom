# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Optional operator-owned single-file scope for custom script workloads."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

from ..specialists.patch_safety import parse_patch_targets

# Schema 1 predates the configurable entry function; its workspaces are resumed and exported as they are.
_SCHEMA_1_FUNCTION = "hyperloom_optimize"


@dataclass(frozen=True)
class OptimizationScope:
    """The sole editable module and the function the user script passes its model through."""

    file: Path
    function: str


def optimization_scope(directory: str | None = None) -> OptimizationScope | None:
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
    version = scope.get("schema_version")
    function = _SCHEMA_1_FUNCTION if version == 1 else scope.get("function")
    if (
        version not in (1, 2)
        or not path.is_absolute()
        or path.suffix != ".py"
        or not str(function or "").isidentifier()
    ):
        raise ValueError("Invalid single-file optimization scope")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Optimization file must be an existing regular Python file")
    return OptimizationScope(file=path.resolve(), function=function)


def check_patch_scope(root: Path, texts: list[str]) -> None:
    """Reject patches outside the sole model artifact before applying them."""
    scope = optimization_scope()
    if scope is None:
        return
    allowed = scope.file
    if root.resolve() != allowed.parent:
        raise ValueError(f"Single-file workload permits changes only to {allowed}")
    for text in texts:
        targets = parse_patch_targets(text).all
        if not targets or set(targets) != {allowed.name}:
            raise ValueError(f"Single-file workload permits changes only to {allowed.name}")


def scope_instructions() -> str:
    """Describe the operator's artifact boundary to optimization agents."""
    scope = optimization_scope()
    if scope is None:
        return ""
    return (
        f"Single-file optimization contract: only {scope.file} may change. "
        f"It must define {scope.function}(model) returning the optimized callable. "
        "The benchmark owns inputs, weights, validation and latency measurement. "
        "Do not edit the user script, the benchmark directory or installed packages. All reusable optimization logic, "
        "including custom kernels and runtime setup, must live in this one Python module. "
        "External environment changes, overlays and auxiliary source files are not deliverables."
    )
