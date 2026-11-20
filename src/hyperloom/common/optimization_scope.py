# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Optional operator-owned single-file scope for custom script workloads."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path


log = logging.getLogger(__name__)

SUPPORTED_SCHEMA_VERSIONS = (1,)


# The entry function's name before a declaration could choose one.
_SCHEMA_1_FUNCTION = "hyperloom_optimize"


@dataclass(frozen=True)
class OptimizationScope:
    """The sole editable module and the function the user script passes its model through."""

    file: Path
    function: str


def _read_scope(directory: str | None = None) -> dict | None:
    """Load and validate the declaration, or return None when none is declared.

    Every failure surfaces as ``ValueError``. The declaration sits on an
    operator-supplied path that is often a network mount, so reading it can raise
    ``OSError``, and a hand-written file can be any shape at all; callers on the
    dispatch and promotion paths should not have to enumerate those.
    """
    if directory is None:
        directory = os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "")
    if not directory:
        return None
    declaration = Path(directory) / "optimization_scope.json"
    try:
        if not declaration.exists():
            return None
        scope = json.loads(declaration.read_text())
        path = Path(scope["file"])
        if (
            scope.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS
            or not path.is_absolute()
            or path.suffix != ".py"
            or not _scope_function(scope).isidentifier()
        ):
            raise ValueError("Invalid single-file optimization scope")
        # Inside the try with the rest: the module often sits on a network mount, so an
        # unreadable parent raises PermissionError here. Callers on the dispatch and
        # promotion paths catch ValueError only, and one escaping OSError takes the tick
        # down instead of costing the gate a cycle.
        if path.is_symlink() or not path.is_file():
            raise ValueError("Optimization file must be an existing regular Python file")
    except (OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable single-file optimization scope: {exc}") from exc
    return scope


def _scope_function(scope: dict) -> str:
    """The entry function the user script calls, which a declaration need not name."""
    return str(scope.get("function") or _SCHEMA_1_FUNCTION)


def optimization_scope(directory: str | None = None) -> OptimizationScope | None:
    """Read the write boundary from the frozen benchmark directory, if declared."""
    scope = _read_scope(directory)
    if scope is None:
        return None
    return OptimizationScope(file=Path(scope["file"]).resolve(), function=_scope_function(scope))


def optimization_file(directory: str | None = None) -> Path | None:
    """The sole editable module, when one is declared."""
    scope = optimization_scope(directory)
    return None if scope is None else scope.file


def check_scope_root(root: Path) -> Path | None:
    """Reject an apply root outside the sole model artifact; return the file it permits, or ``None``.

    The launch checks this before a session exists, so a campaign pointed at the wrong
    tree costs nothing. Patch texts are checked against the returned file separately, by
    the orchestrator that knows how to read one.
    """
    scope = optimization_scope()
    if scope is None:
        return None
    allowed = scope.file
    if root.resolve() != allowed.parent:
        raise ValueError(f"Single-file workload permits changes only to {allowed}")
    return allowed


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
