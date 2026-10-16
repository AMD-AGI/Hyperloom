# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel Forge's run logs as one document, ``{"runs": [...]}``, for the ``kernelforge-v1`` collection mapping.

A log root may hold any number of plain ``kernelforge forge-loop`` runs (``forge_loop``). Each run yields one unit
per KernelForge step that measured a candidate, in the order the logs show them: seq N for forge-loop iteration N. A
run whose logs cannot be read as a run yields one unit holding ``read_error``, which the mapping skips with that reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kernelforge.run_export.files import inventory
from kernelforge.run_export.forge_loop import forge_loop_runs


def _root_runs(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    return forge_loop_runs(inventory(root), set(), root)


def normalize(roots: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
    """Return ``{"runs": [...]}``: every unit of every run under each root, roots in the order given."""

    runs = [unit for root in roots for unit in _root_runs(root)]
    for run in runs:
        origin = run.pop("change_kind", None)
        if origin:
            run["provenance"] = {**run.get("provenance", {}), "candidate_origin": origin.value}
    return {"runs": runs}
