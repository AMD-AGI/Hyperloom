# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel Forge's run logs as one document, ``{"runs": [...]}``, for the ``kernelforge-v1`` collection mapping.

A log root may hold any number of runs in two layouts: an AgentKernelArena driver.log session (``arena``) and a
plain ``kernelforge forge-loop`` run (``forge_loop``). Each run yields one unit per KernelForge step that measured a
candidate, in the order the logs show them: seq 0 for forge-rewrite's PORT step when the run had one, and seq N for
forge-loop iteration N. A run whose logs cannot be read as a run yields one unit holding ``read_error``, which the
mapping skips with that reason. A file one layout read is never read again as part of another.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kernelforge.run_export.arena import session_dirs, session_runs
from kernelforge.run_export.files import inventory
from kernelforge.run_export.forge_loop import forge_loop_runs


def _root_runs(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    records = inventory(root)
    units: list[dict[str, Any]] = []
    claimed: set[Path] = set()
    for session_dir in session_dirs(records):
        found = session_runs(session_dir, records)
        units += found.units
        claimed |= found.claimed
    return units + forge_loop_runs(records, claimed, root)


def normalize(roots: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
    """Return ``{"runs": [...]}``: every unit of every run under each root, roots in the order given."""

    runs = [unit for root in roots for unit in _root_runs(root)]
    for run in runs:
        origin = run.pop("change_kind", None)
        if origin:
            run["provenance"] = {**run.get("provenance", {}), "candidate_origin": origin.value}
    return {"runs": runs}
