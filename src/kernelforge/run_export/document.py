# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel Forge's run logs as one document, ``{"runs": [...]}``, for the ``kernelforge-v1`` collection mapping.

A log root may hold any number of plain ``kernelforge forge-loop`` runs (``forge_loop``). Each run yields one unit
per candidate it archived, in iteration order: seq N for the candidate iteration N measured, so each lane of a fan-out
round is a unit of its own. A run, or a candidate, whose logs cannot be read yields one unit holding ``read_error``,
which the mapping skips with that reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kernelforge.run_export.forge_loop import forge_loop_runs


def normalize(roots: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
    """Return ``{"runs": [...]}``: every unit of every run under each root, roots in the order given."""

    return {"runs": [unit for root in roots for unit in forge_loop_runs(root)]}
