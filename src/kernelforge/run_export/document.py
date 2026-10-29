# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel Forge's run logs as one document, ``{"runs": [...], "raw_logs": [...]}``, for the ``kernelforge-v1`` mapping.

A log root may hold any number of plain ``kernelforge forge-loop`` runs (``forge_loop``). Each run yields one unit
per candidate it archived, in iteration order: seq N for the candidate iteration N measured, so each lane of a fan-out
round is a unit of its own. A run, or a candidate, whose logs cannot be read yields one unit holding ``read_error``,
which the mapping skips with that reason. ``raw_logs`` names the raw logs every Experience links, by paths relative
to the document's directory: only ``NO_RAW_LOGS``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kernelforge.run_export.forge_loop import forge_loop_runs

#: The empty file export-runs writes beside the document, every Experience's one raw log: the global KB refuses a
#: complete Experience without raw logs, and an export collects none.
NO_RAW_LOGS = "raw-logs-not-collected.log"


def normalize(roots: Sequence[Path]) -> dict[str, list[Any]]:
    """Return ``{"runs": [...], "raw_logs": [NO_RAW_LOGS]}``: every unit of every run under each root, roots in the
    order given, and the raw log every Experience links."""

    return {"runs": [unit for root in roots for unit in forge_loop_runs(root)], "raw_logs": [NO_RAW_LOGS]}


def write_document(document: dict[str, list[Any]], output: Path) -> None:
    """Write ``document`` to ``output`` and the empty ``NO_RAW_LOGS`` beside it, truncating one already there."""
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / NO_RAW_LOGS).write_bytes(b"")
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
