# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether every process of a test's server ran the farm's copy of each package the test rebuilt.

The farm is first on the server's ``PYTHONPATH`` and :func:`~hyperloom.measure.farm.verify_farm` checks that a fresh
interpreter finds it there, but that is the start of the server only: a worker starts from its parent's ``sys.path``,
which the framework may have changed by then (aiter puts its meta directory first). A worker that imported the image's
copy runs none of the patch, and the test would report the image's numbers as the candidate's.

Each process notes where it imported every rebuilt package from (:mod:`.import_origin_hook`, installed by the farm). A
test where one of them came from anywhere but the farm ends as an infrastructure failure, ``inert_patch``, naming the file
that was loaded: it is a ``measurement_failed`` that no negative result is written for, whatever the run itself ended
with. A test that ended well although no process imported a rebuilt package at all, or whose processes noted nothing,
ends the same way: the patch never ran, or whether it ran cannot be told.

The notes are read after the run rather than polled at readiness: the processes write them when they import, which is
before the server answers, and the file read is the same.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.contracts.measurement import FailureClass
from hyperloom.measure.farm import Farm
from hyperloom.measure.runner.import_origin_hook import ORIGIN_FIELD, PACKAGE_FIELD
from hyperloom.measure.runner.jit_outcome import failed_outcome

__all__ = ["INERT_PATCH", "ImportNote", "read_import_notes", "refine_imports"]

INERT_PATCH = "inert_patch"
_SHOWN = 3


@dataclass(frozen=True)
class ImportNote:
    """One process's import of one rebuilt package: the process, the package and the file it was executed from."""

    pid: int
    package: str
    origin: str


def read_import_notes(farm: Farm) -> list[ImportNote]:
    """The notes the server's processes left; an unreadable note counts as an import from an unknown place."""
    notes = []
    for path in sorted(farm.import_notes.glob("*.json")) if farm.import_notes.is_dir() else []:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            notes.append(ImportNote(int(document["pid"]), str(document[PACKAGE_FIELD]), str(document[ORIGIN_FIELD])))
        except (OSError, ValueError, KeyError, TypeError):
            notes.append(ImportNote(0, path.stem.partition("-")[2], f"an unreadable note {path.name}"))
    return notes


def _misplaced(farm: Farm, notes: list[ImportNote]) -> str | None:
    wrong = [
        n for n in notes if n.package in farm.packages and not Path(n.origin).is_relative_to(farm.site / n.package)
    ]
    if not wrong:
        return None
    shown = "; ".join(f"process {n.pid} imported {n.package} from {n.origin}" for n in wrong[:_SHOWN])
    more = f" and {len(wrong) - _SHOWN} more" if len(wrong) > _SHOWN else ""
    return f"{INERT_PATCH}: {shown}{more}, not from the farm at {farm.site}, so the patch did not run there"


def _never_imported(farm: Farm, notes: list[ImportNote]) -> str | None:
    missing = [package for package in farm.packages if all(n.package != package for n in notes)]
    if not missing:
        return None
    return (
        f"{INERT_PATCH}: no process of the server noted importing {', '.join(missing)} from the farm at {farm.site}, so "
        "the patch to it never ran, or whether it ran cannot be told"
    )


def refine_imports(document: dict[str, Any], farm: Farm | None) -> dict[str, Any]:
    """``document`` (the outcome the runner reports) as an ``inert_patch`` infrastructure failure when the patch did not
    run in the server, or as it was. A package no process imported counts only for a test that otherwise ended well: a
    server that failed may have failed before it imported anything."""
    if farm is None or not farm.packages:
        return document
    notes = read_import_notes(farm)
    reason = _misplaced(farm, notes)
    if reason is None and document["failure_class"] is None and document["returncode"] == 0:
        reason = _never_imported(farm, notes)
    return document if reason is None else failed_outcome(document, FailureClass.INFRA, reason)
