# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The list of rounds a test ran, in the result the runner signs.

The measurement backend names each round's directory: the report of a measuring round and the discarded warm-up round
that came before it look alike on disk, so a reader that only sees files cannot tell them apart. The runner lists them.
"""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.measure.socket import TestOutcome

__all__ = ["ACCURACY_PHASE", "ROUNDS_FILE", "WARMUP_DIR", "rounds_document", "write_rounds"]

ROUNDS_FILE = "rounds.json"
#: The phase of the round that evaluates accuracy; its entry also lists the scores, as ``AccuracyValue`` documents.
ACCURACY_PHASE = "accuracy"
#: The directory beside a measuring round's, where the old flow keeps the warm-up round it discarded.
WARMUP_DIR = "warmup_round"


def rounds_document(outcome: TestOutcome | None, files_dir: Path) -> list[dict[str, object]]:
    """One entry per round that left its directory below ``files_dir``: point, phase, directory (relative), return code
    and, for an accuracy round, the scores."""
    if outcome is None:
        return []
    root = Path(files_dir).resolve()
    entries = []
    for round_ in outcome.rounds:
        try:
            relative = Path(round_.output_dir).resolve().relative_to(root)
        except ValueError:
            continue  # a round that wrote elsewhere has nothing in the result to point at
        entry: dict[str, object] = {
            "point": round_.point,
            "phase": round_.phase,
            "dir": relative.as_posix(),
            "returncode": round_.returncode,
        }
        if round_.phase == ACCURACY_PHASE:
            entry["accuracy"] = [value.to_document() for value in round_.accuracy]
        entries.append(entry)
    return entries


def write_rounds(files_dir: Path, outcome: TestOutcome | None) -> None:
    """Write ``rounds.json`` into the output files, so the manifest lists it and the signature covers it."""
    files_dir.mkdir(parents=True, exist_ok=True)
    (files_dir / ROUNDS_FILE).write_text(json.dumps(rounds_document(outcome, files_dir)), encoding="utf-8")
