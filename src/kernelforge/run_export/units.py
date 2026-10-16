# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The parts of a normalized unit that every log layout builds the same way."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

#: The objective every unit records: the candidate's mean case speedup over the original kernel.
OBJECTIVE = "kernel_mean_case_speedup@v1"


class CandidateOrigin(str, Enum):
    """Where the candidate a unit measured came from, recorded as its ``candidate_origin`` provenance."""

    SESSION = "session"
    LANE = "lane"
    MERGE = "merge"
    PORT = "port"
    KB_WARM_START = "kb-warm-start"


@dataclass(frozen=True)
class Run:
    """What every unit of one run carries: the run's label in unit ids, its run id, conditions, and provenance."""

    label: str
    run_id: str
    identity: dict[str, Any]
    provenance: dict[str, Any]
    source_ms: float | None = None

    def unit(self, name: str, kind: str, seq: int, files: list[str], **own_provenance: str | None) -> dict[str, Any]:
        """A unit's common keys; ``own_provenance`` is the unit's own provenance, such as the commit it kept."""

        return {
            "unit_id": f"{self.label}/{name}",
            "kind": kind,
            "run_id": self.run_id,
            "seq": seq,
            "identity": self.identity,
            "provenance": {**self.provenance, **{key: value for key, value in own_provenance.items() if value}},
            "log_files": files,
        }


def log_files(paths: Iterable[Path]) -> list[str]:
    """Each path resolved, once, in the order given."""

    return list(dict.fromkeys(str(path.resolve()) for path in paths))


def read_error(unit_id: str, message: str, files: Iterable[Path]) -> dict[str, Any]:
    """The one unit a run gets when its logs cannot be read as a run; the mapping skips it with ``message``."""

    return {"unit_id": unit_id, "kind": "session", "read_error": message, "log_files": log_files(files)}
