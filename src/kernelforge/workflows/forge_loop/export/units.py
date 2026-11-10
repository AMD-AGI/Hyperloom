# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The parts every normalized unit of a run shares, and the unit a run gets when its logs cannot be read."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CandidateOrigin(str, Enum):
    """Where the candidate a unit measured came from, recorded as its ``candidate_origin`` provenance."""

    SESSION = "session"
    LANE = "lane"
    MERGE = "merge"


@dataclass(frozen=True)
class Run:
    """What every unit of one run carries: the run's label in unit ids, its run id, and its conditions."""

    label: str
    run_id: str
    identity: dict[str, Any]

    def unit(self, name: str, seq: int, **own_provenance: Any) -> dict[str, Any]:
        """A unit's common keys; ``own_provenance`` is the unit's own provenance, such as its ``meta.json``."""

        return {
            "unit_id": f"{self.label}/{name}",
            "run_id": self.run_id,
            "seq": seq,
            "identity": self.identity,
            "provenance": {key: value for key, value in own_provenance.items() if value},
        }


def read_error(unit_id: str, message: str) -> dict[str, Any]:
    """The one unit a run, or a candidate, gets when its logs cannot be read; the mapping skips it with ``message``."""

    return {"unit_id": unit_id, "read_error": message}
