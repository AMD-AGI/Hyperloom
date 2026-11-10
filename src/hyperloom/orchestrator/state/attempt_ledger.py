# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The attempts ledger: one row shape, written through one function.

``SharedState.attempts`` is control-plane state -- the per-lever dryness
judgment reads it -- so the write must not sit behind the breakdown timeline
recorder, whose lifetime is one FRAMEWORK_AGENT phase entry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AttemptRow:
    """One attempts-ledger row, shared by every lever.

    ``lever_kind`` through ``after_tput`` are what the dryness judgment reads;
    ``adopted`` is the settlement verdict, not the executor's ``outcome``; the
    rest are the forensic record. Both levers write this one field set, so
    a reader walks them without knowing which arm produced a row.
    """

    lever_kind: str
    outcome: str
    adopted: bool
    gain_pct: float | None
    before_tput: float | None
    after_tput: float | None
    task_id: str = ""
    round_id: str = ""
    fingerprint: str = ""
    variant_name: str = ""
    candidate_id: str = ""
    specialist_task_id: str = ""
    error_class: str = ""
    provenance: str = ""

    @classmethod
    def undelivered_round(
        cls, lever_kind: str, outcome: str, specialist_task_id: str, candidate_id: str
    ) -> "AttemptRow":
        """A specialist round that delivered nothing to measure: not adopted, no gain, no throughput."""
        return cls(
            lever_kind,
            outcome,
            False,
            None,
            None,
            None,
            task_id=specialist_task_id,
            candidate_id=candidate_id,
            specialist_task_id=specialist_task_id,
        )


def record_attempt_row(state: Any, row: AttemptRow) -> None:
    """Append *row*, stamped with the macro-cycle by ``record_attempt``."""
    state.record_attempt(asdict(row))
