# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the AgentX checkpoint did over a session, for the CLOSE report: how many explore tests it judged, how many it
would have stopped (shadow) or stopped (enforce) and how those ended, and why the others went unjudged.

The final outcomes of the tests a shadow checkpoint would have stopped are what an operator switches to ``enforce`` on:
a KEEP among them is a gain enforcing would have lost.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

from hyperloom.common.agentx_early_stop import EarlyStopMode
from hyperloom.common.agentx_objective import AgentXObjective
from hyperloom.contracts.measurement import Validity
from hyperloom_multibranch.contracts.blackboard.records import BlackboardSnapshot, TestRecord
from hyperloom_multibranch.contracts.blackboard.vocab import COMMITTING
from hyperloom_multibranch.toolbox.agentx_checkpoint import judged_by_checkpoint, reference_p50

__all__ = ["CheckpointScope", "CheckpointTally", "checkpoint_tally"]

_KEEP = "KEEP"
_UNJUDGED = "UNJUDGED"


@dataclass(frozen=True)
class CheckpointScope:
    """An AgentX session's checkpoint: its mode and the objective it is graded on."""

    mode: EarlyStopMode
    objective: AgentXObjective


@dataclass(frozen=True)
class CheckpointTally:
    """``judged`` explore tests carry a ruling, ``trailing`` of them trailed their parent by more than the allowed drop,
    and ``outcomes`` counts how those ended (KEEP for a committed one, else the judgement's verdict, UNJUDGED for none).
    ``no_reference`` explores ran unjudged because their parent recorded no checkpoint median, ``no_ruling`` because
    their round never closed the window."""

    scope: CheckpointScope
    judged: int = 0
    trailing: int = 0
    outcomes: Mapping[str, int] = field(default_factory=dict)
    no_reference: int = 0
    no_ruling: int = 0

    def to_document(self) -> dict[str, object]:
        return {
            "mode": self.scope.mode.value,
            "objective": self.scope.objective.value,
            "judged": self.judged,
            "trailing": self.trailing,
            "trailing_outcomes": dict(sorted(self.outcomes.items())),
            "unjudged_no_reference": self.no_reference,
            "unjudged_no_ruling": self.no_ruling,
        }

    def lines(self) -> list[str]:
        """The report's lines."""
        mode, objective = self.scope.mode, self.scope.objective
        if objective is not AgentXObjective.P50_INTERACTIVITY:
            return [f"Mode {mode.value}; no test is judged under {objective.value}."]
        if mode is EarlyStopMode.OFF:
            return ["Mode off; no test was judged."]
        stop = "would have stopped" if mode is EarlyStopMode.SHADOW else "stopped"
        ended = ", ".join(f"{name} {count}" for name, count in sorted(self.outcomes.items())) or "none"
        return [
            f"Mode {mode.value}; explore tests judged: {self.judged}; the checkpoint {stop} {self.trailing}.",
            f"How the {self.trailing} it {stop} ended: {ended}.",
            f"Unjudged: {self.no_reference} with no checkpoint median on the parent, "
            f"{self.no_ruling} whose round never closed the window.",
        ]


def checkpoint_tally(snapshot: BlackboardSnapshot, scope: CheckpointScope) -> CheckpointTally:
    """The tally of ``snapshot``'s explore tests the checkpoint judges under ``scope``, measured or stopped."""
    judged = trailing = no_reference = no_ruling = 0
    outcomes: Counter[str] = Counter()
    for record in snapshot.records:
        if not _in_scope(record, scope):
            continue
        ruling = record.result.agentx_checkpoint if record.result is not None else None
        if ruling is not None:
            judged += 1
            if ruling.trails:
                trailing += 1
                outcomes[_outcome(record)] += 1
        elif _parent_reference(snapshot, record.request.parent_commit) is None:
            no_reference += 1
        else:
            no_ruling += 1
    return CheckpointTally(scope, judged, trailing, dict(outcomes), no_reference, no_ruling)


def _in_scope(record: TestRecord, scope: CheckpointScope) -> bool:
    result = record.result
    ended = result is not None and result.validity is not Validity.MEASUREMENT_FAILED
    return ended and scope.mode is not EarlyStopMode.OFF and judged_by_checkpoint(record.request, scope.objective)


def _outcome(record: TestRecord) -> str:
    if record.conclusion in COMMITTING:
        return _KEEP
    return _UNJUDGED if record.judgement is None else record.judgement.verdict.value.upper()


def _parent_reference(snapshot: BlackboardSnapshot, parent: str) -> float | None:
    """The checkpoint median ``parent`` recorded: the root's baseline, or the judgement that made the commit."""
    baseline = next((b for b in reversed(snapshot.baselines) if b.root_commit == parent), None)
    if baseline is not None:
        return reference_p50(baseline.values)
    made = next((r.judgement for r in snapshot.records if r.commit_id == parent and r.judgement is not None), None)
    return None if made is None else reference_p50(made.recorded_value)
