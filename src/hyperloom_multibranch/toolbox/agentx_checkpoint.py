# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which tests of a session the AgentX checkpoint judges, against which reference, and in which mode.

Only an explore of a candidate on a specialist branch under the P50 interactivity objective is judged, against the
checkpoint median its parent commit recorded (``hyperloom.measure.checkpoint`` runs it). Under total token throughput
P50 is only a guard within a 5% band, and nothing shows the first half hour predicts a throughput verdict. A baseline,
a confirm, an integration or FINAL test, a correctness fix and a retraction measure what they measure on purpose.

The mode is the session's (``RunSettings.agentx_early_stop``), so a resumed session keeps it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from hyperloom_common.coerce import optional_positive_float
from hyperloom.common.agentx_early_stop import EarlyStopMode
from hyperloom.common.agentx_objective import AgentXObjective
from hyperloom.common.perf_metric import GRADED_INTVTY_P50_CHECKPOINT
from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.measure.checkpoint import CheckpointPlan
from hyperloom_multibranch.contracts.blackboard.interfaces import NotRecorded
from hyperloom_multibranch.contracts.blackboard.records import RecordedValue
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.rulegate.fix_case import FIX_CASE_ID
from hyperloom_multibranch.rulegate.rules import RETRACTION_ID

log = logging.getLogger(__name__)

__all__ = ["CheckpointReferences", "judged_by_checkpoint", "reference_p50"]

_MEASURED_ON_PURPOSE = frozenset({FIX_CASE_ID, RETRACTION_ID})


class RecordedValues(Protocol):
    async def recorded_value(self, commit_id: str) -> RecordedValue: ...


def judged_by_checkpoint(request: TestRequest, objective: AgentXObjective) -> bool:
    """Whether the checkpoint judges ``request`` in a session graded on ``objective``, whatever its mode."""
    return (
        objective is AgentXObjective.P50_INTERACTIVITY
        and request.kind is TestKind.TEST
        and request.purpose is Purpose.EXPLORE
        and request.branch_id != INTEGRATION_BRANCH
        and not any(attachment.id in _MEASURED_ON_PURPOSE for attachment in request.attachments)
    )


def reference_p50(recorded: Mapping[str, float]) -> float | None:
    """The checkpoint median among a commit's ``recorded`` values; ``None`` for one measured before it was recorded, or
    whose window never closed."""
    return optional_positive_float(recorded.get(GRADED_INTVTY_P50_CHECKPOINT))


@dataclass(frozen=True)
class CheckpointReferences:
    """The session's checkpoint: its ``mode``, its ``objective``, and the ``commits`` a parent's reference is read from."""

    mode: EarlyStopMode
    objective: AgentXObjective
    commits: RecordedValues

    async def plan(self, request: TestRequest) -> CheckpointPlan | None:
        """The checkpoint ``request``'s measured round is held to; ``None`` when it is not judged. A judged test whose
        parent recorded no checkpoint median runs unjudged, and the log and the CLOSE report say so."""
        if self.mode is EarlyStopMode.OFF or not judged_by_checkpoint(request, self.objective):
            return None
        parent = request.parent_commit
        try:
            reference = reference_p50((await self.commits.recorded_value(parent)).values)
        except NotRecorded as exc:
            log.warning("agentx_checkpoint: an explore on %s runs unjudged: it has no recorded value (%s)", parent, exc)
            return None
        if reference is None:
            log.warning("agentx_checkpoint: an explore on %s runs unjudged: it recorded no checkpoint median", parent)
            return None
        return CheckpointPlan(reference, self.mode)
