# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How much bring-up a session will fund, counted in evidence rather than text.

Every observation lands in exactly one of three separately bounded buckets -- a
strictly deeper ladder stage, a failure digest never seen before, or nothing new
-- so a session records at most :data:`SESSION_OBSERVATION_CEILING` of them. The
counts are folded out of the round ledger rather than held on session state, so
a crash between the observation and its charge cannot resurrect a credit.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from hyperloom.common.bringup import BootObservation, LadderStage, failure_digest
from hyperloom.orchestrator.state.round_store import (
    EVIDENCE_DIGEST,
    EVIDENCE_STAGE,
    RoundEvent,
    RoundStore,
)

#: How many times a boot can get strictly further before it has run out of
#: ladder.
LADDER_ADVANCES: int = len(LadderStage)

#: New failure identities funded at a milestone already reached.
DIGEST_CREDITS: int = 4

#: Observations that showed nothing new the session will sit through.
EVIDENCE_STALL_BUDGET: int = 5

#: The most observations any session can record before the budget is spent.
SESSION_OBSERVATION_CEILING: int = LADDER_ADVANCES + DIGEST_CREDITS + EVIDENCE_STALL_BUDGET

#: The boot got strictly further up the ladder than anything before it.
ADVANCE = "stage_advance"

#: The boot stopped where an earlier one did, for a reason never seen before.
NEW_DIGEST = "new_failure_digest"

#: The boot showed nothing an earlier one had not already shown.
STALL = "no_new_evidence"

#: The session terminal an exhausted budget forces.
STALLED_STOP_REASON = "enablement_stalled"

__all__ = [
    "ADVANCE",
    "DIGEST_CREDITS",
    "EVIDENCE_STALL_BUDGET",
    "LADDER_ADVANCES",
    "NEW_DIGEST",
    "SESSION_OBSERVATION_CEILING",
    "STALL",
    "STALLED_STOP_REASON",
    "ProgressBudget",
    "charge_of",
    "digest_of",
    "fold",
    "round_advanced",
    "session_budget",
    "stage_of",
]


def stage_of(observation: BootObservation | None) -> int:
    """Return how far up the ladder ``observation`` got, as a stage value.

    The greater of ``stage_reached`` and ``stage_failed``, which is the furthest
    point the boot demonstrably got to; ``0`` for ``None``.
    """
    if observation is None:
        return 0
    failed = observation.stage_failed
    return int(max(observation.stage_reached, failed if failed is not None else observation.stage_reached))


def digest_of(observation: BootObservation | None) -> str:
    """Return the failure digest of the wall ``observation`` hit.

    ``""`` when the boot did not stop at a wall, or there is no observation.
    """
    if observation is None or observation.stage_failed is None:
        return ""
    return failure_digest(observation)


def round_advanced(before: BootObservation | None, after: BootObservation | None) -> bool:
    """Whether the boot after a patch is worth keeping the patch for.

    Whether the session can still afford another round is a separate question,
    answered by :func:`session_budget`.

    Args:
        before: The observation the previous round recorded.
        after: The observation this round recorded.

    Returns:
        bool: ``True`` when the boot reached a deeper stage, or stopped at the
        same one for a digest ``before`` did not carry. A boot that got less far
        is never an advance.
    """
    if after is None:
        return False
    reached, previous = stage_of(after), stage_of(before)
    if reached != previous:
        return reached > previous
    digest = digest_of(after)
    return bool(digest) and digest != digest_of(before)


def charge_of(*, stage: int, digest: str, high_water: int, seen: frozenset[str]) -> str:
    """Return :data:`ADVANCE`, :data:`NEW_DIGEST` or :data:`STALL` for one
    observation, given the ``high_water`` stage and the digests already ``seen``.
    """
    if stage > high_water:
        return ADVANCE
    if digest and digest not in seen:
        return NEW_DIGEST
    return STALL


@dataclass(frozen=True)
class ProgressBudget:
    """What the ledger says the session has spent, and what is left.

    Attributes:
        observations: Observations folded in.
        stage_high_water: Furthest ladder stage anything reached.
        advances: Observations that raised the high-water mark.
        digests_spent: Distinct-failure-digest credits consumed.
        stall_spent: Evidence-stall credits consumed.
    """

    observations: int = 0
    stage_high_water: int = 0
    advances: int = 0
    digests_spent: int = 0
    stall_spent: int = 0

    @property
    def digest_credits_left(self) -> int:
        """int: Distinct-failure-digest credits still available."""
        return max(0, DIGEST_CREDITS - self.digests_spent)

    @property
    def stall_credits_left(self) -> int:
        """int: Evidence-stall credits still available."""
        return max(0, EVIDENCE_STALL_BUDGET - self.stall_spent)

    @property
    def exhausted(self) -> bool:
        """bool: Whether the session has stopped being able to fund a round."""
        return self.digest_credits_left <= 0 or self.stall_credits_left <= 0

    @property
    def reason(self) -> str:
        """str: Why the budget is spent; ``""`` while it is not."""
        if self.stall_credits_left <= 0:
            return (
                f"{self.stall_spent} bring-up rounds showed nothing an earlier "
                f"round had not already shown (high water {self.stage_high_water})"
            )
        if self.digest_credits_left <= 0:
            return (
                f"{self.digests_spent} distinct failures were funded at stages "
                f"already reached (high water {self.stage_high_water})"
            )
        return ""


def fold(events: Iterable[RoundEvent]) -> ProgressBudget:
    """Reduce the ledger's observations to a budget.

    Args:
        events: Applied observation events, oldest first; the classification of
            one depends on what came before it.
    """
    high_water = 0
    seen: set[str] = set()
    counts = {ADVANCE: 0, NEW_DIGEST: 0, STALL: 0}
    total = 0
    for event in events:
        total += 1
        stage = int(event.evidence[EVIDENCE_STAGE])
        digest = str(event.evidence[EVIDENCE_DIGEST])
        counts[charge_of(stage=stage, digest=digest, high_water=high_water, seen=frozenset(seen))] += 1
        high_water = max(high_water, stage)
        if digest:
            seen.add(digest)
    return ProgressBudget(
        observations=total,
        stage_high_water=high_water,
        advances=counts[ADVANCE],
        digests_spent=counts[NEW_DIGEST],
        stall_spent=counts[STALL],
    )


async def session_budget(store: RoundStore) -> ProgressBudget:
    """Fold the whole session's observations out of the round ledger."""
    return fold(await store.observations())
