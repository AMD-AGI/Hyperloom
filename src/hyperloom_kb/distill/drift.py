# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a Distill's evidence stands now against the snapshot its version froze.

A version's evidence never changes. Its scope, run again over the schema's visible Experiences, selects members the
snapshot does not hold, and some members of the snapshot are no longer visible; the drift names both and what they do
to the claim. A member the snapshot holds keeps the class it gave it; a new member is judged by the version's check,
or left undetermined when the version has none. Members the author removed never come back as new.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

from hyperloom_kb.distill.explore import judged, newest_first
from hyperloom_kb.distill.model import DistillEntry, EvidenceClass
from hyperloom_kb.schema import Experience, JsonValue


class EvidenceStatus(str, Enum):
    """How a Distill's evidence moved since its snapshot: the first that applies, most pressing first."""

    #: New members contradict the claim.
    CONTRADICTED = "contradicted"
    #: Members that supported the claim are no longer visible.
    WEAKENED = "weakened"
    #: New members that no check classifies.
    UNDETERMINED = "undetermined"
    #: The evidence moved, and nothing it gained or lost tells against the claim.
    STRENGTHENED = "strengthened"
    #: The evidence is the snapshot's.
    CONSISTENT = "consistent"


NEEDS_ATTENTION = frozenset({EvidenceStatus.CONTRADICTED, EvidenceStatus.WEAKENED, EvidenceStatus.UNDETERMINED})
_COUNT_KEYS = {
    EvidenceClass.SUPPORT: "supports",
    EvidenceClass.CONTRADICT: "contradicts",
    EvidenceClass.IRRELEVANT: "irrelevant",
    None: "undetermined",
}


def class_counts(classes: Iterable[EvidenceClass | None]) -> dict[str, int]:
    """How many of ``classes`` there are of each, ``None`` counted as undetermined."""

    counted = Counter(classes)
    return {key: counted[classification] for classification, key in _COUNT_KEYS.items()}


@dataclass(frozen=True)
class Drift:
    """The members a version's evidence gained, newest first, and lost, each with its class."""

    status: EvidenceStatus
    gained: tuple[tuple[Experience, EvidenceClass | None], ...]
    lost: tuple[tuple[str, EvidenceClass], ...]
    current: dict[str, int]

    @property
    def needs_attention(self) -> bool:
        return self.status in NEEDS_ATTENTION

    def summary(self) -> dict[str, JsonValue]:
        """The status and the current counts, which a listing shows beside the snapshot's."""

        return {
            "evidence_status": self.status.value,
            "needs_attention": self.needs_attention,
            "current": dict(self.current),
        }


def _status(gained: Counter[EvidenceClass | None], lost: Counter[EvidenceClass]) -> EvidenceStatus:
    if gained[EvidenceClass.CONTRADICT]:
        return EvidenceStatus.CONTRADICTED
    if lost[EvidenceClass.SUPPORT]:
        return EvidenceStatus.WEAKENED
    if gained[None]:
        return EvidenceStatus.UNDETERMINED
    if gained or lost:
        return EvidenceStatus.STRENGTHENED
    return EvidenceStatus.CONSISTENT


def drift(entry: DistillEntry, visible: Mapping[str, Experience]) -> Drift:
    """How ``entry``'s evidence stands among ``visible``, the schema's visible Experiences by id."""

    evidence = entry.evidence
    snapshot = {member.experience_id: member.classification for member in evidence.members}
    skipped = set(snapshot) | set(evidence.removed)
    gained = tuple(
        (experience, None if evidence.check is None else judged(experience, evidence.check))
        for experience in newest_first(visible.values())
        if experience.id not in skipped and entry.scope.holds(experience) is True
    )
    lost = tuple((experience_id, cls) for experience_id, cls in snapshot.items() if experience_id not in visible)
    kept = [(visible[experience_id], cls) for experience_id, cls in snapshot.items() if experience_id in visible]
    members = kept + list(gained)
    current = {
        **class_counts(cls for _, cls in members),
        "runs": len({experience.run_id for experience, _ in members}),
    }
    status = _status(Counter(cls for _, cls in gained), Counter(cls for _, cls in lost))
    return Drift(status, gained, lost, current)


__all__ = ["NEEDS_ATTENTION", "Drift", "EvidenceStatus", "class_counts", "drift"]
