"""Monotonic update rules for one deterministic Experience id."""

from __future__ import annotations

from enum import Enum

from hyperloom_kb.schema import Experience, ExperienceStatus


class ExperienceConflictError(ValueError):
    """Raised when a write would mutate immutable Experience facts."""


class TransitionKind(str, Enum):
    NOOP = "noop"
    DECIDED = "decided"
    COMPLETED = "completed"


_BEGIN_FIELDS = (
    "id",
    "schema_ref",
    "schema_version",
    "run_id",
    "seq",
    "created_at",
    "parent_id",
    "supersedes",
    "identity",
    "objective",
    "baseline_identity",
    "baseline_value",
    "preconditions",
    "provenance",
)

_DECISION_FIELDS = (
    "reasoning",
    "alternatives",
    "change",
    "rendered_refs",
)


def classify_transition(previous: Experience, incoming: Experience) -> TransitionKind:
    """Validate a same-id write and classify its monotonic state change.

    Replaying an identical snapshot is a no-op. Begin-time facts never change.
    Once decision-time facts are present, they never change. Completion is the
    final transition and an existing complete record is immutable.
    """

    if previous.id != incoming.id:
        raise ExperienceConflictError("cannot transition between different Experience ids")
    if previous == incoming:
        return TransitionKind.NOOP
    if previous.status is ExperienceStatus.COMPLETE:
        raise ExperienceConflictError("a complete Experience is immutable")

    changed_begin = [name for name in _BEGIN_FIELDS if getattr(previous, name) != getattr(incoming, name)]
    if changed_begin:
        raise ExperienceConflictError(f"begin-time fields cannot change: {', '.join(changed_begin)}")

    if previous.change is not None:
        changed_decision = [name for name in _DECISION_FIELDS if getattr(previous, name) != getattr(incoming, name)]
        if changed_decision:
            raise ExperienceConflictError(f"decision-time fields cannot change: {', '.join(changed_decision)}")

    if incoming.status is ExperienceStatus.COMPLETE:
        return TransitionKind.COMPLETED
    if previous.change is None and incoming.change is not None:
        return TransitionKind.DECIDED

    raise ExperienceConflictError("in_progress write is not a valid monotonic transition")


def validate_begin_replay(existing: Experience, replay: Experience) -> None:
    """Allow a repeated ``begin`` to resume an equal or later same-id snapshot."""

    if existing.id != replay.id:
        raise ExperienceConflictError("begin replay resolved to a different Experience id")
    changed_begin = [name for name in _BEGIN_FIELDS if getattr(existing, name) != getattr(replay, name)]
    if changed_begin:
        raise ExperienceConflictError(f"begin replay conflicts with existing fields: {', '.join(changed_begin)}")


def validate_supersession(previous: Experience, correction: Experience) -> None:
    """Validate that ``correction`` is a distinct record correcting ``previous``."""

    if correction.id == previous.id:
        raise ExperienceConflictError("a correction must use a new sequence and id")
    if correction.supersedes != previous.id:
        raise ExperienceConflictError("a correction must name the previous Experience in supersedes")
    if correction.created_at < previous.created_at:
        raise ExperienceConflictError("a correction cannot predate the record it supersedes")


__all__ = [
    "ExperienceConflictError",
    "TransitionKind",
    "classify_transition",
    "validate_begin_replay",
    "validate_supersession",
]
