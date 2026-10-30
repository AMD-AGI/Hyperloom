# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a session learnt, as the few lists every knowledge record and briefing carries.

Each finished test yields at most one lesson:

* ``worked``: a committed or merged improvement. An AgentX session's says its outputs were not checked: that gate reads
  speed and the request error rate, so a change to what the model says is kept all the same (#162).
* ``failed``: a regression or a flat result, which no one should repeat on the same state.
* ``pitfall``: a candidate that failed to run, or ran faster but failed the accuracy check or was blocked.

Hypotheses the specialists confirmed or falsified are the session's ``learned`` list; ``gaps`` say how far a recipe
is from the roofline bound of each metric PRELUDE recorded one for.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

from hyperloom_multibranch.contracts.blackboard.peers import BranchLogType, HypothesisStatus, Validity, Verdict
from hyperloom_multibranch.contracts.blackboard.records import BlackboardSnapshot, SessionProfile, TestRecord
from hyperloom_multibranch.contracts.blackboard.vocab import COMMITTING, Conclusion

MAX_ITEMS = 20
MAX_ITEM_CHARS = 300
UNCHECKED_OUTPUTS = ", outputs not checked"
_WORKED = COMMITTING | {Conclusion.MERGED}
_SETTLED = frozenset({HypothesisStatus.CONFIRMED, HypothesisStatus.FALSIFIED})
_NO_LESSON = frozenset({Verdict.IMPROVED, Verdict.NEEDS_CONFIRM, Verdict.INCONCLUSIVE})


class LessonKind(str, Enum):
    WORKED = "worked"
    FAILED = "failed"
    PITFALL = "pitfall"


@dataclass(frozen=True)
class Lesson:
    """What one test taught: ``direction`` is what it tried, ``parent_commit`` the state it tried it on."""

    kind: LessonKind
    request_id: str
    direction: str
    parent_commit: str
    outcome: str
    gain_pct: float | None = None

    @property
    def text(self) -> str:
        return f"{self.direction} on {self.parent_commit}: {self.outcome}"


@dataclass(frozen=True)
class SessionLessons:
    worked: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    pitfalls: tuple[str, ...] = ()
    learned: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()


def direction(record: TestRecord, snapshot: BlackboardSnapshot) -> str:
    """What a test tried: its intent's target and mechanism, the commits a merge stacks, or its purpose and kind."""

    found = snapshot.intents.get(record.request.intent_fingerprint)
    if found is not None:
        return f"{found.target} / {found.mechanism}"
    merged = [
        commit.commit_id
        for commit in snapshot.committed
        if commit.last_merge is not None and commit.last_merge.request_id == record.request_id
    ]
    if merged:
        return f"merge of {', '.join(merged)}"
    request = record.request
    return f"{request.purpose.value} {request.kind.value} {record.request_id}"


def lesson_of(record: TestRecord, snapshot: BlackboardSnapshot) -> Lesson | None:
    """The lesson of one finished test; ``None`` while it is in flight or unsettled, or when it improved without a
    commit."""

    what = direction(record, snapshot)

    def lesson(kind: LessonKind, outcome: str, gain_pct: float | None = None) -> Lesson:
        return Lesson(kind, record.request_id, what, record.request.parent_commit, outcome, gain_pct)

    if not record.measured:
        return None
    if record.result is not None and record.result.validity is Validity.CANDIDATE_FAILED:
        return lesson(LessonKind.PITFALL, f"the candidate failed ({record.result.failure_reason or 'no reason'})")
    judged = record.judgement
    if judged is None or record.conclusion is None:
        return None
    gain = f"{judged.gain_pct:+.2f}%"
    if record.conclusion is Conclusion.ACCURACY_FAILED:
        return lesson(LessonKind.PITFALL, f"{gain} but failed the accuracy check", judged.gain_pct)
    if record.conclusion in _WORKED:
        kept = f"{gain}, {record.conclusion.value} as {record.commit_id}{_unchecked(snapshot.profile)}"
        return lesson(LessonKind.WORKED, kept, judged.gain_pct)
    if judged.verdict is Verdict.BLOCKED:
        held_back_by = ", ".join(judged.refused_by) or "the accuracy check"
        return lesson(LessonKind.PITFALL, f"{gain} but held back by {held_back_by}", judged.gain_pct)
    if judged.verdict in _NO_LESSON:
        return None
    return lesson(LessonKind.FAILED, f"{judged.verdict.value} at {gain}", judged.gain_pct)


def _unchecked(profile: SessionProfile | None) -> str:
    """What an AgentX session's worked lesson adds; an AgentX trace replay has no input length."""

    return UNCHECKED_OUTPUTS if profile is not None and profile.isl is None else ""


def per_test_lessons(snapshot: BlackboardSnapshot) -> tuple[Lesson, ...]:
    """The lesson of every finished test, in the order the tests were requested."""

    found = (lesson_of(record, snapshot) for record in snapshot.records)
    return tuple(lesson for lesson in found if lesson is not None)


def hypotheses(snapshot: BlackboardSnapshot) -> tuple[str, ...]:
    """Every hypothesis a specialist confirmed or falsified, as ``<status>: <hypothesis>``."""

    return tuple(
        f"{record.entry.hypothesis_status.value}: {record.entry.content}"
        for log in snapshot.branch_logs.values()
        for record in log
        if record.entry.kind is BranchLogType.HYPOTHESIS and record.entry.hypothesis_status in _SETTLED
    )


def roofline_gaps(profile: SessionProfile | None, values: Mapping[str, float]) -> tuple[str, ...]:
    """How far ``values`` are from the roofline bound of each metric that has one."""

    if profile is None:
        return ()
    return tuple(
        f"{metric}: {values[metric]:.1f} is {values[metric] / ceiling:.0%} of the roofline bound {ceiling:.1f}"
        for metric, ceiling in sorted(profile.roofline_ceiling.items())
        if metric in values and ceiling > 0
    )


def session_lessons(snapshot: BlackboardSnapshot, final_commit: str | None = None) -> SessionLessons:
    """The session's lists: improvements best first, failures worst first, each list capped at :data:`MAX_ITEMS`."""

    lessons = per_test_lessons(snapshot)

    def of(kind: LessonKind, best_first: bool = False) -> tuple[str, ...]:
        chosen = [lesson for lesson in lessons if lesson.kind is kind]
        chosen.sort(key=lambda lesson: lesson.gain_pct or 0.0, reverse=best_first)
        return listed(lesson.text for lesson in chosen)

    final = snapshot.value_of(final_commit) if final_commit else {}
    return SessionLessons(
        worked=of(LessonKind.WORKED, best_first=True),
        failed=of(LessonKind.FAILED),
        pitfalls=listed(lesson.text for lesson in lessons if lesson.kind is LessonKind.PITFALL),
        learned=listed(hypotheses(snapshot)),
        gaps=roofline_gaps(snapshot.profile, final),
    )


def clipped(text: str) -> str:
    """``text`` cut to :data:`MAX_ITEM_CHARS`, an ellipsis marking the cut."""

    return text if len(text) <= MAX_ITEM_CHARS else text[: MAX_ITEM_CHARS - 3] + "..."


def listed(texts: Iterable[str]) -> tuple[str, ...]:
    """``texts`` without repeats, each clipped to :data:`MAX_ITEM_CHARS`, at most :data:`MAX_ITEMS` of them."""

    return tuple(dict.fromkeys(map(clipped, texts)))[:MAX_ITEMS]


__all__ = [
    "MAX_ITEMS",
    "MAX_ITEM_CHARS",
    "UNCHECKED_OUTPUTS",
    "Lesson",
    "LessonKind",
    "SessionLessons",
    "clipped",
    "direction",
    "hypotheses",
    "lesson_of",
    "listed",
    "per_test_lessons",
    "roofline_gaps",
    "session_lessons",
]
