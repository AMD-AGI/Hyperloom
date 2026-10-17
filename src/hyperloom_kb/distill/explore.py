# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a group of Experiences looks like, computed the same way every time: the group a scope selects, its counts
and outcome statistics, a cross table over one or two fields, and how each member bears on a claim.

Statistics read only the roles a schema declares: the outcome's decision, the baseline's and outcome's measurements,
and the Repeat Group key; a schema that declares none still selects, groups, and counts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from hyperloom_kb.distill.model import EvidenceClass, Scope, value_at
from hyperloom_kb.query_view import repeat_group_key
from hyperloom_kb.schema import BASELINE, CHANGE, OUTCOME, Experience, ExperienceDeclaration, FieldRole, JsonValue


def select(experiences: Iterable[Experience], scope: Scope) -> list[Experience]:
    """The Experiences ``scope`` holds for; one without a value it names is left out."""

    return [experience for experience in experiences if scope.holds(experience) is True]


def newest_first(experiences: Iterable[Experience]) -> list[Experience]:
    return sorted(experiences, key=lambda e: (e.completed_at or e.created_at, e.id), reverse=True)


def _quantile(ordered: Sequence[float], q: float) -> float:
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: Iterable[float]) -> dict[str, JsonValue]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.5),
        "p75": _quantile(ordered, 0.75),
        "max": ordered[-1],
    }


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


@dataclass(frozen=True)
class GroupView:
    """How to read a schema's roles off its Experiences."""

    declaration: ExperienceDeclaration

    def summary(self, experience: Experience) -> str | None:
        field = self.declaration.role_field(CHANGE, FieldRole.SUMMARY)
        value = experience.change.get(field.name) if field is not None else None
        return value if isinstance(value, str) else None

    def decision(self, experience: Experience) -> str | None:
        field = self.declaration.role_field(OUTCOME, FieldRole.DECISION)
        value = experience.outcome.get(field.name) if field is not None else None
        return value if isinstance(value, str) else None

    def _measurement(self, experience: Experience, category: str) -> float | None:
        field = self.declaration.role_field(category, FieldRole.MEASUREMENT)
        return _number(getattr(experience, category).get(field.name)) if field is not None else None

    def outcome(self, experience: Experience) -> float | None:
        return self._measurement(experience, OUTCOME)

    def change_pct(self, experience: Experience) -> float | None:
        """The outcome's change over the baseline, in percent, when both measurements are there."""

        baseline, outcome = self._measurement(experience, BASELINE), self.outcome(experience)
        if baseline is None or outcome is None or baseline == 0:
            return None
        return 100.0 * (outcome - baseline) / baseline

    def summarize(self, members: Sequence[Experience], member_limit: int) -> dict[str, JsonValue]:
        """Counts, the runs and Repeat Groups the members come from, decisions, and outcome statistics."""

        decisions = Counter(decision for decision in map(self.decision, members) if decision is not None)
        return {
            "count": len(members),
            "runs": len({experience.run_id for experience in members}),
            "repeat_groups": len({repeat_group_key(self.declaration, experience) for experience in members}),
            "decisions": dict(sorted(decisions.items())),
            "measurement": _distribution(v for v in map(self.outcome, members) if v is not None),
            "change_pct": _distribution(v for v in map(self.change_pct, members) if v is not None),
            "members": [experience.id for experience in members[:member_limit]],
            "members_total": len(members),
        }

    def cross(
        self, members: Sequence[Experience], paths: Sequence[str], member_limit: int
    ) -> list[dict[str, JsonValue]]:
        """One cell per combination of values at ``paths``, largest first; a missing value is ``null``."""

        cells: dict[tuple[object, ...], list[Experience]] = {}
        for experience in members:
            cells.setdefault(tuple(value_at(experience, path) for path in paths), []).append(experience)
        ordered = sorted(cells.items(), key=lambda item: (-len(item[1]), repr(item[0])))
        return [
            {"values": dict(zip(paths, key)), **self.summarize(group, member_limit)}  # type: ignore[arg-type]
            for key, group in ordered
        ]


def representatives(
    members: Sequence[tuple[Experience, EvidenceClass]], designated: Sequence[str], limit: int
) -> list[str]:
    """Up to ``limit`` of ``members`` that a page names for the rest: the ``designated`` ones among them, or else every
    contradicting member, then the newest supporting member of each run, newest first."""

    present = {experience.id for experience, _ in members}
    if designated:
        return [experience_id for experience_id in designated if experience_id in present][:limit]
    classes = {experience.id: cls for experience, cls in members}
    ordered = newest_first(experience for experience, _ in members)
    chosen = [experience.id for experience in ordered if classes[experience.id] is EvidenceClass.CONTRADICT]
    runs: set[str] = set()
    for experience in ordered:
        if classes[experience.id] is EvidenceClass.SUPPORT and experience.run_id not in runs:
            runs.add(experience.run_id)
            chosen.append(experience.id)
    return chosen[:limit]


def judged(experience: Experience, check: Scope) -> EvidenceClass:
    """How ``check`` classifies ``experience``: supporting where it holds, irrelevant where a value it names is
    missing."""

    holds = check.holds(experience)
    if holds is None:
        return EvidenceClass.IRRELEVANT
    return EvidenceClass.SUPPORT if holds else EvidenceClass.CONTRADICT


def classify(
    members: Sequence[Experience],
    check: Scope | None,
    annotations: Mapping[str, EvidenceClass],
) -> list[tuple[Experience, EvidenceClass]]:
    """How each member bears on a claim: as the author annotated it, else as ``check`` decides, else supporting.

    Without a check the author claims the group supports the conclusion, and annotates the members that do not.
    """

    def classified(experience: Experience) -> EvidenceClass:
        if experience.id in annotations:
            return annotations[experience.id]
        return EvidenceClass.SUPPORT if check is None else judged(experience, check)

    return [(experience, classified(experience)) for experience in members]


__all__ = ["GroupView", "classify", "judged", "newest_first", "representatives", "select"]
