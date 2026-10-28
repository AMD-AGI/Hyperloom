# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the specialists' tests came to, the menu's branches beside its control: whether the menu pays, by outcomes.

:func:`specialist_outcomes` counts, at CLOSE, each group's explore tests (the branches shown the menu, and the control
branches, drawn for none, :func:`~.menu.in_control`): how many committed, the gain their merged commits brought per GPU
hour, and how many fell on a family the session's model type had tried fewer than :data:`~.menu.EXPLORE_TRIALS` times
before, the exploring ones, with what those committed. A test whose branch was drawn a menu before it, the control's as
well, is read against the last one: whether a line among the first :data:`TOP` covers its family and, where a line
does, the odds it gave beside whether the test committed, the menu's calibration. A control branch's menu was drawn but
never shown, so how often its tests fall on the top families anyway is what the menu's own share is read against.

Each session's outcomes are kept beside the archive (:class:`OutcomePool`), so the studies add them up over sessions.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.notebook.codec import decode, encode
from hyperloom_multibranch.notebook.families import Family
from hyperloom_multibranch.notebook.local.files import encode_json, publish_once
from hyperloom_multibranch.notebook.rsi.archive import SearchTree, TreeArchive
from hyperloom_multibranch.notebook.rsi.family_stats import family_stats
from hyperloom_multibranch.notebook.rsi.history import MODEL_TYPE_LABEL
from hyperloom_multibranch.notebook.rsi.menu import EXPLORE_TRIALS, LoggedMenu, explored, in_control, record_family
from hyperloom_multibranch.notebook.rsi.report import table

#: How many of a menu's first families count as following it.
TOP = 3
OUTCOMES_FORMAT = "multibranch.specialist-outcomes"
OUTCOMES_REPORT = "specialist-outcomes.md"
MENU_GROUP = "menu"
CONTROL_GROUP = "control"
_OUTCOMES = "outcomes"
_SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class GroupOutcome:
    """One group's explore tests, every count adding up over sessions: its ``branches``, ``tests`` and ``commits``,
    the GPU seconds all its tests took and the gain its merged commits brought; the exploring tests and their commits;
    the tests read against a menu drawn before them, those on its first :data:`TOP` families, and those on a family it
    listed with the odds it gave them summed, what they committed and their squared error."""

    branches: int = 0
    tests: int = 0
    commits: int = 0
    gpu_seconds: float = 0.0
    merged_gain_pct: float = 0.0
    exploring_tests: int = 0
    exploring_commits: int = 0
    after_menu: int = 0
    on_top: int = 0
    listed: int = 0
    listed_commits: int = 0
    odds_sum: float = 0.0
    squared_error: float = 0.0

    def __add__(self, other: GroupOutcome) -> GroupOutcome:
        return GroupOutcome(
            **{item.name: getattr(self, item.name) + getattr(other, item.name) for item in dataclasses.fields(self)}
        )

    def row(self, name: str) -> tuple[str, ...]:
        hours = self.gpu_seconds / _SECONDS_PER_HOUR
        exploiting = (self.tests - self.exploring_tests, self.commits - self.exploring_commits)
        return (
            name,
            str(self.branches),
            str(self.tests),
            _rate(self.commits, self.tests),
            _rate(self.merged_gain_pct, hours),
            _rate(self.exploring_tests, self.tests),
            f"{_rate(self.exploring_commits, self.exploring_tests)} / {_rate(*reversed(exploiting))}",
            _rate(self.on_top, self.after_menu),
            f"{_rate(self.odds_sum, self.listed)} / {_rate(self.listed_commits, self.listed)}",
            _rate(self.squared_error, self.listed),
        )


def _rate(part: float, whole: float) -> str:
    return f"{part / whole:.2f}" if whole else "n/a"


HEADERS = (
    "group",
    "branches",
    "explore tests",
    "commit rate",
    "merged gain % per GPU h",
    "exploring share",
    "commit rate exploring / exploiting",
    f"on the menu's top {TOP}",
    "odds given / committed",
    "Brier score",
)


@dataclass(frozen=True)
class SpecialistOutcomes:
    """A session's specialist outcomes by group: :data:`MENU_GROUP` and :data:`CONTROL_GROUP`."""

    session_id: str
    groups: Mapping[str, GroupOutcome] = field(default_factory=dict)

    def render(self, title: str = "") -> str:
        rows = [self.groups.get(name, GroupOutcome()).row(name) for name in (MENU_GROUP, CONTROL_GROUP)]
        return "\n".join([f"# {title or f'Specialist outcomes of {self.session_id}'}", "", *table(HEADERS, rows)])

    def note(self) -> str:
        menu, control = (self.groups.get(name, GroupOutcome()) for name in (MENU_GROUP, CONTROL_GROUP))
        return (
            f"commit rate {_rate(menu.commits, menu.tests)} with the menu, {_rate(control.commits, control.tests)} "
            f"without; on the menu's top {TOP} {_rate(menu.on_top, menu.after_menu)} against "
            f"{_rate(control.on_top, control.after_menu)}"
        )


def _last_menu(menus: Sequence[LoggedMenu], branch: str, before: float) -> LoggedMenu | None:
    return next((menu for menu in reversed(menus) if menu.branch == branch and menu.at <= before), None)


def _test(committed: bool, exploring: bool, shown: LoggedMenu | None, family: Family) -> GroupOutcome:
    """One explore test of ``family``, read against the menu ``shown`` before it, if any: a line lists the test when
    its family covers the test's, as the untried line, a target at one layer with its mechanism open, covers every test
    of that target at that layer."""

    found = GroupOutcome(
        tests=1,
        commits=int(committed),
        exploring_tests=int(exploring),
        exploring_commits=int(exploring and committed),
    )
    if shown is None:
        return found
    lines = [Family.from_label(label) for label in shown.families]
    listing = [index for index, line in enumerate(lines) if line is not None and line.covers(family)]
    found = dataclasses.replace(found, after_menu=1, on_top=int(any(index < TOP for index in listing)))
    if not listing or len(shown.odds) != len(shown.families):
        return found
    odds = shown.odds[listing[0]]
    return dataclasses.replace(
        found, listed=1, listed_commits=int(committed), odds_sum=odds, squared_error=(odds - committed) ** 2
    )


def specialist_outcomes(
    snapshot: NotebookSnapshot, menus: Sequence[LoggedMenu], history: Sequence[SearchTree]
) -> SpecialistOutcomes:
    """The session's outcomes by group, its explore tests read against the ``menus`` its log kept, oldest first, and
    against what the archived ``history`` of its model type had tried."""

    session = snapshot.profile.session_id if snapshot.profile is not None else ""
    kind = snapshot.profile.model_type if snapshot.profile is not None else ""
    typed = [tree for tree in history if tree.session_id != session and tree.task.labels.get(MODEL_TYPE_LABEL) == kind]
    tried = {family.label: stat.tests for family, stat in family_stats(typed).items()}
    group = {
        row.branch: CONTROL_GROUP if in_control(session, row.branch) else MENU_GROUP
        for row in snapshot.progress
        if row.branch != INTEGRATION_BRANCH
    }
    found: dict[str, GroupOutcome] = {}

    def add(branch: str, outcome: GroupOutcome) -> None:
        found[group[branch]] = found.get(group[branch], GroupOutcome()) + outcome

    for branch in group:
        add(branch, GroupOutcome(branches=1))
    for record in snapshot.records:
        branch = record.request.branch
        if branch not in group:
            continue
        add(branch, GroupOutcome(gpu_seconds=record.result.gpu_seconds if record.result is not None else 0.0))
        family = record_family(snapshot, record) if explored(record) else None
        if family is not None:
            exploring = tried.get(family.label, 0.0) < EXPLORE_TRIALS
            shown = _last_menu(menus, branch, record.requested_at)
            add(branch, _test(record.commit_id is not None, exploring, shown, family))
    for item in snapshot.committed:
        if item.branch in group and item.merged_into is not None:
            add(item.branch, GroupOutcome(merged_gain_pct=item.gain_pct))
    return SpecialistOutcomes(session, found)


@dataclass(frozen=True)
class OutcomePool:
    """The sessions' outcomes: ``<root>/outcomes``, beside an archive's trees under ``<root>/trees``."""

    root: Path

    @classmethod
    def beside(cls, archive: TreeArchive) -> OutcomePool:
        return cls(archive.root.parent / _OUTCOMES)

    def record(self, outcomes: SpecialistOutcomes) -> bool:
        """Keep ``outcomes``, once a session; ``False`` when they were kept already."""

        path = self.root / f"{re.sub(r'[^A-Za-z0-9._-]+', '-', outcomes.session_id)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return publish_once(path, encode_json({"format": OUTCOMES_FORMAT, "record": encode(outcomes)}))

    def sessions(self) -> tuple[SpecialistOutcomes, ...]:
        found = []
        for path in sorted(self.root.glob("*.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                if document.get("format") == OUTCOMES_FORMAT:
                    found.append(decode(SpecialistOutcomes, document.get("record")))
            except (OSError, KeyError, TypeError, ValueError):
                continue
        return tuple(found)


def pooled(sessions: Sequence[SpecialistOutcomes]) -> SpecialistOutcomes:
    """The outcomes of ``sessions`` added up, group by group."""

    groups: dict[str, GroupOutcome] = {}
    for outcomes in sessions:
        for name, group in outcomes.groups.items():
            groups[name] = groups.get(name, GroupOutcome()) + group
    return SpecialistOutcomes(f"{len(sessions)} sessions", groups)


__all__ = [
    "CONTROL_GROUP",
    "HEADERS",
    "MENU_GROUP",
    "OUTCOMES_FORMAT",
    "OUTCOMES_REPORT",
    "TOP",
    "GroupOutcome",
    "OutcomePool",
    "SpecialistOutcomes",
    "pooled",
    "specialist_outcomes",
]
