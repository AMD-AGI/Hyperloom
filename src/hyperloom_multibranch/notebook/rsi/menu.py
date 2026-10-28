# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The direction menu: the families a specialist could try next, ranked by what a test of each may bring.

:func:`direction_menu` ranks, for a branch on a commit, the direction families (:mod:`notebook.families`) the archive or
this session holds evidence of by commit odds times the gain within reach over the cost of a test:

* the commit odds are a Thompson draw from the family's Beta distribution: its layer's base rate worth
  :data:`~.family_stats.PRIOR_TRIALS` tests, then the archive's evidence (:mod:`rsi.family_stats`) and this session's,
  at most :data:`~.family_stats.MAX_TRIALS` tests in all, a failure on another commit than the menu's counting
  :data:`ELSEWHERE_FAILURE`. A family tried little draws widely, so now and then it ranks high, while one tried often
  that failed stays low; each branch draws its own, so the branches spread;
* the gain within reach is the family's typical gain where it committed, drawn toward the usual gain of any commit as
  if that were :data:`GAIN_PRIOR_COMMITS` more commits, so a gain counts as much as the commits behind it weigh: one
  commit of a session far from this one moves it little; for an op family no more than the Amdahl ceiling of its ops on
  the commit's profile, for a component no more than the roofline headroom;
* the cost of a test is the median GPU seconds of this session's tests of the family's layer, over that of them all.

A line says whether the family was tried on the commit, another branch holds a claim in it, or the integration head
holds a commit of it. After the :data:`MENU_ITEMS` ranked comes the family with the least evidence for the session's
model type and the most within reach, so exploring never stops, then the untried line: of the directions no test or
claim has acted on, each a target at one layer with its mechanism open, every component but the warm start replay and
each op family whose ops take :data:`HOT_OP_SHARE_PCT` of the commit's profile, the one of the highest score, so a
direction no one has tested yet still comes up. No line offers a family that cannot act on the session's task:
parallelism and communication on a server of one GPU, or an op family none of whose ops the commit's profile shows,
such as moe on a dense model. One branch in four gets no menu (:func:`in_control`): its tests are the menu's control.
Every menu given is logged (:class:`MenuLog`), for CLOSE to read what the specialists chose from it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from hyperloom.common.io import append_jsonl
from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.peers import Intent, IntentLayer, Purpose, TestKind
from hyperloom_multibranch.contracts.notebook.records import (
    NotebookSnapshot,
    OpShares,
    SessionProfile,
    TestRecord,
    amdahl_ceiling_pct,
)
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus
from hyperloom_multibranch.notebook.codec import key_sorted
from hyperloom_multibranch.notebook.families import OP_FAMILIES, Component, Family, Mechanism, family_of, op_family
from hyperloom_multibranch.notebook.rsi.archive import SearchTree
from hyperloom_multibranch.notebook.rsi.family_stats import (
    MAX_TRIALS,
    PRIOR_TRIALS,
    FamilyStat,
    base_rate,
    family_stats,
)
from hyperloom_multibranch.notebook.rsi.frontier import headroom_pct
from hyperloom_multibranch.notebook.rsi.history import MODEL_TYPE_LABEL
from hyperloom_multibranch.notebook.rsi.similarity import task_conditions

log = logging.getLogger(__name__)

MENU_ITEMS = 5
#: Fewer tests of the session's model type than this make a family one to explore.
EXPLORE_TRIALS = 2.0
#: The share of branches that get no menu, the menu's control.
CONTROL_SHARE = 0.25
#: What a failure on another commit than the menu's counts for.
ELSEWHERE_FAILURE = 0.5
#: The gain within reach when no commit's gain is known yet.
DEFAULT_GAIN_PCT = 1.0
#: How many commits the usual gain counts for when a family's typical gain is drawn toward it.
GAIN_PRIOR_COMMITS = 1.0
#: The share of a commit's profile, in percent, an op family's ops take for the menu to offer it before any test.
HOT_OP_SHARE_PCT = 5.0
#: The layers an op family is offered at before any test acts on it there, and those a component is.
OP_LAYERS = (IntentLayer.CONFIG, IntentLayer.SOURCE, IntentLayer.KERNEL)
COMPONENT_LAYERS = (IntentLayer.CONFIG, IntentLayer.SOURCE)
UNTOUCHED = "no test or claim has acted on it at this layer yet"
MENU_LOG = "menus.jsonl"
MENU_LOG_FORMAT = "multibranch.menu-log.v1"
TRIED_HERE = "tried on this commit"
ON_INTEGRATION = "on the integration head"
_OP_FAMILIES = frozenset(name for name, _ in OP_FAMILIES)


def in_control(session_id: str, branch: str) -> bool:
    """Whether ``branch`` is one of the :data:`CONTROL_SHARE` of the session's branches that get no menu: the same
    draw at every launch of its specialist."""

    digest = hashlib.sha256(f"{session_id}:{branch}:menu".encode()).hexdigest()
    return int(digest, 16) < CONTROL_SHARE * 16 ** len(digest)


def explored(record: TestRecord) -> bool:
    """Whether ``record`` is a judged explore test of a specialist branch, what the families' evidence counts."""

    request = record.request
    return (
        request.kind is TestKind.TEST
        and request.purpose is Purpose.EXPLORE
        and request.branch != INTEGRATION_BRANCH
        and record.judged
    )


def record_family(snapshot: NotebookSnapshot, record: TestRecord) -> Family | None:
    """The family of a test of the session, read off its intent and its change; ``None`` without its intent."""

    intent = snapshot.intents.get(record.request.intent_fingerprint or "")
    if intent is None:
        return None
    return family_of(intent.target, intent.mechanism, intent.layer, record.request.delta)


@dataclass
class _Seen:
    """What this session's explore tests of a family came to: commits, misses (one elsewhere counting less), the
    tests themselves, and the gains of its commits."""

    commits: float = 0.0
    misses: float = 0.0
    tests: int = 0
    gains: list[float] = field(default_factory=list)


def _seen(snapshot: NotebookSnapshot, commit: str) -> dict[Family, _Seen]:
    found: dict[Family, _Seen] = {}
    for record in snapshot.records:
        family = record_family(snapshot, record) if explored(record) else None
        if family is None:
            continue
        seen = found.setdefault(family, _Seen())
        seen.tests += 1
        if record.commit_id is not None and record.judgement is not None:
            seen.commits += 1
            seen.gains.append(record.judgement.gain_pct)
        else:
            seen.misses += 1.0 if record.request.parent_commit == commit else ELSEWHERE_FAILURE
    return found


@dataclass(frozen=True)
class FamilyOdds:
    """A family's commit odds as a Beta distribution: its ``commits`` and ``misses`` over its layer's base rate,
    worth :data:`~.family_stats.PRIOR_TRIALS` tests."""

    layer: IntentLayer | None
    commits: float
    misses: float

    @property
    def tests(self) -> float:
        return self.commits + self.misses

    @property
    def mean(self) -> float:
        return (self.commits + base_rate(self.layer) * PRIOR_TRIALS) / (self.tests + PRIOR_TRIALS)

    def draw(self, draws: random.Random) -> float:
        rate = base_rate(self.layer)
        return draws.betavariate(self.commits + rate * PRIOR_TRIALS, self.misses + (1.0 - rate) * PRIOR_TRIALS)


def _odds(family: Family, archived: FamilyStat | None, seen: _Seen | None) -> FamilyOdds:
    """The archive's evidence, its failures elsewhere, and the session's, at most :data:`MAX_TRIALS` tests."""

    commits = (archived.commits if archived is not None else 0.0) + (seen.commits if seen is not None else 0.0)
    missed = (archived.tests - archived.commits) * ELSEWHERE_FAILURE if archived is not None else 0.0
    misses = missed + (seen.misses if seen is not None else 0.0)
    scale = min(1.0, MAX_TRIALS / (commits + misses)) if commits + misses > 0 else 1.0
    return FamilyOdds(family.layer, commits * scale, misses * scale)


def _within_reach(family: Family, typical: float, shares: OpShares | None, headroom: float | None) -> float:
    """``typical`` held to the Amdahl ceiling of an op family's ops in ``shares``, or to the roofline ``headroom`` of a
    component."""

    if family.target in _OP_FAMILIES:
        share = sum(value for op, value in (shares.shares.items() if shares else ()) if op_family(op) == family.target)
        return min(typical, amdahl_ceiling_pct(share)) if share > 0 else typical
    return min(typical, headroom) if headroom is not None else typical


def _applies(family: Family, profile: SessionProfile | None, shares: OpShares | None) -> bool:
    """Whether ``family`` can act on the session's task at all: a server on one GPU runs no collective, so it has no
    parallelism or communication to change, and an op family none of whose ops the commit's profile shows, such as moe
    on a dense model, has no op to change; with no profile of the commit, every op family may."""

    if family.target == Component.PARALLEL.value and profile is not None and profile.tp == 1:
        return False
    if family.target in _OP_FAMILIES and shares is not None:
        return any(share > 0 and op_family(op) == family.target for op, share in shares.shares.items())
    return True


def _untouched(shares: OpShares | None, known: Iterable[Family]) -> dict[Family, str]:
    """The directions no family of ``known`` acts on, each a target at one layer with its mechanism open, and the mark
    its line carries: every component but the warm start replay, whose replays come from the Recipe KB's leads and not
    from a direction to explore, at each of :data:`COMPONENT_LAYERS`, and each op family whose ops take at least
    :data:`HOT_OP_SHARE_PCT` of ``shares`` at each of :data:`OP_LAYERS`. A family of no one layer acts on its target
    at every layer."""

    acted = {(family.target, family.layer) for family in known}
    by_target: dict[str, float] = {}
    for op, share in shares.shares.items() if shares is not None else ():
        target = op_family(op)
        if target is not None:
            by_target[target] = by_target.get(target, 0.0) + share
    offered = [
        (target, layer, f"{UNTOUCHED}; its ops take {share:.0f}% of this commit's profile")
        for target, share in by_target.items()
        if share >= HOT_OP_SHARE_PCT
        for layer in OP_LAYERS
    ]
    offered += [
        (component.value, layer, UNTOUCHED)
        for component in Component
        if component is not Component.WARM_START
        for layer in COMPONENT_LAYERS
    ]
    return {
        Family(target, Mechanism.OTHER, layer): mark
        for target, layer, mark in offered
        if (target, layer) not in acted and (target, None) not in acted
    }


def _costs(snapshot: NotebookSnapshot) -> dict[IntentLayer | None, float]:
    """The median GPU seconds of this session's explore tests of each layer, over that of them all."""

    by_layer: dict[IntentLayer | None, list[float]] = {}
    for record in snapshot.records:
        if not explored(record) or record.result is None or record.result.gpu_seconds <= 0:
            continue
        intent = snapshot.intents.get(record.request.intent_fingerprint or "")
        by_layer.setdefault(intent.layer if intent is not None else None, []).append(record.result.gpu_seconds)
    spent = [value for values in by_layer.values() for value in values]
    usual = statistics.median(spent) if spent else 0.0
    return {layer: statistics.median(values) / usual for layer, values in by_layer.items()} if usual > 0 else {}


def _marks(snapshot: NotebookSnapshot, branch: str, commit: str) -> dict[Family, list[str]]:
    found: dict[Family, list[str]] = {}

    def mark(family: Family | None, text: str) -> None:
        if family is not None and text not in found.setdefault(family, []):
            found[family].append(text)

    for record in snapshot.records:
        if record.request.parent_commit == commit:
            mark(record_family(snapshot, record), TRIED_HERE)
    for claim in snapshot.claims:
        if claim.status is ClaimStatus.HELD and claim.holder != branch:
            intent = claim.intent
            mark(family_of(intent.target, intent.mechanism, intent.layer), f"held by {claim.holder}")
    integration = next((node for node in snapshot.lineage.branches if node.branch == INTEGRATION_BRANCH), None)
    line = {integration.head_commit, *snapshot.lineage.ancestry(integration.head_commit)} if integration else set()
    records = {record.request_id: record for record in snapshot.records}
    for item in snapshot.committed:
        if item.merged_into in line and item.request_id in records:
            mark(record_family(snapshot, records[item.request_id]), ON_INTEGRATION)
    return found


@dataclass(frozen=True)
class MenuItem:
    """One family on the menu: the draw it ranked by, its mean commit odds over its ``tests``, the gain a commit of it
    has within reach, what it scored, its marks, and whether it holds the exploring line or the untried one."""

    family: Family
    draw: float
    odds: float
    tests: float
    gain_pct: float
    score: float
    marks: tuple[str, ...] = ()
    exploring: bool = False
    untried: bool = False

    def render(self) -> str:
        marks = f"; {', '.join(self.marks)}" if self.marks else ""
        lead = "to explore, " if self.exploring else "untried, " if self.untried else ""
        return (
            f"{lead}{self.family.label}: commit odds {self.odds:.2f} over {self.tests:.1f} tests, "
            f"up to {self.gain_pct:+.1f}% a commit{marks}"
        )


@dataclass(frozen=True)
class DirectionMenu:
    """The menu ``branch`` was given on ``commit`` at ``at``; a ``control`` branch's is drawn and logged, not shown."""

    branch: str
    commit: str
    at: float
    control: bool
    items: tuple[MenuItem, ...] = ()

    def lines(self) -> tuple[str, ...]:
        """What the brief shows: nothing to a control branch."""

        return () if self.control else tuple(item.render() for item in self.items)


@dataclass(frozen=True)
class MenuEvidence:
    """What the archive says of each family: weighed for the session (``archived``), and of its model type alone
    (``typed``), which says what is still to explore."""

    archived: Mapping[Family, FamilyStat] = field(default_factory=dict)
    typed: Mapping[Family, FamilyStat] = field(default_factory=dict)


def _headroom(snapshot: NotebookSnapshot, branch: str, commit: str, metric: str) -> float | None:
    ceiling = snapshot.profile.roofline_ceiling.get(metric) if snapshot.profile is not None else None
    row = next((row for row in snapshot.progress if row.branch == branch and row.head_commit == commit), None)
    return headroom_pct(ceiling, row.head_value.get(metric) if row is not None else None)


def direction_menu(
    snapshot: NotebookSnapshot, branch: str, commit: str, evidence: MenuEvidence, *, metric: str
) -> DirectionMenu:
    """The menu of ``branch`` on ``commit``, drawn for this branch, commit and moment: its :data:`MENU_ITEMS` families
    of evidence of the highest score, the little-tried one to explore, and the untouched direction of the highest
    score, the untried line."""

    session = snapshot.profile.session_id if snapshot.profile is not None else ""
    draws = random.Random(hashlib.sha256(f"{session}:{branch}:{commit}:{snapshot.taken_at}".encode()).digest())
    seen = _seen(snapshot, commit)
    gains = [gain for stat in evidence.archived.values() for gain in stat.gains]
    usual = statistics.median([*gains, *(gain for item in seen.values() for gain in item.gains)] or [DEFAULT_GAIN_PCT])
    shares, headroom = snapshot.op_shares.get(commit), _headroom(snapshot, branch, commit, metric)
    costs, marks = _costs(snapshot), _marks(snapshot, branch, commit)
    scoring = _Scoring(draws, usual, shares, headroom, costs)
    scored = [
        scoring.item(family, evidence.archived.get(family), seen.get(family), tuple(marks.get(family, ())))
        for family in sorted({*evidence.archived, *seen}, key=lambda item: item.label)
        if _applies(family, snapshot.profile, shares)
    ]
    ranked = sorted(scored, key=lambda item: (-item.score, item.family.label))
    typed = {family: stat.tests for family, stat in evidence.typed.items()}
    little = [
        item
        for item in ranked[MENU_ITEMS:]
        if typed.get(item.family, 0.0) + (seen[item.family].tests if item.family in seen else 0) < EXPLORE_TRIALS
    ]
    exploring = [replace(max(little, key=lambda item: item.gain_pct), exploring=True)] if little else []
    untouched = _untouched(shares, {*evidence.archived, *seen, *marks})
    fresh = [
        scoring.item(family, None, None, (mark,))
        for family, mark in sorted(untouched.items(), key=lambda item: item[0].label)
        if _applies(family, snapshot.profile, shares)
    ]
    untried = [replace(max(fresh, key=lambda item: (item.score, item.family.label)), untried=True)] if fresh else []
    return DirectionMenu(
        branch, commit, snapshot.taken_at, in_control(session, branch), (*ranked[:MENU_ITEMS], *exploring, *untried)
    )


@dataclass(frozen=True)
class _Scoring:
    """What a menu scores each family by: its branch's draws, the usual gain of a commit, the commit's op shares and
    roofline headroom, and the cost of a test of each layer."""

    draws: random.Random
    usual: float
    shares: OpShares | None
    headroom: float | None
    costs: Mapping[IntentLayer | None, float]

    def item(self, family: Family, archived: FamilyStat | None, here: _Seen | None, marks: tuple[str, ...]) -> MenuItem:
        """``family``'s line: a draw of its odds times the gain within reach over the cost of a test of its layer."""

        odds = _odds(family, archived, here)
        known = [*(archived.gains if archived is not None else ()), *(here.gains if here is not None else ())]
        weight = (archived.commits if archived is not None else 0.0) + (here.commits if here is not None else 0.0)
        typical = statistics.median(known) if known else self.usual
        drawn = (weight * typical + GAIN_PRIOR_COMMITS * self.usual) / (weight + GAIN_PRIOR_COMMITS)
        reach = max(0.0, _within_reach(family, drawn, self.shares, self.headroom))
        draw = odds.draw(self.draws)
        return MenuItem(
            family, draw, odds.mean, odds.tests, reach, draw * reach / self.costs.get(family.layer, 1.0), marks
        )


def direction_note(snapshot: NotebookSnapshot, intent: Intent, delta: RecipeDelta, evidence: MenuEvidence) -> str:
    """What the claim of ``intent`` with ``delta`` tells its specialist of its family: its commit odds, the gain its
    commits made, and how often it was tried on the claimed commit."""

    family = family_of(intent.target, intent.mechanism, intent.layer, delta)
    seen = _seen(snapshot, intent.parent_commit).get(family)
    archived = evidence.archived.get(family)
    odds = _odds(family, archived, seen)
    known = [*(archived.gains if archived is not None else ()), *(seen.gains if seen is not None else ())]
    gain = f", a typical commit {statistics.median(known):+.1f}%" if known else ", no commit yet"
    here = sum(
        1
        for record in snapshot.records
        if record.request.parent_commit == intent.parent_commit and record_family(snapshot, record) == family
    )
    tried = f"; tests of it on this commit: {here}" if here else ""
    return f"your direction is {family.label}: commit odds {odds.mean:.2f} over {odds.tests:.1f} tests{gain}{tried}"


@dataclass(frozen=True)
class LoggedMenu:
    """A menu as the log keeps it: who was given it, on which commit and when, whether it was shown, its families'
    labels in order, the exploring one and the untried one when there are, and the commit odds it gave each."""

    at: float
    branch: str
    commit: str
    control: bool
    families: tuple[str, ...]
    exploring: str = ""
    odds: tuple[float, ...] = ()
    untried: str = ""


@dataclass(frozen=True)
class MenuLog:
    """The session's log of the menus given; a failed append is logged and holds up nobody."""

    path: Path

    def record(self, menu: DirectionMenu) -> None:
        line = {
            "format": MENU_LOG_FORMAT,
            "at": menu.at,
            "branch": menu.branch,
            "commit": menu.commit,
            "control": menu.control,
            "families": [item.family.label for item in menu.items],
            "exploring": next((item.family.label for item in menu.items if item.exploring), ""),
            "untried": next((item.family.label for item in menu.items if item.untried), ""),
            "odds": [round(item.odds, 4) for item in menu.items],
            "draws": [round(item.draw, 4) for item in menu.items],
        }
        try:
            append_jsonl(self.path, key_sorted(line), make_parents=True)
        except OSError as exc:
            log.warning("could not append the menu to %s: %s", self.path, exc)

    def menus(self) -> tuple[LoggedMenu, ...]:
        """The menus logged, oldest first; a line of another format, or one that does not parse, is left out."""

        found = []
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return ()
        for raw in text.splitlines():
            try:
                line = json.loads(raw)
                if line.get("format") != MENU_LOG_FORMAT:
                    continue
                found.append(
                    LoggedMenu(
                        float(line["at"]),
                        str(line["branch"]),
                        str(line["commit"]),
                        bool(line["control"]),
                        tuple(str(label) for label in line["families"]),
                        str(line.get("exploring") or ""),
                        tuple(float(value) for value in line.get("odds") or ()),
                        str(line.get("untried") or ""),
                    )
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
        return tuple(sorted(found, key=lambda item: item.at))


@dataclass
class DirectionMenus:
    """The session's menus: the archived ``trees`` read once for the session's conditions, and the ``log`` of the
    menus given."""

    trees: Sequence[SearchTree] = ()
    metric: str = ""
    log: MenuLog | None = None
    _evidence: MenuEvidence | None = field(default=None, init=False)

    def evidence(self, snapshot: NotebookSnapshot) -> MenuEvidence:
        if self._evidence is None:
            task = task_conditions(snapshot.profile) if snapshot.profile is not None else None
            kind = task.labels.get(MODEL_TYPE_LABEL) if task is not None else None
            typed = [tree for tree in self.trees if kind is not None and tree.task.labels.get(MODEL_TYPE_LABEL) == kind]
            self._evidence = MenuEvidence(family_stats(self.trees, like=task), family_stats(typed))
        return self._evidence

    def menu(self, snapshot: NotebookSnapshot, branch: str, commit: str) -> DirectionMenu:
        """The menu of ``branch`` on ``commit``, logged."""

        found = direction_menu(snapshot, branch, commit, self.evidence(snapshot), metric=self.metric)
        if self.log is not None:
            self.log.record(found)
        return found

    def note(self, snapshot: NotebookSnapshot, intent: Intent, delta: RecipeDelta) -> str:
        return direction_note(snapshot, intent, delta, self.evidence(snapshot))


__all__ = [
    "CONTROL_SHARE",
    "DEFAULT_GAIN_PCT",
    "ELSEWHERE_FAILURE",
    "EXPLORE_TRIALS",
    "HOT_OP_SHARE_PCT",
    "MENU_ITEMS",
    "MENU_LOG",
    "MENU_LOG_FORMAT",
    "ON_INTEGRATION",
    "TRIED_HERE",
    "UNTOUCHED",
    "DirectionMenu",
    "DirectionMenus",
    "FamilyOdds",
    "LoggedMenu",
    "MenuEvidence",
    "MenuItem",
    "MenuLog",
    "direction_menu",
    "direction_note",
    "explored",
    "in_control",
    "record_family",
]
