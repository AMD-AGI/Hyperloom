# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The direction menu: two lists of what a specialist could try next, each ranked by what a test of it may bring.

:func:`direction_menu` draws, for a branch on a commit, two lists of direction families (:mod:`blackboard.families`):

* the settings list: the config-layer families the archive holds evidence of, and the config directions no test or
  claim has acted on, each a target with its mechanism open: every component but the warm start replay, whose replays
  come from the Recipe KB's leads, and each op family whose ops take :data:`HOT_OP_SHARE_PCT` of the profile. A family
  a judged explore test of this session acted on is on neither list: a setting tried is tried for the session. A line
  ranks by a draw of its commit odds times the gain within reach, over the cost of a config test;
* the code list: a line for a target and mechanism changed at the source or the kernel layer, the two layers'
  evidence added up over a base rate between theirs, for each such family the archive or this session holds evidence
  of and for each op family whose ops take :data:`HOT_OP_SHARE_PCT` of the profile while no source or kernel test or
  claim has acted on it. A line ranks by a draw of its commit odds times its ceiling, over the cost of a code test, so
  the largest operators come first, and names the ops it acts on. A tuning line stays whatever its tests on the commit
  came to, its odds falling with each miss; a structural one (:data:`STRUCTURAL`) that missed on the commit goes after
  the others; a line whose last :data:`DROP_MISSES` judged tests on the commit all missed leaves the list, unless a
  commit of it is on the commit's line.

The commit odds are a Thompson draw from the line's Beta distribution: its base rate worth
:data:`~.family_stats.PRIOR_TRIALS` tests, then the archive's evidence (:mod:`rsi.family_stats`) and this session's, at
most :data:`~.family_stats.MAX_TRIALS` tests in all, a failure on another commit than the menu's counting
:data:`ELSEWHERE_FAILURE`. A family tried little draws widely, so now and then it ranks high, while one tried often that
failed stays low; each branch draws its own, so the branches spread. The gain within reach is the family's typical gain
where it committed, drawn toward the usual gain of any commit as if that were :data:`GAIN_PRIOR_COMMITS` more commits,
and no more than its ceiling. The ceiling of an op family is the Amdahl ceiling of its ops on the commit's profile, the
profile of the nearest commit it builds on until it has one of its own (:func:`profiled_commit`); a component's is the
roofline headroom. A family whose ceiling is under the session's bar is on no list: no test of it can clear the bar.
The cost of a test is the median GPU seconds of this session's tests of the list's layers, over that of them all.

No list offers a family that cannot act on the session's task: parallelism and communication on a server of one GPU,
or an op family none of whose ops the profile shows, such as moe on a dense model. A line says whether its family was
tried on the commit, another branch holds a claim in it, or the integration head holds a commit of it. One branch in
four gets no menu (:func:`in_control`): its tests are the menu's control. Every menu given is logged
(:class:`MenuLog`), for CLOSE to read what the specialists chose from it.
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
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.blackboard.peers import Intent, IntentLayer, Purpose, TestKind
from hyperloom_multibranch.contracts.blackboard.records import (
    BlackboardSnapshot,
    OpShares,
    SessionProfile,
    TestRecord,
    amdahl_ceiling_pct,
)
from hyperloom_multibranch.contracts.blackboard.vocab import ClaimStatus
from hyperloom_multibranch.blackboard.codec import key_sorted
from hyperloom_multibranch.blackboard.families import (
    ANY_LAYER,
    OP_FAMILIES,
    Component,
    Family,
    Mechanism,
    family_of,
    op_family,
)
from hyperloom_multibranch.blackboard.rsi.archive import SearchTree
from hyperloom_multibranch.blackboard.rsi.family_stats import (
    MAX_TRIALS,
    PRIOR_TRIALS,
    FamilyStat,
    base_rate,
    family_stats,
)
from hyperloom_multibranch.blackboard.rsi.frontier import headroom_pct
from hyperloom_multibranch.blackboard.rsi.similarity import task_conditions

log = logging.getLogger(__name__)

#: How many lines each list holds.
MENU_ITEMS = 3
#: The share of branches that get no menu, the menu's control.
CONTROL_SHARE = 0.25
#: What a failure on another commit than the menu's counts for.
ELSEWHERE_FAILURE = 0.5
#: The gain within reach when no commit's gain is known yet.
DEFAULT_GAIN_PCT = 1.0
#: How many commits the usual gain counts for when a family's typical gain is drawn toward it.
GAIN_PRIOR_COMMITS = 1.0
#: The share of a commit's profile, in percent, an op family's ops take for a list to offer it before any test.
HOT_OP_SHARE_PCT = 5.0
#: The layers a line of the code list stands for, and what its label says of them.
CODE_LAYERS = (IntentLayer.SOURCE, IntentLayer.KERNEL)
CODE_LAYER = "source or kernel"
#: The mechanisms that change how an operator is built rather than how it is tuned, which one miss says more of.
STRUCTURAL = frozenset(
    {
        Mechanism.FUSION,
        Mechanism.BACKEND,
        Mechanism.PRECISION,
        Mechanism.LAYOUT,
        Mechanism.PARTITIONING,
        Mechanism.OVERHEAD,
    }
)
#: How many judged misses in a row on a commit take a line off the code list.
DROP_MISSES = 3
#: How the ops a line acts on sit under it, the largest first, as many as the brief's line holds.
OPS_LINE = "  ops "
SETTINGS = "settings"
CODE = "code"
UNTOUCHED = "no test or claim has acted on it yet"
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


def profiled_commit(snapshot: BlackboardSnapshot, commit: str) -> str:
    """The commit whose profile stands for ``commit``'s: ``commit`` itself when it was profiled, else the nearest commit
    it builds on that was; empty when none was."""

    return next((item for item in (commit, *snapshot.lineage.ancestry(commit)) if item in snapshot.op_shares), "")


def record_family(snapshot: BlackboardSnapshot, record: TestRecord) -> Family | None:
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


def _seen(snapshot: BlackboardSnapshot, commit: str) -> dict[Family, _Seen]:
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
    """A family's commit odds as a Beta distribution: its ``commits`` and ``misses`` over a base ``rate``, worth
    :data:`~.family_stats.PRIOR_TRIALS` tests."""

    rate: float
    commits: float
    misses: float

    @property
    def tests(self) -> float:
        return self.commits + self.misses

    @property
    def mean(self) -> float:
        return (self.commits + self.rate * PRIOR_TRIALS) / (self.tests + PRIOR_TRIALS)

    def draw(self, draws: random.Random) -> float:
        return draws.betavariate(
            self.commits + self.rate * PRIOR_TRIALS, self.misses + (1.0 - self.rate) * PRIOR_TRIALS
        )


@dataclass(frozen=True)
class _Evidence:
    """What the archive and this session hold of a line's families."""

    archived: tuple[FamilyStat, ...] = ()
    seen: tuple[_Seen, ...] = ()

    @property
    def commits(self) -> float:
        return sum(stat.commits for stat in self.archived) + sum(item.commits for item in self.seen)

    @property
    def gains(self) -> list[float]:
        return [gain for item in (*self.archived, *self.seen) for gain in item.gains]

    def odds(self, rate: float) -> FamilyOdds:
        """The archive's evidence, its failures elsewhere, and the session's, at most :data:`MAX_TRIALS` tests, over
        ``rate``."""

        missed = sum((stat.tests - stat.commits) * ELSEWHERE_FAILURE for stat in self.archived)
        misses = missed + sum(item.misses for item in self.seen)
        total = self.commits + misses
        scale = min(1.0, MAX_TRIALS / total) if total > 0 else 1.0
        return FamilyOdds(rate, self.commits * scale, misses * scale)


def _evidence(
    families: Iterable[Family], archived: Mapping[Family, FamilyStat], seen: Mapping[Family, _Seen]
) -> _Evidence:
    wanted = tuple(families)
    return _Evidence(
        tuple(archived[family] for family in wanted if family in archived),
        tuple(seen[family] for family in wanted if family in seen),
    )


def _code_rate() -> float:
    """The base rate of a code line: between its layers'."""

    return statistics.mean(base_rate(layer) for layer in CODE_LAYERS)


def _stands_for(family: Family) -> tuple[Family, ...]:
    """The families a line of ``family`` stands for: ``family``, or for one of no one layer, a code line, its target
    and mechanism at each of :data:`CODE_LAYERS`."""

    if family.layer is not None:
        return (family,)
    return tuple(replace(family, layer=layer) for layer in CODE_LAYERS)


def _ops(target: str, shares: OpShares | None) -> tuple[tuple[str, float], ...]:
    """The ops of op family ``target`` in ``shares`` with their shares, the largest first."""

    if shares is None or target not in _OP_FAMILIES:
        return ()
    found = [(op, share) for op, share in shares.shares.items() if share > 0 and op_family(op) == target]
    return tuple(sorted(found, key=lambda item: (-item[1], item[0])))


def _ceiling(target: str, shares: OpShares | None, headroom: float | None) -> float | None:
    """What a commit acting on ``target`` can gain at most: the Amdahl ceiling of an op family's ops in ``shares``, the
    roofline ``headroom`` of a component; ``None`` when that is not known."""

    if target in _OP_FAMILIES:
        share = sum(value for _, value in _ops(target, shares))
        return amdahl_ceiling_pct(share) if share > 0 else None
    return headroom


def _applies(family: Family, profile: SessionProfile | None, shares: OpShares | None) -> bool:
    """Whether ``family`` can act on the session's task at all: a server on one GPU runs no collective, so it has no
    parallelism or communication to change, and an op family none of whose ops the commit's profile ``shares`` shows,
    such as moe on a dense model, has no op to change; with no profile of the commit or a commit it builds on, every op
    family may."""

    if family.target == Component.PARALLEL.value and profile is not None and profile.tp == 1:
        return False
    if family.target in _OP_FAMILIES and shares is not None:
        return any(share > 0 and op_family(op) == family.target for op, share in shares.shares.items())
    return True


def _hot_ops(shares: OpShares | None) -> list[str]:
    """The op families whose ops take at least :data:`HOT_OP_SHARE_PCT` of ``shares``."""

    by_target: dict[str, float] = {}
    for op, share in shares.shares.items() if shares is not None else ():
        target = op_family(op)
        if target is not None:
            by_target[target] = by_target.get(target, 0.0) + share
    return sorted(target for target, share in by_target.items() if share >= HOT_OP_SHARE_PCT)


def _acted(known: Iterable[Family], layers: Sequence[IntentLayer]) -> set[str]:
    """The targets a family of ``known`` acts on at one of ``layers``; a family of no one layer acts at every layer."""

    return {family.target for family in known if family.layer is None or family.layer in layers}


def _costs(snapshot: BlackboardSnapshot) -> dict[str, float]:
    """The median GPU seconds of this session's explore tests of each list's layers, over that of them all."""

    by_list: dict[str, list[float]] = {}
    spent: list[float] = []
    for record in snapshot.records:
        if not explored(record) or record.result is None or record.result.gpu_seconds <= 0:
            continue
        spent.append(record.result.gpu_seconds)
        intent = snapshot.intents.get(record.request.intent_fingerprint or "")
        layer = intent.layer if intent is not None else None
        listed = SETTINGS if layer is IntentLayer.CONFIG else CODE if layer in CODE_LAYERS else None
        if listed is not None:
            by_list.setdefault(listed, []).append(record.result.gpu_seconds)
    usual = statistics.median(spent) if spent else 0.0
    return {listed: statistics.median(values) / usual for listed, values in by_list.items()} if usual > 0 else {}


def _marks(snapshot: BlackboardSnapshot, branch: str, commit: str) -> dict[Family, list[str]]:
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


def _code_tests(snapshot: BlackboardSnapshot, commit: str) -> dict[tuple[str, Mechanism], list[bool]]:
    """Whether each judged source or kernel explore test on ``commit`` committed, by target and mechanism, oldest
    first."""

    found: dict[tuple[str, Mechanism], list[bool]] = {}
    for record in sorted(snapshot.records, key=lambda item: item.requested_at):
        here = explored(record) and record.request.parent_commit == commit
        family = record_family(snapshot, record) if here else None
        if family is not None and family.layer in CODE_LAYERS:
            found.setdefault((family.target, family.mechanism), []).append(record.commit_id is not None)
    return found


def _committed_on(snapshot: BlackboardSnapshot, commit: str) -> set[tuple[str, Mechanism]]:
    """The target and mechanism of each source or kernel commit on ``commit`` or a commit it builds on."""

    line = {commit, *snapshot.lineage.ancestry(commit)}
    records = {record.request_id: record for record in snapshot.records}
    found = set()
    for item in snapshot.committed:
        record = records.get(item.request_id) if item.commit_id in line else None
        family = record_family(snapshot, record) if record is not None else None
        if family is not None and family.layer in CODE_LAYERS:
            found.add((family.target, family.mechanism))
    return found


@dataclass(frozen=True)
class MenuItem:
    """One line of the menu: its family, of no one layer on the code list, which folds the source and kernel layers;
    the draw it ranked by, its mean commit odds over its ``tests``, the gain a commit of it has within reach, what it
    scored, its marks, and for an op family the ops it acts on, the largest first, with their shares of the profile of
    ``profile``."""

    family: Family
    draw: float
    odds: float
    tests: float
    gain_pct: float
    score: float
    marks: tuple[str, ...] = ()
    ops: tuple[tuple[str, float], ...] = ()
    profile: str = ""

    @property
    def families(self) -> tuple[Family, ...]:
        """The families the line stands for: its own, or its target and mechanism at each of :data:`CODE_LAYERS`."""

        return _stands_for(self.family)

    @property
    def label(self) -> str:
        if self.family.layer is not None:
            return self.family.label
        return f"{self.family.label.removesuffix(ANY_LAYER)}{CODE_LAYER}"

    def lines(self) -> tuple[str, ...]:
        """The line, and under it the ops it acts on when it names any: the brief clips each line on its own."""

        marks = f"; {', '.join(self.marks)}" if self.marks else ""
        line = f"{self.label}: odds {self.odds:.2f} over {self.tests:.1f} tests, up to {self.gain_pct:+.1f}%{marks}"
        if not self.ops:
            return (line,)
        named = ", ".join(f"{op} {share:.1f}%" for op, share in self.ops)
        total = sum(share for _, share in self.ops)
        return line, f"{OPS_LINE}{total:.1f}% of the profile of {self.profile[:12]}: {named}"


@dataclass(frozen=True)
class DirectionMenu:
    """The menu ``branch`` was given on ``commit`` at ``at``: its settings list and its code list; a ``control``
    branch's is drawn and logged, not shown."""

    branch: str
    commit: str
    at: float
    control: bool
    settings: tuple[MenuItem, ...] = ()
    code: tuple[MenuItem, ...] = ()

    @property
    def lists(self) -> dict[str, tuple[MenuItem, ...]]:
        return {SETTINGS: self.settings, CODE: self.code}

    @property
    def items(self) -> tuple[MenuItem, ...]:
        return (*self.settings, *self.code)

    def lines(self, listed: str) -> tuple[str, ...]:
        """What the brief shows of the list ``listed``, :data:`SETTINGS` or :data:`CODE`: nothing to a control
        branch."""

        return () if self.control else tuple(line for item in self.lists[listed] for line in item.lines())


@dataclass(frozen=True)
class MenuEvidence:
    """What the archive says of each family, weighed for the session."""

    archived: Mapping[Family, FamilyStat] = field(default_factory=dict)


def _headroom(snapshot: BlackboardSnapshot, branch: str, commit: str, metric: str) -> float | None:
    ceiling = snapshot.profile.roofline_ceiling.get(metric) if snapshot.profile is not None else None
    row = next((row for row in snapshot.progress if row.branch == branch and row.head_commit == commit), None)
    return headroom_pct(ceiling, row.head_value.get(metric) if row is not None else None)


@dataclass(frozen=True)
class _Scoring:
    """What a menu scores its lines by: its branch's draws, the usual gain of a commit, the profile standing for the
    commit's (``profiled``'s ``shares``) and the roofline headroom, the cost of a test of each list, and the session's
    bar."""

    draws: random.Random
    usual: float
    shares: OpShares | None
    profiled: str
    headroom: float | None
    costs: Mapping[str, float]
    bar_pct: float | None

    def setting(self, family: Family, evidence: _Evidence, marks: tuple[str, ...]) -> MenuItem | None:
        """``family``'s settings line: a draw of its odds times the gain within reach over the cost of a config test;
        ``None`` when its ceiling is under the bar."""

        ceiling = _ceiling(family.target, self.shares, self.headroom)
        if not self._clears(ceiling):
            return None
        known = evidence.gains
        typical = statistics.median(known) if known else self.usual
        weight = evidence.commits
        drawn = (weight * typical + GAIN_PRIOR_COMMITS * self.usual) / (weight + GAIN_PRIOR_COMMITS)
        reach = min(drawn, ceiling) if ceiling is not None else drawn
        return self._item(family, evidence.odds(base_rate(family.layer)), reach, SETTINGS, marks)

    def code(self, family: Family, evidence: _Evidence, marks: tuple[str, ...]) -> MenuItem | None:
        """``family``'s code line: a draw of its odds times its ceiling over the cost of a code test, the usual gain
        standing in for a ceiling not known; ``None`` when its ceiling is under the bar."""

        ceiling = _ceiling(family.target, self.shares, self.headroom)
        if not self._clears(ceiling):
            return None
        reach = ceiling if ceiling is not None else self.usual
        return self._item(family, evidence.odds(_code_rate()), reach, CODE, marks)

    def _clears(self, ceiling: float | None) -> bool:
        return ceiling is None or self.bar_pct is None or ceiling >= self.bar_pct

    def _item(self, family: Family, odds: FamilyOdds, reach: float, listed: str, marks: tuple[str, ...]) -> MenuItem:
        draw = odds.draw(self.draws)
        reach = max(0.0, reach)
        ops = _ops(family.target, self.shares)
        score = draw * reach / self.costs.get(listed, 1.0)
        return MenuItem(family, draw, odds.mean, odds.tests, reach, score, marks, ops, self.profiled if ops else "")


@dataclass(frozen=True)
class _Known:
    """What a menu's lists are drawn from: the archive's evidence, this session's explore tests of each family, and
    the marks of each family a test or claim of the session acted on."""

    archived: Mapping[Family, FamilyStat]
    seen: Mapping[Family, _Seen]
    marks: Mapping[Family, Sequence[str]]

    @property
    def families(self) -> set[Family]:
        return {*self.archived, *self.seen, *self.marks}


def _top(items: Iterable[MenuItem]) -> tuple[MenuItem, ...]:
    return tuple(sorted(items, key=lambda item: (-item.score, item.label))[:MENU_ITEMS])


def _settings(scoring: _Scoring, known: _Known, profile: SessionProfile | None) -> tuple[MenuItem, ...]:
    """The settings list: the archive's config families and the config directions no test or claim acted on, but for
    those a judged explore test of this session acted on."""

    acted = _acted(known.families, (IntentLayer.CONFIG,))
    targets = [*_hot_ops(scoring.shares), *(item.value for item in Component if item is not Component.WARM_START)]
    untouched = {Family(target, Mechanism.OTHER, IntentLayer.CONFIG) for target in targets if target not in acted}
    archived = {family for family in known.archived if family.layer is IntentLayer.CONFIG}
    found = []
    for family in sorted(archived | untouched, key=lambda item: item.label):
        tried = any(family.covers(other) for other in known.seen)
        if tried or family.target == Component.WARM_START.value or not _applies(family, profile, scoring.shares):
            continue
        marks = (*known.marks.get(family, ()), *((UNTOUCHED,) if family in untouched else ()))
        item = scoring.setting(family, _evidence((family,), known.archived, known.seen), marks)
        if item is not None:
            found.append(item)
    return _top(found)


def _code(
    scoring: _Scoring, known: _Known, profile: SessionProfile | None, snapshot: BlackboardSnapshot, commit: str
) -> tuple[MenuItem, ...]:
    """The code list: a line for each target and mechanism the archive or this session changed at the source or kernel
    layer, and for each hot op family no source or kernel test or claim acted on; a structural line that missed on
    ``commit`` after the others, and none whose last :data:`DROP_MISSES` tests on it missed with no commit of it on
    ``commit``'s line."""

    evidenced = {*known.archived, *known.seen}
    keys = {(family.target, family.mechanism) for family in evidenced if family.layer in CODE_LAYERS}
    acted = _acted(known.families, CODE_LAYERS)
    untouched = {(target, Mechanism.OTHER) for target in _hot_ops(scoring.shares) if target not in acted}
    tests, committed = _code_tests(snapshot, commit), _committed_on(snapshot, commit)
    ranked = []
    for target, mechanism in sorted(keys | untouched, key=lambda key: (key[0], key[1].value)):
        line = Family(target, mechanism, None)
        here = tests.get((target, mechanism), [])
        dropped = len(here) >= DROP_MISSES and not any(here[-DROP_MISSES:]) and (target, mechanism) not in committed
        if dropped or target == Component.WARM_START.value or not _applies(line, profile, scoring.shares):
            continue
        families = _stands_for(line)
        marked = [mark for family in families for mark in known.marks.get(family, ())]
        marks = tuple(dict.fromkeys([*marked, *((UNTOUCHED,) if (target, mechanism) in untouched else ())]))
        item = scoring.code(line, _evidence(families, known.archived, known.seen), marks)
        if item is not None:
            ranked.append((mechanism in STRUCTURAL and not all(here), item))
    ranked.sort(key=lambda pair: (pair[0], -pair[1].score, pair[1].label))
    return tuple(item for _, item in ranked[:MENU_ITEMS])


def direction_menu(
    snapshot: BlackboardSnapshot, branch: str, commit: str, evidence: MenuEvidence, *, metric: str
) -> DirectionMenu:
    """The menu of ``branch`` on ``commit``, drawn for this branch, commit and moment: the :data:`MENU_ITEMS` lines of
    the highest score on each list."""

    session = snapshot.profile.session_id if snapshot.profile is not None else ""
    draws = random.Random(hashlib.sha256(f"{session}:{branch}:{commit}:{snapshot.taken_at}".encode()).digest())
    seen = _seen(snapshot, commit)
    gains = [gain for stat in evidence.archived.values() for gain in stat.gains]
    usual = statistics.median([*gains, *(gain for item in seen.values() for gain in item.gains)] or [DEFAULT_GAIN_PCT])
    profiled = profiled_commit(snapshot, commit)
    bar = snapshot.baseline.bar_pct if snapshot.baseline is not None else None
    headroom = _headroom(snapshot, branch, commit, metric)
    scoring = _Scoring(draws, usual, snapshot.op_shares.get(profiled), profiled, headroom, _costs(snapshot), bar)
    known = _Known(evidence.archived, seen, _marks(snapshot, branch, commit))
    settings = _settings(scoring, known, snapshot.profile)
    code = _code(scoring, known, snapshot.profile, snapshot, commit)
    return DirectionMenu(branch, commit, snapshot.taken_at, in_control(session, branch), settings, code)


def direction_note(snapshot: BlackboardSnapshot, intent: Intent, delta: RecipeDelta, evidence: MenuEvidence) -> str:
    """What the claim of ``intent`` with ``delta`` tells its specialist of its family: its commit odds, the gain its
    commits made, and how often it was tried on the claimed commit."""

    family = family_of(intent.target, intent.mechanism, intent.layer, delta)
    found = _evidence((family,), evidence.archived, _seen(snapshot, intent.parent_commit))
    odds = found.odds(base_rate(family.layer))
    gain = f", a typical commit {statistics.median(found.gains):+.1f}%" if found.gains else ", no commit yet"
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
    labels in order, a code line's at each layer it stands for, the commit odds it gave each, and the list each is
    on."""

    at: float
    branch: str
    commit: str
    control: bool
    families: tuple[str, ...]
    odds: tuple[float, ...] = ()
    sections: tuple[str, ...] = ()


@dataclass(frozen=True)
class MenuLog:
    """The session's log of the menus given; a failed append is logged and holds up nobody."""

    path: Path

    def record(self, menu: DirectionMenu) -> None:
        rows = [
            (listed, item, family) for listed, items in menu.lists.items() for item in items for family in item.families
        ]
        line = {
            "format": MENU_LOG_FORMAT,
            "at": menu.at,
            "branch": menu.branch,
            "commit": menu.commit,
            "control": menu.control,
            "families": [family.label for _, _, family in rows],
            "sections": [listed for listed, _, _ in rows],
            "odds": [round(item.odds, 4) for _, item, _ in rows],
            "draws": [round(item.draw, 4) for _, item, _ in rows],
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
                        tuple(float(value) for value in line.get("odds") or ()),
                        tuple(str(listed) for listed in line.get("sections") or ()),
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

    def evidence(self, snapshot: BlackboardSnapshot) -> MenuEvidence:
        if self._evidence is None:
            task = task_conditions(snapshot.profile) if snapshot.profile is not None else None
            self._evidence = MenuEvidence(family_stats(self.trees, like=task))
        return self._evidence

    def menu(self, snapshot: BlackboardSnapshot, branch: str, commit: str) -> DirectionMenu:
        """The menu of ``branch`` on ``commit``, logged."""

        found = direction_menu(snapshot, branch, commit, self.evidence(snapshot), metric=self.metric)
        if self.log is not None:
            self.log.record(found)
        return found

    def note(self, snapshot: BlackboardSnapshot, intent: Intent, delta: RecipeDelta) -> str:
        return direction_note(snapshot, intent, delta, self.evidence(snapshot))


__all__ = [
    "CODE",
    "CODE_LAYER",
    "CODE_LAYERS",
    "CONTROL_SHARE",
    "DEFAULT_GAIN_PCT",
    "DROP_MISSES",
    "ELSEWHERE_FAILURE",
    "HOT_OP_SHARE_PCT",
    "MENU_ITEMS",
    "MENU_LOG",
    "MENU_LOG_FORMAT",
    "ON_INTEGRATION",
    "OPS_LINE",
    "SETTINGS",
    "STRUCTURAL",
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
    "profiled_commit",
    "record_family",
]
