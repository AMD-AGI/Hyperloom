# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the direction families (:mod:`notebook.families`) of past sessions came to, and whether that tells tests apart.

:func:`family_stats` adds up the families' outcomes over archived sessions: a session of the same model type counts in
full, another half, one on another framework version half again, and each family's evidence is capped at
:data:`MAX_TRIALS` tests, so no rate is ever read as certain. A family's commit odds (:meth:`FamilyStat.odds`) start at
its layer's base rate (:data:`BASE_RATES`), worth :data:`PRIOR_TRIALS` tests: a family tried once, with one commit,
reads 0.39 at the source layer, not 1.0.

:func:`discrimination` is the menu's offline check: leaving each session out, how well the families' commit odds tell
its tests that committed from the rest, beside the layers' base rates alone. An archived direction keeps the family
CLOSE classified from its change; an older one is classified from its words.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.notebook.families import Family, Mechanism, family_of
from hyperloom_multibranch.notebook.rsi.archive import SearchTree, TreeDirection
from hyperloom_multibranch.notebook.rsi.history import MODEL_TYPE_LABEL
from hyperloom_multibranch.notebook.rsi.report import table

#: The families' commit odds before any evidence: the base rate of their layer.
BASE_RATES: Mapping[IntentLayer | None, float] = {
    IntentLayer.CONFIG: 0.12,
    IntentLayer.SOURCE: 0.24,
    IntentLayer.KERNEL: 0.28,
    None: 0.2,
}
#: How many tests the base rate counts as, and the most a family's evidence counts as.
PRIOR_TRIALS = 4.0
MAX_TRIALS = 10.0
#: How much a session of another model type weighs; one on another framework version weighs as much less again.
OTHER_MODEL_WEIGHT = 0.5
OTHER_VERSION_WEIGHT = OTHER_MODEL_WEIGHT
#: What a tie between a committing test and another counts for in the area under the curve.
_TIE = 0.5
FRAMEWORK_VERSION_LABEL = "framework_version"
#: How many families a report lists.
REPORT_ROWS = 20


def direction_family(direction: TreeDirection) -> Family:
    """The family of an archived direction: the one CLOSE kept, or one read off its words."""

    if direction.family_target and direction.family_mechanism:
        return Family(direction.family_target, Mechanism(direction.family_mechanism), direction.layer)
    return family_of(direction.target, direction.mechanism, direction.layer)


def base_rate(layer: IntentLayer | None) -> float:
    return BASE_RATES.get(layer, BASE_RATES[None])


@dataclass(frozen=True)
class FamilyStat:
    """A family's evidence: ``tests`` and ``commits``, weighted and capped, and the best gains its commits reached."""

    tests: float = 0.0
    commits: float = 0.0
    gains: tuple[float, ...] = ()

    def odds(self, layer: IntentLayer | None) -> float:
        """The family's commit odds: its evidence over the base rate of ``layer``, worth :data:`PRIOR_TRIALS`
        tests."""

        return (self.commits + base_rate(layer) * PRIOR_TRIALS) / (self.tests + PRIOR_TRIALS)

    @property
    def typical_gain(self) -> float | None:
        ordered = sorted(self.gains)
        return ordered[len(ordered) // 2] if ordered else None


def _weight(tree: SearchTree, like: SearchTree | None) -> float:
    if like is None:
        return 1.0
    theirs, ours = tree.task.labels, like.task.labels
    weight = 1.0 if theirs.get(MODEL_TYPE_LABEL) == ours.get(MODEL_TYPE_LABEL) else OTHER_MODEL_WEIGHT
    if theirs.get(FRAMEWORK_VERSION_LABEL) != ours.get(FRAMEWORK_VERSION_LABEL):
        weight *= OTHER_VERSION_WEIGHT
    return weight


def family_stats(trees: Sequence[SearchTree], like: SearchTree | None = None) -> dict[Family, FamilyStat]:
    """Each family's evidence over ``trees``, weighed against ``like``, the session to rank for, and capped at
    :data:`MAX_TRIALS` tests a family."""

    tests: dict[Family, float] = {}
    commits: dict[Family, float] = {}
    gains: dict[Family, list[float]] = {}
    for tree in trees:
        weight = _weight(tree, like)
        for branch in tree.branches:
            if not branch.steered:
                continue
            for direction in branch.directions:
                family = direction_family(direction)
                tests[family] = tests.get(family, 0.0) + weight * direction.tests
                commits[family] = commits.get(family, 0.0) + weight * direction.commits
                if direction.commits and direction.best_gain_pct is not None:
                    gains.setdefault(family, []).append(direction.best_gain_pct)
    found = {}
    for family, seen in tests.items():
        scale = min(1.0, MAX_TRIALS / seen) if seen else 1.0
        found[family] = FamilyStat(seen * scale, commits[family] * scale, tuple(gains.get(family, ())))
    return found


@dataclass(frozen=True)
class Discrimination:
    """How well commit odds tell the tests that committed from the rest, leaving each session out: the area under
    the curve of the families' odds and of the layers' base rates alone, 0.5 being chance, over ``tests`` tests."""

    family_auc: float | None
    layer_auc: float | None
    tests: int
    commits: int

    @property
    def tells_apart(self) -> bool:
        """Whether the families tell the tests apart better than their layers alone, so a menu may rank by them."""

        return self.family_auc is not None and self.layer_auc is not None and self.family_auc > self.layer_auc

    def render(self) -> str:
        def shown(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        return (
            f"Over {self.tests} archived tests, {self.commits} of them committing, leaving each session out: the "
            f"families' commit odds tell them apart with an AUC of {shown(self.family_auc)}, the layers' base rates "
            f"alone {shown(self.layer_auc)} (0.5 is chance)."
        )


def _auc(scored: Sequence[tuple[float, bool]]) -> float | None:
    """The chance a committing test scores above one that did not, ties counting half; ``None`` without both."""

    hits = [score for score, committed in scored if committed]
    misses = [score for score, committed in scored if not committed]
    if not hits or not misses:
        return None
    above = sum(1.0 if hit > miss else _TIE if hit == miss else 0.0 for hit in hits for miss in misses)
    return above / (len(hits) * len(misses))


def discrimination(trees: Sequence[SearchTree]) -> Discrimination:
    """The menu's offline check over ``trees``: every test scored by its family's odds from the other sessions."""

    by_family: list[tuple[float, bool]] = []
    by_layer: list[tuple[float, bool]] = []
    for tree in trees:
        stats = family_stats([other for other in trees if other.session_id != tree.session_id], like=tree)
        for branch in tree.branches:
            if not branch.steered:
                continue
            for direction in branch.directions:
                odds = stats.get(direction_family(direction), FamilyStat()).odds(direction.layer)
                for index in range(direction.tests):
                    committed = index < direction.commits
                    by_family.append((odds, committed))
                    by_layer.append((base_rate(direction.layer), committed))
    commits = sum(1 for _, committed in by_family if committed)
    return Discrimination(_auc(by_family), _auc(by_layer), len(by_family), commits)


@dataclass(frozen=True)
class FamilyReport:
    """The offline check and the families with the most evidence, each with its tests, commits and odds."""

    check: Discrimination
    rows: tuple[tuple[Family, FamilyStat], ...]

    def render(self) -> str:
        rows = [
            (
                family.label,
                f"{stat.tests:.1f}",
                f"{stat.commits:.1f}",
                f"{stat.odds(family.layer):.2f}",
                "n/a" if stat.typical_gain is None else f"{stat.typical_gain:+.2f}",
            )
            for family, stat in self.rows
        ]
        headers = ("family", "tests", "commits", "commit odds", "typical best gain %")
        return "\n".join([self.check.render(), "", *table(headers, rows)])


def family_report(trees: Sequence[SearchTree], limit: int = REPORT_ROWS) -> FamilyReport:
    """The offline check over ``trees`` and their ``limit`` families with the most tests, before the cap."""

    stats = family_stats(trees)
    ranked = sorted(stats.items(), key=lambda item: (-item[1].tests, item[0].label))
    return FamilyReport(discrimination(trees), tuple(ranked[:limit]))


__all__ = [
    "BASE_RATES",
    "FRAMEWORK_VERSION_LABEL",
    "MAX_TRIALS",
    "OTHER_MODEL_WEIGHT",
    "OTHER_VERSION_WEIGHT",
    "PRIOR_TRIALS",
    "REPORT_ROWS",
    "Discrimination",
    "FamilyReport",
    "FamilyStat",
    "base_rate",
    "direction_family",
    "discrimination",
    "family_report",
    "family_stats",
]
