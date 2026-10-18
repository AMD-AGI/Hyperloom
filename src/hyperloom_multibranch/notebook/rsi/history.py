# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A digest of the archived sessions like this one, for the scheduler's judge to weigh beside the frontier scores.

The frontier prior folds history into one number per branch; the judge also decides how many branches to open, what
K to give them and when the session has done what such sessions do. :func:`history_digest` answers that from the
archive: how many similar sessions there are and where they came from; how much of their final gain they had by a
quarter, half and three quarters of their budget, sessions being compared on the fraction of their own budget; their
final gain, median and p90; from multi-branch sessions only (a backfilled session has one line of work), how many
branches they opened, how many of those committed, at which test the first commit came, how many branches still
without a commit after 4 and after 8 tests committed later, and why branches were stopped; the directions that paid
off most; and, given where this session stands, the median similar session at the same fraction of its budget.

Every statistic weighs a session by its similarity, and only sessions at least half similar count; with fewer than
1.5 sessions' worth, the digest says history is too thin rather than draw from it. The judge, and gongzheng, decide
what to make of it. :func:`session_history` digests the history for a running session from its own snapshot.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, NotRecorded
from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import CommitOrigin, LineageKind
from hyperloom_multibranch.notebook.rsi.archive import (
    SOURCE_NOTEBOOK,
    SearchTree,
    TreeBranch,
    WeightedTree,
    session_start,
    similar_trees,
    tree_from_snapshot,
)
from hyperloom_multibranch.notebook.rsi.similarity import MIN_SIMILARITY, TaskConditions

PACE_FRACTIONS = (0.25, 0.5, 0.75)
MEDIAN, P90 = 0.5, 0.9
NO_COMMIT_AFTER = (4, 8)
MIN_SESSIONS = 1.5
MAX_CHARS = 1500
TOP_DIRECTIONS = 5
TOP_STOP_REASONS = 3


@dataclass(frozen=True)
class BranchHistory:
    """What the branches of the similar multi-branch sessions did.

    ``late_commits`` maps N to (branches that committed later, branches without a commit after N tests).
    """

    sessions: int
    mean_opened: float
    committed_share: float
    median_first_commit_test: float | None
    late_commits: dict[int, tuple[int, int]] = field(default_factory=dict)
    stop_reasons: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class DirectionHistory:
    """One direction across the similar sessions: in how many it was tried, its tests, commits and best gain."""

    target: str
    mechanism: str
    layer: IntentLayer | None
    sessions: int
    tests: int
    commits: int
    best_gain_pct: float | None


@dataclass(frozen=True)
class SoFar:
    """How far this session has come: ``fraction`` of its budget spent (0 to 1), its best gain over its baseline."""

    fraction: float
    gain_pct: float


@dataclass(frozen=True)
class Standing:
    """This session's best gain at ``fraction`` of its budget, beside the median similar session's at that fraction."""

    fraction: float
    gain_pct: float
    typical_gain_pct: float


@dataclass(frozen=True)
class HistoryDigest:
    """The digest; ``enough`` is false when the similar sessions are worth fewer than :data:`MIN_SESSIONS`.

    ``pace`` maps a fraction of the budget to the median share of their final gain similar sessions had by then.
    ``note`` says why a digest is empty when there was nothing to digest from.
    """

    enough: bool
    sessions: int
    effective_sessions: float
    top_similarities: tuple[float, ...] = ()
    multi_branch: int = 0
    backfilled: int = 0
    pace: dict[float, float] = field(default_factory=dict)
    median_final_gain_pct: float | None = None
    p90_final_gain_pct: float | None = None
    branches: BranchHistory | None = None
    directions: tuple[DirectionHistory, ...] = ()
    standing: Standing | None = None
    note: str = ""

    def render(self, max_chars: int = MAX_CHARS) -> str:
        """The digest as prompt text of at most ``max_chars``; a line that does not fit is left out, never cut."""

        kept: list[str] = []
        used = 0
        for line in _lines(self):
            if used + len(line) + 1 <= max_chars:
                kept.append(line)
                used += len(line) + 1
        return "\n".join(kept)


@dataclass
class _Tally:
    sessions: int = 0
    tests: int = 0
    commits: int = 0
    best_gain_pct: float | None = None


def _quantile(pairs: Sequence[tuple[float, float]], q: float) -> float | None:
    """The weighted ``q``-quantile of ``(value, weight)`` pairs: the smallest value whose weight reaches ``q``."""

    kept = sorted((value, weight) for value, weight in pairs if weight > 0)
    total = sum(weight for _, weight in kept)
    running = 0.0
    for value, weight in kept:
        running += weight
        if running >= q * total * (1 - 1e-9):
            return value
    return None


def _specialists(tree: SearchTree) -> list[TreeBranch]:
    return [b for b in tree.branches if b.name != INTEGRATION_BRANCH and b.origin is not LineageKind.ROOT]


def _best_by(tree: SearchTree, t_s: float) -> float:
    return max((c.gain_pct for b in tree.branches for c in b.commits if c.t_s <= t_s), default=0.0)


def _own_commits(branch: TreeBranch) -> list[float]:
    return sorted(c.t_s for c in branch.commits if c.origin in (CommitOrigin.COMMIT, CommitOrigin.SPLIT))


def _pace(similar: Sequence[WeightedTree]) -> dict[float, float]:
    pace = {}
    for fraction in PACE_FRACTIONS:
        shares = [
            (_best_by(item.tree, fraction * item.tree.budget_seconds) / item.tree.final_gain_pct, item.weight)
            for item in similar
            if item.tree.final_gain_pct > 0 and item.tree.budget_seconds > 0
        ]
        share = _quantile(shares, MEDIAN)
        if share is not None:
            pace[fraction] = share
    return pace


def _branch_history(similar: Sequence[WeightedTree]) -> BranchHistory | None:
    multi = [item for item in similar if item.tree.source == SOURCE_NOTEBOOK and _specialists(item.tree)]
    if not multi:
        return None
    opened, committed, first_tests = [], [], []
    late = {n: [0, 0] for n in NO_COMMIT_AFTER}
    reasons: Counter[str] = Counter()
    for item in multi:
        branches = _specialists(item.tree)
        opened.append((float(len(branches)), item.weight))
        committed.append((sum(1 for b in branches if _own_commits(b)) / len(branches), item.weight))
        for branch in branches:
            commits, tests = _own_commits(branch), sorted(branch.test_times)
            if commits:
                first_tests.append((float(branch.tests_by(commits[0])), item.weight))
            for n in NO_COMMIT_AFTER:
                if len(tests) >= n and not any(t <= tests[n - 1] for t in commits):
                    late[n][1] += 1
                    late[n][0] += bool(commits)
            if branch.stop_reason:
                reasons[branch.stop_reason] += 1
    total = sum(item.weight for item in multi)
    return BranchHistory(
        sessions=len(multi),
        mean_opened=sum(count * weight for count, weight in opened) / total,
        committed_share=sum(share * weight for share, weight in committed) / total,
        median_first_commit_test=_quantile(first_tests, MEDIAN),
        late_commits={n: (later, of) for n, (later, of) in late.items() if of},
        stop_reasons=tuple(reasons.most_common(TOP_STOP_REASONS)),
    )


def _directions(similar: Sequence[WeightedTree]) -> tuple[DirectionHistory, ...]:
    tallies: dict[tuple[str, str, IntentLayer | None], _Tally] = {}
    for item in similar:
        seen = set()
        for branch in _specialists(item.tree):
            for direction in branch.directions:
                key = (direction.target.strip().lower(), direction.mechanism.strip().lower(), direction.layer)
                tally = tallies.setdefault(key, _Tally())
                tally.tests += direction.tests
                tally.commits += direction.commits
                gain = direction.best_gain_pct
                if gain is not None:
                    tally.best_gain_pct = gain if tally.best_gain_pct is None else max(tally.best_gain_pct, gain)
                if key not in seen:
                    seen.add(key)
                    tally.sessions += 1
    ranked = sorted(
        (
            DirectionHistory(target, mechanism, layer, t.sessions, t.tests, t.commits, t.best_gain_pct)
            for (target, mechanism, layer), t in tallies.items()
        ),
        key=lambda d: (-d.commits, -(d.best_gain_pct if d.best_gain_pct is not None else -1e9), -d.sessions),
    )
    return tuple(ranked[:TOP_DIRECTIONS])


def history_digest(
    history: Sequence[SearchTree],
    task: TaskConditions,
    *,
    so_far: SoFar | None = None,
    min_similarity: float = MIN_SIMILARITY,
    min_sessions: float = MIN_SESSIONS,
) -> HistoryDigest:
    """Digest the archived ``history`` for a session on ``task``.

    With ``so_far``, how far this session has come, the digest also stands it beside the similar ones.
    """

    similar = similar_trees(history, task, minimum=min_similarity)
    worth = sum(item.weight for item in similar)
    if worth < min_sessions:
        return HistoryDigest(enough=False, sessions=len(similar), effective_sessions=worth)
    finals = [(item.tree.final_gain_pct, item.weight) for item in similar]
    standing = None
    if so_far is not None:
        then = [
            (_best_by(item.tree, so_far.fraction * item.tree.budget_seconds), item.weight)
            for item in similar
            if item.tree.budget_seconds > 0
        ]
        typical = _quantile(then, MEDIAN)
        if typical is not None:
            standing = Standing(so_far.fraction, so_far.gain_pct, typical)
    multi_branch = sum(1 for item in similar if item.tree.source == SOURCE_NOTEBOOK)
    return HistoryDigest(
        enough=True,
        sessions=len(similar),
        effective_sessions=worth,
        top_similarities=tuple(item.weight for item in similar[:3]),
        multi_branch=multi_branch,
        backfilled=len(similar) - multi_branch,
        pace=_pace(similar),
        median_final_gain_pct=_quantile(finals, MEDIAN),
        p90_final_gain_pct=_quantile(finals, P90),
        branches=_branch_history(similar),
        directions=_directions(similar),
        standing=standing,
    )


def session_history(
    snapshot: NotebookSnapshot, history: Sequence[SearchTree], metric: str, *, budget_seconds: float
) -> HistoryDigest:
    """The digest for the session ``snapshot`` shows, standing it at the share of ``budget_seconds`` it has spent."""

    try:
        tree = tree_from_snapshot(snapshot, metric, budget_seconds=budget_seconds)
    except (NotRecorded, ValueError) as exc:
        return HistoryDigest(enough=False, sessions=0, effective_sessions=0.0, note=f"no digest yet: {exc}")
    spent = (snapshot.taken_at - session_start(snapshot)) / budget_seconds if budget_seconds > 0 else None
    fraction = None if spent is None else min(1.0, max(0.0, spent))
    so_far = None if fraction is None else SoFar(fraction, tree.final_gain_pct)
    return history_digest(history, tree.task, so_far=so_far)


def _pct(value: float) -> str:
    return f"{value:+.1f}%"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _share(value: float) -> str:
    return f"{value * 100:.0f}%"


def _outcome_lines(digest: HistoryDigest) -> list[str]:
    lines = []
    if digest.median_final_gain_pct is not None and digest.p90_final_gain_pct is not None:
        lines.append(
            f"Final gain over baseline: median {_pct(digest.median_final_gain_pct)}, "
            f"p90 {_pct(digest.p90_final_gain_pct)}."
        )
    if digest.pace:
        reached = ", ".join(f"{_share(share)} by {_share(fraction)}" for fraction, share in digest.pace.items())
        lines.append(f"Share of their final gain reached, by share of their budget: {reached}.")
    standing = digest.standing
    if standing is not None:
        gap = standing.gain_pct - standing.typical_gain_pct
        side = "ahead of" if gap > 0 else "behind" if gap < 0 else "level with"
        lines.append(
            f"This session: {_pct(standing.gain_pct)} at {_share(standing.fraction)} of its budget; the median "
            f"similar session had {_pct(standing.typical_gain_pct)} by then ({side} it by {abs(gap):.1f} points)."
        )
    return lines


def _branch_lines(branches: BranchHistory | None) -> list[str]:
    if branches is None:
        return []
    first = (
        f" the first commit came at test {branches.median_first_commit_test:.0f} (median)."
        if branches.median_first_commit_test is not None
        else " none committed."
    )
    lines = [
        f"Branches, from {_count(branches.sessions, 'multi-branch session')}: {branches.mean_opened:.1f} opened "
        f"on average; {_share(branches.committed_share)} of them committed at least once;{first}",
        *(
            f"Branches without a commit after {n} tests: {later} of {of} committed later."
            for n, (later, of) in branches.late_commits.items()
        ),
    ]
    if branches.stop_reasons:
        lines.append(f"Stopped for: {', '.join(f'{reason} ({count})' for reason, count in branches.stop_reasons)}.")
    return lines


def _direction_line(direction: DirectionHistory) -> str:
    layer = f" [{direction.layer.value}]" if direction.layer is not None else ""
    best = f", best {_pct(direction.best_gain_pct)}" if direction.best_gain_pct is not None else ""
    return (
        f"- {direction.target} / {direction.mechanism}{layer}: {_count(direction.sessions, 'session')}, "
        f"{_count(direction.tests, 'test')}, {_count(direction.commits, 'commit')}{best}"
    )


def _lines(digest: HistoryDigest) -> list[str]:
    if digest.note:
        return [f"History: {digest.note}"]
    if not digest.enough:
        return [
            f"History: too thin to go by ({_count(digest.sessions, 'similar session')}, worth "
            f"{digest.effective_sessions:.1f}; {MIN_SESSIONS} needed)."
        ]
    head = (
        f"History: {_count(digest.sessions, 'similar session')} (worth {digest.effective_sessions:.1f}, similarity "
        f"up to {digest.top_similarities[0]:.2f}): {digest.multi_branch} multi-branch, {digest.backfilled} "
        "backfilled single-lineage."
    )
    directions = (
        ["Directions tried, most commits first:", *map(_direction_line, digest.directions)] if digest.directions else []
    )
    return [head, *_outcome_lines(digest), *_branch_lines(digest.branches), *directions]


__all__ = [
    "MAX_CHARS",
    "MIN_SESSIONS",
    "NO_COMMIT_AFTER",
    "PACE_FRACTIONS",
    "BranchHistory",
    "DirectionHistory",
    "HistoryDigest",
    "SoFar",
    "Standing",
    "history_digest",
    "session_history",
]
