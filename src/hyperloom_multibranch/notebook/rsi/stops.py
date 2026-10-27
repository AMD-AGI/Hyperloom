# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a session's stops and probations came to: the stop audit CLOSE writes and the studies read.

:func:`stop_audit` reads a search tree. For every steered branch that stopped it records how long the branch had
stalled by then, as the stop rule counts it: its tests judged since its last commit that did not fail to merge, or
since it opened, and the minutes since; who stopped it and why; the probation it was held through in that stall, if
any; and, given the stall odds the judge read, ``p(k)``, its odds of a commit within its next four tests. For every
probation it records what came of it: how many of its tests the branch ran, whether a commit came within them, and
whether the branch stopped before it ran them.

Over the judge's stops it gives what the scheduling playbook's evaluation reads: the median stall at a stop; the
early stops, at most :data:`EARLY_STOP_TESTS` judged tests past a commit; the probations that committed; and what the
stops were worth by the stall odds, ``q - p(k)`` summed over the stops whose ``p(k)`` could be read, ``q`` being a new
branch's odds of a commit within its first four tests. Above 0, the slots the stops freed were likelier to commit
than the branches they stopped.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hyperloom_multibranch.contracts.notebook.vocab import CommitOrigin
from hyperloom_multibranch.notebook.rsi.archive import SearchTree, TreeBranch, TreeProbation
from hyperloom_multibranch.notebook.rsi.history import (
    DEFAULT_WORKLOAD,
    HORIZON_TESTS,
    MODEL_TYPE_LABEL,
    WORKLOAD_LABEL,
    StallOdds,
    judged_tests,
    stall_odds,
)
from hyperloom_multibranch.notebook.rsi.report import NOT_AVAILABLE, number, share, table
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions

#: The cause the scheduler names a judge's stop by, before the reason the judge gave: its ``StopCause.JUDGE``.
JUDGE_CAUSE = "judge"
#: How few judged tests past a commit make a stop early.
EARLY_STOP_TESTS = 2
#: How much of a stop's reason a row of the audit keeps.
REASON_CHARS = 90
_MINUTE = 60.0


@dataclass(frozen=True)
class ProbationOutcome:
    """What came of one probation of ``branch``: of the ``tests`` it gave from ``started_min``, the branch ran ``ran``;
    ``committed`` when a commit that did not fail to merge came within them, ``stopped`` when the branch stopped
    before it ran them all."""

    branch: str
    started_min: float
    tests: int
    ran: int
    committed: bool
    stopped: bool


@dataclass(frozen=True)
class StopRecord:
    """One stop of a steered branch: at ``stopped_min``, by ``cause`` for ``reason``, ``stalled_tests`` judged tests
    and ``stalled_min`` minutes past its last commit that did not fail to merge, or since it opened, with ``commits``
    of its own by then. ``kept_odds`` is ``p(stalled_tests)`` by the stall odds, ``None`` where they could not tell or
    were not given; ``probation`` what came of the probation the stall held it through."""

    branch: str
    stopped_min: float
    cause: str
    reason: str
    stalled_tests: int
    stalled_min: float
    commits: int
    kept_odds: float | None = None
    probation: ProbationOutcome | None = None

    @property
    def by_judge(self) -> bool:
        return self.cause == JUDGE_CAUSE

    @property
    def early(self) -> bool:
        """Whether the branch was stopped at most :data:`EARLY_STOP_TESTS` judged tests past a commit of its own."""

        return self.commits > 0 and self.stalled_tests <= EARLY_STOP_TESTS


@dataclass(frozen=True)
class StopAudit:
    """The stops and probations of one session, with ``new_branch_odds``, ``q``, and the ``stall_tests`` bar of the
    stall odds read beside them."""

    session_id: str
    stops: tuple[StopRecord, ...] = ()
    probations: tuple[ProbationOutcome, ...] = ()
    new_branch_odds: float | None = None
    stall_tests: int | None = None

    @property
    def judged(self) -> tuple[StopRecord, ...]:
        """The judge's stops."""

        return tuple(stop for stop in self.stops if stop.by_judge)

    @property
    def median_stall(self) -> float | None:
        stalls = [stop.stalled_tests for stop in self.judged]
        return statistics.median(stalls) if stalls else None

    @property
    def early_stops(self) -> int:
        return sum(1 for stop in self.judged if stop.early)

    @property
    def below_bar(self) -> int:
        """The judge's stops before the stop bar; 0 without one."""

        if self.stall_tests is None:
            return 0
        return sum(1 for stop in self.judged if stop.stalled_tests < self.stall_tests)

    @property
    def stop_value(self) -> float | None:
        """``q - p(k)`` summed over the judge's stops whose ``p(k)`` could be read; ``None`` without ``q`` or any."""

        if self.new_branch_odds is None:
            return None
        values = [self.new_branch_odds - stop.kept_odds for stop in self.judged if stop.kept_odds is not None]
        return sum(values) if values else None

    def counts(self) -> dict[str, int]:
        return {
            "stops": len(self.stops),
            "judge_stops": len(self.judged),
            "early_stops": self.early_stops,
            "below_bar": self.below_bar,
            "probations": len(self.probations),
            "probations_committed": sum(1 for item in self.probations if item.committed),
        }

    def note(self) -> str:
        """One line: the judge's stops, their median stall, the early ones and those before the bar, the probations
        and what the stops were worth."""

        if not self.judged:
            return "the judge stopped no branch"
        stopped = (
            f"the judge stopped {len(self.judged)} branches at a median of {self.median_stall:g} judged tests past "
            f"their last commit, {self.early_stops} of them at most {EARLY_STOP_TESTS} past one"
        )
        if self.stall_tests is not None:
            stopped += f" and {self.below_bar} before the stop bar of {self.stall_tests}"
        parts = [stopped]
        counts = self.counts()
        if self.probations:
            parts.append(f"{counts['probations_committed']} of {counts['probations']} probations committed")
        if self.stop_value is not None:
            parts.append(f"the stops were worth {self.stop_value:+.2f} commits by the stall odds")
        return "; ".join(parts)

    def render(self) -> str:
        """The audit as Markdown: the note, a row for each stop, and one for each probation."""

        note = self.note()
        bar = str(self.stall_tests) if self.stall_tests is not None else NOT_AVAILABLE
        lines = [
            f"# Stop audit of {self.session_id}",
            "",
            f"{note[0].upper()}{note[1:]}.",
            "",
            f"A branch opened mid-session commits within its first {HORIZON_TESTS} judged tests: "
            f"{_odds(self.new_branch_odds)}; the stop bar stands at {bar} judged tests. p(k) is a stopped branch's odds of a commit within its next "
            f"{HORIZON_TESTS}.",
            "",
            *table(
                (
                    "branch",
                    "stopped at (min)",
                    "by",
                    "stalled tests",
                    "stalled min",
                    "commits",
                    "p(k)",
                    "probation",
                    "reason",
                ),
                [_stop_row(stop) for stop in self.stops],
            ),
        ]
        if self.probations:
            rows = [
                (item.branch, f"{item.started_min:.0f}", item.tests, item.ran, _yes(item.committed), _yes(item.stopped))
                for item in self.probations
            ]
            lines.extend(["", *table(("branch", "from (min)", "tests", "ran", "committed", "stopped early"), rows)])
        return "\n".join(lines)


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def _odds(value: float | None) -> str:
    return NOT_AVAILABLE if value is None else f"{value:.0%}"


def _stop_row(stop: StopRecord) -> tuple[object, ...]:
    held = NOT_AVAILABLE
    if stop.probation is not None:
        held = f"{stop.probation.ran}/{stop.probation.tests}, {'committed' if stop.probation.committed else 'none'}"
    reason = stop.reason.replace("|", "/").replace("\n", " ")
    if len(reason) > REASON_CHARS:
        reason = reason[: REASON_CHARS - 1].rstrip() + "…"
    return (
        stop.branch,
        f"{stop.stopped_min:.0f}",
        stop.cause or NOT_AVAILABLE,
        stop.stalled_tests,
        f"{stop.stalled_min:.0f}",
        stop.commits,
        _odds(stop.kept_odds),
        held,
        reason,
    )


def _anchors(branch: TreeBranch) -> list[float]:
    """When the branch's commits that end a stall landed: every one of its own but those whose merge failed."""

    return sorted(commit.t_s for commit in branch.commits if not commit.merge_failed)


def _own(branch: TreeBranch) -> list[float]:
    return sorted(
        commit.t_s for commit in branch.commits if commit.origin is CommitOrigin.COMMIT and not commit.merge_failed
    )


def _probation(branch: TreeBranch, judged: Sequence[float], held: TreeProbation) -> ProbationOutcome:
    window = [when for when in judged if when > held.t_s][: held.tests]
    end = window[-1] if len(window) == held.tests else math.inf
    return ProbationOutcome(
        branch=branch.name,
        started_min=held.t_s / _MINUTE,
        tests=held.tests,
        ran=len(window),
        committed=any(held.t_s < when <= end for when in _own(branch)),
        stopped=branch.closed_s is not None and branch.closed_s < end,
    )


def _stop(branch: TreeBranch, closed_s: float, odds: StallOdds | None) -> StopRecord:
    judged = judged_tests(branch)
    since = max((when for when in _anchors(branch) if when <= closed_s), default=branch.opened_s)
    stalled = sum(1 for when in judged if since < when <= closed_s)
    cause, said, reason = branch.stop_reason.partition(":")
    held = [item for item in branch.probations if since <= item.t_s <= closed_s]
    return StopRecord(
        branch=branch.name,
        stopped_min=closed_s / _MINUTE,
        cause=cause.strip() if said else "",
        reason=reason.strip() if said else branch.stop_reason,
        stalled_tests=stalled,
        stalled_min=(closed_s - since) / _MINUTE,
        commits=sum(1 for when in _own(branch) if when <= closed_s),
        kept_odds=odds.after_stall(stalled) if odds is not None else None,
        probation=_probation(branch, judged, held[-1]) if held else None,
    )


def stop_audit(tree: SearchTree, odds: StallOdds | None = None) -> StopAudit:
    """The audit of the stops and probations of ``tree``'s steered branches, read against ``odds`` when given."""

    steered = [branch for branch in tree.branches if branch.steered]
    stops = [_stop(branch, branch.closed_s, odds) for branch in steered if branch.closed_s is not None]
    probations = [_probation(branch, judged_tests(branch), held) for branch in steered for held in branch.probations]
    return StopAudit(
        session_id=tree.session_id,
        stops=tuple(sorted(stops, key=lambda stop: stop.stopped_min)),
        probations=tuple(sorted(probations, key=lambda item: item.started_min)),
        new_branch_odds=odds.new_branch_odds if odds is not None else None,
        stall_tests=odds.stall_tests if odds is not None else None,
    )


@dataclass(frozen=True)
class StallStudy:
    """The stall odds of the archive for each model type and workload it holds, and each archived session's stop
    audit, read against the odds of the rest of the archive and of the session itself."""

    odds: Mapping[str, StallOdds]
    audits: tuple[StopAudit, ...]

    def render(self) -> str:
        """The study as Markdown: the odds of each kind of session, then a row for each session's stops."""

        lines: list[str] = []
        for kind, odds in self.odds.items():
            lines.extend([f"## {kind}", "", odds.render(), ""])
        rows = [
            (
                audit.session_id,
                len(audit.judged),
                number(audit.median_stall, 1),
                audit.early_stops,
                audit.below_bar,
                share(audit.counts()["probations_committed"], len(audit.probations)),
                number(audit.stop_value),
            )
            for audit in self.audits
        ]
        stalls = [stop.stalled_tests for audit in self.audits for stop in audit.judged]
        early = sum(audit.early_stops for audit in self.audits)
        below = sum(audit.below_bar for audit in self.audits)
        total = (
            f"Over the archive the judge stopped {len(stalls)} branches at a median of "
            f"{number(statistics.median(stalls) if stalls else None, 1)} judged tests past their last commit, "
            f"{early} of them at most {EARLY_STOP_TESTS} past one and {below} before their session's stop bar."
        )
        headers = (
            "session",
            "judge stops",
            "median stall",
            "early stops",
            "before the bar",
            "probations committed",
            "worth",
        )
        return "\n".join([*lines, "## Stops", "", total, "", *table(headers, rows)])


def _kind(task: TaskConditions) -> str:
    labels = task.labels
    return f"{labels.get(MODEL_TYPE_LABEL, 'any model type')}, {labels.get(WORKLOAD_LABEL, DEFAULT_WORKLOAD)}"


def stall_study(trees: Sequence[SearchTree]) -> StallStudy:
    """The stall odds and the stop audits of the archived ``trees``."""

    odds: dict[str, StallOdds] = {}
    for tree in trees:
        odds.setdefault(_kind(tree.task), stall_odds(trees, tree.task))
    audits = tuple(stop_audit(tree, stall_odds(trees, tree.task, session=tree)) for tree in trees)
    return StallStudy(dict(sorted(odds.items())), audits)


__all__ = [
    "EARLY_STOP_TESTS",
    "JUDGE_CAUSE",
    "REASON_CHARS",
    "ProbationOutcome",
    "StallStudy",
    "StopAudit",
    "StopRecord",
    "stall_study",
    "stop_audit",
]
