# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The evaluation ladder a new playbook climbs, from what costs nothing to what costs sessions.

* R0, :func:`r0`: the static check (:func:`~hyperloom_multibranch.notebook.rsi.playbook.problems`).
* R1, :func:`r1`: the scheduling replay of :mod:`rsi.replay`, no LLM and no GPU. A change to a number passes when its
  paired gain over the champion stays above 0 across the archived sessions both ways beyond the data; a change to a
  rule cannot be replayed and goes on to R2.
* R2, :func:`r2`: the judge's own decision points (:func:`archived_points`), the prompts the archived sessions' judge
  was asked, asked again with the candidate's playbook and the champion's. A point counts where its outcome is known: a stop at most
  :data:`EARLY_STOP_TESTS` judged tests past a commit, which the playbook should have prevented, and a stop at or past
  the stall bar, which it should keep. The candidate passes when it gets more of them right than the champion. A rule
  change also replays :data:`JUDGED_SESSIONS` sessions :data:`JUDGED_RUNS` times each under either playbook with the
  judge asked every round (:func:`r2_sessions`), and passes when its best gain beats the champion's, paired run by
  run.
* R3, :func:`r3`: the challenger sessions, each recorded at CLOSE as a :class:`Trial` with its task
  (:func:`session_task`) and its decision measures (:func:`session_measures`), and paired with the champion's session
  of its task that started nearest it (:func:`pairs`): the run entry runs the two side by side. A playbook is crowned
  once at least :data:`MIN_TRIALS` pairs beat the champion's sessions on the measure it named, on average, by more
  than the noise band, while their gain per GPU hour stays within the band; it is retired once :data:`MAX_TRIALS` of
  its sessions have not. The noise band of a measure is its spread within a task over the champion's sessions of the
  pairs' tasks (:func:`noise_band`), the repeats the run entry runs to measure it: until there are :data:`MIN_TRIALS`
  of one task, nothing is crowned. A challenger session that broke the safety envelope played the champion from then
  on, and pairs with nothing.

At every CLOSE the replay reruns over the archive with the session's tree in it, against the champion
(:func:`replay_standing`), and says which of its conclusions moved since the last CLOSE (:func:`moved`).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import re
import statistics
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from hyperloom_multibranch.contracts.notebook.vocab import CommitOrigin, LineageKind
from hyperloom_multibranch.notebook.codec import decode, encode
from hyperloom_multibranch.notebook.local.files import encode_json, publish_once
from hyperloom_multibranch.notebook.rsi.archive import SearchTree, written_by
from hyperloom_multibranch.notebook.rsi.history import judged_tests, stall_odds
from hyperloom_multibranch.notebook.rsi.playbook import Crowned, Playbook, PlaybookPool, Retired, changes, problems
from hyperloom_multibranch.notebook.rsi.replay import (
    OPTIMISTIC,
    PESSIMISTIC,
    REPLAY_RUNS,
    Draws,
    PolicyRow,
    SchedulingComparison,
    compare,
    stops_in,
    variants,
)
from hyperloom_multibranch.notebook.rsi.similarity import MODEL, WORKLOAD, model_name, workload_kind
from hyperloom_multibranch.notebook.rsi.stops import EARLY_STOP_TESTS, StopAudit

#: Asks the LLM with a system prompt and a prompt, and returns its answer.
Complete = Callable[[str, str], Awaitable[str]]
TRIAL_FORMAT = "multibranch.scheduling-playbook-trial"
#: Each decision measure, and whether more of it is better (1) or less (-1).
DIRECTION: Mapping[str, int] = {
    "early_stops": -1,
    "probation_commits": -1,
    "stop_value": 1,
    "first_commit_minutes": -1,
    "opened_committed": 1,
}
#: The whole-session measure every trial must not lose on.
GAIN_PER_GPU_HOUR = "gain_per_gpu_hour"
MIN_TRIALS = 3
MAX_TRIALS = 6
#: How far apart a challenger session and the champion's session it pairs with may start: the two run side by side.
PAIR_SECONDS = 48 * 3600.0
#: The labels and sizes of one task beside its workload's kind: the model, the GPU, the framework and its version,
#: which the image fixes, the precision, the parallelism, the concurrency and the sequence lengths.
TASK_LABELS = (MODEL, "gpu", "framework", "framework_version", "precision")
TASK_SIZES = ("tp", "conc", "isl", "osl")
#: How many runs a session gets when CLOSE reruns the scheduling replay, fewer than a study's so CLOSE stays short.
CLOSE_REPLAY_RUNS = 50
STANDING_FORMAT = "multibranch.scheduling-replay-standing"
BETTER = "better"
WORSE = "worse"
UNCLEAR = "unclear"
#: How many of the judge's decision points R2 asks again, each once under either playbook.
R2_POINTS = 15
#: The judged replay of a rule change: sessions, runs each, the judge asked every round under either playbook; about
#: 2000 calls in all over twelve-hour sessions.
JUDGED_SESSIONS = 3
JUDGED_RUNS = 5
#: What the judge of a judged replay is told it is.
JUDGE_SYSTEM_PROMPT = (
    "You are the scheduling judge of a multi-branch optimization session. Every round you read the running branches' "
    "stall evidence and the scheduling playbook, and decide which branches to stop. Follow the playbook, its numbers, "
    "its rules and its safety envelope; stop a branch only where they say to."
)
_TRIALS = "trials"
_REPLAYS = "replays"
_JSON = "*.json"
_ENCODING = "utf-8"
_MINUTE = 60.0
_Kept = TypeVar("_Kept")
_CANDIDATE_SECTION = "The scheduling playbook to follow in this round:"


@dataclass(frozen=True)
class Rung:
    """What one rung said of a playbook: ``passed``, or why not; ``skipped`` when it cannot judge that playbook."""

    rung: str
    passed: bool
    reason: str
    skipped: bool = False


def r0(candidate: Playbook, parent: Playbook | None) -> Rung:
    found = problems(candidate, parent)
    return Rung("R0", not found, "; ".join(found) or "within its bounds, one change of its parent")


def r1(
    candidate: Playbook, champion: Playbook, trees: Sequence[SearchTree], *, runs: int = REPLAY_RUNS, seed: str = ""
) -> Rung:
    """R1: ``candidate`` beside ``champion`` in the scheduling replay of ``trees``, both ways beyond the data."""

    if "rules" in changes(champion, candidate):
        return Rung("R1", True, "a rule change: the replay cannot play it", skipped=True)
    lows = []
    pair = {champion.label: champion, candidate.label: candidate}
    for beyond in (OPTIMISTIC, PESSIMISTIC):
        found = compare(trees, pair, baseline=champion.label, drawn=Draws(runs=runs, beyond=beyond, seed=seed))
        row = next(item for item in found.rows if item.label == candidate.label)
        lows.append((beyond, row.gain_difference, row.difference_low))
    passed = all(low > 0 for _, _, low in lows)
    said = "; ".join(f"{beyond} {gain:+.2f} points (low {low:+.2f})" for beyond, gain, low in lows)
    return Rung("R1", passed, f"best gain against {champion.label}: {said}")


@dataclass(frozen=True)
class DecisionPoint:
    """One branch the archived judge stopped, at ``at`` on the session clock, ``stalled`` judged tests past its last
    commit, with the prompt the judge was asked; ``keep`` when the stop should go, a stop right after a commit, and
    not when it should stay, a stop past the stall bar."""

    session_id: str
    branch: str
    at: float
    stalled: int
    keep: bool
    prompt: str


def _stall_at(tree: SearchTree, name: str, t_s: float) -> tuple[int, bool] | None:
    """The judged tests of ``name`` at ``t_s`` since its last commit that did not fail to merge, or since it opened,
    and whether it had made one; ``None`` for a branch the tree lacks."""

    branch = next((item for item in tree.branches if item.name == name), None)
    if branch is None:
        return None
    own = [c.t_s for c in branch.commits if c.origin is CommitOrigin.COMMIT and not c.merge_failed and c.t_s <= t_s]
    since = max(own, default=branch.opened_s)
    return sum(1 for when in judged_tests(branch) if since < when <= t_s), bool(own)


def decision_points(
    tree: SearchTree, calls: Sequence[Mapping[str, object]], *, bar: int, start: float | None = None
) -> tuple[DecisionPoint, ...]:
    """The stops of ``tree``'s judge, from its logged ``calls`` (``at``, ``prompt``, ``reply``), whose outcome is
    known under a stall bar of ``bar``; ``start`` is when the session began, by default its archive's."""

    began = start if start is not None else (tree.started_at or 0.0)
    points = []
    for call in calls:
        at, prompt, reply = call.get("at"), call.get("prompt"), call.get("reply")
        if not isinstance(at, (int, float)) or not isinstance(prompt, str) or not isinstance(reply, str):
            continue
        for name in stops_in(reply):
            when = float(at) - began
            found = _stall_at(tree, name, when)
            if found is None:
                continue
            stalled, committed = found
            early = committed and stalled <= EARLY_STOP_TESTS
            if early or stalled >= bar:
                points.append(DecisionPoint(tree.session_id, name, when, stalled, early, prompt))
    return tuple(points)


def archived_points(
    trees: Sequence[SearchTree], decisions: Mapping[str, Sequence[Mapping[str, object]]]
) -> tuple[DecisionPoint, ...]:
    """The decision points of ``trees`` whose judge's calls ``decisions`` keeps by session, each tree's at the stop bar
    the stall odds of the other trees set, its own branches counted in."""

    found: list[DecisionPoint] = []
    for tree in trees:
        calls = decisions.get(tree.session_id, ())
        if calls:
            others = [other for other in trees if other.session_id != tree.session_id]
            found += decision_points(tree, calls, bar=stall_odds(others, tree.task, session=tree).stall_tests)
    return tuple(found)


@dataclass(frozen=True)
class ReplayJudge:
    """The judge R2 asks again: ``complete`` with the judge's ``system`` prompt."""

    complete: Complete
    system: str = JUDGE_SYSTEM_PROMPT

    async def ask(self, prompt: str) -> str:
        return await self.complete(self.system, prompt)


def _asked(point: DecisionPoint, playbook: Playbook) -> str:
    return f"{point.prompt}\n\n{_CANDIDATE_SECTION}\n{playbook.render()}"


async def r2(
    candidate: Playbook,
    champion: Playbook,
    points: Sequence[DecisionPoint],
    judge: ReplayJudge,
    *,
    limit: int = R2_POINTS,
) -> Rung:
    """R2: the decision ``points`` asked again of ``judge`` under ``candidate`` and ``champion``; the first ``limit``
    of them, the stops to undo and those to keep in turn."""

    undo = [point for point in points if point.keep]
    keep = [point for point in points if not point.keep]
    chosen = [point for pair in zip(undo, keep) for point in pair] + undo[len(keep) :] + keep[len(undo) :]
    chosen = chosen[:limit]
    if not chosen:
        return Rung("R2", True, "no archived decision point with a known outcome", skipped=True)
    right = {candidate.label: 0, champion.label: 0}
    for point in chosen:
        for playbook in (candidate, champion):
            stops = point.branch in stops_in(await judge.ask(_asked(point, playbook)))
            right[playbook.label] += int(stops != point.keep)
    passed = right[candidate.label] > right[champion.label]
    said = f"right on {right[candidate.label]} of {len(chosen)} points, {champion.label} on {right[champion.label]}"
    return Rung("R2", passed, said)


async def r2_sessions(
    candidate: Playbook,
    champion: Playbook,
    trees: Sequence[SearchTree],
    judge: ReplayJudge,
    *,
    drawn: Draws = Draws(runs=JUDGED_RUNS),
) -> Rung:
    """R2 for a rule change: ``trees`` replayed under ``candidate`` and ``champion`` as ``drawn`` says, ``judge``
    asked every round; it passes when the candidate's best gain beats the champion's, paired run by run. A change of
    numbers only is R1's to replay."""

    if "rules" not in changes(champion, candidate):
        return Rung("R2", True, "a change of numbers: R1 replays it", skipped=True)
    loop = asyncio.get_running_loop()

    def decide(prompt: str) -> str:
        return asyncio.run_coroutine_threadsafe(judge.ask(prompt), loop).result()

    pair = {champion.label: champion, candidate.label: candidate}
    judged = dataclasses.replace(drawn, decide=decide)
    found = await asyncio.to_thread(compare, trees, pair, baseline=champion.label, drawn=judged)
    row = next(item for item in found.rows if item.label == candidate.label)
    said = (
        f"judged replay of {len(found.sessions)} sessions, {drawn.runs} runs each: best gain "
        f"{row.gain_difference:+.2f} points against {champion.label} ({row.difference_low:+.2f} to "
        f"{row.difference_high:+.2f})"
    )
    return Rung("R2", row.gain_difference > 0, said)


@dataclass(frozen=True)
class Trial:
    """One session under ``played``, with ``champion`` the champion then: the ``task`` it ran (:func:`session_task`),
    when it started, whether it ``fell_back`` to the champion on a breach of the safety envelope, and its decision
    measures and gain per GPU hour; a measure is missing where the session gave it nothing to count."""

    session_id: str
    played: str
    champion: str
    task: str = ""
    started_at: float | None = None
    fell_back: bool = False
    measures: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Pair:
    """A challenger session and the champion's session of its task it is read against."""

    challenger: Trial
    champion: Trial

    def difference(self, measure: str) -> float | None:
        """How much better the challenger did on ``measure``, by :data:`DIRECTION` (more gain per GPU hour is
        better); ``None`` when either session did not measure it."""

        ours, theirs = self.challenger.measures.get(measure), self.champion.measures.get(measure)
        if ours is None or theirs is None:
            return None
        return DIRECTION.get(measure, 1) * (ours - theirs)


def session_task(tree: SearchTree) -> str:
    """The task ``tree`` ran, as one key: its workload's kind, its :data:`TASK_LABELS` and :data:`TASK_SIZES` and its
    budget. Two sessions pair only when their keys are one."""

    labels, sizes = tree.task.labels, tree.task.sizes
    named = [
        f"{key}={model_name(labels[key]) if key == MODEL else labels[key].lower()}"
        for key in TASK_LABELS
        if key in labels
    ]
    sized = [f"{key}={sizes[key]:g}" for key in TASK_SIZES if key in sizes]
    return " ".join(
        [f"{WORKLOAD}={workload_kind(tree.task)}", *named, *sized, f"budget={tree.budget_seconds / _MINUTE:g}m"]
    )


def session_measures(tree: SearchTree, audit: StopAudit) -> dict[str, float]:
    """The decision measures of ``tree``: its judge's early stops, the share of its probations that committed, what
    its stops were worth, a new branch's median minutes to its first commit, the share of new branches that
    committed, and its best gain per GPU hour."""

    opened = [branch for branch in tree.branches if branch.steered and branch.origin is not LineageKind.ROOT]
    firsts = [
        (min(c.t_s for c in branch.commits if c.origin is CommitOrigin.COMMIT) - branch.opened_s) / _MINUTE
        for branch in opened
        if any(c.origin is CommitOrigin.COMMIT for c in branch.commits)
    ]
    gpu_hours = sum(branch.gpu_seconds for branch in tree.branches if branch.steered) / 3600.0
    found: dict[str, float] = {"early_stops": float(audit.early_stops)}
    if audit.probations:
        found["probation_commits"] = sum(1 for item in audit.probations if item.committed) / len(audit.probations)
    if audit.stop_value is not None:
        found["stop_value"] = audit.stop_value
    if firsts:
        found["first_commit_minutes"] = statistics.median(firsts)
    if opened:
        found["opened_committed"] = len(firsts) / len(opened)
    if gpu_hours > 0:
        found[GAIN_PER_GPU_HOUR] = tree.final_gain_pct / gpu_hours
    return found


def _keep(folder: Path, session_id: str, fmt: str, record: object) -> bool:
    path = folder / f"{re.sub(r'[^A-Za-z0-9._-]+', '-', session_id)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return publish_once(path, encode_json({"format": fmt, "record": encode(record)}))


def _kept(folder: Path, kind: type[_Kept], fmt: str) -> list[_Kept]:
    found = []
    for path in sorted(folder.glob(_JSON)):
        try:
            document = json.loads(path.read_text(encoding=_ENCODING))
            if document.get("format") == fmt:
                found.append(decode(kind, document.get("record")))
        except (OSError, KeyError, TypeError, ValueError):
            continue
    return found


def record_trial(pool: PlaybookPool, trial: Trial) -> bool:
    """Keep ``trial`` beside the pool's playbooks, once a session; ``False`` when it was kept already."""

    return _keep(pool.root / _TRIALS, trial.session_id, TRIAL_FORMAT, trial)


def trials(pool: PlaybookPool) -> tuple[Trial, ...]:
    """The trials kept beside ``pool``, of the sessions that started by the time it is read as of."""

    return tuple(
        item for item in _kept(pool.root / _TRIALS, Trial, TRIAL_FORMAT) if written_by(pool.as_of, item.started_at)
    )


def pairs(candidate: str, champion: str, kept: Sequence[Trial]) -> tuple[Pair, ...]:
    """Each session that played ``candidate`` against ``champion`` and kept to it, beside the session of ``champion``
    on its task that started nearest it, within :data:`PAIR_SECONDS`; a champion session pairs once at most, and the
    challenger sessions take theirs in the order they started."""

    started = {trial.session_id: trial.started_at for trial in kept if trial.started_at is not None}

    def gap(first: Trial, second: Trial) -> float:
        return abs(started[first.session_id] - started[second.session_id])

    mine = [
        trial
        for trial in kept
        if trial.session_id in started
        and trial.played == candidate
        and trial.champion == champion
        and not trial.fell_back
    ]
    free = [trial for trial in kept if trial.session_id in started and trial.played == champion]
    found = []
    for trial in sorted(mine, key=lambda item: started[item.session_id]):
        near = [item for item in free if item.task == trial.task and gap(item, trial) <= PAIR_SECONDS]
        if near:
            partner = min(near, key=lambda item: gap(item, trial))
            free.remove(partner)
            found.append(Pair(trial, partner))
    return tuple(found)


def noise_band(sessions: Sequence[Trial], measure: str) -> float | None:
    """The spread of ``measure`` within a task over ``sessions``: the standard deviation pooled over their tasks, so a
    difference between tasks is no noise; ``None`` with less to go on than :data:`MIN_TRIALS` sessions of one task."""

    by_task: dict[str, list[float]] = {}
    for trial in sessions:
        if measure in trial.measures:
            by_task.setdefault(trial.task, []).append(trial.measures[measure])
    groups = [values for values in by_task.values() if len(values) > 1]
    freedom = sum(len(values) - 1 for values in groups)
    if freedom < MIN_TRIALS - 1:
        return None
    return math.sqrt(sum((len(values) - 1) * statistics.variance(values) for values in groups) / freedom)


def _fell(sessions: Sequence[Trial], label: str) -> str:
    count = sum(1 for trial in sessions if trial.played == label and trial.fell_back)
    return f"; {count} of its sessions fell back to the champion" if count else ""


def r3(candidate: Playbook, champion: Playbook, kept: Sequence[Trial]) -> Rung:
    """R3: ``candidate``'s sessions paired with ``champion``'s (:func:`pairs`), on the measure it named and on gain per
    GPU hour, each against its noise band over ``champion``'s sessions of the pairs' tasks."""

    measure = candidate.measure
    if measure not in DIRECTION:
        return Rung("R3", False, f"{candidate.label} names no decision measure")
    differences = [
        (pair.challenger.task, pair.difference(measure), pair.difference(GAIN_PER_GPU_HOUR))
        for pair in pairs(candidate.label, champion.label, kept)
    ]
    measured = [(task, gain, rate) for task, gain, rate in differences if gain is not None and rate is not None]
    fell = _fell(kept, candidate.label)
    if len(measured) < MIN_TRIALS:
        return Rung("R3", False, f"{len(measured)} of its {MIN_TRIALS} paired sessions so far{fell}", skipped=True)
    tasks = {task for task, _, _ in measured}
    theirs = [trial for trial in kept if trial.played == champion.label and trial.task in tasks]
    band, floor_band = noise_band(theirs, measure), noise_band(theirs, GAIN_PER_GPU_HOUR)
    if band is None or floor_band is None:
        return Rung(
            "R3", False, f"too few repeats of {champion.label} on these tasks to know the noise band", skipped=True
        )
    better = statistics.fmean(gain for _, gain, _ in measured)
    rate = statistics.fmean(rate for _, _, rate in measured)
    passed = better > band and rate >= -floor_band
    said = (
        f"{measure} better by {better:.3g} over {len(measured)} paired sessions, band {band:.3g}; "
        f"gain per GPU hour {rate:+.3g}, band {floor_band:.3g}{fell}"
    )
    return Rung("R3", passed, said)


def settle(pool: PlaybookPool, now: float) -> tuple[str, ...]:
    """Crown the oldest playbook on trial that passes R3, or retire those whose :data:`MAX_TRIALS` sessions did not;
    what was done, a line each."""

    kept, champion, done = trials(pool), pool.champion(), []
    for candidate in pool.on_trial():
        verdict = r3(candidate, champion, kept)
        played = sum(1 for trial in kept if trial.played == candidate.label)
        if verdict.passed:
            pool.crown(Crowned(candidate.label, now, verdict.reason))
            done.append(f"crowned {candidate.label}: {verdict.reason}")
            break
        if played >= MAX_TRIALS:
            pool.retire(Retired(candidate.label, now, verdict.reason))
            done.append(f"retired {candidate.label}: {verdict.reason}")
    return tuple(done)


def trial_of(tree: SearchTree, audit: StopAudit, played: str, champion: str, *, fell_back: bool = False) -> Trial:
    task, measures = session_task(tree), session_measures(tree, audit)
    return Trial(tree.session_id, played, champion, task, tree.started_at, fell_back, measures)


@dataclass(frozen=True)
class Difference:
    """A playbook's mean paired difference in best gain from the champion, and its interval's ends."""

    mean: float
    low: float
    high: float

    @classmethod
    def of(cls, row: PolicyRow) -> Difference:
        return cls(row.gain_difference, row.difference_low, row.difference_high)


@dataclass(frozen=True)
class StandingRow:
    """One playbook's paired difference from the champion with h(k) optimistic and pessimistic beyond the data."""

    label: str
    optimistic: Difference
    pessimistic: Difference

    @property
    def reading(self) -> str:
        """:data:`BETTER` when both intervals sit above 0, :data:`WORSE` when both sit below, :data:`UNCLEAR`
        otherwise."""

        if self.optimistic.low > 0 and self.pessimistic.low > 0:
            return BETTER
        if self.optimistic.high < 0 and self.pessimistic.high < 0:
            return WORSE
        return UNCLEAR


@dataclass(frozen=True)
class ReplayStanding:
    """What the scheduling replay said at the CLOSE of ``session_id``, at ``at``: each playbook against ``champion``
    over the ``sessions`` archived then, ``runs`` runs each."""

    session_id: str
    at: float
    champion: str
    sessions: int
    runs: int
    rows: tuple[StandingRow, ...] = ()

    def readings(self) -> dict[str, str]:
        return {row.label: row.reading for row in self.rows}


def replay_standing(
    trees: Sequence[SearchTree], pool: PlaybookPool, *, session_id: str, at: float, runs: int = CLOSE_REPLAY_RUNS
) -> tuple[ReplayStanding, tuple[SchedulingComparison, ...]]:
    """The scheduling replay of ``trees`` against ``pool``'s champion, both ways beyond the data: the default variants
    (H0 and the old rule among them) and the playbooks on trial. The standing it comes to, and the comparisons."""

    champion = pool.champion()
    playbooks = {**variants(), **{playbook.label: playbook for playbook in pool.on_trial()}, champion.label: champion}
    found = tuple(
        compare(trees, playbooks, baseline=champion.label, drawn=Draws(runs=runs, beyond=beyond))
        for beyond in (OPTIMISTIC, PESSIMISTIC)
    )
    held, dropped = ({row.label: row for row in comparison.rows} for comparison in found)
    rows = tuple(
        StandingRow(label, Difference.of(row), Difference.of(dropped[label]))
        for label, row in held.items()
        if label != champion.label
    )
    return ReplayStanding(session_id, at, champion.label, len(trees), runs, rows), found


def record_standing(pool: PlaybookPool, standing: ReplayStanding) -> bool:
    """Keep ``standing`` beside the pool's playbooks, once a session; ``False`` when it was kept already."""

    return _keep(pool.root / _REPLAYS, standing.session_id, STANDING_FORMAT, standing)


def standings(pool: PlaybookPool) -> tuple[ReplayStanding, ...]:
    """The standings kept beside ``pool`` by the time it is read as of, the earliest first."""

    kept = (
        item for item in _kept(pool.root / _REPLAYS, ReplayStanding, STANDING_FORMAT) if written_by(pool.as_of, item.at)
    )
    return tuple(sorted(kept, key=lambda item: item.at))


def moved(previous: ReplayStanding | None, current: ReplayStanding) -> tuple[str, ...]:
    """The readings of ``current`` that moved since ``previous`` against the same champion, a line each."""

    if previous is None or previous.champion != current.champion:
        return ()
    before = previous.readings()
    return tuple(
        f"{label}: {before[label]} -> {reading}"
        for label, reading in current.readings().items()
        if label in before and before[label] != reading
    )


def standing_report(
    standing: ReplayStanding, comparisons: Sequence[SchedulingComparison], previous: ReplayStanding | None
) -> str:
    """The rerun as CLOSE writes it: what replays better and worse than the champion both ways beyond the data, what
    moved since ``previous``, and the comparisons."""

    readings = standing.readings()
    lines = [
        f"Scheduling replay at the CLOSE of {standing.session_id}, over {standing.sessions} archived sessions, "
        f"{standing.runs} runs each, against the champion {standing.champion}:",
        "",
        *(
            f"- {reading} both ways: {', '.join(label for label, found in readings.items() if found == reading) or 'none'}"
            for reading in (BETTER, WORSE, UNCLEAR)
        ),
        "",
    ]
    if previous is None or previous.champion != standing.champion:
        lines.append(f"No earlier CLOSE read the replay against {standing.champion}.")
    else:
        shifted = moved(previous, standing)
        lines.append(f"Since the CLOSE of {previous.session_id}: {'; '.join(shifted) or 'no reading moved'}.")
    return "\n".join([*lines, *(f"\n{comparison.render()}" for comparison in comparisons)])


__all__ = [
    "BETTER",
    "CLOSE_REPLAY_RUNS",
    "DIRECTION",
    "GAIN_PER_GPU_HOUR",
    "MAX_TRIALS",
    "MIN_TRIALS",
    "PAIR_SECONDS",
    "JUDGED_RUNS",
    "JUDGED_SESSIONS",
    "JUDGE_SYSTEM_PROMPT",
    "R2_POINTS",
    "STANDING_FORMAT",
    "TASK_LABELS",
    "TASK_SIZES",
    "TRIAL_FORMAT",
    "UNCLEAR",
    "WORSE",
    "Complete",
    "DecisionPoint",
    "Difference",
    "Pair",
    "ReplayJudge",
    "ReplayStanding",
    "Rung",
    "StandingRow",
    "Trial",
    "archived_points",
    "decision_points",
    "moved",
    "noise_band",
    "pairs",
    "r0",
    "r1",
    "r2",
    "r2_sessions",
    "r3",
    "record_standing",
    "record_trial",
    "replay_standing",
    "session_measures",
    "session_task",
    "settle",
    "standing_report",
    "standings",
    "trial_of",
    "trials",
]
