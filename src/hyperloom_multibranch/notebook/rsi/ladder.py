# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The evaluation ladder a new playbook climbs, from what costs nothing to what costs sessions.

* R0, :func:`r0`: the static check (:func:`~hyperloom_multibranch.notebook.rsi.playbook.problems`).
* R1, :func:`r1`: the scheduling replay of :mod:`rsi.replay`, no LLM and no GPU. A change to a number passes when its
  paired gain over the champion stays above 0 across the archived sessions both ways beyond the data; a change to a
  rule cannot be replayed and goes on to R2.
* R2, :func:`r2`: the judge's own decision points, the prompts the archived sessions' judge was asked, asked again
  with the candidate's playbook and the champion's. A point counts where its outcome is known: a stop at most
  :data:`EARLY_STOP_TESTS` judged tests past a commit, which the playbook should have prevented, and a stop at or past
  the stall bar, which it should keep. The candidate passes when it gets more of them right than the champion.
* R3, :func:`r3`: the challenger sessions, one in four, each recorded at CLOSE as a :class:`Trial` with its decision
  measures (:func:`session_measures`). A playbook is crowned once at least :data:`MIN_TRIALS` of its sessions beat the
  champion's on the measure it named by more than the noise band, while their gain per GPU hour stays within the band
  of the champion's; it is retired once :data:`MAX_TRIALS` have not. The noise band of a measure is the spread of the
  champion's own sessions of the same model type (:func:`noise_band`): until there are :data:`MIN_TRIALS` of them,
  nothing is crowned.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field

from hyperloom_multibranch.contracts.notebook.vocab import CommitOrigin, LineageKind
from hyperloom_multibranch.notebook.codec import decode, encode
from hyperloom_multibranch.notebook.local.files import encode_json, publish_once
from hyperloom_multibranch.notebook.rsi.archive import SearchTree
from hyperloom_multibranch.notebook.rsi.history import MODEL_TYPE_LABEL, judged_tests
from hyperloom_multibranch.notebook.rsi.playbook import Crowned, Playbook, PlaybookPool, Retired, changes, problems
from hyperloom_multibranch.notebook.rsi.replay import OPTIMISTIC, PESSIMISTIC, REPLAY_RUNS, compare
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
#: How many of the judge's decision points R2 asks again, each once under either playbook.
R2_POINTS = 15
#: The key of a judge reply's stop orders.
_STOP_KEY = "stop"
_TRIALS = "trials"
_JSON = "*.json"
_ENCODING = "utf-8"
_MINUTE = 60.0
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
        found = compare(trees, pair, runs=runs, beyond=beyond, seed=seed, baseline=champion.label)
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


def _stopped(reply: str) -> tuple[str, ...]:
    """The branches a judge reply stops; none when it holds no JSON object with a stop list."""

    start, end = reply.find("{"), reply.rfind("}")
    try:
        document = json.loads(reply[start : end + 1]) if 0 <= start < end else None
    except json.JSONDecodeError:
        return ()
    orders = document.get(_STOP_KEY) if isinstance(document, dict) else None
    return tuple(str(item.get("branch")) for item in orders or () if isinstance(item, Mapping))


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
        for name in _stopped(reply):
            when = float(at) - began
            found = _stall_at(tree, name, when)
            if found is None:
                continue
            stalled, committed = found
            early = committed and stalled <= EARLY_STOP_TESTS
            if early or stalled >= bar:
                points.append(DecisionPoint(tree.session_id, name, when, stalled, early, prompt))
    return tuple(points)


def _asked(point: DecisionPoint, playbook: Playbook) -> str:
    return f"{point.prompt}\n\n{_CANDIDATE_SECTION}\n{playbook.render()}"


async def r2(
    candidate: Playbook,
    champion: Playbook,
    points: Sequence[DecisionPoint],
    complete: Complete,
    system: str,
    *,
    limit: int = R2_POINTS,
) -> Rung:
    """R2: the decision ``points`` asked again under ``candidate`` and ``champion``, through ``complete`` with the
    judge's ``system`` prompt; the first ``limit`` of them, the stops to undo and those to keep in turn."""

    undo = [point for point in points if point.keep]
    keep = [point for point in points if not point.keep]
    chosen = [point for pair in zip(undo, keep) for point in pair] + undo[len(keep) :] + keep[len(undo) :]
    chosen = chosen[:limit]
    if not chosen:
        return Rung("R2", True, "no archived decision point with a known outcome", skipped=True)
    right = {candidate.label: 0, champion.label: 0}
    for point in chosen:
        for playbook in (candidate, champion):
            stops = point.branch in _stopped(await complete(system, _asked(point, playbook)))
            right[playbook.label] += int(stops != point.keep)
    passed = right[candidate.label] > right[champion.label]
    said = f"right on {right[candidate.label]} of {len(chosen)} points, {champion.label} on {right[champion.label]}"
    return Rung("R2", passed, said)


@dataclass(frozen=True)
class Trial:
    """One session under ``played``, with ``champion`` the champion then: its decision measures and its gain per GPU
    hour; a measure is missing where the session gave it nothing to count."""

    session_id: str
    played: str
    champion: str
    model_type: str
    measures: Mapping[str, float] = field(default_factory=dict)


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


def record_trial(pool: PlaybookPool, trial: Trial) -> bool:
    """Keep ``trial`` beside the pool's playbooks, once a session; ``False`` when it was kept already."""

    path = pool.root / _TRIALS / f"{re.sub(r'[^A-Za-z0-9._-]+', '-', trial.session_id)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return publish_once(path, encode_json({"format": TRIAL_FORMAT, "record": encode(trial)}))


def trials(pool: PlaybookPool) -> tuple[Trial, ...]:
    found = []
    for path in sorted((pool.root / _TRIALS).glob(_JSON)):
        try:
            document = json.loads(path.read_text(encoding=_ENCODING))
            if document.get("format") == TRIAL_FORMAT:
                found.append(decode(Trial, document.get("record")))
        except (OSError, KeyError, TypeError, ValueError):
            continue
    return tuple(found)


def noise_band(sessions: Sequence[Trial], measure: str) -> float | None:
    """The spread of ``measure`` over ``sessions``, its standard deviation; ``None`` with fewer than
    :data:`MIN_TRIALS` that measured it."""

    values = [trial.measures[measure] for trial in sessions if measure in trial.measures]
    return statistics.stdev(values) if len(values) >= MIN_TRIALS else None


def _mean(sessions: Sequence[Trial], measure: str) -> float | None:
    values = [trial.measures[measure] for trial in sessions if measure in trial.measures]
    return statistics.fmean(values) if values else None


def r3(candidate: Playbook, champion: Playbook, kept: Sequence[Trial]) -> Rung:
    """R3: ``candidate``'s sessions beside ``champion``'s of the same model types, on the measure it named."""

    mine = [trial for trial in kept if trial.played == candidate.label]
    types = {trial.model_type for trial in mine}
    theirs = [trial for trial in kept if trial.played == champion.label and trial.model_type in types]
    measure = candidate.measure
    if measure not in DIRECTION:
        return Rung("R3", False, f"{candidate.label} names no decision measure")
    if len(mine) < MIN_TRIALS:
        return Rung("R3", False, f"{len(mine)} of its {MIN_TRIALS} trial sessions so far", skipped=True)
    band, floor_band = noise_band(theirs, measure), noise_band(theirs, GAIN_PER_GPU_HOUR)
    if band is None or floor_band is None:
        return Rung("R3", False, f"too few sessions of {champion.label} to know the noise band", skipped=True)
    ours, base = _mean(mine, measure), _mean(theirs, measure)
    rate, base_rate = _mean(mine, GAIN_PER_GPU_HOUR), _mean(theirs, GAIN_PER_GPU_HOUR)
    if ours is None or base is None or rate is None or base_rate is None:
        return Rung("R3", False, f"its sessions did not measure {measure}", skipped=True)
    better = DIRECTION[measure] * (ours - base)
    passed = better > band and rate >= base_rate - floor_band
    said = (
        f"{measure} {ours:.3g} against {base:.3g} over {len(mine)} and {len(theirs)} sessions, band {band:.3g}; "
        f"gain per GPU hour {rate:.3g} against {base_rate:.3g}, band {floor_band:.3g}"
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


def trial_of(tree: SearchTree, audit: StopAudit, played: str, champion: str) -> Trial:
    model_type = tree.task.labels.get(MODEL_TYPE_LABEL, "")
    return Trial(tree.session_id, played, champion, model_type, session_measures(tree, audit))


__all__ = [
    "DIRECTION",
    "GAIN_PER_GPU_HOUR",
    "MAX_TRIALS",
    "MIN_TRIALS",
    "R2_POINTS",
    "TRIAL_FORMAT",
    "Complete",
    "DecisionPoint",
    "Rung",
    "Trial",
    "decision_points",
    "noise_band",
    "r0",
    "r1",
    "r2",
    "r3",
    "record_trial",
    "session_measures",
    "settle",
    "trial_of",
    "trials",
]
