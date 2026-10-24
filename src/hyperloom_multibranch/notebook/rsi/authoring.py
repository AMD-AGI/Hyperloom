# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE: the scheduler's LLM writes frontier scorers as expressions, one generation a session.

:func:`survey` sets every pooled expression beside the champion (``frontier-v1`` until one earned the place) on the
archived sessions like this one: prospectively, each on the sessions that started after it was written, which is what
decides, and in hindsight, on all of them, for ideas. :func:`feedback_for` looks back on the earlier generations:
whether each refinement ranks better than the parent it refined, on the sessions that started after it, and the rounds
where each parent to refine misranked worst, with the features that told its pick from the branch that gained the most.
:func:`authoring_prompt` is what the LLM reads: the expression syntax and the features of :mod:`rsi.expression`,
``frontier-v1`` written as an expression, both tables, how this session's branches went, the pool's expressions with
the parents each refined, the ones retired, that feedback, and the parents to refine: the champion, and the challengers
whose lead over it has the highest low end. It answers with one JSON object,
``{"candidates": [{"label": ..., "expression": ..., "rationale": ..., "parents": [...]}]}``, of at most
:data:`~hyperloom_multibranch.notebook.rsi.scorers.MAX_PER_SESSION` candidates, at most one of them with no parents: a
new idea. :func:`read_candidates` turns the answer into pool records and says why it refused any.

:func:`author_scorers` is the whole step for the scheduler: given ``complete``, which asks the LLM with a system prompt
and a prompt and returns its answer, it refuses a candidate that ranks the branches like ``frontier-v1``, a pooled or
a recently retired expression in more than :data:`DUPLICATE_SHARE` of the rounds, too many for it to lead that one by
the gate's margin, writes the rest to the pool, and retires every pooled expression clearly worse than the champion:
behind it over the evidence the gate asks for, the high end of its lead below 0. It says what it did as a
:class:`~hyperloom_multibranch.notebook.rsi.session.CloseStep`; what the call
cost is the scheduler's to record, as ``UsageSource.SCORER``. A candidate is evidence only on the sessions that start
after it is written (:mod:`rsi.scorers`), so the LLM may read the whole archive without a candidate fitted to it
winning on it.

:func:`session_challenger` is PRELUDE's call for a 1-in-4 exploration session, which the scheduler picks: the first
challenger :func:`parents` offers after the champion, from the same standings, which crowns nothing; promotion stays
:func:`~hyperloom_multibranch.notebook.rsi.session.session_frontier_scorer`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.notebook.rsi.archive import NO_ARCHIVE, SKIPPED, SearchTree, TreeArchive, similar_trees
from hyperloom_multibranch.notebook.rsi.calibration import (
    MIN_IMPROVEMENT,
    MIN_ROUNDS,
    MIN_SESSIONS,
    VARIATIONS,
    CalibrationReport,
    CalibrationSettings,
    Comparison,
    PolicyChoice,
    Registered,
    Replayed,
    calibrate,
    calibrate_against,
    ranked_alike,
    rankings,
    replayed,
    scorer_baseline,
)
from hyperloom_multibranch.notebook.rsi.expression import (
    FEATURES,
    MAX_CHARS,
    ExpressionError,
    ExpressionScorer,
    features,
)
from hyperloom_multibranch.notebook.rsi.frontier import (
    FRONTIER_V1,
    FRONTIER_V1_LABEL,
    HOUR,
    MIN_EFFECTIVE_SESSIONS,
    OBSERVATION_FLOOR_PCT,
    SCORER_LABEL,
    FrontierPolicy,
    FrontierScorer,
)
from hyperloom_multibranch.notebook.rsi.report import number, table
from hyperloom_multibranch.notebook.rsi.scorers import (
    MAX_PER_SESSION,
    PoolRecord,
    RetiredRecord,
    ScorerPool,
    ScorerSource,
    current_champion,
    scorer_named,
)
from hyperloom_multibranch.notebook.rsi.session import (
    FAILED,
    NO_PROFILE,
    STEP_ERRORS,
    WRITTEN,
    CloseStep,
    archived_history,
    named_frontier_choice,
)
from hyperloom_multibranch.notebook.rsi.similarity import task_conditions

log = logging.getLogger(__name__)

#: Asks the LLM with a system prompt and a prompt, and returns its answer.
Complete = Callable[[str, str], Awaitable[str]]
MAX_RATIONALE_CHARS = 600
#: How many parents the prompt offers to refine: the champion and the best challengers.
MAX_PARENTS = 3
#: A candidate that ranks the branches like another scorer in more than this share of the rounds is a duplicate: a
#: round's correlation moves by 2 at most, so it could not lead that scorer by MIN_IMPROVEMENT on them.
DUPLICATE_SHARE = 1 - MIN_IMPROVEMENT / 2
#: How many of the most recently retired expressions the prompt shows and a candidate is checked against.
RECENT_RETIRED = 12
#: How many rounds the prompt shows of each parent to refine, those it misranked worst, from as many sessions.
MISRANKED_ROUNDS = 3
REFINEMENT_HEADINGS = (
    "refinement",
    "parent",
    "rank correlation",
    "the parent's",
    "lead interval",
    "rounds",
    "sessions",
    "verdict",
)
#: ``frontier-v1`` under the default policy as an expression; the envelope's new-branch floor, which lifts an
#: expression's score, never lifts one ``frontier-v1`` took from the prior.
FRONTIER_V1_EXPRESSION = (
    f"max(prior_remaining_pct if prior_effective_sessions >= {MIN_EFFECTIVE_SESSIONS} else trend_pct, "
    f"{OBSERVATION_FLOOR_PCT} * observing)"
)
NOT_ARCHIVED = "the session's search tree is not in the archive"
NO_CANDIDATES = "the LLM wrote no candidate"
_MINUTE = 60.0
_SAME = 1e-9

AUTHORING_SYSTEM_PROMPT = """You write frontier scorers for a multi-branch LLM inference optimization session.
The session runs specialist branches in parallel. Every round its judge spends the budget where a branch's frontier
score, the gain in points over the baseline expected of one more unit of budget, is highest, and stops the branches
scored lowest. A scorer is one arithmetic expression over the features you are given. It is judged by how well its
scores rank the branches by the gain each made afterwards (the Spearman correlation, round by round), beside the
champion, and only on sessions that start after you write it: a formula fitted to the sessions you read gains nothing.
Each session's close is one generation. Refine the parents you are given, naming them in parents, and write at most one
candidate with no parents, a new idea. You are shown whether each earlier refinement ranked better than its parent and
where each parent misranked worst: fix what those show in general, not for those rounds, and build on the changes that
helped rather than those that did not. Write formulas that rank by what foretells a branch's later gain in general,
simple enough that their rationale says why. Answer with one JSON object and nothing else:
{"candidates": [{"label": "...", "expression": "...", "rationale": "...", "parents": ["..."]}]}"""


@dataclass(frozen=True)
class Survey:
    """Where the pool stands at a CLOSE: ``standings`` sets each pooled expression beside the champion on the sessions
    that started after it was written, ``hindsight`` on all the sessions; ``pooled`` are the records offered,
    ``retired`` the :data:`RECENT_RETIRED` most recently retired, and ``champion`` the champion's label."""

    standings: CalibrationReport
    hindsight: CalibrationReport
    pooled: tuple[PoolRecord, ...]
    retired: tuple[RetiredRecord, ...]
    champion: str


def _readable(expressions: Mapping[str, str]) -> dict[str, ExpressionScorer]:
    """The scorers of ``expressions``, by label; one that no longer reads is left out with a warning."""

    found = {}
    for label, text in expressions.items():
        try:
            found[label] = ExpressionScorer.read(label, text)
        except ExpressionError as exc:
            log.warning("left out the frontier scorer %s: %s", label, exc)
    return found


def survey(similar: Sequence[SearchTree], pool: ScorerPool) -> Survey:
    """The pool's expressions beside its champion on the archived sessions ``similar`` to this one."""

    pooled = pool.expressions()
    held = current_champion(pool)
    champion = held.scorer.label if held is not None else FRONTIER_V1_LABEL
    readable = _readable({record.label: record.expression for record in pooled})
    prospective = [
        Registered(readable[record.label], record.registered_at)
        for record in pooled
        if record.label in readable and record.label != champion
    ]
    standings = calibrate_against(similar, prospective, held)
    rows = [scorer_baseline(scorer) for scorer in readable.values()]
    hindsight = calibrate(similar, variations=VARIATIONS[:1], baselines=rows)
    return Survey(standings, hindsight, tuple(pooled), pool.retired()[-RECENT_RETIRED:], champion)


def _clearly_worse(item: Comparison) -> bool:
    return (
        item.sessions >= MIN_SESSIONS
        and item.rounds >= MIN_ROUNDS
        and item.difference_high is not None
        and item.difference_high < 0
    )


def parents(found: Survey) -> tuple[str, ...]:
    """The labels the next candidates refine, :data:`MAX_PARENTS` at most: the champion, and the pooled challengers
    whose lead over it has the highest low end, none of them clearly worse than it."""

    pooled = {record.label for record in found.pooled}
    led = [
        item
        for item in found.standings.comparisons
        if item.label in pooled and item.difference_low is not None and not _clearly_worse(item)
    ]
    best = sorted(led, key=lambda item: item.difference_low or 0.0, reverse=True)[: MAX_PARENTS - 1]
    return (found.champion, *(item.label for item in best))


def session_challenger(
    snapshot: NotebookSnapshot, history: Sequence[SearchTree], defaults: FrontierPolicy, pool: ScorerPool | None = None
) -> PolicyChoice | None:
    """The scorer a 1-in-4 exploration session scores with, under ``defaults``: the first challenger :func:`parents`
    offers after the champion, on the archived sessions like this one, from ``pool``, by default the one beside the
    environment's archive. ``None`` when the session has no profile, there is no pool, or no challenger qualifies.
    It crowns nothing."""

    target = pool if pool is not None else ScorerPool.from_env()
    if snapshot.profile is None or target is None:
        return None
    found = survey(tuple(item.tree for item in similar_trees(history, task_conditions(snapshot.profile))), target)
    label = next((item for item in parents(found) if item != found.champion), None)
    compared = None if label is None else found.standings.comparison(label)
    if label is None or compared is None:
        return None
    lead = f"{number(compared.difference_low)} to {number(compared.difference_high)}"
    evidence = f"{label}: exploration challenger, its lead over {found.champion} {lead}"
    return named_frontier_choice(label, defaults, evidence, target)


@dataclass(frozen=True)
class Refinement:
    """A pooled expression beside ``parent``, a scorer it refined: ``compared``, labelled with the expression, on the
    same rounds of the sessions that started after it was written."""

    parent: str
    compared: Comparison


@dataclass(frozen=True)
class Misranked:
    """A round a scorer misranked: ``picked``, the branch it scored highest, went on to gain less than ``best``, the
    branch that gained the most; ``differed`` holds each feature that told the two apart, with its value for each."""

    picked: Replayed
    best: Replayed
    differed: tuple[tuple[str, float, float], ...] = ()


@dataclass(frozen=True)
class Feedback:
    """What the earlier generations show the next: each refinement beside its parent, and the rounds each parent to
    refine misranked worst, by label, or why its rounds could not be replayed."""

    refinements: tuple[Refinement, ...] = ()
    misranked: Mapping[str, tuple[Misranked, ...]] = field(default_factory=dict)
    failures: Mapping[str, str] = field(default_factory=dict)


def _beside_parent(
    similar: Sequence[SearchTree], parent: FrontierScorer, children: Sequence[Registered]
) -> CalibrationReport:
    """``children`` beside ``parent`` on ``similar``, each on the sessions that started after it was written."""

    since = min((item.since for item in children if item.since is not None), default=None)
    rows = (scorer_baseline(parent, since), *(scorer_baseline(item.scorer, item.since) for item in children))
    return calibrate(similar, variations=(), baselines=rows, settings=CalibrationSettings(reference=parent.label))


def _refinements(
    similar: Sequence[SearchTree], pool: ScorerPool, pooled: Sequence[PoolRecord]
) -> tuple[Refinement, ...]:
    """Each of ``pooled`` that refines parents beside each of them, by parent: a parent that is no longer to be found,
    or no longer reads, is left out."""

    readable = _readable({record.label: record.expression for record in pooled if record.parents})
    children: dict[str, list[Registered]] = {}
    for record in pooled:
        for parent in record.parents if record.label in readable else ():
            children.setdefault(parent, []).append(Registered(readable[record.label], record.registered_at))
    found: list[Refinement] = []
    for label, registered in children.items():
        try:
            parent = scorer_named(label, pool)
        except ExpressionError as exc:
            log.warning("left out the refinements of %s: %s", label, exc)
            continue
        if parent is not None:
            found += (Refinement(label, item) for item in _beside_parent(similar, parent, registered).comparisons)
    return tuple(found)


def _shortfall(picked: Replayed, best: Replayed) -> float:
    return best.gain_after_pct - picked.gain_after_pct


def _differed(picked: Replayed, best: Replayed) -> tuple[tuple[str, float, float], ...]:
    mine = features(picked.branch, picked.t_s, picked.context, picked.readings)
    theirs = features(best.branch, best.t_s, best.context, best.readings)
    return tuple((name, mine[name], theirs[name]) for name in FEATURES if abs(mine[name] - theirs[name]) > _SAME)


def _misranked(similar: Sequence[SearchTree], scorer: FrontierScorer) -> tuple[Misranked, ...]:
    """The rounds of ``similar`` where the branch ``scorer`` scored highest fell furthest short of the branch that gained
    the most, the worst of each session, :data:`MISRANKED_ROUNDS` at most, the worst first. Of branches it scored alike,
    the one that gained the most is its pick."""

    worst: dict[str, tuple[Replayed, Replayed]] = {}
    for branches in replayed(similar, scorer):
        top = max(item.score for item in branches)
        picked = max((item for item in branches if item.score >= top - _SAME), key=lambda item: item.gain_after_pct)
        best = max(branches, key=lambda item: item.gain_after_pct)
        held = worst.get(picked.session_id)
        if _shortfall(picked, best) > (_shortfall(*held) if held is not None else _SAME):
            worst[picked.session_id] = (picked, best)
    ranked = sorted(worst.values(), key=lambda pair: _shortfall(*pair), reverse=True)[:MISRANKED_ROUNDS]
    return tuple(Misranked(picked, best, _differed(picked, best)) for picked, best in ranked)


def feedback_for(similar: Sequence[SearchTree], pool: ScorerPool, found: Survey) -> Feedback:
    """What the earlier generations of ``pool`` show on the archived sessions ``similar`` to this one: each pooled
    refinement beside its parents, and the rounds each of :func:`parents` misranked worst."""

    misranked: dict[str, tuple[Misranked, ...]] = {}
    failures: dict[str, str] = {}
    for label in parents(found):
        try:
            scorer = scorer_named(label, pool)
            if scorer is not None:
                misranked[label] = _misranked(similar, scorer)
        except (ArithmeticError, ValueError) as exc:
            failures[label] = f"{type(exc).__name__}: {exc}"
    return Feedback(_refinements(similar, pool, found.pooled), misranked, failures)


def _tree_lines(tree: SearchTree) -> list[str]:
    lines = [
        f"Session {tree.session_id}: a budget of {tree.budget_seconds / HOUR:.1f} h, the best gain "
        f"{tree.final_gain_pct:+.2f}% over the baseline."
    ]
    for branch in tree.branches:
        if branch.name == INTEGRATION_BRANCH:
            continue
        closed = "ran to the end" if branch.closed_s is None else f"stopped at {branch.closed_s / _MINUTE:.0f} min"
        why = f" ({branch.stop_reason})" if branch.stop_reason else ""
        lines.append(
            f"- {branch.name}: opened at {branch.opened_s / _MINUTE:.0f} min from {branch.fork_gain_pct:+.2f}%, "
            f"{closed}{why}; {len(branch.test_times)} tests, {len(branch.commits)} commits, "
            f"ended at {branch.final_gain_pct:+.2f}%"
        )
    return lines


def _pooled_line(record: PoolRecord) -> str:
    refines = f"refines {', '.join(record.parents)}" if record.parents else "a new idea"
    return f"- {record.label}: {record.expression} ({refines}; {record.rationale})"


def _verdict(item: Comparison) -> str:
    """Whether the refinement ``item`` ranks better than its parent, on the evidence the gate asks for."""

    if item.sessions < MIN_SESSIONS or item.rounds < MIN_ROUNDS:
        return "too little evidence"
    if item.difference_low is not None and item.difference_low > 0:
        return "better"
    if item.difference_high is not None and item.difference_high < 0:
        return "worse"
    return "no clear difference"


def _refinement_lines(found: Sequence[Refinement]) -> list[str]:
    heading = (
        "Whether each change helped: each refinement beside the parent it refined, on the same rounds of the sessions "
        "that started after it was written."
    )
    shown = [item for item in found if item.compared.rounds]
    rows = [
        (
            item.compared.label,
            item.parent,
            number(item.compared.rank_correlation),
            number(item.compared.current_correlation),
            f"{number(item.compared.difference_low)} to {number(item.compared.difference_high)}",
            item.compared.rounds,
            item.compared.sessions,
            _verdict(item.compared),
        )
        for item in shown
    ]
    waiting = ", ".join(dict.fromkeys(item.compared.label for item in found if not item.compared.rounds))
    return [
        heading,
        *(["", *table(REFINEMENT_HEADINGS, rows)] if rows else ["- none yet"]),
        *(["", f"Too new to tell, no session like this one started after them: {waiting}."] if waiting else []),
    ]


def _misranked_line(label: str, item: Misranked) -> str:
    picked, best = item.picked, item.best
    differed = ", ".join(f"{name} {number(mine)} / {number(theirs)}" for name, mine, theirs in item.differed)
    return (
        f"- {label}, session {picked.session_id} at {picked.t_s / _MINUTE:.0f} min: it scored {picked.branch.name} "
        f"highest ({number(picked.score)}), which gained {picked.gain_after_pct:+.2f}% after; {best.branch.name} "
        f"gained {best.gain_after_pct:+.2f}% (scored {number(best.score)}). Where they differed, "
        f"{picked.branch.name} / {best.branch.name}: {differed or 'nowhere'}."
    )


def _feedback_lines(feedback: Feedback) -> list[str]:
    lines = [
        *_refinement_lines(feedback.refinements),
        "",
        "Where each parent misranked worst on the archived sessions like this one, a round a session: the branch it "
        "scored highest beside the branch that went on to gain the most, and the features where the two differed, "
        "the first's value then the second's.",
    ]
    for label, found in feedback.misranked.items():
        none = f"- {label}: nowhere, the branch it scored highest gained the most in every round"
        lines += [_misranked_line(label, item) for item in found] or [none]
    lines += [f"- {label}: its rounds could not be replayed: {why}" for label, why in feedback.failures.items()]
    return [*lines, ""]


def authoring_prompt(tree: SearchTree, found: Survey, feedback: Feedback | None = None) -> str:
    """What the LLM reads at the CLOSE of ``tree``'s session: the syntax, the features, ``frontier-v1``, where the pool
    stands, the session, the pool's expressions and the ones retired, ``feedback`` when given, and the parents."""

    taken = ", ".join([*(record.label for record in found.pooled), *(item.label for item in found.retired)])
    retired = [f"- {item.label}: {item.expression} ({item.evidence})" for item in found.retired] or ["- none yet"]
    return "\n".join(
        [
            "Syntax: numbers, + - * / **, comparisons, and, or, not (true is 1, false 0), a if test else b, and the "
            "functions min and max (2 to 8 arguments), abs, sqrt, log1p and clip(x, low, high); at most "
            f"{MAX_CHARS} characters. Nothing else: no other names, no attributes, no strings.",
            "",
            "Features, each a number for the branch as it stood when it was scored:",
            *(f"- {name}: {meaning}" for name, meaning in FEATURES.items()),
            "",
            "Whatever a scorer says, the session then holds every score to the same envelope: a young branch keeps a "
            "floor, a candidate waiting for its confirm lifts the score to its gain, the roofline and Amdahl ceilings "
            "cap it, and it is never negative. A formula needs none of these.",
            "",
            f"A label matches {SCORER_LABEL.pattern}, names a new formula, and is none the pool holds: "
            f"{taken or 'none yet'}.",
            "",
            f"The built-in scorer {FRONTIER_V1_LABEL}, as an expression: {FRONTIER_V1_EXPRESSION}; the young floor "
            "never lifts a score it takes from the prior. In the tables, current is it under each session's policy.",
            "",
            f"The champion, the scorer to beat: {found.champion}. A candidate takes its place when it leads it by "
            f"{MIN_IMPROVEMENT} of rank correlation on the same rounds, the low end of its interval above 0, over at "
            f"least {MIN_SESSIONS} sessions and {MIN_ROUNDS} rounds, all of them sessions that started after it was "
            "written. Where the pool stands on that count, which decides:",
            found.standings.render(),
            "",
            "Every scorer on all the archived sessions like this one, in hindsight: for ideas, it decides nothing.",
            found.hindsight.render(),
            "",
            "This session's branches:",
            *_tree_lines(tree),
            "",
            "The pool's expressions:",
            *(_pooled_line(record) for record in found.pooled),
            "",
            "Retired, clearly worse than the champion; a candidate that ranks the branches like one is refused:",
            *retired,
            "",
            *(_feedback_lines(feedback) if feedback is not None else ()),
            f"Parents to refine: {', '.join(parents(found))}. Write at most {MAX_PER_SESSION} candidates, each naming "
            "in parents the labels it refines, at most one of them a new idea with none; fewer, or none, when nothing "
            "is worth trying.",
        ]
    )


@dataclass(frozen=True)
class Authored:
    """What :func:`read_candidates` made of an answer: the records it accepted, and why it refused the rest."""

    accepted: tuple[PoolRecord, ...]
    refused: tuple[str, ...] = ()


def _candidate(item: object, provenance: Mapping[str, str], now: float, known: Collection[str]) -> PoolRecord:
    """``item`` as a pool record, its parents those of ``known``; raises ``ValueError`` saying why it is none."""

    if not isinstance(item, dict):
        raise ValueError("a candidate is an object with a label, an expression and a rationale")
    label, text, rationale = (item.get(key) for key in ("label", "expression", "rationale"))
    if not (isinstance(label, str) and isinstance(text, str) and isinstance(rationale, str)):
        raise ValueError("a candidate's label, expression and rationale are strings")
    named = item.get("parents") or []
    if not isinstance(named, list) or not all(isinstance(parent, str) for parent in named):
        raise ValueError("a candidate's parents are a list of labels")
    scorer = ExpressionScorer.read(label, text)
    return PoolRecord(
        scorer.label,
        ScorerSource.EXPRESSION,
        now,
        expression=scorer.expression.text,
        rationale=rationale.strip()[:MAX_RATIONALE_CHARS],
        provenance=dict(provenance),
        parents=tuple(dict.fromkeys(parent for parent in named if parent in known and parent != scorer.label)),
    )


def read_candidates(answer: str, provenance: Mapping[str, str], now: float, known: Collection[str] = ()) -> Authored:
    """The candidates of ``answer`` as pool records registered ``now``, each with ``provenance`` and the parents it
    names among ``known``; the first :data:`~hyperloom_multibranch.notebook.rsi.scorers.MAX_PER_SESSION` of them only,
    and one new idea at most."""

    start, end = answer.find("{"), answer.rfind("}")
    try:
        document = json.loads(answer[start : end + 1]) if 0 <= start < end else None
    except json.JSONDecodeError:
        document = None
    candidates = document.get("candidates") if isinstance(document, dict) else None
    if not isinstance(candidates, list):
        return Authored((), ("the answer holds no JSON object with a list of candidates",))
    accepted: list[PoolRecord] = []
    refused = []
    if len(candidates) > MAX_PER_SESSION:
        refused.append(f"{len(candidates) - MAX_PER_SESSION} past the first {MAX_PER_SESSION}")
    for position, item in enumerate(candidates[:MAX_PER_SESSION], start=1):
        try:
            record = _candidate(item, provenance, now, known)
        except (ExpressionError, ValueError) as exc:
            refused.append(f"candidate {position}: {exc}")
            continue
        if any(other.label == record.label for other in accepted):
            refused.append(f"candidate {position}: the label {record.label} twice")
        elif not record.parents and any(not other.parents for other in accepted):
            refused.append(f"candidate {position}: a second new idea, with no parents")
        else:
            accepted.append(record)
    return Authored(tuple(accepted), tuple(refused))


def _distinct(authored: Authored, similar: Sequence[SearchTree], found: Survey) -> Authored:
    """``authored`` less the candidates that rank the branches like ``frontier-v1``, a pooled or a recently retired
    expression, or an earlier candidate, in more than :data:`DUPLICATE_SHARE` of the rounds."""

    known = {item.label: item.expression for item in found.retired}
    known.update((record.label, record.expression) for record in found.pooled)
    offered: dict[str, FrontierScorer] = {FRONTIER_V1_LABEL: FRONTIER_V1, **_readable(known)}
    ranked = {label: rankings(similar, scorer) for label, scorer in offered.items()}
    kept, refused = [], list(authored.refused)
    for record in authored.accepted:
        mine = rankings(similar, ExpressionScorer.read(record.label, record.expression))
        twin = next((label for label, theirs in ranked.items() if ranked_alike(mine, theirs) > DUPLICATE_SHARE), None)
        if twin is not None:
            refused.append(f"{record.label}: ranks the branches like {twin}")
            continue
        kept.append(record)
        ranked[record.label] = mine
    return Authored(tuple(kept), tuple(refused))


def _retire(pool: ScorerPool, found: Survey, now: float) -> int:
    """Retire every pooled expression the standings find clearly worse than the champion; how many were retired."""

    pooled = {record.label: record for record in found.pooled if record.label != found.champion}
    retired = 0
    for item in found.standings.comparisons:
        record = pooled.get(item.label)
        if record is None or not _clearly_worse(item):
            continue
        evidence = (
            f"behind {found.champion} by {number(item.difference_low)} to {number(item.difference_high)} over "
            f"{item.rounds} rounds of {item.sessions} sessions"
        )
        try:
            pool.retire(RetiredRecord(item.label, now, evidence, record.expression))
        except (OSError, ValueError) as exc:
            log.warning("could not retire the frontier scorer %s: %s", item.label, exc)
            continue
        retired += 1
    return retired


def _register(pool: ScorerPool, authored: Authored) -> Authored:
    """``authored``'s candidates as the pool took them, and why it took none of the rest."""

    written, refused = [], list(authored.refused)
    for record in authored.accepted:
        try:
            written.append(pool.register(record))
        except (OSError, ValueError) as exc:
            refused.append(f"{record.label}: {exc}")
    return Authored(tuple(written), tuple(refused))


async def author_scorers(
    snapshot: NotebookSnapshot,
    complete: Complete,
    *,
    model: str,
    archive: TreeArchive | None = None,
    pool: ScorerPool | None = None,
) -> CloseStep:
    """CLOSE, after the search tree is archived: one generation of frontier scorers written as expressions. Ask the LLM,
    through ``complete``, write what the syntax, the pool and the duplicate check accept to ``pool``, by default the
    one beside ``archive``, by default the environment's, and retire what the standings find clearly worse than the
    champion. ``model`` names the LLM in each record's provenance. A step that cannot be done says why and never stops
    CLOSE."""

    profile = snapshot.profile
    if profile is None:
        return CloseStep(SKIPPED, NO_PROFILE)
    target = archive if archive is not None else TreeArchive.from_env()
    if target is None:
        return CloseStep(SKIPPED, NO_ARCHIVE)
    history = archived_history(target)
    own = next((tree for tree in history if tree.session_id == profile.session_id), None)
    if own is None:
        return CloseStep(SKIPPED, NOT_ARCHIVED)
    writable = pool if pool is not None else ScorerPool.beside(target)
    similar = tuple(item.tree for item in similar_trees(history, own.task))
    try:
        found = await asyncio.to_thread(survey, similar, writable)
        feedback = await asyncio.to_thread(feedback_for, similar, writable, found)
    except STEP_ERRORS as exc:
        log.warning("CLOSE could not calibrate the pool's frontier scorers: %s", exc)
        return CloseStep(FAILED, f"{type(exc).__name__}: {exc}")
    try:
        answer = await complete(AUTHORING_SYSTEM_PROMPT, authoring_prompt(own, found, feedback))
    except Exception as exc:
        log.warning("CLOSE could not ask the LLM for frontier scorers: %s", exc, exc_info=True)
        return CloseStep(FAILED, f"{type(exc).__name__}: {exc}")
    known = {record.label for record in writable.records()} | {FRONTIER_V1_LABEL, found.champion}
    provenance = {"model": model, "session_id": profile.session_id}
    authored = read_candidates(answer, provenance, snapshot.taken_at, known)
    distinct = await asyncio.to_thread(_distinct, authored, similar, found)
    done = _register(writable, distinct)
    retired = _retire(writable, found, snapshot.taken_at)
    counts = {"written": len(done.accepted), "refused": len(done.refused), "retired": retired}
    reason = "; ".join(done.refused) or ("" if done.accepted else NO_CANDIDATES)
    return CloseStep(WRITTEN if done.accepted else SKIPPED, reason, counts)


__all__ = [
    "AUTHORING_SYSTEM_PROMPT",
    "DUPLICATE_SHARE",
    "FRONTIER_V1_EXPRESSION",
    "MAX_PARENTS",
    "MAX_RATIONALE_CHARS",
    "MISRANKED_ROUNDS",
    "NOT_ARCHIVED",
    "NO_ARCHIVE",
    "NO_CANDIDATES",
    "RECENT_RETIRED",
    "REFINEMENT_HEADINGS",
    "Authored",
    "Complete",
    "Feedback",
    "Misranked",
    "Refinement",
    "Survey",
    "author_scorers",
    "authoring_prompt",
    "feedback_for",
    "parents",
    "read_candidates",
    "session_challenger",
    "survey",
]
