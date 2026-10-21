# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE: the scheduler's LLM writes frontier scorers as expressions, for the sessions after it to choose from.

:func:`authoring_prompt` is what the LLM reads: the expression syntax and the features of :mod:`rsi.expression`, the
frontier calibration over the archived sessions like this one with the pool's expressions beside ``frontier-v1``, how
this session's branches went, and the labels the pool holds. It answers with one JSON object,
``{"candidates": [{"label": ..., "expression": ..., "rationale": ...}]}``, of at most
:data:`~hyperloom_multibranch.notebook.rsi.scorers.MAX_PER_SESSION` candidates. :func:`read_candidates` turns the
answer into pool records and says why it refused any. :func:`author_scorers` is the whole step for the scheduler: given
``complete``, which asks the LLM with a system prompt and a prompt and returns its answer, it writes what it accepted
to the pool and says what it did as a :class:`~hyperloom_multibranch.notebook.rsi.session.CloseStep`. What the call
cost is the scheduler's to record, as ``UsageSource.SCORER``.

A candidate is evidence only on the sessions that start after it is written (:mod:`rsi.scorers`), so the LLM may read
the whole archive without a candidate fitted to it winning on it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.notebook.rsi.archive import SKIPPED, SearchTree, TreeArchive, similar_trees
from hyperloom_multibranch.notebook.rsi.calibration import CalibrationReport, calibrate, scorer_baseline
from hyperloom_multibranch.notebook.rsi.expression import FEATURES, MAX_CHARS, ExpressionError, ExpressionScorer
from hyperloom_multibranch.notebook.rsi.frontier import HOUR, SCORER_LABEL
from hyperloom_multibranch.notebook.rsi.scorers import (
    MAX_PER_SESSION,
    PoolRecord,
    ScorerPool,
    ScorerSource,
)
from hyperloom_multibranch.notebook.rsi.session import (
    FAILED,
    NO_PROFILE,
    STEP_ERRORS,
    WRITTEN,
    CloseStep,
    archived_history,
)

log = logging.getLogger(__name__)

#: Asks the LLM with a system prompt and a prompt, and returns its answer.
Complete = Callable[[str, str], Awaitable[str]]
MAX_RATIONALE_CHARS = 600
NO_ARCHIVE = "no_archive_configured"
NOT_ARCHIVED = "the session's search tree is not in the archive"
_MINUTE = 60.0

AUTHORING_SYSTEM_PROMPT = """You write frontier scorers for a multi-branch LLM inference optimization session.
The session runs specialist branches in parallel. Every round its judge spends the budget where a branch's frontier
score, the gain in points over the baseline expected of one more unit of budget, is highest, and stops the branches
scored lowest. A scorer is one arithmetic expression over the features you are given. It is judged by how well its
scores rank the branches by the gain each made afterwards (the Spearman correlation, round by round), and only on
sessions that start after you write it: a formula fitted to the sessions you read gains nothing. Write formulas that
rank by what foretells a branch's later gain in general, simple enough that their rationale says why.
Answer with one JSON object and nothing else:
{"candidates": [{"label": "...", "expression": "...", "rationale": "..."}]}"""


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


def authoring_prompt(tree: SearchTree, report: CalibrationReport, pool: Sequence[PoolRecord]) -> str:
    """What the LLM reads at the CLOSE of ``tree``'s session: the syntax, the features, ``report``, the session, and
    ``pool``'s labels."""

    taken = ", ".join(item.label for item in pool) or "none yet"
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
            f"A label matches {SCORER_LABEL.pattern}, names a new formula, and is none the pool holds: {taken}.",
            "",
            "How each scorer ranked the branches of the archived sessions like this one, in hindsight:",
            report.render(),
            "",
            "This session's branches:",
            *_tree_lines(tree),
            "",
            "The pool's expressions:",
            *(f"- {item.label}: {item.expression} ({item.rationale})" for item in pool),
            "",
            f"Write at most {MAX_PER_SESSION} candidates; fewer, or none, when nothing is worth trying.",
        ]
    )


@dataclass(frozen=True)
class Authored:
    """What :func:`read_candidates` made of an answer: the records it accepted, and why it refused the rest."""

    accepted: tuple[PoolRecord, ...]
    refused: tuple[str, ...] = ()


def _candidate(item: object, provenance: Mapping[str, str], now: float) -> PoolRecord:
    """``item`` as a pool record; raises ``ValueError`` saying why it is none."""

    if not isinstance(item, dict):
        raise ValueError("a candidate is an object with a label, an expression and a rationale")
    label, text, rationale = (item.get(key) for key in ("label", "expression", "rationale"))
    if not (isinstance(label, str) and isinstance(text, str) and isinstance(rationale, str)):
        raise ValueError("a candidate's label, expression and rationale are strings")
    scorer = ExpressionScorer.read(label, text)
    return PoolRecord(
        scorer.label,
        ScorerSource.EXPRESSION,
        now,
        expression=scorer.expression.text,
        rationale=rationale.strip()[:MAX_RATIONALE_CHARS],
        provenance=dict(provenance),
    )


def read_candidates(answer: str, provenance: Mapping[str, str], now: float) -> Authored:
    """The candidates of ``answer`` as pool records registered ``now``, each with ``provenance``; the first
    :data:`~hyperloom_multibranch.notebook.rsi.scorers.MAX_PER_SESSION` of them only."""

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
    for number, item in enumerate(candidates[:MAX_PER_SESSION], start=1):
        try:
            record = _candidate(item, provenance, now)
        except (ExpressionError, ValueError) as exc:
            refused.append(f"candidate {number}: {exc}")
            continue
        if any(other.label == record.label for other in accepted):
            refused.append(f"candidate {number}: the label {record.label} twice")
            continue
        accepted.append(record)
    return Authored(tuple(accepted), tuple(refused))


def _prompt(history: Sequence[SearchTree], tree: SearchTree, pool: ScorerPool) -> str:
    pooled = pool.expressions()
    rows = []
    for record in pooled:
        try:
            rows.append(scorer_baseline(ExpressionScorer.read(record.label, record.expression)))
        except ExpressionError as exc:
            log.warning("left out the pooled frontier scorer %s: %s", record.label, exc)
    report = calibrate(tuple(item.tree for item in similar_trees(history, tree.task)), baselines=rows)
    return authoring_prompt(tree, report, pooled)


async def author_scorers(
    snapshot: NotebookSnapshot,
    complete: Complete,
    *,
    model: str,
    archive: TreeArchive | None = None,
    pool: ScorerPool | None = None,
) -> CloseStep:
    """CLOSE, after the search tree is archived: ask the LLM, through ``complete``, for frontier scorers written as
    expressions, and write the ones it gives that the syntax and the pool accept to ``pool``, by default the one beside
    ``archive``, by default the environment's. ``model`` names the LLM in each record's provenance. A step that cannot
    be done says why and never stops CLOSE."""

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
    try:
        prompt = await asyncio.to_thread(_prompt, history, own, writable)
    except STEP_ERRORS as exc:
        log.warning("CLOSE could not calibrate the pool's frontier scorers: %s", exc)
        return CloseStep(FAILED, f"{type(exc).__name__}: {exc}")
    try:
        answer = await complete(AUTHORING_SYSTEM_PROMPT, prompt)
    except Exception as exc:
        log.warning("CLOSE could not ask the LLM for frontier scorers: %s", exc, exc_info=True)
        return CloseStep(FAILED, f"{type(exc).__name__}: {exc}")
    authored = read_candidates(answer, {"model": model, "session_id": profile.session_id}, snapshot.taken_at)
    refused = list(authored.refused)
    written = 0
    for record in authored.accepted:
        try:
            writable.register(record)
        except (OSError, ValueError) as exc:
            refused.append(f"{record.label}: {exc}")
            continue
        written += 1
    return CloseStep(WRITTEN if written else SKIPPED, "; ".join(refused), {"written": written, "refused": len(refused)})


__all__ = [
    "AUTHORING_SYSTEM_PROMPT",
    "MAX_RATIONALE_CHARS",
    "NOT_ARCHIVED",
    "NO_ARCHIVE",
    "Authored",
    "Complete",
    "author_scorers",
    "authoring_prompt",
    "read_candidates",
]
